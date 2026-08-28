import argparse
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from load_dataset import load_dataset

logging.basicConfig(level=logging.INFO)

WINDOW_SIZE = 6
STRIDE = 1

LANDMARKS = [
    10, 34, 35, 36, 47, 50, 53, 67, 69, 70, 100, 101, 104, 108, 109, 111, 116,
    117, 118, 119, 121, 123, 124, 126, 127, 139, 143, 147, 151, 187, 189, 203,
    205, 206, 207, 216, 222, 228, 230, 234, 244, 264, 266, 276, 280, 282, 283,
    299, 300, 329, 330, 337, 338, 340, 346, 347, 348, 353, 355, 368, 371, 372,
    411, 417, 423, 425, 426, 427, 436, 441, 444, 446, 448, 450, 452, 454, 464,
]

# Only landmarks present in LANDMARKS are used; empty regions are skipped at runtime.
LANDMARK_REGIONS = {
    # High Priority Zones (Best for rPPG)
    "high_prio_forehead": [10, 67, 69, 104, 108, 109, 151, 299, 337, 338],
    "high_prio_left_cheek": [36, 47, 50, 100, 101, 116, 117, 118, 119, 123, 126, 147, 187, 203, 205,
                             206, 207, 216],
    "high_prio_right_cheek": [266, 280, 329, 330, 346, 347, 348, 355, 371, 411, 423, 425, 426, 427,
                              436]
}

# Pre-compute: for each region, the positional indices into sig (indexed by LANDMARKS order).
# Regions with no overlap with LANDMARKS are excluded.
_LM_TO_IDX = {lm: i for i, lm in enumerate(LANDMARKS)}
_LM_SET = set(LANDMARKS)
_REGION_SIG_INDICES = {
    name: [_LM_TO_IDX[lm] for lm in lms if lm in _LM_SET]
    for name, lms in LANDMARK_REGIONS.items()
    if any(lm in _LM_SET for lm in lms)
}

SUPPORTED_METHODS = ['green', 'pos', 'omit', 'chrom']


def extract_background_signals(videofilepath, bg_fraction=0.1):
    """Return mean RGB for top-left and top-right corner regions across all frames.

    Returns two arrays of shape [num_frames, 3].
    """
    cap = cv2.VideoCapture(str(videofilepath))
    bg1, bg2 = [], []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        bh = max(1, int(h * bg_fraction))
        bw = max(1, int(w * bg_fraction))
        rgb = frame[..., ::-1].astype(np.float64)
        bg1.append(rgb[:bh, :bw].mean(axis=(0, 1)))
        bg2.append(rgb[:bh, w - bw:].mean(axis=(0, 1)))
    cap.release()
    return np.array(bg1), np.array(bg2)


def _compute_bvp(windowed_sig, fps, method, method_map, RGB_sig_to_BVP):
    """Convert windowed RGB signal to BVP for the given method."""
    if method == 'pos':
        return RGB_sig_to_BVP(windowed_sig, fps=fps, device_type='cpu',
                               method=method_map['pos'], params={'fps': fps})
    return RGB_sig_to_BVP(windowed_sig, fps=fps, device_type='cpu', method=method_map[method])


def _compute_snr(bvp_windows, bpm_windows, fps, Welch):
    """Per-window signal-to-noise ratio (dB) of a BVP signal.

    Self-referenced against each estimator's own dominant frequency (no ground-truth
    HR needed), following the de Haan et al. (2013) formulation used by pyVHR's
    ``get_SNR``: power within +/-0.2 Hz of the peak and its first harmonic vs. the
    rest of the physiological band.
    """
    interv_hz = 0.2  # +/- 0.2 Hz band window around HR peak and harmonic
    snr = np.zeros(len(bvp_windows), dtype=np.float32)

    for i, (bvp_win, bpm_win) in enumerate(zip(bvp_windows, bpm_windows)):
        bvp_win = np.atleast_2d(bvp_win)
        bpm_win = np.atleast_1d(bpm_win)

        if bvp_win.size == 0 or bpm_win.size == 0:
            continue

        # pfreqs: (n_freqs,) in Hz | power: (n_channels, n_freqs) or (n_freqs,)
        pfreqs, power = Welch(bvp_win, fps)

        # Ensure power has at least 2 dimensions: (n_channels, n_freqs)
        if power.ndim == 1:
            power = power[np.newaxis, :]

        # Average PSD across channels if multiple channels exist per window
        avg_power = np.mean(power, axis=0)

        win_snr = np.zeros(len(bpm_win), dtype=np.float32)

        for e, peak_bpm in enumerate(bpm_win):
            peak_hz = peak_bpm / 60.0  # Convert peak BPM to Hz to match pfreqs

            # Mask fundamental (f0) and first harmonic (2*f0) in Hz
            sig_mask = (np.abs(pfreqs - peak_hz) <= interv_hz) | \
                       (np.abs(pfreqs - 2 * peak_hz) <= interv_hz)

            sig_power = avg_power[sig_mask].sum()
            noise_power = avg_power[~sig_mask].sum()

            if noise_power > 1e-8 and sig_power > 0:
                win_snr[e] = 10 * np.log10(sig_power / noise_power)
            else:
                win_snr[e] = 0.0

        snr[i] = np.median(win_snr)

    return snr.tolist()


def write_video(output_path, images, fps):
    height, width = images[0].shape[:2]
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height), True)
    for image in images:
        video.write(image[..., ::-1].copy())
    video.release()


def process_video(instance, method, calc_bpm=False, landmarks_video_dir=None,
                  bg_subtract=False, per_region=False):
    videofilepath = instance.path
    video_type = instance.annotation.get_label("authenticity")
    try:
        from pyVHR.extraction import SkinExtractionConvexHull
        from pyVHR.extraction.utils import get_fps, sig_windowing
        from pyVHR.extraction.sig_processing import SignalProcessing
        from pyVHR.BPM.BPM import BVP_to_BPM, BPM_median
        from pyVHR.BPM.utils import Welch
        from pyVHR.analysis.pipeline import RGB_sig_to_BVP
        from pyVHR.BVP.methods import cpu_GREEN, cpu_POS, cpu_OMIT, cpu_CHROM

        method_map = {
            'green': cpu_GREEN, 'pos': cpu_POS, 'omit': cpu_OMIT, 'chrom': cpu_CHROM,
        }

        sig_processing = SignalProcessing()
        sig_processing.set_landmarks(LANDMARKS)
        sig_processing.set_square_patches_side(28.0)
        sig_processing.choose_cuda_device(0)
        sig_processing.skin_extractor = SkinExtractionConvexHull('GPU')

        if landmarks_video_dir:
            sig_processing.set_visualize_skin_and_landmarks(
                visualize_skin=True, visualize_landmarks=True,
                visualize_landmarks_number=True, visualize_patch=True,
            )

        logging.info(f"Processing video: {videofilepath}")

        # sig shape from pyVHR: (num_patches, num_frames, 3)
        sig = sig_processing.extract_patches(str(videofilepath), "squares", "mean")
        fps = get_fps(str(videofilepath))

        if landmarks_video_dir:
            landmarks_video_path = os.path.join(
                landmarks_video_dir, videofilepath.stem + '_landmarks.mp4'
            )
            write_video(landmarks_video_path, sig_processing.get_visualize_patches(), fps)

        timesES = None
        if per_region:
            # For each region, compute BVP independently.
            # sig shape from pyVHR: (num_frames, num_patches, 3)
            region_bvps = {}
            for region_name, sig_indices in _REGION_SIG_INDICES.items():
                sig_region = sig[:, sig_indices, :].mean(axis=1, keepdims=True)  # (num_frames, 1, 3)
                windowed_region, timesES = sig_windowing(sig_region, WINDOW_SIZE, STRIDE, fps)
                region_bvps[region_name] = _compute_bvp(
                    windowed_region, fps, method, method_map, RGB_sig_to_BVP,
                )
            bvp = region_bvps
        else:
            windowed_sig, timesES = sig_windowing(sig, WINDOW_SIZE, STRIDE, fps)
            bvp = _compute_bvp(windowed_sig, fps, method, method_map, RGB_sig_to_BVP)

        if bg_subtract:
            # Extract background from two corner regions, average into one "patch"
            bg1_sig, bg2_sig = extract_background_signals(videofilepath)
            bg_sig = ((bg1_sig + bg2_sig) / 2.0)[:, np.newaxis, :]  # [frames, 1, 3]
            windowed_bg, _ = sig_windowing(bg_sig, WINDOW_SIZE, STRIDE, fps)
            bg_bvp = _compute_bvp(windowed_bg, fps, method, method_map, RGB_sig_to_BVP)

            if per_region:
                for region_name, region_bvp in bvp.items():
                    n = region_bvp.shape[-1]
                    face_fft = np.fft.rfft(region_bvp, axis=-1)
                    bg_fft = np.fft.rfft(bg_bvp, axis=-1)
                    clean_mag = np.clip(np.abs(face_fft) - np.abs(bg_fft), 0, None)
                    bvp[region_name] = np.fft.irfft(
                        clean_mag * np.exp(1j * np.angle(face_fft)), n=n, axis=-1,
                    )
            else:
                # Subtract in frequency domain based on FFT magnitudes; preserve face phase
                n = bvp.shape[-1]
                face_fft = np.fft.rfft(bvp, axis=-1)
                bg_fft = np.fft.rfft(bg_bvp, axis=-1)
                clean_mag = np.clip(np.abs(face_fft) - np.abs(bg_fft), 0, None)
                bvp = np.fft.irfft(clean_mag * np.exp(1j * np.angle(face_fft)), n=n, axis=-1)

        if bvp is None or (isinstance(bvp, dict) and not bvp):
            logging.warning(f"No BVP extracted for {videofilepath}. Skipping.")
            return None

        if calc_bpm:
            if per_region:
                # Each region is collapsed to a single averaged patch before windowing
                region_bpmES = {r: [np.atleast_1d(b) for b in BVP_to_BPM(b_arr, fps, minHz=0.5, maxHz=4.0)]
                                for r, b_arr in bvp.items()}
                region_median = {r: BPM_median(b) for r, b in region_bpmES.items()}
                bpmES = region_bpmES
                bpm = {r: m[0] for r, m in region_median.items()}
                uncertainty = {r: m[1] for r, m in region_median.items()}
                snr = {r: _compute_snr(bvp[r], region_bpmES[r], fps, Welch) for r in bvp}
            else:
                bpmES = [np.atleast_1d(b) for b in BVP_to_BPM(bvp, fps, minHz=0.5, maxHz=4.0)]
                bpm, uncertainty = BPM_median(bpmES)
                snr = _compute_snr(bvp, bpmES, fps, Welch)
        else:
            bpmES, bpm, uncertainty, snr = None, None, None, None

        return {
            "Filename": str(videofilepath),
            "BVPS": bvp,
            "timesES": timesES,
            "Type": video_type,
            "BPMES": bpmES,
            "BPM": bpm,
            "Uncertainty": uncertainty,
            "SNR": snr,
            "FPS": fps,
        }
    except Exception as e:
        logging.error(f"Failed to process {videofilepath}: {e}")
        return None
    finally:
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description='Extract rPPG signals from a video dataset using pyVHR.'
    )
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (see load_dataset.py for supported names).')
    parser.add_argument('--dataset-dir', type=str, required=True,
                        help='Root directory of the dataset on disk.')
    parser.add_argument('--method', type=str, required=True, choices=SUPPORTED_METHODS,
                        help='rPPG extraction method.')
    parser.add_argument('--output-dir', type=str, default='extracted_signals',
                        help='Directory to save the output NDJSON file (default: extracted_signals/).')
    parser.add_argument('--calc-bpm', action='store_true',
                        help='Also compute BPM and per-window SNR (dB) estimates from extracted BVP signals.')
    parser.add_argument('--landmarks-video-dir', type=str, default=None,
                        help='If set, save landmark visualisation videos to this directory.')
    parser.add_argument('--bg-subtract', action='store_true',
                        help='Subtract background FFT map (averaged from two corner regions) '
                             'from the face FFT map to remove global illumination artefacts.')
    parser.add_argument('--per-region', action='store_true',
                        help='Extract BVP per landmark region (as defined in LANDMARK_REGIONS) '
                             'instead of aggregating all patches. Within each region, patches are '
                             'averaged before BVP extraction. Output BVPS is a dict mapping region '
                             'name to a (num_windows, window_samples) array.')
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, path=args.dataset_dir)
    if dataset is None:
        raise ValueError(f"Unknown dataset: '{args.dataset}'. See load_dataset.py for supported names.")

    os.makedirs(args.output_dir, exist_ok=True)
    if args.landmarks_video_dir:
        os.makedirs(args.landmarks_video_dir, exist_ok=True)

    start_time = time.perf_counter()

    if args.landmarks_video_dir:
        results = [
            process_video(instance, args.method, args.calc_bpm, args.landmarks_video_dir,
                          args.bg_subtract, args.per_region)
            for instance in tqdm(dataset, total=len(dataset))
        ]
    else:
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = list(tqdm(
                executor.map(process_video, dataset, repeat(args.method), repeat(args.calc_bpm),
                             repeat(None), repeat(args.bg_subtract), repeat(args.per_region)),
                total=len(dataset),
            ))

    results = [r for r in results if r is not None]

    suffix = '_bg_subtracted' if args.bg_subtract else ''
    region_suffix = '_per_region' if args.per_region else ''
    outfile = os.path.join(
        args.output_dir,
        f"extracted_signals_{dataset.dataset_name}_{args.method}{suffix}{region_suffix}.json",
    )
    pd.DataFrame(results).to_json(outfile, orient='records', lines=True)
    logging.info(f"Saved extracted data to {outfile}")
    logging.info(f"Total time: {time.perf_counter() - start_time:.2f}s")


if __name__ == '__main__':
    main()