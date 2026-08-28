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

# Sample every Nth frame; compression artifacts are stable within a clip so dense
# sampling adds cost without adding signal.
FRAME_STRIDE = 10

# JPEG re-encode qualities to sweep when estimating the quality-factor proxy.
JPEG_QUALITY_SWEEP = list(range(20, 100, 10))

_FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'


def _detect_face_crop(frame_bgr, cascade, last_bbox):
    """Detect the largest face in a frame; fall back to the previous bbox if none is found."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))
    if len(faces) == 0:
        return last_bbox

    # Largest detected face by area
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return (x, y, w, h)


def _blockiness_score(gray_crop, block_size=8):
    """
    Blind 8x8 blockiness metric: mean horizontal gradient at block-grid boundaries
    relative to the mean gradient elsewhere. Block-based codecs (JPEG, H.264) leave
    discontinuities aligned to the block grid, so a ratio above ~1 indicates visible
    blocking artifacts regardless of the source codec.
    """
    h, w = gray_crop.shape
    if h < block_size * 2 or w < block_size * 2:
        return np.nan
    gray = gray_crop.astype(np.float64)

    # Horizontal gradient at vertical block boundaries vs. interior columns
    col_diff = np.abs(np.diff(gray, axis=1))
    boundary_cols = np.arange(block_size - 1, w - 1, block_size)
    non_boundary_mask = np.ones(col_diff.shape[1], dtype=bool)
    non_boundary_mask[boundary_cols] = False

    interior_energy = col_diff[:, non_boundary_mask].mean()
    if boundary_cols.size == 0 or interior_energy < 1e-8:
        return np.nan
    boundary_energy = col_diff[:, boundary_cols].mean()
    return float(boundary_energy / interior_energy)


def _estimate_jpeg_quality(gray_crop, quality_sweep=JPEG_QUALITY_SWEEP):
    """
    JPEG quality-factor proxy: the source video is H.264, not JPEG, so this is not a
    literal quality factor. It re-encodes the crop at each sweep quality and picks the
    one whose round-trip distortion is smallest, i.e. the JPEG quality the crop's
    existing compression artifacts most resemble — a codec-agnostic scalar for "how
    degraded does this frame already look."
    """
    best_quality, best_error = quality_sweep[0], np.inf
    for q in quality_sweep:
        ok, enc = cv2.imencode('.jpg', gray_crop, [cv2.IMWRITE_JPEG_QUALITY, q])
        if not ok:
            continue
        decoded = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
        error = float(np.mean((gray_crop.astype(np.float64) - decoded.astype(np.float64)) ** 2))
        if error < best_error:
            best_error, best_quality = error, q
    return float(best_quality)


def _hf_energy_ratio(gray_crop):
    """Fraction of 2-D spectral energy outside the central half-radius of the crop's
    FFT (compression suppresses high spatial frequencies first)."""
    if gray_crop.size == 0:
        return np.nan
    spectrum = np.fft.fftshift(np.fft.fft2(gray_crop.astype(np.float64)))
    power = np.abs(spectrum) ** 2

    h, w = power.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_radius = np.sqrt(cy ** 2 + cx ** 2) + 1e-8

    hf_mask = radius > 0.5 * max_radius
    total = power.sum() + 1e-8
    return float(power[hf_mask].sum() / total)


def _aggregate(values, name):
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f'{name}_mean': np.nan, f'{name}_std': np.nan}
    return {f'{name}_mean': float(arr.mean()), f'{name}_std': float(arr.std())}


def process_video(instance, frame_stride=FRAME_STRIDE):
    videofilepath = instance.path
    video_type = instance.annotation.get_label("authenticity")

    # Built per-worker (not pickled across processes) since cv2 objects aren't picklable.
    cascade = cv2.CascadeClassifier(_FACE_CASCADE_PATH)

    cap = cv2.VideoCapture(str(videofilepath))
    blockiness, qualities, hf_ratios = [], [], []
    residual_blockiness, residual_hf_ratios = [], []
    last_bbox = None
    prev_gray_crop, prev_bbox = None, None
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Only run face (re-)detection on every Nth frame; the detector is the
            # expensive part, cropping/differencing below is cheap.
            if frame_idx % frame_stride == 0:
                bbox = _detect_face_crop(frame, cascade, last_bbox)
                if bbox is not None:
                    last_bbox = bbox

            if last_bbox is not None:
                x, y, w, h = last_bbox
                gray_crop = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
                if gray_crop.size:
                    if frame_idx % frame_stride == 0:
                        blockiness.append(_blockiness_score(gray_crop))
                        qualities.append(_estimate_jpeg_quality(gray_crop))
                        hf_ratios.append(_hf_energy_ratio(gray_crop))

                        # Temporal (inter-frame) artifacts: block-based video codecs spend
                        # most of their bits on motion-compensated PREDICTION residuals, not
                        # on independently-coded frames, so the purely spatial features above
                        # miss most of the compression signature. Reuse the same blockiness /
                        # HF-energy probes on the single-step frame difference to capture that.
                        # Requires a crop from the immediately preceding frame with the SAME
                        # bbox - skip at a redetection boundary where the bbox just moved, to
                        # avoid a spurious jump artifact contaminating the residual.
                        if (prev_gray_crop is not None and prev_bbox == last_bbox
                                and prev_gray_crop.shape == gray_crop.shape):
                            residual = cv2.absdiff(gray_crop, prev_gray_crop)
                            residual_blockiness.append(_blockiness_score(residual))
                            residual_hf_ratios.append(_hf_energy_ratio(residual))

                    # Tracked every frame (not just every Nth) so the residual above is
                    # always a true adjacent-frame difference, matching the timescale a
                    # codec actually predicts over.
                    prev_gray_crop, prev_bbox = gray_crop, last_bbox
            frame_idx += 1
    except Exception as e:
        logging.error(f"Failed to process {videofilepath}: {e}")
        return None
    finally:
        cap.release()

    if not blockiness:
        logging.warning(f"No face detected in {videofilepath}. Skipping.")
        return None

    return {
        "Filename": str(videofilepath),
        "Type": video_type,
        **_aggregate(blockiness, 'blockiness'),
        **_aggregate(qualities, 'jpeg_quality'),
        **_aggregate(hf_ratios, 'hf_energy_ratio'),
        **_aggregate(residual_blockiness, 'residual_blockiness'),
        **_aggregate(residual_hf_ratios, 'residual_hf_energy_ratio'),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Extract image-domain compression-artifact features (blockiness, '
                    'JPEG-quality proxy, high-frequency spectral energy - both from single '
                    'frames and from frame-to-frame residuals, to capture inter-frame/GOP '
                    'compression as well as intra-frame compression) directly from face-crop '
                    'pixels. Unlike BVP-derived features, these never pass through the rPPG '
                    'projection, so they can serve as a genuine baseline for whether a '
                    'classifier is keying on codec artifacts rather than cardiac information. '
                    'Output feeds baseline_train.py --feature-set image-compression.'
    )
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (see load_dataset.py for supported names).')
    parser.add_argument('--dataset-dir', type=str, required=True,
                        help='Root directory of the dataset on disk.')
    parser.add_argument('--output-dir', type=str, default='extracted_signals',
                        help='Directory to save the output NDJSON file (default: extracted_signals/).')
    parser.add_argument('--frame-stride', type=int, default=FRAME_STRIDE,
                        help=f'Sample every Nth frame for feature extraction (default: {FRAME_STRIDE}).')
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, path=args.dataset_dir)
    if dataset is None:
        raise ValueError(f"Unknown dataset: '{args.dataset}'. See load_dataset.py for supported names.")

    os.makedirs(args.output_dir, exist_ok=True)
    start_time = time.perf_counter()

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(
            executor.map(process_video, dataset, repeat(args.frame_stride)),
            total=len(dataset),
        ))

    results = [r for r in results if r is not None]

    outfile = os.path.join(
        args.output_dir, f"compression_features_{dataset.dataset_name}.json",
    )
    pd.DataFrame(results).to_json(outfile, orient='records', lines=True)
    logging.info(f"Saved compression features to {outfile}")
    logging.info(f"Total time: {time.perf_counter() - start_time:.2f}s")


if __name__ == '__main__':
    main()