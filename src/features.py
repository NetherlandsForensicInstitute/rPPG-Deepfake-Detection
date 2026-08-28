import numpy as np
from scipy import stats
from scipy.signal import coherence

from src.filter import LOW_HZ, HIGH_HZ
from src.utils import _summary_stats, flatten, average_patch_axis


def _bpm_row_features(row) -> dict:
    """
    Defines features for the BPM baseline given a BVP signal as input.

    Returns a dict containing the following features:
    - Mean BPM across windows
    - STD of BPM across windows
    - BPM-median uncertainty

    Requires --calc-bpm extraction with the extraction/extract_signals.py script.
    """
    feats = _summary_stats(flatten(row.get('BPMES')), 'bpm')
    feats.update(_summary_stats(flatten(row.get('Uncertainty')), 'uncertainty'))
    return feats


def _snr_row_features(row) -> dict:
    """
    Defines features for the SNR baseline given a BVP signal as input.

    Returns a dict containing the following features:
    - Mean per-window rPPG signal-to-noise ratio (dB)
    - STD of per-window rPPG signal-to-noise ratio (dB)

    Requires --calc-bpm extraction with the extraction/extract_signals.py script.
    """
    return _summary_stats(flatten(row.get('SNR')), 'snr')


def _handcrafted_rPPG_row_features(row, fs: float = 30.0) -> dict:
    """
    Defines features for the handcrafted rPPG features baseline given a BVP signal as input.

    Args:
        row: Row containing 'BVPS' key with BVP signal data.
        fs: Sampling rate of the BVP signal in Hz (default: 30 Hz for standard video).

    Returns a dict containing the following features:
    - Mean per-window peak/total in-band spectral power ratio
    - Mean per-window BVP signal variance
    - Mean per-window zero-crossing rate
    - Mean per-window BVP signal mean
    - Mean per-window BVP signal standard deviation
    - Mean per-window BVP skewness
    - Mean per-window BVP kurtosis
    - Mean per-window BVP peak-to-peak amplitude
    - Mean per-window BVP crest factor
    - Mean per-window peak heart rate (BPM)
    - Mean per-window spectral entropy

    Returns:
        dict: Handcrafted rPPG feature dictionary.
    """
    # Keys to initialize in case of empty window
    feature_keys = [
        'bvp_mean', 'bvp_std_mean', 'bvp_variance_mean', 'bvp_skewness_mean', 'bvp_kurtosis_mean',
        'bvp_p2p_mean', 'bvp_crest_factor_mean', 'bvp_zero_crossing_rate_mean',
        'bvp_peak_freq_bpm_mean', 'bvp_spectral_entropy_mean', 'bvp_peak_power_ratio_mean'
    ]

    # Get BVP signal
    bvp = row.get('BVPS')

    # Average signals over patch axis
    if isinstance(bvp, dict):
        arrays = [average_patch_axis(v) for v in bvp.values() if v]
        arrays = [a for a in arrays if a.size]
        windows = np.concatenate(arrays, axis=0) if arrays else np.empty((0, 0))
    else:
        windows = average_patch_axis(bvp) if bvp else np.empty((0, 0))

    if windows.size == 0 or windows.shape[-1] < 2:
        return {key: np.nan for key in feature_keys}

    n_samples = windows.shape[-1]

    # Statistical Features
    means = np.mean(windows, axis=-1)
    stds = np.std(windows, axis=-1)
    variances = np.var(windows, axis=-1)
    skews = stats.skew(windows, axis=-1)
    kurt = stats.kurtosis(windows, axis=-1)

    # Morphological / Amplitude Features
    p2p = np.ptp(windows, axis=-1)  # Peak-to-peak amplitude
    # Crest factor: peak amplitude divided by RMS value
    rms = np.sqrt(np.mean(windows ** 2, axis=-1))
    max_abs = np.max(np.abs(windows), axis=-1)
    crest_factor = np.divide(max_abs, rms, out=np.zeros_like(max_abs), where=rms != 0)

    # Temporal / Dynamic Features
    zero_centered = windows - means[:, np.newaxis]
    signs = np.sign(zero_centered)
    zero_crossings = (np.diff(signs, axis=-1) != 0).sum(axis=-1) / n_samples

    # Spectral Features (Frequency Domain)
    # Compute FFT per window along the last axis
    fft_vals = np.abs(np.fft.rfft(zero_centered, axis=-1))
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)

    # Filter frequencies to human cardiac range (src.filter.LOW_HZ / HIGH_HZ)
    cardiac_mask = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)

    if np.any(cardiac_mask):
        cardiac_fft = fft_vals[:, cardiac_mask]
        cardiac_freqs = freqs[cardiac_mask]

        # Dominant frequency converted to Beats Per Minute (BPM)
        peak_freq_idx = np.argmax(cardiac_fft, axis=-1)
        peak_freqs_bpm = cardiac_freqs[peak_freq_idx] * 60.0

        # Spectral Entropy within cardiac band (measure of pulse periodicity vs noise)
        norm_fft = cardiac_fft / (np.sum(cardiac_fft, axis=-1, keepdims=True) + 1e-8)
        spectral_entropy = -np.sum(norm_fft * np.log2(norm_fft + 1e-8), axis=-1)

        # Peak power ratio: power in peak frequency neighborhood vs total cardiac power
        # (not a true SNR estimate - see _snr_row_features for the de Haan-based SNR)
        peak_power = np.max(cardiac_fft ** 2, axis=-1)
        total_power = np.sum(cardiac_fft ** 2, axis=-1) + 1e-8
        peak_power_ratio = peak_power / total_power
    else:
        peak_freqs_bpm = np.full(windows.shape[0], np.nan)
        spectral_entropy = np.full(windows.shape[0], np.nan)
        peak_power_ratio = np.full(windows.shape[0], np.nan)

    return {
        'bvp_mean': float(np.nanmean(means)),
        'bvp_std_mean': float(np.nanmean(stds)),
        'bvp_variance_mean': float(np.nanmean(variances)),
        'bvp_skewness_mean': float(np.nanmean(skews)),
        'bvp_kurtosis_mean': float(np.nanmean(kurt)),
        'bvp_p2p_mean': float(np.nanmean(p2p)),
        'bvp_crest_factor_mean': float(np.nanmean(crest_factor)),
        'bvp_zero_crossing_rate_mean': float(np.nanmean(zero_crossings)),
        'bvp_peak_freq_bpm_mean': float(np.nanmean(peak_freqs_bpm)),
        'bvp_spectral_entropy_mean': float(np.nanmean(spectral_entropy)),
        'bvp_peak_power_ratio_mean': float(np.nanmean(peak_power_ratio)),
    }


def _phase_shuffle_batch(signal: np.ndarray, rng: np.random.Generator, n_shuffles: int) -> np.ndarray:
    """
    Randomize the Fourier phase of a real 1-D signal while preserving its magnitude
    spectrum, batched over n_shuffles independent draws.

    Used to build a null distribution for coherence: a phase-randomized signal has the
    same power spectrum (and autocorrelation structure) as the original but no genuine
    phase relationship to another signal, so any coherence with it must come from shared
    broadband/spectral shape alone, not real synchrony.
    """
    n = signal.size
    spectrum = np.fft.rfft(signal)
    magnitudes = np.abs(spectrum)

    # Draw independent random phases per shuffle, per frequency bin
    phases = rng.uniform(0, 2 * np.pi, size=(n_shuffles, spectrum.size))

    # DC (and Nyquist, if n is even) must stay real-valued for the signal to stay real
    phases[:, 0] = 0.0
    if n % 2 == 0:
        phases[:, -1] = 0.0

    shuffled_spectra = magnitudes[np.newaxis, :] * np.exp(1j * phases)
    return np.fft.irfft(shuffled_spectra, n=n, axis=-1)


def _spatial_coherence_row_features(row, fs: float = 30.0, stride: float = 1.0, n_shuffles: int = 50, seed: int = 0) -> dict:
    """
    Defines features for the spatial coherence baseline given a BVP signal as input.

    Computes magnitude-squared coherence (MSC) between facial-region BVP signals via
    Welch's method, averaged over the cardiac band (src.filter.LOW_HZ-HIGH_HZ).

    Returns a dict containing:
    - Mean/std/10th-percentile cardiac-band MSC across region-pairs
    - Mean z-score of observed MSC against the phase-shuffle null

    Requires --per-region extraction with the extraction/extract_signals.py script.
    """
    # Get BVP signal
    bvp = row.get('BVPS')
    empty = {
        'msc_cardiac_mean': np.nan, 'msc_cardiac_std': np.nan, 'msc_cardiac_p10': np.nan,
        'msc_null_z_mean': np.nan,
    }
    if not isinstance(bvp, dict) or len(bvp) < 2:
        return empty

    # Get regions and average over patch axis
    regions = list(bvp.keys())
    signals = {r: average_patch_axis(bvp[r]) for r in regions}
    signals = {r: s for r, s in signals.items() if s.size}
    if len(signals) < 2:
        return empty

    # Reconstruct one continuous per-region signal from the overlapping windows
    stride_frames = max(1, int(round(stride * fs)))
    reconstructed = {}
    for r, windows in signals.items():
        if windows.shape[0] == 1:
            reconstructed[r] = windows[0]
            continue
        tails = [windows[i, -stride_frames:] for i in range(1, windows.shape[0])]
        reconstructed[r] = np.concatenate([windows[0]] + tails)
    signals = reconstructed

    rng = np.random.default_rng(seed)

    # Loop over region-pairs and calculate cardiac-band MSC on the reconstructed signals
    msc_values = []
    z_scores = []
    region_names = list(signals.keys())
    for i in range(len(region_names)):
        for j in range(i + 1, len(region_names)):
            wa, wb = signals[region_names[i]], signals[region_names[j]]
            n_samples = min(wa.shape[-1], wb.shape[-1])
            wa, wb = wa[:n_samples], wb[:n_samples]

            # Need multiple Welch segments per estimate, or MSC is trivially 1 everywhere
            if wa.std() < 1e-8 or wb.std() < 1e-8 or n_samples < 16:
                continue
            nperseg = min(64, max(16, n_samples // 3))

            freqs, cxy = coherence(wa, wb, fs=fs, nperseg=nperseg)
            band_mask = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
            if not np.any(band_mask):
                continue
            observed = float(np.mean(cxy[band_mask]))
            msc_values.append(observed)

            # Phase-shuffle null: both signals independently phase-randomized, so any
            # residual coherence reflects marginal spectral overlap, not real synchrony
            shuffled_wa = _phase_shuffle_batch(wa, rng, n_shuffles)
            shuffled_wb = _phase_shuffle_batch(wb, rng, n_shuffles)
            _, cxy_null = coherence(shuffled_wa, shuffled_wb, fs=fs, nperseg=nperseg, axis=-1)
            null_band = cxy_null[:, band_mask].mean(axis=-1)
            null_mean, null_std = null_band.mean(), null_band.std()
            if null_std > 1e-8:
                z_scores.append((observed - null_mean) / null_std)

    # Calculate mean, std, 10th percentile of calculated MSC values and mean null z-score
    msc_values = np.array(msc_values, dtype=float)
    msc_values = msc_values[np.isfinite(msc_values)]
    if msc_values.size == 0:
        return empty
    z_scores = np.array(z_scores, dtype=float)
    z_scores = z_scores[np.isfinite(z_scores)]
    return {
        'msc_cardiac_mean': float(msc_values.mean()),
        'msc_cardiac_std': float(msc_values.std()),
        'msc_cardiac_p10': float(np.percentile(msc_values, 10)),
        'msc_null_z_mean': float(z_scores.mean()) if z_scores.size else np.nan,
    }


def _image_compression_row_features(row) -> dict:
    """
    Read precomputed features for the image-domain compression baseline.

    Requires --data to point at the output of extraction/extract_compression_features.py,
    not a BVP signals file.
    """
    keys = [
        'blockiness_mean', 'blockiness_std',
        'jpeg_quality_mean', 'jpeg_quality_std',
        'hf_energy_ratio_mean', 'hf_energy_ratio_std',
        'residual_blockiness_mean', 'residual_blockiness_std',
        'residual_hf_energy_ratio_mean', 'residual_hf_energy_ratio_std',
    ]
    return {key: row.get(key, np.nan) for key in keys}


_FEATURE_FUNCS = {
    'bpm':               [_bpm_row_features],
    'snr':                [_snr_row_features],
    'handcrafted':         [_bpm_row_features, _snr_row_features, _handcrafted_rPPG_row_features],
    'spatial-coherence':   [_spatial_coherence_row_features],
    'image-compression':   [_image_compression_row_features],
}