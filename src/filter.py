import numpy as np
from scipy.signal import butter, filtfilt


# Physiological heart rate range
LOW_HZ = 0.5
HIGH_HZ = 4

_MODES = ("bandpass", "bandstop", "lowpass", "highpass")


def filter_heart_rate_frequency(
    signal: np.ndarray,
    fps: float,
    mode: str = "bandpass",
    order: int = 4,
    axis: int = -1,
) -> np.ndarray:
    """
    Filter a BVP signal relative to the physiological heart rate band (0.5-4 Hz).

    Args:
        signal: BVP signal array (1-D or N-D). For N-D arrays filtering is applied along `axis`.
        fps: Sampling frequency in Hz.
        mode: One of:
            "bandpass" - keep only 0.5-4 Hz (the physiological heart rate band).
            "bandstop" - remove 0.5-4 Hz, keeping everything else (the band-stop control:
                         if a classifier still performs well on this signal, the physiological
                         band is not what it was using).
            "lowpass"  - keep only frequencies below 0.5 Hz (illumination drift / slow motion,
                         no heartbeat).
            "highpass" - keep only frequencies above 4 Hz (tracking jitter / high-frequency
                         noise, no heartbeat).
        order: Butterworth filter order.
        axis: Axis along which to apply the filter (default: -1, i.e. the last axis).

    Returns:
        Filtered signal as a numpy array with the same shape as input.
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")

    # Get filter cutoff frequencies
    nyq = fps / 2.0
    low = LOW_HZ / nyq
    high = HIGH_HZ / nyq

    # Apply filter depending on the mode
    if mode in ("bandpass", "bandstop"):
        b, a = butter(order, [low, high], btype=mode)
    elif mode == "lowpass":
        b, a = butter(order, low, btype="lowpass")
    else:  # highpass
        b, a = butter(order, high, btype="highpass")

    return filtfilt(b, a, signal, axis=axis)