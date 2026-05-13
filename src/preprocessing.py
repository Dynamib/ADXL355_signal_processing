"""Preprocessing: gravity removal and Butterworth bandpass filtering."""

import numpy as np
from scipy.signal import butter, filtfilt


def remove_gravity(accel):
    """Subtract per-axis mean to remove DC / gravity component.

    Args:
        accel: (N, 3) or (N,) acceleration array.

    Returns:
        Zero-mean acceleration, same shape as input.
    """
    return accel - accel.mean(axis=0)


def butter_bandpass(data, lowcut, highcut, fs, order=4):
    """Apply zero-phase Butterworth bandpass filter.

    Implements the knowledge base reference:
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    Args:
        data: 1-D signal array.
        lowcut: low cutoff frequency in Hz.
        highcut: high cutoff frequency in Hz.
        fs: sampling rate in Hz.
        order: Butterworth filter order.

    Returns:
        Filtered signal, same length as input.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def preprocess_per_axis(accel_dc, fs, lowcut=1.0, highcut=200.0, order=4):
    """Apply bandpass filter to each axis independently.

    Args:
        accel_dc: (N, 3) zero-mean acceleration.
        fs: sampling rate in Hz.
        lowcut, highcut: filter cutoff frequencies in Hz.
        order: Butterworth filter order.

    Returns:
        accel_bp: (N, 3) bandpass-filtered acceleration.
    """
    accel_bp = np.empty_like(accel_dc)
    for i in range(3):
        accel_bp[:, i] = butter_bandpass(accel_dc[:, i], lowcut, highcut, fs, order)
    return accel_bp
