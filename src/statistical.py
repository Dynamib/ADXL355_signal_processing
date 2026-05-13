"""Statistical verification and anomaly detection.

Implements knowledge base Step 3:
  - Welch PSD estimation
  - Spectral feature extraction (centroid, bandwidth, kurtosis, band energies)
  - Sliding-window spectral analysis
  - Z-score anomaly detection

Since data lacks clinical labels, the primary analysis is unsupervised anomaly
detection over sliding windows.
"""

import numpy as np
import pandas as pd
from scipy.signal import welch


def compute_welch_psd(signal, fs, nperseg=1024, noverlap=512):
    """Compute Welch Power Spectral Density.

    Returns:
        freqs: frequency bins in Hz.
        psd: power spectral density in signal_units^2 / Hz.
    """
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return freqs, psd


def spectral_centroid(freqs, psd):
    """Weighted mean frequency: sum(f_i * P_i) / sum(P_i)."""
    total = psd.sum()
    if total == 0:
        return 0.0
    return (freqs * psd).sum() / total


def spectral_bandwidth(freqs, psd):
    """RMS frequency spread around centroid."""
    fc = spectral_centroid(freqs, psd)
    total = psd.sum()
    if total == 0:
        return 0.0
    return np.sqrt(((freqs - fc) ** 2 * psd).sum() / total)


def spectral_kurtosis(freqs, psd):
    """How peaked or flat the spectrum is.

    High value -> narrowband (tonal).  Low -> broadband (noisy).
    """
    fc = spectral_centroid(freqs, psd)
    total = psd.sum()
    if total == 0:
        return 0.0
    sigma2 = ((freqs - fc) ** 2 * psd).sum() / total
    if sigma2 < 1e-20:
        return 0.0
    return ((freqs - fc) ** 4 * psd).sum() / (total * sigma2 ** 2) - 3.0


def band_energy_ratios(freqs, psd, bands):
    """Fraction of total PSD energy within each frequency band.

    Args:
        freqs: frequency axis.
        psd: PSD values.
        bands: dict of {name: (low_hz, high_hz)}.

    Returns:
        dict of {name: fraction (0-1)}.
    """
    total = psd.sum()
    if total == 0:
        return {name: 0.0 for name in bands}
    ratios = {}
    for name, (flo, fhi) in bands.items():
        mask = (freqs >= flo) & (freqs < fhi)
        ratios[f"E_{name}"] = psd[mask].sum() / total
    return ratios


def extract_spectral_features(freqs, psd, bands):
    """Extract all spectral features for one PSD.

    Returns a flat dict with all features.
    """
    feats = {
        'centroid': spectral_centroid(freqs, psd),
        'bandwidth': spectral_bandwidth(freqs, psd),
        'spectral_kurtosis': spectral_kurtosis(freqs, psd),
    }
    feats.update(band_energy_ratios(freqs, psd, bands))
    return feats


def sliding_spectral_features(signal, fs, window_dur=2.0, overlap=0.75, bands=None):
    """Sliding-window spectral feature extraction.

    Args:
        signal: 1-D signal array.
        fs: sampling rate in Hz.
        window_dur: window duration in seconds.
        overlap: window overlap ratio (0-1).
        bands: dict of {name: (low_hz, high_hz)}.

    Returns:
        pd.DataFrame with columns: t_center, centroid, bandwidth,
        spectral_kurtosis, E_low, E_tremor, E_beta, E_gamma, E_high.
    """
    if bands is None:
        bands = {}

    n_total = len(signal)
    win_samples = int(window_dur * fs)
    step_samples = int(win_samples * (1 - overlap))
    if step_samples < 1:
        step_samples = 1

    feats = []
    centers = []

    for start in range(0, n_total - win_samples + 1, step_samples):
        segment = signal[start:start + win_samples]
        nperseg = min(256, win_samples)
        f, psd = compute_welch_psd(segment, fs, nperseg=nperseg, noverlap=nperseg // 2)
        feats.append(extract_spectral_features(f, psd, bands))
        centers.append((start + win_samples / 2) / fs)

    df = pd.DataFrame(feats)
    df.insert(0, 't_center', centers)
    return df


def detect_anomalies_zscore(df, feature_cols, z_threshold=3.0):
    """Detect windows where features deviate by more than z_threshold std from median.

    A window is flagged if ANY of the feature columns exceeds the threshold.

    Args:
        df: DataFrame from sliding_spectral_features.
        feature_cols: list of column names to check.
        z_threshold: number of median absolute deviations for flagging.

    Returns:
        anomaly_mask: boolean array (n_windows,) where True = anomaly.
    """
    anomaly_mask = np.zeros(len(df), dtype=bool)
    for col in feature_cols:
        if col not in df.columns:
            continue
        vals = df[col].values
        median = np.median(vals)
        mad = np.median(np.abs(vals - median))
        if mad < 1e-20:
            continue
        z_scores = np.abs(vals - median) / mad
        anomaly_mask |= (z_scores > z_threshold)
    return anomaly_mask
