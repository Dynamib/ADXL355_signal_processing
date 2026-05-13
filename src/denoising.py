"""Blind deconvolution and non-Gaussian noise handling.

Implements knowledge base Step 1:
  - MED (Minimum Entropy Deconvolution) via kurtosis maximization
  - Hilbert envelope extraction
  - Log transform for heavy-tailed noise stabilization
"""

import numpy as np
from scipy.signal import hilbert, convolve, correlate
from scipy.linalg import solve, toeplitz


def kurtosis(x):
    """Excess kurtosis: E[(x-mu)^4] / sigma^4 - 3.

    Gaussian: ~0.  Impulsive: > 0.  Sub-Gaussian: < 0.
    """
    mu = x.mean()
    sigma2 = ((x - mu) ** 2).mean()
    if sigma2 < 1e-20:
        return 0.0
    return ((x - mu) ** 4).mean() / (sigma2 ** 2) - 3.0


def _autocorr_toeplitz(x, filter_length):
    """Build Toeplitz autocorrelation matrix of size (filter_length, filter_length)."""
    r = correlate(x, x, mode='full')
    center = len(r) // 2
    r_vals = r[center:center + filter_length]
    return toeplitz(r_vals)


def minimum_entropy_deconvolution(signal, filter_length=30, max_iter=50, tol=1e-6):
    """Minimum Entropy Deconvolution — maximize kurtosis via iterative deconvolution.

    Based on Wiggins (1978) / Lee & Nandi approach.

    Algorithm per iteration:
      1. y = signal * f  (convolve, mode='same')
      2. g = cross_correlation(signal, y^3)  trimmed to filter_length
      3. solve R_xx @ f_new = g  (Toeplitz linear system)
      4. normalize f_new
      5. check convergence

    Args:
        signal: 1-D input signal.
        filter_length: length of the MED inverse filter.
        max_iter: maximum iterations.
        tol: convergence tolerance on filter change.

    Returns:
        y: deconvolved output signal (same length as input).
        f: learned inverse filter (length = filter_length).
    """
    x = np.asarray(signal, dtype=np.float64)
    n = len(x)

    # Build Toeplitz autocorrelation matrix (used each iteration)
    R_xx = _autocorr_toeplitz(x, filter_length)

    # Initialize filter as delta at center
    f = np.zeros(filter_length, dtype=np.float64)
    f[filter_length // 2] = 1.0

    # Add small regularization for numerical stability
    R_reg = R_xx + 1e-6 * np.eye(filter_length)

    for iteration in range(max_iter):
        # Convolve input with current filter
        y = convolve(x, f, mode='same')

        # Cross-correlation of input with cubed output
        y3 = y ** 3
        g_full = correlate(x, y3, mode='full')
        # Extract filter_length samples centered
        center = len(g_full) // 2
        start = center - filter_length // 2
        g = g_full[start:start + filter_length]

        # Solve linear system
        f_new = solve(R_reg, g, assume_a='sym')

        # Normalize
        norm = np.linalg.norm(f_new)
        if norm < 1e-20:
            break
        f_new = f_new / norm

        # Check convergence
        delta = np.linalg.norm(f_new - f)
        f = f_new
        if delta < tol:
            break

    # Final output
    y = convolve(x, f, mode='same')
    return y, f


def hilbert_envelope(signal):
    """Compute analytic signal envelope via Hilbert transform.

    envelope = |hilbert(signal)|
    """
    analytic = hilbert(signal)
    return np.abs(analytic)


def log_stabilize(envelope, eps=1e-10):
    """Log-transform envelope to handle heavy-tailed noise.

    log_envelope = log(envelope + eps)

    Converts multiplicative noise to additive noise and suppresses
    the influence of large-amplitude outliers (hand tremor "jerks").
    """
    return np.log(envelope + eps)


def denoise_per_axis(accel_bp, fs, filter_length=30, max_iter=50):
    """Full denoising pipeline applied to each axis.

    For each axis:
      1. MED: enhance impulsive features
      2. Hilbert envelope
      3. Log transform

    Args:
        accel_bp: (N, 3) bandpass-filtered acceleration.
        fs: sampling rate in Hz.
        filter_length: MED filter length in samples.
        max_iter: maximum MED iterations.

    Returns:
        dict with keys:
          'med': (N, 3) MED output
          'filters': (L, 3) learned MED filters
          'envelope': (N, 3) Hilbert envelope
          'log_envelope': (N, 3) log-stabilized envelope
          'kurt_before': (3,) kurtosis of bandpass signal
          'kurt_after': (3,) kurtosis of MED output
    """
    n_samples, n_axes = accel_bp.shape
    med_out = np.empty_like(accel_bp)
    filters = np.empty((filter_length, n_axes), dtype=np.float64)
    envelope = np.empty_like(accel_bp)
    log_envelope = np.empty_like(accel_bp)
    kurt_before = np.empty(n_axes)
    kurt_after = np.empty(n_axes)

    for i in range(n_axes):
        sig = accel_bp[:, i]
        kurt_before[i] = kurtosis(sig)
        med_out[:, i], filters[:, i] = minimum_entropy_deconvolution(
            sig, filter_length=filter_length, max_iter=max_iter
        )
        kurt_after[i] = kurtosis(med_out[:, i])
        envelope[:, i] = hilbert_envelope(med_out[:, i])
        log_envelope[:, i] = log_stabilize(envelope[:, i])

    return {
        'med': med_out,
        'filters': filters,
        'envelope': envelope,
        'log_envelope': log_envelope,
        'kurt_before': kurt_before,
        'kurt_after': kurt_after,
    }
