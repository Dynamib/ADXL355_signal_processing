"""Time-frequency analysis: STFT and SST (Synchrosqueezed Transform).

Implements knowledge base Step 2:
  - STFT with Hann window, configurable overlap and FFT size
  - SST via ssqueezepy for high-resolution time-frequency representation
"""

import numpy as np
from scipy.signal import stft as scipy_stft, get_window


def compute_stft(signal, fs, nperseg=512, noverlap=384, nfft=1024, window='hann'):
    """Compute Short-Time Fourier Transform.

    Uses scipy.signal.stft.

    Args:
        signal: 1-D signal array.
        fs: sampling rate in Hz.
        nperseg: samples per segment (window length).
        noverlap: overlap between segments.
        nfft: FFT points (>= nperseg for zero-padding).
        window: window type ('hann', 'hamming', 'kaiser', etc.).

    Returns:
        f: (nfft//2+1,) frequency bins in Hz.
        t: (n_frames,) time bins in seconds.
        Zxx: (nfft//2+1, n_frames) complex STFT coefficients.
    """
    f, t, Zxx = scipy_stft(signal, fs=fs, window=window, nperseg=nperseg,
                           noverlap=noverlap, nfft=nfft, boundary=None)
    return f, t, Zxx


def compute_sst(signal, fs, nv=32):
    """Compute Synchrosqueezed CWT (SST).

    Implements the knowledge base reference:
        Tx, *_ = sq.ssq_cwt(x, ('gmw', {'beta': 2}), fs=fs, nv=32)

    Args:
        signal: 1-D signal array.
        fs: sampling rate in Hz.
        nv: number of voices per octave.

    Returns:
        Tx: 2-D SST magnitude array (frequencies x time).
        freqs: frequency axis values.
        t: time axis values.
        Wx: CWT coefficients.
    """
    import ssqueezepy as sq
    Tx, Wx, freqs, *_ = sq.ssq_cwt(
        signal, ('gmw', {'beta': 2}), fs=fs, nv=nv
    )
    t = np.linspace(0, len(signal) / fs, Tx.shape[1])
    return Tx, freqs, t, Wx
