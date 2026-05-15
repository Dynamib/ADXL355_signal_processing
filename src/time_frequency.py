"""Time-frequency analysis: STFT and SST (Synchrosqueezed Transform).

Implements knowledge base Step 2:
  - STFT with Hann window, configurable overlap and FFT size
  - SST via ssqueezepy for high-resolution time-frequency representation
"""

import numpy as np
from scipy.signal import stft as scipy_stft, get_window, resample


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


def compute_sst(signal, fs, nv=32, fs_sst=300.0):
    """Compute Synchrosqueezed CWT (SST) with optional downsampling.

    Long signals at high sampling rates cause 15+ GiB memory allocation
    in ssqueezepy. If fs > fs_sst, the signal is low-pass filtered and
    decimated to fs_sst before SST computation.

    Args:
        signal: 1-D signal array.
        fs: sampling rate in Hz.
        nv: number of voices per octave.
        fs_sst: target sampling rate for SST (Hz). Downsample if fs exceeds this.

    Returns:
        Tx: 2-D SST magnitude array (frequencies x time).
        freqs: frequency axis values.
        t: time axis values.
        Wx: CWT coefficients.
    """
    import ssqueezepy as sq

    sig = signal
    fs_used = fs

    if fs > fs_sst:
        n_target = int(len(sig) * fs_sst / fs)
        sig = resample(sig, n_target)
        fs_used = fs_sst

    Tx, Wx, freqs, *_ = sq.ssq_cwt(
        sig, ('gmw', {'beta': 2}), fs=fs_used, nv=nv
    )
    t = np.linspace(0, len(signal) / fs, Tx.shape[1])
    return Tx, freqs, t, Wx
