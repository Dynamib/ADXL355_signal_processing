"""Visualization functions for ADXL355 moxibustion signal processing.

Each function creates a matplotlib Figure, optionally saves it, and returns it.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config as CFG


def _ensure_dir():
    CFG.FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, name):
    _ensure_dir()
    path = CFG.FIGURE_DIR / name
    fig.savefig(path, dpi=CFG.FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_raw_time_series(t, accel, filename="01_raw_time_series.png"):
    """4-panel: X, Y, Z acceleration + their combined overlay."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    names = CFG.AXIS_NAMES
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i in range(3):
        ax = axes[i]
        ax.plot(t, accel[:, i], color=colors[i], linewidth=0.3)
        mu = accel[:, i].mean()
        ax.set_ylabel(f'{names[i]} (g)')
        ax.set_title(f'{names[i]}-axis (mean = {mu:.4f} g)')
        ax.grid(True, alpha=0.3)

    for i in range(3):
        axes[3].plot(t, accel[:, i], color=colors[i], linewidth=0.3, alpha=0.7)
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('Acceleration (g)')
    axes[3].set_title('All axes overlay')
    axes[3].legend(names)
    axes[3].grid(True, alpha=0.3)

    fig.suptitle('Raw ADXL355 Acceleration Time Series', fontsize=14)
    fig.tight_layout()
    _save(fig, filename)
    return fig


def plot_bandpass_filtered(t, accel_raw, accel_bp, axis_idx=0, filename="03_bandpass_filtered.png"):
    """Compare raw vs bandpass-filtered signal for one axis."""
    name = CFG.AXIS_NAMES[axis_idx]
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(t, accel_raw[:, axis_idx], color='gray', linewidth=0.3)
    axes[0].set_ylabel(f'{name} (g)')
    axes[0].set_title(f'{name}-axis: Raw (DC-coupled)')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, accel_bp[:, axis_idx], color='#1f77b4', linewidth=0.3)
    axes[1].set_ylabel(f'{name} (g)')
    axes[1].set_title(f'{name}-axis: Bandpass {CFG.BP_LOWCUT}-{CFG.BP_HIGHCUT} Hz (order={CFG.BP_ORDER})')
    axes[1].grid(True, alpha=0.3)

    from scipy.signal import welch
    f_raw, psd_raw = welch(accel_raw[:, axis_idx], fs=CFG.FS_TARGET, nperseg=CFG.WELCH_NPERSEG)
    f_bp, psd_bp = welch(accel_bp[:, axis_idx], fs=CFG.FS_TARGET, nperseg=CFG.WELCH_NPERSEG)

    mask = (f_raw >= 0.1) & (f_raw <= 250)
    axes[2].semilogy(f_raw[mask], psd_raw[mask], 'gray', alpha=0.6, label='Raw', linewidth=0.8)
    axes[2].semilogy(f_bp[mask], psd_bp[mask], '#1f77b4', label='Bandpass', linewidth=0.8)
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('PSD (g²/Hz)')
    axes[2].set_title('Welch PSD Comparison')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f'Bandpass Filtering — {name}-axis', fontsize=14)
    fig.tight_layout()
    _save(fig, filename.replace('.png', f'_{name}.png'))
    return fig


def plot_med_envelope(t, bp_signal, med_signal, envelope, log_envelope,
                      kurt_before, kurt_after, axis_name="X",
                      filename="04_envelope_kurtosis.png"):
    """4-panel: bandpass, MED output, Hilbert envelope, log envelope."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(t, bp_signal, color='#1f77b4', linewidth=0.3)
    axes[0].set_ylabel('Accel (g)')
    axes[0].set_title(f'{axis_name}-axis: Bandpass-filtered (kurt={kurt_before:.2f})')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, med_signal, color='#d62728', linewidth=0.3)
    axes[1].set_ylabel('Accel (g)')
    axes[1].set_title(f'{axis_name}-axis: MED output (kurt={kurt_after:.2f})')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, envelope, color='#ff7f0e', linewidth=0.5)
    axes[2].set_ylabel('Amplitude (g)')
    axes[2].set_title('Hilbert Envelope')
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(t, log_envelope, color='#2ca02c', linewidth=0.5)
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('log(g)')
    axes[3].set_title('Log-stabilized Envelope')
    axes[3].grid(True, alpha=0.3)

    fig.suptitle(f'MED Deconvolution & Envelope Analysis — {axis_name}-axis', fontsize=14)
    fig.tight_layout()
    _save(fig, filename.replace('.png', f'_{axis_name}.png'))
    return fig


def plot_psd_comparison(psd_data, filename="02_psd_comparison.png"):
    """Overlay Welch PSD for multiple signals. psd_data is list of (freqs, psd, label)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (f, psd, label) in enumerate(psd_data):
        mask = (f >= 0.1) & (f <= 250)
        ax.semilogy(f[mask], psd[mask], color=colors[i % len(colors)],
                     label=label, linewidth=0.8)

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (g²/Hz)')
    ax.set_title('Welch PSD: Three-Axis Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, filename)
    return fig


def plot_stft_spectrogram(f, t, Zxx, fs, title="STFT Spectrogram",
                          filename="05_stft_spectrogram.png"):
    """Single STFT spectrogram with dB color scale."""
    fig, ax = plt.subplots(figsize=(14, 5))
    Zxx_db = 20 * np.log10(np.abs(Zxx) + 1e-20)
    vmax = np.percentile(Zxx_db, 95)
    vmin = vmax - 60

    mesh = ax.pcolormesh(t, f, Zxx_db, shading='gouraud',
                          vmin=vmin, vmax=vmax, cmap=CFG.CMAP)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label='Magnitude (dB)')
    fig.tight_layout()
    _save(fig, filename)
    return fig


def plot_sst_spectrogram(Tx, freqs, t, title="SST Spectrogram",
                         filename="06_sst_spectrogram.png"):
    """SST spectrogram with dB color scale."""
    fig, ax = plt.subplots(figsize=(14, 5))
    mag_db = 20 * np.log10(np.abs(Tx) + 1e-20)
    vmax = np.percentile(mag_db, 95)
    vmin = vmax - 60

    mesh = ax.pcolormesh(t, freqs, mag_db, shading='gouraud',
                          vmin=vmin, vmax=vmax, cmap=CFG.CMAP)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)
    ax.set_ylim(0, CFG.BP_HIGHCUT + 20)
    fig.colorbar(mesh, ax=ax, label='Magnitude (dB)')
    fig.tight_layout()
    _save(fig, filename)
    return fig


def plot_fft_spectrum(t, accel, filename="09_fft_spectrum.png"):
    """FFT magnitude spectrum for each axis, with log-frequency and log-magnitude axes.

    Unlike PSD (which estimates power density), this shows the raw FFT magnitude
    |FFT(signal)| in dB, which is more intuitive for seeing individual frequency peaks.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    fs = CFG.FS_TARGET

    for i, ax in enumerate(axes):
        signal = accel[:, i] - accel[:, i].mean()
        n = len(signal)
        fft = np.fft.rfft(signal)
        mag = np.abs(fft) / n
        freq = np.fft.rfftfreq(n, 1 / fs)

        mask = (freq >= 0.5) & (freq <= 250)
        mag_db = 20 * np.log10(mag[mask] + 1e-20)

        ax.semilogx(freq[mask], mag_db, color=colors[i], linewidth=0.5)
        ax.set_ylabel('Magnitude (dB)')
        ax.set_title(f'{CFG.AXIS_NAMES[i]}-axis FFT Spectrum')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(mag_db.max() - 80, mag_db.max() + 5)

    axes[-1].set_xlabel('Frequency (Hz)')
    fig.suptitle('FFT Magnitude Spectrum (DC-removed, 3 axes)', fontsize=14)
    fig.tight_layout()
    _save(fig, filename)
    return fig


def plot_sliding_features(df, anomaly_mask=None, axis_name="X",
                          filename="07_sliding_window_features.png"):
    """Multi-panel sliding-window spectral features over time."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # Panel 1: Spectral centroid + bandwidth
    ax = axes[0]
    ax.plot(df['t_center'], df['centroid'], 'b-', linewidth=0.8, label='Centroid')
    ax.fill_between(df['t_center'],
                     df['centroid'] - df['bandwidth'],
                     df['centroid'] + df['bandwidth'],
                     alpha=0.2, color='b', label='Bandwidth')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(f'{axis_name}-axis: Spectral Centroid & Bandwidth')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Panel 2: Spectral kurtosis
    ax = axes[1]
    ax.plot(df['t_center'], df['spectral_kurtosis'], color='#d62728', linewidth=0.8)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Kurtosis')
    ax.set_title('Spectral Kurtosis (high = tonal, low = broadband)')
    ax.grid(True, alpha=0.3)

    # Panel 3: Band energy ratios
    ax = axes[2]
    band_cols = [c for c in df.columns if c.startswith('E_')]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for col, color in zip(band_cols, colors):
        ax.plot(df['t_center'], df[col], color=color, linewidth=0.8, label=col)
    ax.set_ylabel('Energy Ratio')
    ax.set_title('Band Energy Ratios')
    ax.legend(loc='upper right', ncol=3, fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 4: Anomaly indicator
    ax = axes[3]
    if anomaly_mask is not None and anomaly_mask.any():
        ax.scatter(df['t_center'][anomaly_mask],
                    np.ones(anomaly_mask.sum()),
                    color='red', s=20, marker='|', label='Anomaly')
    ax.plot(df['t_center'], df['centroid'].values, alpha=0.3, linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Centroid (Hz)')
    ax.set_title('Anomaly Detection (red = z-score > 3)')
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Sliding-Window Spectral Analysis — {axis_name}-axis '
                 f'({CFG.WINDOW_DURATION_SEC}s window, {CFG.WINDOW_OVERLAP_RATIO*100:.0f}% overlap)',
                 fontsize=14)
    fig.tight_layout()
    _save(fig, filename.replace('.png', f'_{axis_name}.png'))
    return fig


def plot_anomaly_heatmap(df, axis_name="X", filename="08_anomaly_heatmap.png"):
    """Heatmap of all spectral features over time, normalized per-column."""
    feature_cols = [c for c in df.columns if c != 't_center']
    if not feature_cols:
        return None

    data = df[feature_cols].values
    data_norm = (data - np.median(data, axis=0)) / (np.std(data, axis=0) + 1e-10)

    fig, ax = plt.subplots(figsize=(14, 6))
    mesh = ax.pcolormesh(df['t_center'], np.arange(len(feature_cols)),
                          data_norm.T, shading='auto', cmap='RdBu_r',
                          vmin=-4, vmax=4)
    ax.set_yticks(np.arange(len(feature_cols)))
    ax.set_yticklabels(feature_cols)
    ax.set_xlabel('Time (s)')
    ax.set_title(f'{axis_name}-axis: Spectral Feature Heatmap (z-score normalized)')
    fig.colorbar(mesh, ax=ax, label='Z-score')
    fig.tight_layout()
    _save(fig, filename.replace('.png', f'_{axis_name}.png'))
    return fig
