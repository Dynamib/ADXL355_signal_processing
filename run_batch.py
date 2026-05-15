#!/usr/bin/env python3
"""Batch process all CSV files in Data/ folder.

Runs the full signal processing pipeline on each file, saving
results to output/<dataset_name>/figures/.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import (FS_TARGET, AXIS_NAMES, BP_LOWCUT, BP_HIGHCUT, BP_ORDER,
                    MED_FILTER_LENGTH, MED_MAX_ITER,
                    STFT_NPERSEG, STFT_NOVERLAP, STFT_NFFT, SST_NV,
                    WELCH_NPERSEG, WELCH_NOVERLAP,
                    WINDOW_DURATION_SEC, WINDOW_OVERLAP_RATIO,
                    SPECTRAL_BANDS, Z_THRESHOLD, SST_FS)

from src.io_utils import load_unix_and_resample
from src.preprocessing import remove_gravity, preprocess_per_axis
from src.denoising import denoise_per_axis
from src.time_frequency import compute_stft, compute_sst
from src.statistical import (compute_welch_psd, sliding_spectral_features,
                              detect_anomalies_zscore)
from src.visualization import (plot_raw_time_series, plot_fft_spectrum,
                                plot_psd_comparison, plot_bandpass_filtered,
                                plot_med_envelope, plot_stft_spectrogram,
                                plot_sst_spectrogram, plot_sliding_features,
                                plot_anomaly_heatmap)

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "Data"
OUTPUT_BASE = PROJECT_ROOT / "output"


def process_one_file(csv_path, output_dir):
    """Run full pipeline on one CSV file, save results to output_dir."""
    name = csv_path.stem
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Monkey-patch visualization's save path
    import src.visualization as viz
    import config as CFG
    orig_fig_dir = CFG.FIGURE_DIR
    CFG.FIGURE_DIR = fig_dir

    print(f"\n  [{name}] Step 0: Loading...")
    t, accel_raw = load_unix_and_resample(str(csv_path), FS_TARGET)
    duration = t[-1]
    print(f"    Samples: {len(t)}, duration: {duration:.1f}s")

    accel_dc = remove_gravity(accel_raw)
    for i, ax in enumerate(AXIS_NAMES):
        print(f"    {ax}: mean={accel_raw[:, i].mean():.4f}g")

    viz.plot_raw_time_series(t, accel_raw)
    viz.plot_fft_spectrum(t, accel_raw)

    # Step 1: Denoising
    print(f"  [{name}] Step 1: Denoising...")
    accel_bp = preprocess_per_axis(accel_dc, FS_TARGET, BP_LOWCUT, BP_HIGHCUT, BP_ORDER)
    dn = denoise_per_axis(accel_bp, FS_TARGET, MED_FILTER_LENGTH, MED_MAX_ITER)

    for i, ax in enumerate(AXIS_NAMES):
        kb, ka = dn['kurt_before'][i], dn['kurt_after'][i]
        print(f"    {ax} kurtosis: {kb:.2f} -> {ka:.2f}  (Δ{ka - kb:+.2f})")

    # PSD comparison
    from scipy.signal import welch
    psd_flat = []
    for i, ax_name in enumerate(AXIS_NAMES):
        f1, psd1 = welch(accel_dc[:, i], fs=FS_TARGET, nperseg=WELCH_NPERSEG, noverlap=WELCH_NOVERLAP)
        f2, psd2 = welch(accel_bp[:, i], fs=FS_TARGET, nperseg=WELCH_NPERSEG, noverlap=WELCH_NOVERLAP)
        psd_flat.append((f1, psd1, f'{ax_name} raw'))
        psd_flat.append((f2, psd2, f'{ax_name} bandpass'))
    viz.plot_psd_comparison(psd_flat)

    for i, ax in enumerate(AXIS_NAMES):
        viz.plot_bandpass_filtered(t, accel_dc, accel_bp, axis_idx=i)
        viz.plot_med_envelope(t, accel_bp[:, i], dn['med'][:, i],
                              dn['envelope'][:, i], dn['log_envelope'][:, i],
                              dn['kurt_before'][i], dn['kurt_after'][i],
                              axis_name=ax)

    # Step 2: Time-frequency
    print(f"  [{name}] Step 2: Time-frequency...")
    for i, ax in enumerate(AXIS_NAMES):
        signal = dn['med'][:, i]
        f_stft, t_stft, Zxx = compute_stft(signal, FS_TARGET,
                                           nperseg=STFT_NPERSEG, noverlap=STFT_NOVERLAP,
                                           nfft=STFT_NFFT, window='hann')
        viz.plot_stft_spectrogram(f_stft, t_stft, Zxx, FS_TARGET,
                                  title=f"STFT: {ax}-axis",
                                  filename=f"05_stft_{ax}.png")
        try:
            Tx, freqs_sst, t_sst, _ = compute_sst(signal, FS_TARGET, nv=SST_NV, fs_sst=SST_FS)
            viz.plot_sst_spectrogram(Tx, freqs_sst, t_sst,
                                     title=f"SST: {ax}-axis",
                                     filename=f"06_sst_{ax}.png")
        except Exception as e:
            print(f"    WARNING SST {ax}: {e}")

    # Step 3: Statistical
    print(f"  [{name}] Step 3: Statistical...")
    for i, ax in enumerate(AXIS_NAMES):
        signal = dn['med'][:, i]
        df = sliding_spectral_features(signal, FS_TARGET,
                                       window_dur=WINDOW_DURATION_SEC,
                                       overlap=WINDOW_OVERLAP_RATIO,
                                       bands=SPECTRAL_BANDS)
        feature_cols = ['centroid', 'bandwidth', 'spectral_kurtosis']
        feature_cols += [c for c in df.columns if c.startswith('E_')]
        anomalies = detect_anomalies_zscore(df, feature_cols, Z_THRESHOLD)
        n_a = anomalies.sum()
        print(f"    {ax}: {n_a}/{len(df)} windows flagged")
        if n_a > 0:
            df[anomalies].to_csv(fig_dir / f"anomaly_windows_{ax}.csv", index=False)
        viz.plot_sliding_features(df, anomalies, axis_name=ax)
        viz.plot_anomaly_heatmap(df, axis_name=ax)

    # Restore original figure dir
    CFG.FIGURE_DIR = orig_fig_dir

    print(f"  [{name}] Done -> {fig_dir}")
    return dn['kurt_before'], dn['kurt_after']


def main():
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        print("No CSV files found in Data/")
        return

    print(f"Found {len(csv_files)} dataset(s) in {DATA_DIR}")
    print("=" * 60)

    for csv_path in csv_files:
        name = csv_path.stem
        print(f"\n{'='*60}")
        print(f"Processing: {name}")
        print(f"{'='*60}")
        out_dir = OUTPUT_BASE / name
        process_one_file(csv_path, out_dir)

    print(f"\n{'='*60}")
    print(f"All done. Results in {OUTPUT_BASE}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
