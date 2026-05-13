#!/usr/bin/env python3
"""ADXL355 Moxibustion Vibration Signal Processing Pipeline.

Full pipeline implementing the signal processing knowledge base:
  Step 0: Load, resample, remove gravity
  Step 1: Bandpass filter, MED deconvolution, envelope analysis
  Step 2: STFT + SST time-frequency analysis
  Step 3: Sliding-window spectral features + anomaly detection
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import (DATA_CSV, FS_TARGET, SKIP_ROWS, AXIS_NAMES,
                    BP_LOWCUT, BP_HIGHCUT, BP_ORDER,
                    MED_FILTER_LENGTH, MED_MAX_ITER,
                    STFT_NPERSEG, STFT_NOVERLAP, STFT_NFFT,
                    SST_NV, WELCH_NPERSEG, WELCH_NOVERLAP,
                    WINDOW_DURATION_SEC, WINDOW_OVERLAP_RATIO,
                    SPECTRAL_BANDS, Z_THRESHOLD, FIGURE_DIR)

from src.io_utils import load_adxl_csv, resample_to_uniform
from src.preprocessing import remove_gravity, preprocess_per_axis
from src.denoising import denoise_per_axis, kurtosis
from src.time_frequency import compute_stft, compute_sst
from src.statistical import (compute_welch_psd, sliding_spectral_features,
                              detect_anomalies_zscore)
from src.visualization import (plot_raw_time_series, plot_psd_comparison,
                                plot_bandpass_filtered, plot_med_envelope,
                                plot_stft_spectrogram, plot_sst_spectrogram,
                                plot_sliding_features, plot_anomaly_heatmap)

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def banner(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def main():
    print("ADXL355 Moxibustion Vibration Signal Processing Pipeline")
    print(f"Data: {DATA_CSV}")

    # =====================================================================
    # Step 0: Load, resample, remove gravity
    # =====================================================================
    banner("Step 0: Loading & Resampling")
    time_ms, accel_raw, temp = load_adxl_csv(str(DATA_CSV), SKIP_ROWS)
    duration_raw = (time_ms[-1] - time_ms[0]) / 1000.0
    print(f"  Raw samples: {len(time_ms)}, duration: {duration_raw:.1f} s")

    t, accel_uniform = resample_to_uniform(time_ms, accel_raw, FS_TARGET)
    duration = t[-1]
    print(f"  Resampled to {FS_TARGET} Hz: {len(t)} samples, {duration:.1f} s")

    accel_dc = remove_gravity(accel_uniform)
    for i, name in enumerate(AXIS_NAMES):
        print(f"  {name}-axis mean (gravity): {accel_uniform[:, i].mean():.4f} g")

    # Plot raw time series
    print("  Saving raw time series plot...")
    plot_raw_time_series(t, accel_uniform)

    # =====================================================================
    # Step 1: Denoising — bandpass + MED + envelope
    # =====================================================================
    banner("Step 1: Denoising & Feature Extraction")

    print(f"  Bandpass filter: {BP_LOWCUT}-{BP_HIGHCUT} Hz, order={BP_ORDER}")
    accel_bp = preprocess_per_axis(accel_dc, FS_TARGET, BP_LOWCUT, BP_HIGHCUT, BP_ORDER)

    print(f"  MED filter length: {MED_FILTER_LENGTH}, max iter: {MED_MAX_ITER}")
    dn = denoise_per_axis(accel_bp, FS_TARGET, MED_FILTER_LENGTH, MED_MAX_ITER)

    for i, name in enumerate(AXIS_NAMES):
        kb, ka = dn['kurt_before'][i], dn['kurt_after'][i]
        delta_k = ka - kb
        print(f"  {name}-axis kurtosis: {kb:.2f} -> {ka:.2f} (delta={delta_k:+.2f})")

    # Plot PSD comparison (raw vs bandpass for each axis)
    psd_data = []
    from scipy.signal import welch
    for i, name in enumerate(AXIS_NAMES):
        f1, psd1 = welch(accel_dc[:, i], fs=FS_TARGET, nperseg=WELCH_NPERSEG, noverlap=WELCH_NOVERLAP)
        f2, psd2 = welch(accel_bp[:, i], fs=FS_TARGET, nperseg=WELCH_NPERSEG, noverlap=WELCH_NOVERLAP)
        psd_data.append((f1, psd1, f'{name} raw (DC-free)'))
        psd_data.append((f2, psd2, f'{name} bandpass'))
    # Flatten pairs into alternating list
    psd_flat = []
    for i in range(0, len(psd_data), 2):
        psd_flat.append(psd_data[i])
        psd_flat.append(psd_data[i + 1])
    plot_psd_comparison(psd_flat)

    # Plot bandpass comparison and MED/envelope for each axis
    for i, name in enumerate(AXIS_NAMES):
        print(f"  Plotting {name}-axis preprocessing...")
        plot_bandpass_filtered(t, accel_dc, accel_bp, axis_idx=i)
        plot_med_envelope(t, accel_bp[:, i], dn['med'][:, i],
                          dn['envelope'][:, i], dn['log_envelope'][:, i],
                          dn['kurt_before'][i], dn['kurt_after'][i],
                          axis_name=name)

    # =====================================================================
    # Step 2: Time-frequency analysis (STFT + SST)
    # =====================================================================
    banner("Step 2: Time-Frequency Analysis")

    for i, name in enumerate(AXIS_NAMES):
        signal = dn['med'][:, i]

        # STFT
        print(f"  {name}-axis STFT...")
        f_stft, t_stft, Zxx = compute_stft(
            signal, FS_TARGET,
            nperseg=STFT_NPERSEG, noverlap=STFT_NOVERLAP,
            nfft=STFT_NFFT, window='hann')
        plot_stft_spectrogram(
            f_stft, t_stft, Zxx, FS_TARGET,
            title=f"STFT: {name}-axis (MED output)",
            filename=f"05_stft_{name}.png")

        # SST
        print(f"  {name}-axis SST...")
        try:
            Tx, freqs_sst, t_sst, _ = compute_sst(signal, FS_TARGET, nv=SST_NV)
            plot_sst_spectrogram(
                Tx, freqs_sst, t_sst,
                title=f"SST: {name}-axis (MED output)",
                filename=f"06_sst_{name}.png")
        except Exception as e:
            print(f"  WARNING: SST failed for {name}-axis: {e}")

    # =====================================================================
    # Step 3: Statistical verification — sliding window + anomaly detection
    # =====================================================================
    banner("Step 3: Statistical Verification")

    for i, name in enumerate(AXIS_NAMES):
        signal = dn['med'][:, i]

        print(f"  {name}-axis: sliding window spectral features...")
        df = sliding_spectral_features(
            signal, FS_TARGET,
            window_dur=WINDOW_DURATION_SEC,
            overlap=WINDOW_OVERLAP_RATIO,
            bands=SPECTRAL_BANDS)

        feature_cols = ['centroid', 'bandwidth', 'spectral_kurtosis']
        feature_cols += [c for c in df.columns if c.startswith('E_')]

        anomalies = detect_anomalies_zscore(df, feature_cols, Z_THRESHOLD)
        n_anomaly = anomalies.sum()
        n_total = len(df)
        print(f"  {name}-axis: {n_anomaly}/{n_total} windows flagged "
              f"({100 * n_anomaly / max(n_total, 1):.1f}%)")

        if n_anomaly > 0:
            anomaly_times = df['t_center'][anomalies].values
            anom_path = FIGURE_DIR / f"anomaly_windows_{name}.csv"
            df[anomalies].to_csv(anom_path, index=False)
            print(f"    Anomalous time windows saved to: {anom_path}")

        plot_sliding_features(df, anomalies, axis_name=name)
        plot_anomaly_heatmap(df, axis_name=name)

    # =====================================================================
    # Summary
    # =====================================================================
    banner("Pipeline Complete")
    print(f"  All figures saved to: {FIGURE_DIR}")
    print(f"\n  Summary of kurtosis changes (MED effect):")
    for i, name in enumerate(AXIS_NAMES):
        kb, ka = dn['kurt_before'][i], dn['kurt_after'][i]
        print(f"    {name}-axis: {kb:.2f} -> {ka:.2f}  ({ka - kb:+.2f})")

    print("\n  Knowledge Base alignment:")
    print(f"    Step 1: Bandpass {BP_LOWCUT}-{BP_HIGHCUT}Hz + MED + Hilbert envelope -> DONE")
    print(f"    Step 2: STFT (Hann {STFT_NPERSEG}pt, {STFT_NOVERLAP}ol) + SST (Morlet, nv={SST_NV}) -> DONE")
    print(f"    Step 3: Sliding window spectral features + z-score anomaly detection -> DONE")
    print()

    return t, accel_dc, accel_bp, dn


if __name__ == "__main__":
    main()
