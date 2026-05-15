"""Data loading and resampling utilities for ADXL355 CSV data."""

import numpy as np
import pandas as pd


def load_adxl_csv(filepath, skip_rows=2):
    """Load ADXL355 CSV and return time (ms), acceleration (N,3) in g, temperature (N,) in C.

    CSV format:
      Row 0: column headers "Time,X,Y,Z,Temp"
      Row 1: "Connect Success. Format: Time(ms),X(g),Y(g),Z(g),Temp(C)"
      Row 2+: comma-separated float values
    """
    df = pd.read_csv(filepath, skiprows=skip_rows, header=None,
                     names=["Time", "X", "Y", "Z", "Temp"])
    df = df.dropna(subset=["Time", "X", "Y", "Z"])
    time_ms = df["Time"].values.astype(np.float64)
    accel = df[["X", "Y", "Z"]].values.astype(np.float64)
    temp = df["Temp"].values.astype(np.float64)
    return time_ms, accel, temp


def resample_to_uniform(time_ms, accel, fs_target=1000.0):
    """Resample irregularly-sampled data to a uniform grid via linear interpolation.

    Args:
        time_ms: (N,) raw timestamps in milliseconds.
        accel: (N, 3) acceleration in g.
        fs_target: target sampling rate in Hz.

    Returns:
        t_uniform: (M,) uniform time grid in seconds starting from 0.
        accel_uniform: (M, 3) resampled acceleration.
    """
    time_sec = (time_ms - time_ms[0]) / 1000.0
    duration = time_sec[-1]
    n_samples = int(duration * fs_target) + 1
    t_uniform = np.arange(n_samples) / fs_target

    accel_uniform = np.empty((n_samples, 3), dtype=np.float64)
    for i in range(3):
        accel_uniform[:, i] = np.interp(t_uniform, time_sec, accel[:, i])

    return t_uniform, accel_uniform


def load_and_resample_pipeline(filepath, fs_target=1000.0, skip_rows=2):
    """Convenience: load CSV, resample, return uniform time + acceleration.

    Returns:
        t: (M,) time in seconds.
        accel: (M, 3) resampled acceleration in g.
    """
    time_ms, accel, _ = load_adxl_csv(filepath, skip_rows)
    t, accel_uniform = resample_to_uniform(time_ms, accel, fs_target)
    return t, accel_uniform


def load_unix_csv(filepath):
    """Load newer-format ADXL355 CSV with Unix timestamps.

    CSV format:
      Row 0: column headers "timestamp,accel_x,accel_y,accel_z"
      Row 1+: Unix timestamp (seconds), accel_x, accel_y, accel_z (g)

    Returns:
        time_sec: (N,) timestamps in seconds, relative to first sample.
        accel: (N, 3) acceleration in g.
    """
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["timestamp", "accel_x", "accel_y", "accel_z"])
    ts = df["timestamp"].values.astype(np.float64)
    time_sec = ts - ts[0]
    accel = df[["accel_x", "accel_y", "accel_z"]].values.astype(np.float64)
    return time_sec, accel


def resample_to_uniform_sec(time_sec, accel, fs_target=1000.0):
    """Resample to uniform grid from time-in-seconds input.

    Args:
        time_sec: (N,) timestamps in seconds from 0.
        accel: (N, 3) acceleration in g.
        fs_target: target sampling rate in Hz.

    Returns:
        t_uniform: (M,) uniform time grid in seconds.
        accel_uniform: (M, 3) resampled acceleration.
    """
    duration = time_sec[-1]
    n_samples = int(duration * fs_target) + 1
    t_uniform = np.arange(n_samples) / fs_target

    accel_uniform = np.empty((n_samples, 3), dtype=np.float64)
    for i in range(3):
        accel_uniform[:, i] = np.interp(t_uniform, time_sec, accel[:, i])

    return t_uniform, accel_uniform


def load_unix_and_resample(filepath, fs_target=1000.0):
    """Convenience: load Unix-timestamp CSV, resample.

    Returns:
        t: (M,) time in seconds.
        accel: (M, 3) resampled acceleration in g.
    """
    time_sec, accel = load_unix_csv(filepath)
    return resample_to_uniform_sec(time_sec, accel, fs_target)
