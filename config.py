"""Centralized parameters for ADXL355 moxibustion vibration signal processing."""

from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent
DATA_CSV = PROJECT_ROOT.parent / "ADXL355" / "adxl_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURE_DIR = OUTPUT_DIR / "figures"

# --- Data loading ---
SKIP_ROWS = 2
AXIS_NAMES = ["X", "Y", "Z"]
FS_TARGET = 1000.0  # Hz, target uniform resampling rate

# --- Step 1: Bandpass filter ---
BP_LOWCUT = 1.0      # Hz, above breathing (0.2-0.5 Hz)
BP_HIGHCUT = 200.0   # Hz, below Nyquist (500 Hz)
BP_ORDER = 4

# --- Step 1b: MED (Minimum Entropy Deconvolution) ---
MED_FILTER_LENGTH = 30
MED_MAX_ITER = 50
MED_CONV_TOL = 1e-6

# --- Step 2: STFT ---
STFT_NPERSEG = 512
STFT_NOVERLAP = 384   # 75% overlap
STFT_NFFT = 1024

# --- Step 2b: SST ---
SST_NV = 32
SST_FS = 300.0  # Downsample to this rate before SST (avoids 15GiB memory blowup)

# --- Step 3: Welch PSD ---
WELCH_NPERSEG = 1024
WELCH_NOVERLAP = 512  # 50% overlap

# --- Sliding window ---
WINDOW_DURATION_SEC = 2.0
WINDOW_OVERLAP_RATIO = 0.75

# --- Spectral frequency bands ---
SPECTRAL_BANDS = {
    "low":     (1.0, 8.0),
    "tremor":  (8.0, 12.0),
    "beta":    (12.0, 30.0),
    "gamma":   (30.0, 100.0),
    "high":    (100.0, 200.0),
}

# --- Anomaly detection ---
Z_THRESHOLD = 3.0

# --- Visualization ---
FIGURE_DPI = 150
CMAP = "inferno"
