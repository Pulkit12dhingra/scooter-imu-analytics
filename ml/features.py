"""
ml/features.py
Sliding-window feature extraction from preprocessed IMU DataFrames.

Each window produces 39 features:
  - 9 stats × 4 signals (gyr_x, gyr_y, gyr_z, magnitude) = 36
  - 3 cross-axis Pearson correlations

Stats per window: mean, std, min, max, rms, zero-crossing-rate, fft_bin1-3
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple

WINDOW_S = 1.0   # seconds per window
STEP_S   = 0.5   # seconds between windows (50 % overlap)


# ── Per-axis features ────────────────────────────────────────────────────── #

def _window_features(w: np.ndarray) -> np.ndarray:
    """9 features for a 1-D signal window."""
    mean  = w.mean()
    std   = w.std() + 1e-12          # avoid zero-std issues
    mn    = w.min()
    mx    = w.max()
    rms   = float(np.sqrt((w ** 2).mean()))
    zcr   = float((np.diff(np.sign(w)) != 0).sum()) / len(w)

    # FFT: first 3 non-DC frequency bins
    fft_mag = np.abs(np.fft.rfft(w - mean))[1:4]
    if len(fft_mag) < 3:
        fft_mag = np.pad(fft_mag, (0, 3 - len(fft_mag)))

    return np.array([mean, std, mn, mx, rms, zcr, *fft_mag], dtype=np.float64)


# ── Main API ─────────────────────────────────────────────────────────────── #

def extract_windows(
    df: pd.DataFrame,
    window_s: float = WINDOW_S,
    step_s: float   = STEP_S,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slide a window over the preprocessed DataFrame and compute features.

    Parameters
    ----------
    df       : wide-format preprocessed DataFrame (output of preprocess())
    window_s : window length in seconds
    step_s   : step size between windows in seconds

    Returns
    -------
    X          : (n_windows, 39) float64 feature matrix
    timestamps : (n_windows,) array of window-centre timestamps
    """
    dt        = df["dt"].median()
    win_size  = max(2, int(window_s / dt))
    step_size = max(1, int(step_s   / dt))

    signals = df[["gyr_x", "gyr_y", "gyr_z", "gyr_magnitude"]].to_numpy(dtype=np.float64)
    ts      = df["timestamp"].to_numpy()

    rows, mid_ts = [], []
    for s in range(0, len(signals) - win_size + 1, step_size):
        e = s + win_size
        w = signals[s:e]                              # (win_size, 4)

        feats = np.concatenate([_window_features(w[:, i]) for i in range(4)])

        # Cross-axis Pearson correlations (handle constant windows)
        corr = []
        for i, j in [(0, 1), (1, 2), (0, 2)]:
            c = np.corrcoef(w[:, i], w[:, j])[0, 1]
            corr.append(0.0 if np.isnan(c) else c)
        feats = np.append(feats, corr)

        rows.append(feats)
        mid_ts.append(ts[s + win_size // 2])

    X = np.array(rows, dtype=np.float64)
    np.nan_to_num(X, copy=False)
    return X, np.array(mid_ts)


def feature_names() -> list[str]:
    """Column names for the X matrix produced by extract_windows()."""
    stats = ["mean", "std", "min", "max", "rms", "zcr", "fft1", "fft2", "fft3"]
    names = [f"{ax}_{s}" for ax in ["gyrx", "gyry", "gyrz", "mag"] for s in stats]
    names += ["corr_xy", "corr_yz", "corr_xz"]
    return names
