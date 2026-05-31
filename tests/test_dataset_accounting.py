"""
tests/test_dataset_accounting.py
Unit tests for Stage 1 dataset accounting and window calculations.
"""
import numpy as np
import pandas as pd
import pytest

from ml.features import extract_windows


def calculate_expected_windows(n_samples: int, win_size: int, step_size: int) -> int:
    """Exact complete-window accounting formula."""
    if n_samples < win_size:
        return 0
    return 1 + (n_samples - win_size) // step_size


def test_window_accounting_boundary_cases():
    """Verify window counting under N < W, N == W, and N > W conditions."""
    win_size = 50
    step_size = 25

    # N < W
    assert calculate_expected_windows(30, win_size, step_size) == 0
    # N == W
    assert calculate_expected_windows(50, win_size, step_size) == 1
    # N == W + step_size - 1
    assert calculate_expected_windows(74, win_size, step_size) == 1
    # N == W + step_size
    assert calculate_expected_windows(75, win_size, step_size) == 2
    # Standard 10-min 50 Hz dataset (30,000 samples)
    assert calculate_expected_windows(30000, win_size, step_size) == 1199


def test_extract_windows_agrees_with_accounting_formula():
    """Verify ml.features.extract_windows produces exact expected count."""
    n_samples = 200
    timestamps = pd.date_range("2024-03-15 10:00:00", periods=n_samples, freq="20ms")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "dt": np.full(n_samples, 0.02),
        "gyr_x": np.zeros(n_samples),
        "gyr_y": np.zeros(n_samples),
        "gyr_z": np.zeros(n_samples),
        "gyr_magnitude": np.zeros(n_samples),
    })

    # 1.0s window @ 50 Hz = 50 samples, 0.5s step = 25 samples
    X, mid_ts = extract_windows(df, window_s=1.0, step_s=0.5)
    expected = calculate_expected_windows(n_samples, 50, 25)
    assert len(X) == expected
    assert len(mid_ts) == expected
    assert len(X) == 7  # 1 + (200 - 50) // 25 = 1 + 6 = 7

