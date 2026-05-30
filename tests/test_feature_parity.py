"""tests/test_feature_parity.py
Strict numerical feature parity tests comparing Python ml/features.py
against C++ feature_extractor implementation (Stage 7).
"""
import subprocess
from pathlib import Path
import numpy as np
import pytest

from ml.features import _window_features, feature_names
from ml.train_synthetic import extract_features_from_window

FEATURE_RUNNER_BIN = Path("embedded/build/feature_runner")


def run_cpp_features(samples_array: np.ndarray, dt_s: float = 0.02) -> np.ndarray:
    """Run C++ feature_runner on a (N, 3) gyro array and parse 39 features."""
    assert FEATURE_RUNNER_BIN.exists()
    n_samples = len(samples_array)

    # Build CSV input
    lines = ["sample_index,elapsed_s,dt_s,gyr_x_rad_s,gyr_y_rad_s,gyr_z_rad_s"]
    for i in range(n_samples):
        gx, gy, gz = samples_array[i]
        lines.append(f"{i},{i*dt_s:.6f},{dt_s:.6f},{gx:.10e},{gy:.10e},{gz:.10e}")
    csv_input = "\n".join(lines) + "\n"

    proc = subprocess.run(
        [str(FEATURE_RUNNER_BIN)],
        input=csv_input,
        text=True,
        capture_output=True,
        check=True,
    )
    tokens = [float(v) for v in proc.stdout.strip().split(",")]
    return np.array(tokens, dtype=np.float64)


def compute_python_features(samples_array: np.ndarray) -> np.ndarray:
    """Compute reference Python 39 features on a (N, 3) gyro array."""
    gx, gy, gz = samples_array[:, 0], samples_array[:, 1], samples_array[:, 2]
    mag = np.sqrt(gx**2 + gy**2 + gz**2)
    w_4d = np.column_stack([gx, gy, gz, mag])
    return extract_features_from_window(w_4d)


@pytest.mark.parametrize("seed", [1, 42, 99, 123])
def test_random_signal_feature_parity(seed):
    """Verify all 39 features match between C++ and Python on pseudo-random windows."""
    rng = np.random.default_rng(seed)
    # 50 samples of 3 gyro axes
    samples = rng.standard_normal((50, 3)) * 0.5

    py_feats = compute_python_features(samples)
    cpp_feats = run_cpp_features(samples)

    assert len(py_feats) == 39
    assert len(cpp_feats) == 39
    np.testing.assert_allclose(
        cpp_feats,
        py_feats,
        rtol=1e-5,
        atol=1e-6,
        err_msg=f"Feature mismatch on seed {seed}",
    )


def test_constant_and_zero_signal_parity():
    """Verify constant/zero signals produce finite values and zero correlations."""
    zeros = np.zeros((50, 3), dtype=np.float64)
    py_zeros = compute_python_features(zeros)
    cpp_zeros = run_cpp_features(zeros)

    assert np.all(np.isfinite(cpp_zeros))
    np.testing.assert_allclose(cpp_zeros, py_zeros, atol=1e-6)

    # Constant non-zero signal
    consts = np.ones((50, 3), dtype=np.float64) * 0.25
    py_consts = compute_python_features(consts)
    cpp_consts = run_cpp_features(consts)

    assert np.all(np.isfinite(cpp_consts))
    np.testing.assert_allclose(cpp_consts, py_consts, atol=1e-6)


def test_opposite_phase_and_ramp_parity():
    """Verify linear ramp and opposite phase axes correlations match (-1.0)."""
    t = np.linspace(0, 1, 50)
    gx = t
    gy = -2.0 * t  # Perfect negative correlation
    gz = np.sin(2 * np.pi * 3.0 * t)  # 3 Hz sine
    samples = np.column_stack([gx, gy, gz])

    py_feats = compute_python_features(samples)
    cpp_feats = run_cpp_features(samples)

    np.testing.assert_allclose(cpp_feats, py_feats, rtol=1e-5, atol=1e-6)
    # Check corr_xy is approximately -1.0
    corr_xy = cpp_feats[36]
    assert abs(corr_xy - (-1.0)) < 1e-5


def test_zero_crossing_rate_parity_with_exact_zeros():
    """Verify zero-crossing rate matches when values pass through exact zero."""
    gx = np.array([-1.0, 0.0, 1.0, -1.0, 0.0] * 10)  # 50 samples
    gy = np.zeros(50)
    gz = np.ones(50) * 0.05
    samples = np.column_stack([gx, gy, gz])

    py_feats = compute_python_features(samples)
    cpp_feats = run_cpp_features(samples)

    np.testing.assert_allclose(cpp_feats, py_feats, rtol=1e-5, atol=1e-6)
