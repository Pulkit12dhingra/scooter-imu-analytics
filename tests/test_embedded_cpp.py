"""tests/test_embedded_cpp.py
Unit tests for the Stage 6 C++ stream parser executable.
"""
from pathlib import Path
import subprocess
import pytest

BINARY_PATH = Path("embedded/build/edge_scoot")


@pytest.fixture(scope="session", autouse=True)
def ensure_edge_scoot_binary():
    """Ensure edge_scoot binary is compiled before tests run."""
    if not BINARY_PATH.exists():
        cmake_bin = "/opt/homebrew/bin/cmake"
        subprocess.run([cmake_bin, "-S", "embedded", "-B", "embedded/build"], check=True)
        subprocess.run([cmake_bin, "--build", "embedded/build"], check=True)
    assert BINARY_PATH.exists()


def test_edge_scoot_valid_csv():
    """Verify edge_scoot parses valid CSV sample lines."""
    input_data = (
        "sample_index,elapsed_s,dt_s,gyr_x_rad_s,gyr_y_rad_s,gyr_z_rad_s\n"
        "0,0.000000,0.020000,0.0150000,-0.0200000,0.0050000\n"
        "1,0.020000,0.020000,0.0100000,-0.0180000,0.0040000\n"
    )
    proc = subprocess.run(
        [str(BINARY_PATH)],
        input=input_data,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0
    assert "Total samples: 2" in proc.stderr
    assert "Predictions emitted: 0" in proc.stderr


def test_edge_scoot_empty_stream():
    """Verify edge_scoot exits cleanly with 0 on empty input."""
    proc = subprocess.run(
        [str(BINARY_PATH)],
        input="",
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0
    assert "Total samples: 0" in proc.stderr


def test_edge_scoot_malformed_columns():
    """Verify edge_scoot rejects rows with wrong column count."""
    input_data = "0,0.0,0.02,0.01,0.02\n"  # Only 5 columns
    proc = subprocess.run(
        [str(BINARY_PATH)],
        input=input_data,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 1
    assert "Expected exactly 6 columns" in proc.stderr


def test_edge_scoot_nan_rejection():
    """Verify edge_scoot rejects NaN values."""
    input_data = "0,0.0,0.02,NaN,0.02,0.01\n"
    proc = subprocess.run(
        [str(BINARY_PATH)],
        input=input_data,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 1
    assert "NaN or Inf detected" in proc.stderr
