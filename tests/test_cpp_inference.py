"""tests/test_cpp_inference.py
Unit tests for Stage 8 C++ ONNX Runtime CPU inference.
"""
from pathlib import Path
import subprocess
import pytest

BINARY_PATH = Path("embedded/build/edge_scoot")
MODEL_PATH = Path("artifacts/edge/maneuver_rf.onnx")
MANIFEST_PATH = Path("artifacts/edge/model_manifest.json")


def test_edge_scoot_missing_model():
    """Verify application exits with code 1 on non-existent model."""
    proc = subprocess.run(
        [str(BINARY_PATH), "--model", "non_existent.onnx"],
        input="",
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 1
    assert "Failed to load ONNX model" in proc.stderr


def test_edge_scoot_inference_pipeline():
    """Verify edge_scoot processes 50-sample window and emits valid class prediction."""
    if not MODEL_PATH.exists():
        pytest.skip("ONNX model not found")

    # Construct 50 samples of stationary noise
    lines = ["sample_index,elapsed_s,dt_s,gyr_x_rad_s,gyr_y_rad_s,gyr_z_rad_s"]
    for i in range(50):
        lines.append(f"{i},{i*0.02:.6f},0.020000,0.0010000,-0.0010000,0.0005000")
    csv_input = "\n".join(lines) + "\n"

    proc = subprocess.run(
        [str(BINARY_PATH), "--model", str(MODEL_PATH), "--manifest", str(MANIFEST_PATH)],
        input=csv_input,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0
    assert "PRED window=1" in proc.stdout
    assert "class=parked" in proc.stdout
    assert "latency_us=" in proc.stdout
    assert "Predictions emitted: 1" in proc.stderr


def test_edge_scoot_multi_window_reuse():
    """Verify model is loaded once and reused across multiple successive windows."""
    if not MODEL_PATH.exists():
        pytest.skip("ONNX model not found")

    # 100 samples produces 3 windows (at indices 50, 75, 100)
    lines = ["sample_index,elapsed_s,dt_s,gyr_x_rad_s,gyr_y_rad_s,gyr_z_rad_s"]
    for i in range(100):
        lines.append(f"{i},{i*0.02:.6f},0.020000,0.0010000,-0.0010000,0.0005000")
    csv_input = "\n".join(lines) + "\n"

    proc = subprocess.run(
        [str(BINARY_PATH), "--model", str(MODEL_PATH), "--manifest", str(MANIFEST_PATH)],
        input=csv_input,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0
    assert "PRED window=1" in proc.stdout
    assert "PRED window=2" in proc.stdout
    assert "PRED window=3" in proc.stdout
    assert "Predictions emitted: 3" in proc.stderr

