"""tests/test_end_to_end_pipe.py
Integration tests for Stage 9 end-to-end simulator-to-C++ inference pipeline.
"""
import subprocess
from pathlib import Path
import pytest

BINARY_PATH = Path("embedded/build/edge_scoot")
MODEL_PATH = Path("artifacts/edge/maneuver_rf.onnx")
MANIFEST_PATH = Path("artifacts/edge/model_manifest.json")
SIMULATOR_SCRIPT = Path("scripts/live_imu_simulator.py")


def test_end_to_end_simulator_to_cpp_pipe(tmp_path):
    """Verify live Python simulator pipes into C++ edge_scoot executable."""
    if not BINARY_PATH.exists() or not MODEL_PATH.exists():
        pytest.skip("Prerequisites missing")

    gt_path = tmp_path / "ground_truth.csv"

    sim_cmd = [
        "python3", "-u", str(SIMULATOR_SCRIPT),
        "--mode", "batch",
        "--rate", "50",
        "--format", "csv",
        "--duration", "2.0",  # 100 samples -> 3 windows (50, 75, 100)
        "--seed", "42",
        "--transport", "stdout",
        "--record-ground-truth", str(gt_path),
    ]

    cpp_cmd = [
        str(BINARY_PATH),
        "--model", str(MODEL_PATH),
        "--manifest", str(MANIFEST_PATH),
    ]

    p_sim = subprocess.Popen(sim_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    p_cpp = subprocess.Popen(cpp_cmd, stdin=p_sim.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    p_sim.stdout.close()  # Allow p_sim to receive SIGPIPE if p_cpp exits

    cpp_out, cpp_err = p_cpp.communicate()
    sim_out, sim_err = p_sim.communicate()

    assert p_sim.returncode == 0
    assert p_cpp.returncode == 0

    lines = [ln for ln in cpp_out.strip().split("\n") if ln.startswith("PRED")]
    assert len(lines) == 3
    assert "PRED window=1" in lines[0]
    assert "PRED window=2" in lines[1]
    assert "PRED window=3" in lines[2]

    # Verify ground truth was recorded separately
    assert gt_path.exists()
    gt_lines = gt_path.read_text().strip().split("\n")
    assert len(gt_lines) == 101  # Header + 100 samples

