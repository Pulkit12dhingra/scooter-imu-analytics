"""tests/test_live_imu_simulator.py
Unit tests for the procedural live IMU simulator (Stage 1).
"""
import io
import math
import sys
import time
from pathlib import Path

import pytest

# Ensure scripts directory is importable
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import live_imu_simulator as sim
from live_imu_simulator import (
    GROUND_TRUTH_CSV_COLUMNS,
    SENSOR_CSV_COLUMNS,
    ImuSample,
    LiveScooterSimulator,
    main,
    validate_args,
)


def test_simulator_no_dataset_dependency():
    """Verify simulator starts procedurally without reading external datasets."""
    simulator = LiveScooterSimulator(sample_rate_hz=50.0, seed=123, mode="batch")
    samples = list(simulator.samples(duration_s=0.2))
    assert len(samples) == 10
    assert simulator.sample_index == 10


def test_batch_mode_fast_execution():
    """Verify batch mode does not sleep and generates 500 samples rapidly (< 0.5s)."""
    simulator = LiveScooterSimulator(sample_rate_hz=50.0, seed=42, mode="batch")
    start = time.monotonic()
    samples = list(simulator.samples(duration_s=10.0))  # 10s * 50Hz = 500 samples
    elapsed = time.monotonic() - start

    assert len(samples) == 500
    assert elapsed < 0.5, f"Batch mode took too long ({elapsed:.3f}s), expected < 0.5s"


def test_live_mode_timing():
    """Verify live mode approximates configured sampling cadence."""
    simulator = LiveScooterSimulator(sample_rate_hz=50.0, seed=42, mode="live", speed=1.0)
    start = time.monotonic()
    samples = list(simulator.samples(duration_s=0.1))  # 5 samples = 100 ms
    elapsed = time.monotonic() - start

    assert len(samples) == 5
    # Should take at least ~80 ms due to real-time pacing
    assert elapsed >= 0.07, f"Live mode elapsed time {elapsed:.3f}s too fast for 50 Hz"


def test_sample_schema_and_finite_values():
    """Verify every sample has finite numbers and sensor fields match schema."""
    simulator = LiveScooterSimulator(sample_rate_hz=50.0, seed=99, mode="batch")
    for sample in simulator.samples(duration_s=2.0):
        assert isinstance(sample.sample_index, int)
        assert isinstance(sample.elapsed_s, float)
        assert abs(sample.dt_s - 0.02) < 1e-5

        for val in (sample.gyr_x_rad_s, sample.gyr_y_rad_s, sample.gyr_z_rad_s):
            assert math.isfinite(val)
            assert not math.isnan(val)

        # Check sensor-only dictionary
        sensor_dict = sample.to_sensor_dict()
        assert set(sensor_dict.keys()) == set(SENSOR_CSV_COLUMNS)
        assert "scenario" not in sensor_dict
        assert "maneuver_label" not in sensor_dict
        assert "condition_label" not in sensor_dict

        # Check ground truth dictionary
        gt_dict = sample.to_ground_truth_dict()
        assert set(gt_dict.keys()) == set(GROUND_TRUTH_CSV_COLUMNS)


def test_determinism_with_same_seed():
    """Two batch simulations with identical seed must yield identical readings."""
    sim1 = LiveScooterSimulator(sample_rate_hz=50.0, seed=42, mode="batch")
    sim2 = LiveScooterSimulator(sample_rate_hz=50.0, seed=42, mode="batch")

    samples1 = list(sim1.samples(duration_s=5.0))
    samples2 = list(sim2.samples(duration_s=5.0))

    assert len(samples1) == len(samples2)
    for s1, s2 in zip(samples1, samples2):
        assert s1.sample_index == s2.sample_index
        assert s1.elapsed_s == s2.elapsed_s
        assert s1.gyr_x_rad_s == s2.gyr_x_rad_s
        assert s1.gyr_y_rad_s == s2.gyr_y_rad_s
        assert s1.gyr_z_rad_s == s2.gyr_z_rad_s
        assert s1.scenario == s2.scenario
        assert s1.maneuver_label == s2.maneuver_label
        assert s1.condition_label == s2.condition_label


def test_different_seeds_produce_different_sequences():
    """Different seeds should generate differing readings or scenario choices."""
    sim1 = LiveScooterSimulator(sample_rate_hz=50.0, seed=10, mode="batch")
    sim2 = LiveScooterSimulator(sample_rate_hz=50.0, seed=20, mode="batch")

    samples1 = list(sim1.samples(duration_s=5.0))
    samples2 = list(sim2.samples(duration_s=5.0))

    # At least some signals should differ
    signals1 = [(s.gyr_x_rad_s, s.gyr_y_rad_s, s.gyr_z_rad_s) for s in samples1]
    signals2 = [(s.gyr_x_rad_s, s.gyr_y_rad_s, s.gyr_z_rad_s) for s in samples2]
    assert signals1 != signals2


def test_cli_argument_validation():
    """Verify validation rejects invalid rates, durations, speeds, and ports."""
    # Invalid sample rate
    with pytest.raises(ValueError, match="--rate"):
        args = sim.parse_args(["--rate", "-10"])
        validate_args(args)

    # Invalid duration
    with pytest.raises(ValueError, match="--duration"):
        args = sim.parse_args(["--duration", "0"])
        validate_args(args)

    # Invalid speed
    with pytest.raises(ValueError, match="--speed"):
        args = sim.parse_args(["--speed", "-1"])
        validate_args(args)

    # Invalid port
    with pytest.raises(ValueError, match="--port"):
        args = sim.parse_args(["--port", "70000"])
        validate_args(args)


def test_cli_end_to_end_batch_stdout_csv(monkeypatch, tmp_path):
    """Verify end-to-end batch execution emits sensor-only CSV over stdout."""
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout_buf)
    monkeypatch.setattr(sys, "stderr", stderr_buf)

    gt_path = tmp_path / "ground_truth.csv"
    ret = main([
        "--mode", "batch",
        "--format", "csv",
        "--duration", "1.0",
        "--seed", "42",
        "--record-ground-truth", str(gt_path),
    ])

    assert ret == 0
    stdout_lines = stdout_buf.getvalue().strip().split("\n")
    # 1 header + 50 sample lines = 51 lines
    assert len(stdout_lines) == 51
    assert stdout_lines[0] == ",".join(SENSOR_CSV_COLUMNS)

    # Ensure no label in stdout
    assert "scenario" not in stdout_lines[0]
    assert "maneuver_label" not in stdout_lines[0]

    # Verify separate ground truth file
    assert gt_path.exists()
    gt_lines = gt_path.read_text().strip().split("\n")
    assert len(gt_lines) == 51
    assert gt_lines[0] == ",".join(GROUND_TRUTH_CSV_COLUMNS)

