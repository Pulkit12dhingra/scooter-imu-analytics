"""tests/test_simulator_scenarios.py
Signal-level scenario validation and window labeling tests (Stage 2).
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from live_imu_simulator import LiveScooterSimulator
from ml.window_labeling import label_window


def test_turn_directions_and_opposite_signs():
    """Confirm left_turn and right_turn have opposite sign yaw rate profiles."""
    sim_left = LiveScooterSimulator(sample_rate_hz=50.0, seed=1, mode="batch")
    sim_left.force_scenario("left_turn", duration_s=3.0, turn_sign=1.0, turn_peak=0.4)
    left_samples = list(sim_left.samples(duration_s=3.0))

    sim_right = LiveScooterSimulator(sample_rate_hz=50.0, seed=1, mode="batch")
    sim_right.force_scenario("right_turn", duration_s=3.0, turn_sign=-1.0, turn_peak=0.4)
    right_samples = list(sim_right.samples(duration_s=3.0))

    left_yaw = [s.gyr_y_rad_s for s in left_samples]
    right_yaw = [s.gyr_y_rad_s for s in right_samples]

    # Peak yaw rate for left turn should be positive, right turn negative
    assert max(left_yaw) > 0.25
    assert min(right_yaw) < -0.25
    assert np.mean(left_yaw) > 0.10
    assert np.mean(right_yaw) < -0.10


def test_lane_change_opposing_pulses():
    """Confirm lane_change contains a pair of opposing yaw pulses."""
    sim = LiveScooterSimulator(sample_rate_hz=50.0, seed=42, mode="batch")
    sim.force_scenario("lane_change", duration_s=3.0)
    samples = list(sim.samples(duration_s=3.0))
    yaw = np.array([s.gyr_y_rad_s for s in samples])

    # Should exhibit both significant positive and significant negative excursions
    assert np.max(yaw) > 0.10
    assert np.min(yaw) < -0.08
    # Zero-crossing rate of de-meaned yaw should reflect at least 2 pulses
    signs = np.sign(yaw - np.mean(yaw))
    zero_crossings = np.sum(np.diff(signs) != 0)
    assert zero_crossings >= 2


def test_wear_vibration_frequency_content():
    """Confirm wear_vibration produces dominant energy around ~8 Hz."""
    sim = LiveScooterSimulator(sample_rate_hz=50.0, seed=42, mode="batch")
    sim.force_scenario("wear_vibration", duration_s=4.0)
    samples = list(sim.samples(duration_s=4.0))

    gx = np.array([s.gyr_x_rad_s for s in samples])
    n = len(gx)
    dt = 1.0 / 50.0

    # 50 Hz satisfies Nyquist limit (25 Hz > 8 Hz)
    fft_vals = np.abs(np.fft.rfft(gx - np.mean(gx)))
    freqs = np.fft.rfftfreq(n, d=dt)

    peak_freq = freqs[np.argmax(fft_vals[1:]) + 1]
    assert 7.0 <= peak_freq <= 9.0, f"Expected peak ~8 Hz, got {peak_freq:.2f} Hz"


def test_fall_event_transient_spike():
    """Confirm synthetic fall scenario produces an energetic multi-axis spike."""
    sim = LiveScooterSimulator(sample_rate_hz=50.0, seed=7, mode="batch")
    sim.force_scenario("fall", duration_s=1.5)
    samples = list(sim.samples(duration_s=1.5))

    mags = np.array([
        math.sqrt(s.gyr_x_rad_s**2 + s.gyr_y_rad_s**2 + s.gyr_z_rad_s**2)
        for s in samples
    ])
    assert np.max(mags) >= 1.0, f"Expected fall peak >= 1.0 rad/s, got {np.max(mags):.2f}"


def test_parked_and_straight_characteristics():
    """Confirm parked is stationary noise and straight is low-amplitude."""
    sim_parked = LiveScooterSimulator(sample_rate_hz=50.0, seed=42, mode="batch")
    sim_parked.force_scenario("parked", duration_s=2.0)
    parked_mags = [
        math.sqrt(s.gyr_x_rad_s**2 + s.gyr_y_rad_s**2 + s.gyr_z_rad_s**2)
        for s in sim_parked.samples(duration_s=2.0)
    ]
    # Parked magnitude is very small (< 0.03 rad/s)
    assert np.mean(parked_mags) < 0.02
    assert np.max(parked_mags) < 0.05

    sim_straight = LiveScooterSimulator(sample_rate_hz=50.0, seed=42, mode="batch")
    sim_straight.force_scenario("straight", duration_s=2.0)
    straight_mags = [
        math.sqrt(s.gyr_x_rad_s**2 + s.gyr_y_rad_s**2 + s.gyr_z_rad_s**2)
        for s in sim_straight.samples(duration_s=2.0)
    ]
    assert np.mean(straight_mags) > np.mean(parked_mags)
    assert np.max(straight_mags) < 0.10


def test_window_labeling_purity_policy():
    """Test 80% purity rule for sustained maneuver classes."""
    # 50 samples with 45 straight (90%) -> straight
    maneuvers = ["straight"] * 45 + ["left_turn"] * 5
    scenarios = ["straight"] * 45 + ["left_turn"] * 5
    mags = np.ones(50) * 0.05
    label, condition = label_window(maneuvers, scenarios, mags, min_purity=0.8)
    assert label == "straight"
    assert condition == ""

    # Ambiguous transition: 30 straight, 20 left_turn (60% < 80%) -> None
    maneuvers_amb = ["straight"] * 30 + ["left_turn"] * 20
    scenarios_amb = ["straight"] * 30 + ["left_turn"] * 20
    label_amb, _ = label_window(maneuvers_amb, scenarios_amb, mags, min_purity=0.8)
    assert label_amb is None


def test_window_labeling_event_aware_fall():
    """Test event-aware labeling for fall."""
    # Window with fall scenario AND strong impact peak -> fall
    maneuvers = ["straight"] * 40 + ["fall"] * 10
    scenarios = ["straight"] * 40 + ["fall"] * 10
    mags = np.array([0.05] * 45 + [1.4] * 5)
    label, _ = label_window(maneuvers, scenarios, mags, min_purity=0.8)
    assert label == "fall"

    # Window with fall scenario label but NO energy spike (< 0.8 rad/s) -> falls back to purity check
    mags_low = np.array([0.05] * 50)
    label_low, _ = label_window(maneuvers, scenarios, mags_low, min_purity=0.8)
    # Since straight is 40/50 = 80%, it gets straight
    assert label_low == "straight"

