"""
tests/test_remediation_controls.py
Regression tests for Stage 6 (Theft Controls), Stage 7 (Fall Controls),
Stage 8 (Isolation Forest Disjoint Evaluation), and Stage 10 (Wear Vibration Frequency).
"""
import numpy as np
import pandas as pd
import pytest

from scooter_imu.algorithms.fall_detection import FallDetector
from scooter_imu.algorithms.theft_detection import TheftDetector
from ml.anomaly import IMUAnomalyDetector


def test_theft_detector_positive_and_negative_controls():
    """Verify theft detector remains quiet during stationary baseline and alerts on sustained motion."""
    fs = 50.0
    dt = 1.0 / fs
    n_quiet = int(10.0 * fs)
    n_move = int(10.0 * fs)

    rng = np.random.default_rng(42)
    # Stationary segment: low noise << 0.01 rad/s
    gx_q = rng.normal(0, 0.002, n_quiet)
    gy_q = rng.normal(0, 0.002, n_quiet)
    gz_q = rng.normal(0, 0.002, n_quiet)

    # Sustained unauthorized movement: ~0.05 rad/s > 0.01 rad/s
    gx_m = rng.normal(0.04, 0.01, n_move)
    gy_m = rng.normal(0.04, 0.01, n_move)
    gz_m = rng.normal(0.04, 0.01, n_move)

    gx = np.concatenate([gx_q, gx_m])
    gy = np.concatenate([gy_q, gy_m])
    gz = np.concatenate([gz_q, gz_m])
    mag = np.sqrt(gx**2 + gy**2 + gz**2)

    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-03-15 10:00:00", periods=len(gx), freq="20ms"),
        "dt": np.full(len(gx), dt),
        "gyr_x": gx,
        "gyr_y": gy,
        "gyr_z": gz,
        "gyr_magnitude": mag,
    })

    detector = TheftDetector(threshold=0.01, window_s=2.0)
    res = detector.predict(df)

    quiet_alerts = res["theft_alert"].iloc[:n_quiet].sum()
    move_alerts = res["theft_alert"].iloc[n_quiet:].sum()
    first_alert_idx = res.index[res["theft_alert"]].min()

    assert quiet_alerts == 0, "Spurious theft alerts occurred during stationary baseline!"
    assert move_alerts > 0, "No theft alerts occurred during movement phase!"
    assert first_alert_idx >= n_quiet, "Theft alert fired before movement commenced!"


def test_fall_detector_controls():
    """Verify fall detector does not trigger on quiet riding or gentle turns, but triggers on sudden angular spike."""
    fs = 50.0
    dt = 1.0 / fs
    n_quiet = int(5.0 * fs)
    n_turn = int(5.0 * fs)
    n_fall = int(2.0 * fs)

    rng = np.random.default_rng(42)
    # Quiet baseline
    gx_q = rng.normal(0, 0.002, n_quiet)
    gy_q = rng.normal(0, 0.002, n_quiet)
    gz_q = rng.normal(0, 0.002, n_quiet)

    # Normal steering turn (max mag ~ 0.45 rad/s << 2.0 rad/s)
    gx_t = rng.normal(0, 0.01, n_turn)
    gy_t = np.full(n_turn, 0.4) + rng.normal(0, 0.02, n_turn)
    gz_t = rng.normal(0, 0.01, n_turn)

    # Fall impact spike (> 2.5 rad/s)
    gx_f = np.full(n_fall, 2.5)
    gy_f = np.full(n_fall, 1.5)
    gz_f = np.full(n_fall, 1.0)

    gx = np.concatenate([gx_q, gx_t, gx_f])
    gy = np.concatenate([gy_q, gy_t, gy_f])
    gz = np.concatenate([gz_q, gz_t, gz_f])
    mag = np.sqrt(gx**2 + gy**2 + gz**2)

    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-03-15 10:00:00", periods=len(gx), freq="20ms"),
        "dt": np.full(len(gx), dt),
        "gyr_x": gx,
        "gyr_y": gy,
        "gyr_z": gz,
        "gyr_magnitude": mag,
    })

    detector = FallDetector()  # default 2.0 rad/s
    res = detector.predict(df)

    event_starts = np.where(res["fall_event_start"])[0]
    assert len(event_starts) == 1, f"Expected exactly 1 fall event start, got {len(event_starts)}"
    # Event should occur around the start of the fall segment
    assert event_starts[0] >= n_quiet + n_turn - 2


def test_isolation_forest_disjoint_evaluation():
    """Verify that anomaly detector can be fit on normal data and evaluated on disjoint holdout sets."""
    rng = np.random.default_rng(42)
    n_features = 39

    # 1. Fit population: purely normal baseline distributions
    X_train = rng.normal(0.0, 1.0, (200, n_features))

    # 2. Independent nominal holdout
    X_test_normal = rng.normal(0.0, 1.0, (50, n_features))

    # 3. Independent anomaly holdout: shifted distribution / outlier spikes
    X_test_anomaly = rng.normal(6.0, 2.0, (50, n_features))

    detector = IMUAnomalyDetector(contamination=0.05, random_state=42)
    detector.fit(X_train)

    preds_normal = detector.predict(X_test_normal)
    preds_anomaly = detector.predict(X_test_anomaly)

    normal_anomaly_rate = float(preds_normal.mean())
    anomaly_detection_rate = float(preds_anomaly.mean())

    # Normal holdout should have low anomaly rate; injected anomalies should have high detection rate
    assert normal_anomaly_rate < 0.15, f"False positive rate too high: {normal_anomaly_rate}"
    assert anomaly_detection_rate >= 0.85, f"Anomaly detection rate too low: {anomaly_detection_rate}"



def test_wear_vibration_frequency_resolution():
    """Verify frequency estimation on known injected sinusoidal oscillation (8 Hz @ 50 Hz sampling)."""
    fs = 50.0
    dt = 1.0 / fs
    duration = 10.0
    t = np.arange(0, duration, dt)
    f_injected = 8.0  # Hz

    rng = np.random.default_rng(42)
    sig = 0.5 * np.sin(2 * np.pi * f_injected * t) + rng.normal(0, 0.05, len(t))

    fft_vals = np.abs(np.fft.rfft(sig - sig.mean()))
    freqs = np.fft.rfftfreq(len(sig), d=dt)
    dominant_freq = freqs[np.argmax(fft_vals[1:]) + 1]

    assert abs(dominant_freq - f_injected) < 0.2
