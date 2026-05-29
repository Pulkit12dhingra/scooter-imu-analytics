# Simulator Design & Signal Contract

This document describes the design, signal specifications, and execution interfaces of the procedural live IMU simulator implemented in `scripts/live_imu_simulator.py`.

## 1. Design Purpose

The simulator generates synthetic gyroscope measurements in real time or high-speed batch mode without reading or replaying historical recordings. It acts as the sensor frontend for both the offline ML training pipeline (Batch Mode) and the online C++ embedded inference engine (Live Mode).

## 2. Sensor & Ground-Truth Separation

The simulator enforces a strict separation between sensor readings and scenario metadata:

### A. Sensor-Only Stream (Inference Interface)
Emitted to `stdout` (or sent via UDP):
- **Columns (CSV):** `sample_index,elapsed_s,dt_s,gyr_x_rad_s,gyr_y_rad_s,gyr_z_rad_s`
- **JSON Lines Format:** `{"sample_index": 0, "elapsed_s": 0.0, "dt_s": 0.02, "gyr_x_rad_s": 0.001, "gyr_y_rad_s": -0.002, "gyr_z_rad_s": 0.0005}`
- **Zero Label Leakage:** Scenario names and maneuver labels are **never** included in this stream.

### B. Ground-Truth Stream (Evaluation & Training Interface)
Optionally recorded to a dedicated CSV file via `--record-ground-truth <path>`:
- **Columns:** `sample_index,elapsed_s,scenario,maneuver_label,condition_label`
- Associates each generated sample index with the true underlying scenario and classification target.

## 3. Scenarios & Physical Signal Conventions

All angular rates are modeled in **radians per second (`rad/s`)**. Nominal sampling cadence is 50 Hz (`dt = 0.02 s`).

| Scenario Name | Duration Range | Maneuver Target | Condition Label | Signal Characteristics |
|---|---|---|---|---|
| `parked` | 5.0 – 12.0 s | `parked` | `""` | Sensor baseline noise + micro-movements (<0.005 rad/s). |
| `straight` | 6.0 – 15.0 s | `straight` | `""` | Low-amplitude riding oscillations (0.01 – 0.02 rad/s). |
| `left_turn` | 2.0 – 4.0 s | `left_turn` | `""` | Positive yaw envelope (`gyr_y` peak +0.18 to +0.65 rad/s). |
| `right_turn` | 2.0 – 4.0 s | `right_turn` | `""` | Negative yaw envelope (`gyr_y` peak -0.18 to -0.65 rad/s). |
| `lane_change` | 2.0 – 4.0 s | `lane_change` | `""` | Two opposing Gaussian yaw pulses in quick succession. |
| `sharp_turn` | 1.5 – 3.0 s | `sharp_turn` | `""` | Strong turn envelope (peak 1.22 to 1.55 rad/s). |
| `bumpy_road` | 4.0 – 9.0 s | `bumpy_road` | `""` | High-frequency pitch disturbance (1.4 – 2.2 Hz) with impulse noise. |
| `fall` | 1.0 – 2.0 s | `fall` | `""` | Multi-axis transient spike (simulated event test pulse). |
| `aggressive` | 3.0 – 7.0 s | `aggressive` | `""` | Rapid, oscillatory high-amplitude yaw reversals (~0.7 rad/s @ 0.85 Hz). |
| `wear_vibration` | 4.0 – 8.0 s | `straight` | `wear_vibration` | Narrow-band mechanical vibration overlay (~8 Hz harmonic). |

## 4. Execution Modes

1. **Live Mode (`--mode live`):**
   - Synchronizes sample emission to the wall clock using high-resolution monotonic deadlines.
   - Paced to nominal sensor sample rate (e.g., 50 Hz = ~20 ms per sample).
   - Supports `--speed <multiplier>` for accelerated real-time simulation.

2. **Batch Mode (`--mode batch`):**
   - Completely bypasses sleep calls.
   - Computes and emits samples synchronously at maximum host CPU throughput (typically >5,000 samples/sec).
   - Feeds offline training data generation and automated unit testing.

## 5. Command-Line Reference

```bash
# Stream sensor CSV to stdout for 10 seconds:
uv run python scripts/live_imu_simulator.py --mode live --format csv --duration 10

# Generate 60 seconds of batch data with deterministic seed and save ground truth:
uv run python scripts/live_imu_simulator.py --mode batch --duration 60 --seed 42 \
  --record-ground-truth artifacts/edge/ground_truth.csv > /dev/null
```

