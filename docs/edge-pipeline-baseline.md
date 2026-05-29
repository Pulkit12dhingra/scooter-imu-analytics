# Edge Pipeline: Baseline Audit Report

This report documents the baseline state of `scooter-imu-analytics` prior to the embedded ML edge implementation (Stage 0).

## 1. Repository & Codebase Overview

- **Repository:** `scooter-imu-analytics`
- **Initial Git Status:** Clean tree with untracked plan and simulator scripts.
- **Project Purpose:** Multi-use-case analytics platform for scooter IMU/gyroscope data.
- **Existing Capabilities:**
  - 8 rule-based algorithms (`TurnIndicator`, `FallDetector`, `RoadQualityAnalyzer`, `RidingStyleScorer`, `LaneClassifier`, `TheftDetector`, `PathReconstructor`, `WearDetector`).
  - Supervised Random Forest maneuver classifier (`ml/classifiers.py`).
  - Unsupervised Isolation Forest anomaly detector (`ml/anomaly.py`).
  - K-Means trip risk profiler (`ml/risk_profiler.py`).
  - FastAPI service & Streamlit dashboard.

## 2. Feature & Label Contract Baseline

### Feature Contract (`ml/features.py`)
- **Signal Inputs:** `gyr_x`, `gyr_y`, `gyr_z`, and `gyr_magnitude` (computed from gyro axes).
- **Windowing:** 1.0 s window duration (`WINDOW_S = 1.0`), 0.5 s step duration (`STEP_S = 0.5`, 50% overlap).
- **Window Length:** At 50 Hz, 50 samples per window, advanced by 25 samples per step.
- **Per-Signal Statistics (9 metrics × 4 signals = 36 features):**
  1. Mean (`w.mean()`)
  2. Population Standard Deviation + 1e-12 (`w.std() + 1e-12`, ddof=0)
  3. Minimum (`w.min()`)
  4. Maximum (`w.max()`)
  5. RMS (`sqrt(mean(w**2))`)
  6. Zero-Crossing Rate (`(diff(sign(w)) != 0).sum() / len(w)`)
  7. FFT Bin 1 Magnitude (`abs(rfft(w - mean))[1]`)
  8. FFT Bin 2 Magnitude (`abs(rfft(w - mean))[2]`)
  9. FFT Bin 3 Magnitude (`abs(rfft(w - mean))[3]`)
- **Cross-Axis Pearson Correlations (3 features):**
  - `corr_xy`, `corr_yz`, `corr_xz` (NaN values from constant signals replaced by 0.0).
- **Total Features:** Exactly 39 features in canonical order defined by `ml.features.feature_names()`.

### Label Contract (`ml/labeler.py`)
- **9 Canonical Classes:**
  1. `parked`
  2. `straight`
  3. `left_turn`
  4. `right_turn`
  5. `lane_change`
  6. `sharp_turn`
  7. `fall`
  8. `bumpy_road`
  9. `aggressive`

## 3. Test Suite Baseline Execution

Executed command:
```bash
uv run --locked pytest
```

### Results
- **Platform:** macOS 27.0.1 (arm64), Python 3.11.13, pytest-9.1.1, pluggy-1.6.0
- **Total Tests Collected:** 13
- **Passing Tests:** 13 (100%)
  - `tests/test_fall_detection.py` (3 passed)
  - `tests/test_riding_style.py` (3 passed)
  - `tests/test_road_quality.py` (3 passed)
  - `tests/test_turn_indicator.py` (4 passed)
- **Failing Tests:** 0
- **Execution Duration:** 0.10s

## 4. Git & Environment Hygiene

- `.gitignore` updated:
  - Removed `uv.lock` from `.gitignore` so lockfile is version-controlled for reproducible environments.
  - Added build output (`embedded/build/`, `build/`, `*.o`, `*.dylib`, etc.) and artifact directories (`artifacts/`, `ml/models/*.onnx`).
- Available disk space confirmed at ~23 GiB.
- Apple Clang 21.0.0 is verified and ready for C++17 compilation.

