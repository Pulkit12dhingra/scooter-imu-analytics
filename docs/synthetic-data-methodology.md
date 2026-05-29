# Synthetic Data Methodology & Training Pipeline

This document details the generation methodology, windowing parameters, labeling rules, and split isolation for procedural synthetic training data in `scooter-imu-analytics`.

## 1. Zero Historical Data Dependency

Traditional supervised training in the repository relied on a single fixed 10-minute synthetic ride (`Data/raw/ScooterIMUData.csv`). EdgeScoot replaces this with online, procedural batch generation (`ml/train_synthetic.py`) driven by `LiveScooterSimulator`:
- Zero dependence on recorded CSV files.
- Generates arbitrary volume of episodes with independent random seeds.
- Independent variance across duration, amplitudes, sensor bias, and Gaussian noise.

## 2. Windowing & Feature Extraction

- **Sampling Rate:** 50 Hz nominal (`dt = 0.02 s`).
- **Window Length:** 1.0 second (50 samples).
- **Step Size:** 0.5 seconds (25 samples, 50% overlap).
- **Extracted Features:** Exactly 39 features per window, matching the canonical contract in `ml.features.feature_names()`:
  - 9 stats × 4 signals (`gyr_x`, `gyr_y`, `gyr_z`, `gyr_magnitude`) = 36 features
  - 3 cross-axis Pearson correlations (`corr_xy`, `corr_yz`, `corr_xz`) = 3 features

## 3. Window Labeling & Transition Policy

Because adjacent windows advance with a 50% overlap, windows crossing scenario boundaries can contain mixed physical behaviors. The policy in `ml/window_labeling.py` enforces:

1. **80% Purity Threshold for Sustained Maneuvers:**
   - A window is assigned a maneuver class only if at least 80% (>= 40 of 50 samples) share that single maneuver label.
   - Mixed boundary windows (< 80% purity) are marked as ambiguous and dropped from supervised training sets.
2. **Event-Aware Rule for Short Transients (`fall`):**
   - The synthetic `fall` event is a short impulse (~1.0s to 1.5s).
   - A window receives the `fall` label only if the window overlaps the synthetic fall scenario AND the peak gyro magnitude reaches at least `0.80 rad/s`.
3. **Condition Decoupling (`wear_vibration`):**
   - Wear vibration overlay represents mechanical condition, not vehicle path maneuver.
   - For maneuver classification, it is classified as `straight`, with an optional `condition_label = "wear_vibration"`.

## 4. Episode-Level Split Isolation

Due to 50% window overlap, random window-level splitting causes massive data leakage between train and test. EdgeScoot strictly splits at the **Episode / Seed level**:
- Train episodes, validation episodes, and test episodes use mutually disjoint seeds (e.g. Train: `[101, 102, 103, 104]`, Val: `[201, 202]`, Test: `[301, 302]`).
- No sample from a simulated ride ever appears in another split.

## 5. Limitations & Real-World Caveats

> [!WARNING] Synthetic Limitations
> Models trained on procedural synthetic signals evaluate algorithmic consistency within the synthetic family. High test accuracy on synthetic data does **not** validate real-world fall detection or scooter safety on physical roads.

