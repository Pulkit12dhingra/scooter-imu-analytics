# EdgeScoot Notebook and Pipeline Remediation Audit Report

**Date:** 2026-10-12  
**Baseline Test Suite Status:** 51 passed, 1 skipped (52 tests total)  
**Package Manager:** `uv` (`/Users/pulkit/.local/bin/uv`)  
**Target Environment:** macOS (Apple Silicon arm64)  

---

## 1. Repository & Notebook Inventory

There are 12 Jupyter notebooks located in the repository:

| Notebook Path | Purpose | Code Cells | Cells with Outputs | Classification | Baseline State |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `notebooks/00_eda.ipynb` | Exploratory data analysis on raw IMU CSV | 8 | 7 | Canonical | Working; labels Y-axis as `deg/s` despite rad/s values |
| `notebooks/01_turn_indicator.ipynb` | Turn detection & sensitivity | 7 | 4 | Exploratory | Working; threshold analysis |
| `notebooks/02_fall_detection.ipynb` | Fall detector threshold demonstration | 4 | 3 | Canonical | Uses `threshold=1.2` vs default `2.0`; triggers false positives |
| `notebooks/03_road_quality.ipynb` | Road roughness & surface scoring | 7 | 6 | Exploratory | Working |
| `notebooks/04_riding_style.ipynb` | Jerk & aggressive acceleration metrics | 7 | 6 | Exploratory | Working |
| `notebooks/05_theft_detection.ipynb` | Stationary theft alerting | 3 | 2 | Canonical | Evaluates entire moving trip as parked; alerts throughout |
| `notebooks/06_lane_classifier.ipynb` | Heuristic lane change detector | 4 | 3 | Exploratory | Working |
| `notebooks/07_path_reconstruction.ipynb` | 2D illustrative path dead-reckoning | 5 | 4 | Exploratory | Gyro-only relative illustration without metric ground truth |
| `notebooks/08_wear_detection.ipynb` | Bearing wear FFT band-power analysis | 4 | 0 | Canonical | **Unexecuted**; outputs empty |
| `ml/notebooks/01_maneuver_classification.ipynb` | RF maneuver classifier on windows | 7 | 7 | Canonical | **Overlapping window leakage** via random `train_test_split` |
| `ml/notebooks/02_anomaly_detection.ipynb` | Isolation Forest anomaly detection | 6 | 6 | Canonical | Evaluates anomaly scores on fit population |
| `ml/notebooks/03_risk_profiling.ipynb` | K-Means fleet risk clustering | 8 | 8 | Canonical | Clusters synthetic trips |

---

## 2. Review Findings Matrix (R1 – R13)

| ID | Finding Description | Status | Evidence & Details | Corrective Action |
| :--- | :--- | :--- | :--- | :--- |
| **R1** | Inconsistent dataset accounting & stale outputs | `CONFIRMED` | Notebook outputs display 30,000 samples and 1,199 windows from legacy single-trip CSV runs. | Implement standard dataset accounting diagnostics and verify window formulas `1 + floor((N - W) / H)`. |
| **R2** | Inconsistent path casing (`Data/` vs `data/`) and cwd dependence | `CONFIRMED` | `notebooks/00_eda.ipynb` references `../Data/raw/ScooterIMUData.csv`, other notebooks reference `../data/...`. Git tracks `Data/`. Fails on case-sensitive filesystems. | Standardize project root discovery via marker (`pyproject.toml`) and ensure case consistency. |
| **R3** | Discrepancy between `deg/s` and `rad/s` | `CONFIRMED` | Raw data columns are named `GYR_X_DEG` etc., but generated in `rad/s` by `scripts/generate_trip_data.py`. Algorithms in `src/scooter_imu/algorithms/` use `rad/s` thresholds. Plots in EDA and notebooks label axes `deg/s`. | Clarify documentation and plot labels that signal values and algorithm thresholds are canonical `rad/s`. |
| **R4** | Random train/test split on overlapping windows | `CONFIRMED` | `ml/notebooks/01_maneuver_classification.ipynb` cell 5 uses `train_test_split(X, y, test_size=0.2, stratify=y)`. 50% overlap between adjacent windows causes test set data leakage. | Replace with group-based episode splitting or disjoint blocked split with boundary purging. |
| **R5** | Fixed-schedule labeling vs randomized simulator ground truth | `CONFIRMED` | Legacy labeling maps fixed timestamps rather than true underlying generator states. | Standardize window-to-label policy based on actual simulator ground truth with 80% purity rule. |
| **R6** | Theft detector demonstration evaluates full ride as parked | `CONFIRMED` | `notebooks/05_theft_detection.ipynb` passes the entire 10-minute trip to `TheftDetector`, producing 25,515 theft alerts. | Construct controlled parked negative baseline followed by distinct moving event. |
| **R7** | Fall detector threshold mismatch and startup false positive | `CONFIRMED` | `notebooks/02_fall_detection.ipynb` passes `threshold=1.2` (generating 5 events) while detector default is `2.0` (generating 1 event). | Harmonize threshold to detector standard and demonstrate clear positive and negative controls. |
| **R8** | Isolation Forest fit and evaluation populations overlap | `CONFIRMED` | `ml/notebooks/02_anomaly_detection.ipynb` evaluates anomaly scores on the training trace. | Evaluate anomaly scores on held-out independent nominal and anomaly episodes. |
| **R9** | Synthetic trip clustering diversity and interpretation | `CONFIRMED` | K-means risk profiling clusters synthetic noise with fixed trip schedules. | Clarify synthetic clustering limitations and explore varied trip parameters. |
| **R10** | Wear detection notebook unexecuted from clean kernel | `CONFIRMED` | `notebooks/08_wear_detection.ipynb` has 0 cell outputs. | Execute cleanly via automated runner and validate 5–20 Hz band power on known synthetic frequency. |
| **R11** | Path reconstruction presented without velocity/GPS limitations | `CONFIRMED` | `notebooks/07_path_reconstruction.ipynb` produces relative trajectory from gyro integration only. | Clearly document unit-speed and drift assumptions in markdown and plot titles. |
| **R12** | Interactive session dependencies & stale artifacts | `CONFIRMED` | Notebooks rely on `ROOT = Path.cwd().parent...` which fails if working directory varies. | Add robust root discovery and clean execution validation. |
| **R13** | ML feature contract stability across Python and C++ | `ALREADY_FIXED` | 39-feature contract is verified and tested under `tests/test_feature_parity.py` and `tests/test_cpp_inference.py`. | Maintain regression tests across all stages. |

