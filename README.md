# Scooter IMU Analytics

Multi-use-case analytics platform for scooter gyroscope (IMU) data. Built on top of an Ather scooter dataset, the project ships eight rule-based analysis algorithms, three ML models, a Streamlit dashboard, a FastAPI REST API, and a suite of Jupyter notebooks — all managed with [uv](https://docs.astral.sh/uv/).

## Use cases

| # | Algorithm | What it detects |
|---|-----------|-----------------|
| 1 | Turn Indicator | Left/right turn events and their intensity level |
| 2 | Fall Detection | Multi-axis magnitude spikes indicating accidents |
| 3 | Road Quality | Rolling RMS severity score per road segment |
| 4 | Riding Style | Jerk-based smoothness score (0–100) |
| 5 | Theft Detection | Unauthorized movement of a parked scooter |
| 6 | Lane Classifier | Straight / lane change / turn / sharp turn classification |
| 7 | Path Reconstruction | Dead-reckoning heading from GYR_Y integration |
| 8 | Wear Detection | FFT band-power anomalies indicating bearing/suspension wear |

### ML models

| # | Model | Technique | What it does |
|---|-------|-----------|--------------|
| M1 | ManeuverClassifier | Random Forest (300 trees) | Replaces the rule-based LaneClassifier with a data-driven 9-class maneuver classifier trained on 39-feature sliding windows |
| M2 | IMUAnomalyDetector | Isolation Forest | Detects abnormal riding events (falls, unusual vibration) without hard-coded thresholds |
| M3 | RiskProfiler | K-Means (k=3) | Clusters fleet trips into Low / Medium / High risk tiers from 16 trip-level features |

## Project structure

```
src/scooter_imu/
    data/               # loader.py + preprocessor.py
    algorithms/         # base.py + one module per use case (8 total)
    api/                # FastAPI app + routes (POST /ingest, GET /status)
    dashboard/          # Streamlit multi-page dashboard

ml/
    features.py         # sliding-window feature extraction (39 features / window)
    labeler.py          # ground-truth labels from trip segment schedule
    classifiers.py      # ManeuverClassifier (Random Forest)
    anomaly.py          # IMUAnomalyDetector (Isolation Forest)
    risk_profiler.py    # RiskProfiler (K-Means) + aggregate_trip()
    train_models.py     # CLI to train and save all three models
    models/             # saved .joblib files (git-ignored)
    notebooks/
        01_maneuver_classification.ipynb
        02_anomaly_detection.ipynb
        03_risk_profiling.ipynb

notebooks/              # 00_eda through 08_wear_detection (rule-based analysis)
scripts/
    preprocess.py       # raw CSV → parquet CLI
    generate_trip_data.py  # synthetic trip data generator
tests/                  # pytest suite (13 tests)
data/
    raw/                # ScooterIMUData.csv
    processed/          # output of scripts/preprocess.py
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11.

```bash
# Core + dev tools (notebooks, tests, linter)
uv sync --extra dev

# Add ML dependencies (scikit-learn, xgboost, joblib)
uv sync --extra dev --extra ml

# Install the package in editable mode
uv pip install -e .
```

## Usage

**Preprocess raw CSV:**

```bash
uv run python scripts/preprocess.py \
    --input  data/raw/ScooterIMUData.csv \
    --output data/processed/clean.parquet
```

**Run the interactive dashboard:**

```bash
uv run streamlit run src/scooter_imu/dashboard/app.py
```

**Start the REST API** (Swagger UI at `/docs`):

```bash
uv run uvicorn scooter_imu.api.main:app --reload
```

**Run tests:**

```bash
uv run pytest
```

**Open rule-based analysis notebooks:**

```bash
uv run jupyter lab notebooks/
```

**Train all ML models** (saves to `ml/models/`):

```bash
uv run python ml/train_models.py
# Options:
#   --data  data/raw/ScooterIMUData.csv   (raw CSV for classifier + anomaly)
#   --fleet 30                             (synthetic trips for risk profiler)
#   --out   ml/models                      (output directory)
```

**Open ML notebooks:**

```bash
uv run jupyter lab ml/notebooks/
```

**Regenerate synthetic trip data:**

```bash
uv run python scripts/generate_trip_data.py
```

## Data schema

The raw CSV (`ScooterIMUData.csv`) is long-format with one row per axis reading:

| Column | Description |
|--------|-------------|
| `ts` | Sensor timestamp |
| `received_ts` | Server-received timestamp |
| `device_uuid` | Scooter identifier |
| `data_item_name` | Axis name: `GYR_X_DEG`, `GYR_Y_DEG`, or `GYR_Z_DEG` |
| `value` | Angular velocity in rad/s |

The preprocessor pivots this to wide format and adds `sensor_lag_s`, `gyr_magnitude`, and `dt`.

## Architecture

### Rule-based algorithms

All algorithms implement the `BaseDetector` interface:

```python
class BaseDetector(ABC):
    def predict(self, df: pd.DataFrame) -> pd.DataFrame: ...
```

Each detector receives the preprocessed wide-format DataFrame and returns it with additional result columns appended — making them trivially composable:

```python
from scooter_imu.data.loader import load_raw
from scooter_imu.data.preprocessor import preprocess
from scooter_imu.algorithms.turn_indicator import TurnIndicator
from scooter_imu.algorithms.fall_detection import FallDetector

df = preprocess(load_raw("data/raw/ScooterIMUData.csv"))
df = TurnIndicator().predict(df)
df = FallDetector().predict(df)
```

### ML pipeline

```python
import sys
sys.path.insert(0, "ml")
sys.path.insert(0, "src")

from scooter_imu.data.loader import load_raw
from scooter_imu.data.preprocessor import preprocess
from features import extract_windows
from labeler import label_timestamps, CLASS_NAMES
from classifiers import ManeuverClassifier
from anomaly import IMUAnomalyDetector
from risk_profiler import RiskProfiler

df = preprocess(load_raw("data/raw/ScooterIMUData.csv"))

# Maneuver classification
X, mid_ts = extract_windows(df)               # (n_windows, 39)
mc = ManeuverClassifier.load()
labels = mc.predict(X)                         # string class names per window

# Anomaly detection
det = IMUAnomalyDetector.load()
scores = det.anomaly_score(X)                  # [0, 1] per window

# Trip risk scoring (requires all algorithms to have run first)
rp = RiskProfiler.load()
cluster, label = rp.predict_one(df)            # e.g. (2, "High Risk")
```

## Algorithm logic


All algorithms operate on the wide-format DataFrame produced by the preprocessor. Input columns used by every algorithm: `timestamp`, `gyr_x`, `gyr_y`, `gyr_z`, `gyr_magnitude`, `dt`.

### Preprocessing

The raw long-format CSV is pivoted to wide format, then three derived columns are appended:

```
sensor_lag_s  = (received_ts - ts).total_seconds()
gyr_magnitude = sqrt(gyr_x² + gyr_y² + gyr_z²)        [rad/s]
dt            = timestamp.diff()  (filled 0.02 s for first row)
```

---

### 1. Turn Indicator

**Input signal:** `gyr_y` (yaw rate, deg/s)  
**Thresholds:** dead-band = 0.05 rad/s (~3 deg/s), sharp-turn = 1.2 rad/s (~69 deg/s)

```
turn_direction:
    gyr_y >  dead_band  →  "Left"
    gyr_y < -dead_band  →  "Right"
    otherwise           →  NA

turn_level:
    |gyr_y| <= dead_band             →  0  (straight)
    dead_band < |gyr_y| <= sharp_turn →  1  (normal turn)
    |gyr_y| > sharp_turn              →  2  (sharp turn)
```

---

### 2. Fall Detection

**Input signal:** `gyr_magnitude` (deg/s)  
**Thresholds:** magnitude threshold = 2.0 rad/s (~115 deg/s), rolling window = 100 ms

```
window_size   = max(1, int(0.1 / median_dt))
magnitude_peak = gyr_magnitude.rolling(window_size, center=True).max()
fall_event     = magnitude_peak > threshold
fall_event_start = fall_event AND NOT fall_event.shift(1)   (rising edge)
```

---

### 3. Road Quality

**Input signal:** `gyr_x` (pitch rate, deg/s)  
**Parameters:** window = 5 s, severity scale = 0.5

```
window_size   = max(1, int(5.0 / median_dt))
road_rms      = sqrt( mean(gyr_x²) over rolling window )
road_severity = clip(road_rms / 0.5,  0, 1)             [0 = smooth, 1 = severe]
```

---

### 4. Riding Style

**Input signal:** `gyr_y` (yaw rate, deg/s)  
**Parameters:** jerk scale = 5.0 deg/s², smoothing window = 10 s

```
jerk         = diff(gyr_y) / dt                          [deg/s²]
window_size  = max(1, int(10.0 / median_dt))
aggression   = clip( quantile_95(|jerk|, window) / 5.0,  0, 1 )
smooth_score = round((1 - aggression) × 100, 1)          [0 = aggressive, 100 = smooth]
```

---

### 5. Theft Detection

**Input signal:** `gyr_magnitude` (deg/s)  
**Thresholds:** stationary threshold = 0.01 rad/s (~0.6 deg/s), movement window = 2 s

```
is_moving      = gyr_magnitude > 0.01
window_size    = max(1, int(2.0 / median_dt))
movement_ratio = mean(is_moving) over rolling window
theft_alert    = movement_ratio > 0.5
```

---

### 6. Lane Classifier

**Input signal:** `gyr_y` (yaw rate, deg/s)  
**Thresholds:** turn magnitude = 0.08 rad/s (~5 deg/s), sustained duration = 0.5 s, sharp threshold = 1.2 rad/s (~69 deg/s)

```
active       = (|gyr_y| >= 0.08).astype(int)
min_samples  = max(1, int(0.5 / median_dt))
sustained    = rolling_sum(active, min_samples) >= min_samples

maneuver (applied in order, each overrides previous):
    default           →  "straight"
    active == 1       →  "lane_change"
    sustained == True →  "turn"
    |gyr_y| > 1.2     →  "sharp_turn"
```

---

### 7. Path Reconstruction

**Input signals:** `gyr_y` (rad/s), `dt`  
**Parameter:** initial heading = 0.0 deg

Dead-reckoning integration of yaw rate to recover heading:

```
heading_delta_deg = degrees(gyr_y) × dt            [incremental heading change]
cumulative        = initial_heading + cumsum(heading_delta_deg)
heading_deg       = ((cumulative + 180) % 360) − 180   [normalised to −180 … +180]
```

---

### 8. Wear Detection

**Input signal:** `gyr_x` (pitch rate, deg/s)  
**Parameters:** FFT window = 30 s, wear frequency band = 5–20 Hz

```
fs          = 1.0 / median_dt                          [sampling frequency, Hz]
half_window = max(4, int((30.0 / median_dt) / 2))      [samples per half-window]

Per sample, extract segment x[i−half_window : i+half_window]:
    freqs, psd = periodogram(segment, fs=fs)
    band_power = mean(psd[ 5 Hz ≤ freqs ≤ 20 Hz ])

mean_bp     = mean(band_power over all samples)
std_bp      = std(band_power over all samples)
wear_anomaly = band_power > (mean_bp + 2 × std_bp)
```

---

## Dataset

`data/raw/ScooterIMUData.csv` is a synthetic 10-minute trip (30,000 timestamps at 50 Hz) generated by `scripts/generate_trip_data.py`. It covers all eight algorithm use cases in a structured sequence:

| Time (s) | Event |
|---|---|
| 0–30 | Parked |
| 30–90 | Straight riding |
| 60–90 | Left turn (+0.30 rad/s yaw) |
| 150–180 | Bumpy road (high pitch rate) |
| 180–210 | Right turn (−0.38 rad/s yaw) |
| 270–330 | Lane changes (±0.18 rad/s) |
| 360–380 | Sharp right turn (−1.3 rad/s) |
| 380–420 | Wear vibration (8 Hz periodic) |
| 420–430 | Fall event (magnitude ~5.3 rad/s) |
| 480–510 | Aggressive riding |
| 570–600 | Parked |

Call `generate_trip_df(seed)` from `scripts/generate_trip_data.py` to generate additional statistically distinct trips (same event structure, different random noise).

`Data/Ather_Submission.html` contains the original exploratory data analysis notebook exported from Jupyter.

