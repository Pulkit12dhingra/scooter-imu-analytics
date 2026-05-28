"""
ml/train_models.py
CLI to train all three ML models and save them to ml/models/.

Usage (from project root):
    uv run python ml/train_models.py
    uv run python ml/train_models.py --fleet 50 --out ml/models
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ml/ and src/ importable regardless of where the script is invoked from
ML_DIR   = Path(__file__).resolve().parent
ROOT_DIR = ML_DIR.parent
sys.path.insert(0, str(ML_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "scripts"))

import numpy as np
import pandas as pd

from scooter_imu.data.loader      import load_raw
from scooter_imu.data.preprocessor import preprocess

from features      import extract_windows
from labeler       import label_timestamps, CLASS_NAMES, CLASS_TO_IDX
from classifiers   import ManeuverClassifier
from anomaly       import IMUAnomalyDetector
from risk_profiler import RiskProfiler, aggregate_trip

import generate_trip_data as gtd


# ── Helpers ──────────────────────────────────────────────────────────────── #

def _run_algorithms(df: pd.DataFrame) -> pd.DataFrame:
    """Annotate df with all 8 rule-based algorithm outputs."""
    from scooter_imu.algorithms.turn_indicator      import TurnIndicator
    from scooter_imu.algorithms.fall_detection      import FallDetector
    from scooter_imu.algorithms.road_quality        import RoadQualityAnalyzer
    from scooter_imu.algorithms.riding_style        import RidingStyleScorer
    from scooter_imu.algorithms.lane_classifier     import LaneClassifier

    for algo in [TurnIndicator(), FallDetector(), RoadQualityAnalyzer(),
                 RidingStyleScorer(), LaneClassifier()]:
        df = algo.predict(df)
    return df


def _load(data_path: str) -> pd.DataFrame:
    return preprocess(load_raw(data_path))


def _synth_preprocessed(seed: int) -> pd.DataFrame:
    """Generate a synthetic trip and return a preprocessed wide-format DataFrame."""
    return preprocess(gtd.generate_trip_df(seed))


# ── Model trainers ───────────────────────────────────────────────────────── #

def train_classifier(df: pd.DataFrame, out_dir: Path) -> None:
    print("\n── ManeuverClassifier (Random Forest) ──────────────────────────")
    X, mid_ts = extract_windows(df)
    y = label_timestamps(mid_ts, df["timestamp"].iloc[0])

    valid = y >= 0
    X, y_int = X[valid], y[valid]
    print(f"  Windows: {len(X)}  Classes: {np.unique(y_int).tolist()}")

    mc     = ManeuverClassifier()
    scores = mc.cross_validate(X, y_int, cv=5)
    print(f"  5-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    mc.fit(X, y_int, CLASS_NAMES)
    mc.save(out_dir / "maneuver_rf.joblib")

    # Quick train-set report
    y_pred = mc.clf.predict(X)
    from sklearn.metrics import accuracy_score
    print(f"  Train accuracy: {accuracy_score(y_int, y_pred):.3f}")


def train_anomaly_detector(df: pd.DataFrame, out_dir: Path) -> None:
    print("\n── IMUAnomalyDetector (Isolation Forest) ────────────────────────")
    X, mid_ts = extract_windows(df)
    y = label_timestamps(mid_ts, df["timestamp"].iloc[0])

    ANOMALOUS = {CLASS_TO_IDX.get("fall", -1), CLASS_TO_IDX.get("parked", -1)}
    normal    = np.array([i >= 0 and i not in ANOMALOUS for i in y])
    X_normal  = X[normal]
    print(f"  Normal windows: {X_normal.shape[0]} / {len(X)} total")

    det = IMUAnomalyDetector(contamination=0.05)
    det.fit(X_normal)

    # Quick sanity check on the fall segment
    FALL_IDX = CLASS_TO_IDX.get("fall", -1)
    fall_mask = y == FALL_IDX
    if fall_mask.sum() > 0:
        fall_scores = det.anomaly_score(X[fall_mask])
        print(f"  Fall-window anomaly scores: mean={fall_scores.mean():.3f}  "
              f"min={fall_scores.min():.3f}  max={fall_scores.max():.3f}")

    det.save(out_dir / "anomaly_iforest.joblib")


def train_risk_profiler(n_trips: int, out_dir: Path) -> None:
    print(f"\n── RiskProfiler (K-Means, fleet of {n_trips} synthetic trips) ──")
    feature_rows: list[dict] = []

    for seed in range(n_trips):
        trip_df = _run_algorithms(_synth_preprocessed(seed))
        feature_rows.append(aggregate_trip(trip_df))
        if (seed + 1) % 10 == 0:
            print(f"  Processed trip {seed + 1}/{n_trips}")

    rp = RiskProfiler(n_clusters=3)
    rp.fit(feature_rows)
    rp.save(out_dir / "risk_kmeans.joblib")

    labels, counts = np.unique(rp.predict(feature_rows), return_counts=True)
    for lbl, cnt in zip(labels, counts):
        print(f"  Cluster {lbl} ({['Low','Medium','High'][lbl]} Risk): {cnt} trips")


# ── CLI ──────────────────────────────────────────────────────────────────── #

def main() -> None:
    parser = argparse.ArgumentParser(description="Train all ML models")
    parser.add_argument(
        "--data", default="data/raw/ScooterIMUData.csv",
        help="Raw CSV for classifier + anomaly detector (default: data/raw/ScooterIMUData.csv)",
    )
    parser.add_argument(
        "--fleet", type=int, default=30,
        help="Synthetic trips for risk profiler (default: 30)",
    )
    parser.add_argument(
        "--out", default="ml/models",
        help="Directory to save trained models (default: ml/models)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load(args.data)
    print(f"Loaded {len(df)} rows from {args.data}")

    train_classifier(df, out_dir)
    train_anomaly_detector(df, out_dir)
    train_risk_profiler(args.fleet, out_dir)

    print(f"\nAll models saved to {out_dir}/")


if __name__ == "__main__":
    main()
