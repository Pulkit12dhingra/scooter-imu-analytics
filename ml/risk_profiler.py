"""
ml/risk_profiler.py
Trip-level feature aggregation and K-Means risk-tier clustering.

Typical workflow:
  1. Run all 8 rule-based algorithms on each trip DataFrame.
  2. Call aggregate_trip() to reduce each trip to a 1-D feature row.
  3. Build a fleet-level matrix and fit RiskProfiler.
  4. Predict cluster (0 = Low, 1 = Medium, 2 = High Risk) for any new trip.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "risk_kmeans.joblib"

RISK_LABELS = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
RISK_COLORS = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}


# ── Trip aggregation ─────────────────────────────────────────────────────── #

def aggregate_trip(df: pd.DataFrame) -> dict[str, float]:
    """
    Collapse a fully-annotated trip DataFrame into a scalar feature row.

    Parameters
    ----------
    df : wide-format DataFrame that has been processed by all 8 algorithms.
         Tolerates missing columns — uses 0 as fallback.
    """
    n = len(df)

    def col(c: str) -> pd.Series:
        return df[c] if c in df.columns else pd.Series(np.zeros(n))

    duration_s = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds()

    return {
        # Rotational intensity
        "mean_magnitude":    float(col("gyr_magnitude").mean()),
        "max_magnitude":     float(col("gyr_magnitude").max()),
        "p95_magnitude":     float(col("gyr_magnitude").quantile(0.95)),

        # Turning behaviour
        "pct_left_turn":     float((col("turn_direction") == "Left").mean()),
        "pct_right_turn":    float((col("turn_direction") == "Right").mean()),
        "pct_sharp_turn":    float((col("turn_level") == 2).mean()),

        # Safety events
        "n_falls":           float(col("fall_event_start").sum()),
        "pct_fall_duration": float(col("fall_event").mean()),

        # Road exposure
        "mean_road_severity":float(col("road_severity").mean()),
        "pct_severe_road":   float((col("road_severity") > 0.6).mean()),

        # Riding style
        "mean_smooth_score": float(col("smooth_score").mean()),
        "min_smooth_score":  float(col("smooth_score").min()),
        "pct_low_score":     float((col("smooth_score") < 50).mean()),

        # Maneuver mix
        "pct_lane_change":   float((col("maneuver") == "lane_change").mean()),
        "pct_straight":      float((col("maneuver") == "straight").mean()),

        # Trip meta
        "duration_s":        duration_s,
    }


# ── Clustering model ─────────────────────────────────────────────────────── #

class RiskProfiler:
    """
    K-Means risk-tier clustering on fleet-level trip feature rows.

    Cluster labels are ordered so that 0 = Low Risk, 1 = Medium, 2 = High.
    Ordering is based on the 'mean_magnitude' centroid dimension.
    """

    def __init__(self, n_clusters: int = 3, random_state: int = 42) -> None:
        self.n_clusters  = n_clusters
        self.scaler      = StandardScaler()
        self.km          = KMeans(n_clusters=n_clusters, n_init=20,
                                  random_state=random_state)
        self._feature_names: list[str] = []
        self._label_map: dict[int, int] = {}

    def _to_matrix(self, feature_rows: list[dict]) -> np.ndarray:
        df = pd.DataFrame(feature_rows)
        if not self._feature_names:
            self._feature_names = list(df.columns)
        return df[self._feature_names].to_numpy(dtype=np.float64)

    def fit(self, feature_rows: list[dict]) -> "RiskProfiler":
        X = self._to_matrix(feature_rows)
        X_s = self.scaler.fit_transform(X)
        self.km.fit(X_s)
        self._order_by_risk()
        return self

    def _order_by_risk(self) -> None:
        """Map raw cluster indices so lowest-magnitude cluster = 0 (Low Risk)."""
        if "mean_magnitude" in self._feature_names:
            idx = self._feature_names.index("mean_magnitude")
        else:
            idx = 0
        order = np.argsort(self.km.cluster_centers_[:, idx])
        self._label_map = {int(old): int(new) for new, old in enumerate(order)}

    def predict(self, feature_rows: list[dict]) -> np.ndarray:
        X   = self._to_matrix(feature_rows)
        X_s = self.scaler.transform(X)
        raw = self.km.predict(X_s)
        return np.array([self._label_map.get(int(r), int(r)) for r in raw])

    def predict_one(self, trip_df: pd.DataFrame) -> tuple[int, str]:
        """Convenience: pass a single annotated trip DataFrame."""
        feats   = aggregate_trip(trip_df)
        cluster = int(self.predict([feats])[0])
        return cluster, RISK_LABELS[cluster]

    def cluster_centers_df(self) -> pd.DataFrame:
        """Return centroid values in original (un-scaled) feature space."""
        centers_scaled = self.km.cluster_centers_
        centers_orig   = self.scaler.inverse_transform(centers_scaled)
        # Re-order rows by risk level
        ordered = np.zeros_like(centers_orig)
        for raw, mapped in self._label_map.items():
            ordered[mapped] = centers_orig[raw]
        return pd.DataFrame(
            ordered, columns=self._feature_names,
            index=[RISK_LABELS[i] for i in range(self.n_clusters)],
        )

    def save(self, path: str | Path = DEFAULT_MODEL_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "km": self.km, "scaler": self.scaler,
            "feature_names": self._feature_names,
            "label_map": self._label_map,
        }, path)
        print(f"RiskProfiler saved → {path}")

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH) -> "RiskProfiler":
        obj = joblib.load(path)
        rp  = cls.__new__(cls)
        rp.n_clusters      = obj["km"].n_clusters
        rp.km              = obj["km"]
        rp.scaler          = obj["scaler"]
        rp._feature_names  = obj["feature_names"]
        rp._label_map      = obj["label_map"]
        return rp
