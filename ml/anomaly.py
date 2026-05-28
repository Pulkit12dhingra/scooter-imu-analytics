"""
ml/anomaly.py
Unsupervised anomaly detection using Isolation Forest.

Train ONLY on normal-riding feature windows. At inference the model
assigns an anomaly score to every window — no labels required.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "anomaly_iforest.joblib"


class IMUAnomalyDetector:
    """
    Isolation-Forest anomaly detector for IMU windows.

    Usage
    -----
        det = IMUAnomalyDetector()
        det.fit(X_normal)                # train on normal-riding windows
        scores = det.anomaly_score(X)    # high → anomalous  (0–1 normalised)
        flags  = det.predict(X)          # True = anomaly

    The raw `decision_function` output is negative-more-anomalous;
    `anomaly_score` inverts and normalises this to [0, 1].
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 200,
        random_state: int = 42,
    ) -> None:
        self.ifo = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            n_jobs=-1,
            random_state=random_state,
        )
        self._score_min: float = 0.0
        self._score_range: float = 1.0

    # ── Training ─────────────────────────────────────────────────────────── #

    def fit(self, X_normal: np.ndarray) -> "IMUAnomalyDetector":
        """Train on normal-riding feature windows."""
        self.ifo.fit(X_normal)
        # Calibrate score normalisation on the training set
        raw = self.ifo.score_samples(X_normal)
        self._score_min   = float(raw.min())
        self._score_range = float(raw.max() - raw.min()) + 1e-9
        return self

    # ── Inference ────────────────────────────────────────────────────────── #

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Normalised anomaly score in [0, 1].
        0 = clearly normal · 1 = very anomalous.
        """
        raw = self.ifo.score_samples(X)
        # Invert: lower decision score → higher anomaly score
        score = 1.0 - (raw - self._score_min) / self._score_range
        return np.clip(score, 0.0, 1.0)

    def predict(self, X: np.ndarray, threshold: float = 0.65) -> np.ndarray:
        """
        Return boolean anomaly flag (True = anomaly).

        Parameters
        ----------
        threshold : anomaly_score cutoff (default 0.65)
        """
        return self.anomaly_score(X) >= threshold

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Raw sklearn decision function (lower = more anomalous)."""
        return self.ifo.decision_function(X)

    # ── Persistence ──────────────────────────────────────────────────────── #

    def save(self, path: str | Path = DEFAULT_MODEL_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "ifo": self.ifo,
            "score_min": self._score_min,
            "score_range": self._score_range,
        }, path)
        print(f"IMUAnomalyDetector saved → {path}")

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH) -> "IMUAnomalyDetector":
        obj = joblib.load(path)
        det = cls.__new__(cls)
        det.ifo           = obj["ifo"]
        det._score_min    = obj["score_min"]
        det._score_range  = obj["score_range"]
        return det
