"""
ml/classifiers.py
Supervised maneuver classifier — Random Forest with sklearn interface.

The model replaces the heuristic LaneClassifier + TurnIndicator pair with
a data-driven classifier trained on windowed IMU features.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "maneuver_rf.joblib"


class ManeuverClassifier:
    """
    Wraps a Random Forest for IMU maneuver classification.

    Usage
    -----
        mc = ManeuverClassifier()
        scores = mc.cross_validate(X, y_int, class_names)
        mc.fit(X_train, y_int_train, class_names)
        y_pred_str = mc.predict(X_test)
        mc.save()

        # Reload later
        mc2 = ManeuverClassifier.load()
    """

    def __init__(self, n_estimators: int = 300, random_state: int = 42) -> None:
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            min_samples_leaf=3,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )
        self.le: LabelEncoder = LabelEncoder()
        self._fitted: bool = False

    # ── Training ─────────────────────────────────────────────────────────── #

    def fit(
        self,
        X: np.ndarray,
        y_int: np.ndarray,
        class_names: list[str],
    ) -> "ManeuverClassifier":
        """
        Fit on feature matrix X and integer label array y_int.

        Parameters
        ----------
        X           : (n_windows, n_features) feature matrix
        y_int       : (n_windows,) integer class indices
        class_names : ordered list so that class_names[i] maps to label i
        """
        self.le.classes_ = np.array(class_names)
        self.clf.fit(X, y_int)
        self._fitted = True
        return self

    def cross_validate(
        self,
        X: np.ndarray,
        y_int: np.ndarray,
        cv: int = 5,
    ) -> np.ndarray:
        """Return per-fold accuracy scores (array of length cv)."""
        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        return cross_val_score(self.clf, X, y_int, cv=kf, scoring="accuracy")

    # ── Inference ────────────────────────────────────────────────────────── #

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return string class names (e.g. 'left_turn')."""
        return self.le.inverse_transform(self.clf.predict(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability matrix (n_windows, n_classes)."""
        return self.clf.predict_proba(X)

    # ── Evaluation helpers ───────────────────────────────────────────────── #

    def report(self, X: np.ndarray, y_int: np.ndarray) -> str:
        y_pred = self.clf.predict(X)
        return classification_report(
            y_int, y_pred, target_names=list(self.le.classes_), zero_division=0
        )

    def confusion(self, X: np.ndarray, y_int: np.ndarray) -> np.ndarray:
        return confusion_matrix(y_int, self.clf.predict(X))

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.clf.feature_importances_

    @property
    def class_names(self) -> list[str]:
        return list(self.le.classes_)

    # ── Persistence ──────────────────────────────────────────────────────── #

    def save(self, path: str | Path = DEFAULT_MODEL_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"clf": self.clf, "le": self.le}, path)
        print(f"ManeuverClassifier saved → {path}")

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH) -> "ManeuverClassifier":
        obj = joblib.load(path)
        mc = cls.__new__(cls)
        mc.clf     = obj["clf"]
        mc.le      = obj["le"]
        mc._fitted = True
        return mc
