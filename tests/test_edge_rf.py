"""tests/test_edge_rf.py
Unit tests for Stage 4 Random Forest baseline training and evaluation.
"""
from pathlib import Path
import joblib
import numpy as np

from ml.classifiers import ManeuverClassifier
from ml.labeler import CLASS_NAMES
from ml.train_edge_rf import train_and_evaluate_rf


def test_train_and_evaluate_rf_end_to_end(tmp_path):
    """Verify RF model trains, evaluates on held-out test split, and produces metrics."""
    out_dir = tmp_path / "edge_artifacts"

    results = train_and_evaluate_rf(
        train_seeds=[1, 2, 3],
        val_seeds=[10],
        test_seeds=[20],
        n_estimators=20,
        max_depth=8,
        episode_duration_s=30.0,
        output_dir=out_dir,
        random_state=42,
    )

    # Verify metrics structure
    assert "test_metrics" in results
    assert "accuracy" in results["test_metrics"]
    assert "macro_f1" in results["test_metrics"]
    assert results["test_metrics"]["accuracy"] > results["majority_baseline"]["accuracy"]

    # Verify per-class support and names
    assert results["class_names"] == CLASS_NAMES
    assert len(results["per_class_metrics"]) == len(CLASS_NAMES)

    # Verify model artifact saved and loadable
    model_path = out_dir / "maneuver_rf.joblib"
    assert model_path.exists()
    loaded_mc = joblib.load(model_path)
    assert isinstance(loaded_mc, ManeuverClassifier)

    # Verify inference on a test sample produces valid class
    dummy_input = np.zeros((1, 39), dtype=np.float64)
    pred = loaded_mc.predict(dummy_input)
    assert len(pred) == 1
    assert pred[0] in CLASS_NAMES

