"""ml/train_edge_rf.py
Train and evaluate Random Forest baseline on synthetic simulator data (Stage 4).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.dummy import DummyClassifier

# Make root and ml importable
ROOT_DIR = Path(__file__).resolve().parent.parent
ML_DIR = ROOT_DIR / "ml"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from ml.classifiers import ManeuverClassifier
from ml.labeler import CLASS_NAMES, CLASS_TO_IDX
from ml.train_synthetic import generate_dataset_splits


def train_and_evaluate_rf(
    train_seeds: list[int],
    val_seeds: list[int],
    test_seeds: list[int],
    n_estimators: int = 100,
    max_depth: int | None = 15,
    episode_duration_s: float = 45.0,
    output_dir: Path | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train Random Forest on synthetic data and evaluate on held-out test episodes."""
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate episode-split dataset
    dataset = generate_dataset_splits(
        train_seeds=train_seeds,
        val_seeds=val_seeds,
        test_seeds=test_seeds,
        episode_duration_s=episode_duration_s,
    )

    X_train, y_train = dataset["train"]["X"], dataset["train"]["y"]
    X_val, y_val = dataset["val"]["X"], dataset["val"]["y"]
    X_test, y_test = dataset["test"]["X"], dataset["test"]["y"]

    # 2. Train model
    mc = ManeuverClassifier(n_estimators=n_estimators, random_state=random_state)
    mc.clf.max_depth = max_depth
    mc.fit(X_train, y_train, CLASS_NAMES)

    # 3. Evaluate majority class baseline on test set
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_preds = dummy.predict(X_test)
    dummy_acc = float(accuracy_score(y_test, dummy_preds))
    dummy_f1 = float(f1_score(y_test, dummy_preds, average="macro", zero_division=0))

    # 4. Evaluate Random Forest on held-out test set
    y_test_pred_names = mc.predict(X_test)
    y_test_pred_idx = np.array([CLASS_TO_IDX[c] for c in y_test_pred_names], dtype=np.int32)

    test_acc = float(accuracy_score(y_test, y_test_pred_idx))
    test_f1 = float(f1_score(y_test, y_test_pred_idx, average="macro", zero_division=0))

    report_dict = classification_report(
        y_test,
        y_test_pred_idx,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_test_pred_idx, labels=list(range(len(CLASS_NAMES)))).tolist()

    # Per-class metrics
    per_class = {}
    for c in CLASS_NAMES:
        if c in report_dict:
            per_class[c] = {
                "precision": round(report_dict[c]["precision"], 4),
                "recall": round(report_dict[c]["recall"], 4),
                "f1": round(report_dict[c]["f1-score"], 4),
                "support": int(report_dict[c]["support"]),
            }

    results: dict[str, Any] = {
        "model_parameters": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
        },
        "dataset_metadata": dataset["metadata"],
        "splits_sample_count": {
            "train": int(len(X_train)),
            "val": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "majority_baseline": {
            "accuracy": round(dummy_acc, 4),
            "macro_f1": round(dummy_f1, 4),
        },
        "test_metrics": {
            "accuracy": round(test_acc, 4),
            "macro_f1": round(test_f1, 4),
        },
        "per_class_metrics": per_class,
        "confusion_matrix": cm,
        "class_names": CLASS_NAMES,
    }

    if output_dir is not None:
        # Save model joblib
        model_path = output_dir / "maneuver_rf.joblib"
        joblib.dump(mc, model_path)
        results["saved_model_path"] = str(model_path)

        # Save metrics report
        report_path = output_dir / "baseline_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        results["saved_report_path"] = str(report_path)

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of trees (default: 100)")
    parser.add_argument("--max-depth", type=int, default=15, help="Max depth of trees (default: 15)")
    parser.add_argument("--train-episodes", type=int, default=6, help="Number of train episodes")
    parser.add_argument("--val-episodes", type=int, default=2, help="Number of val episodes")
    parser.add_argument("--test-episodes", type=int, default=2, help="Number of test episodes")
    parser.add_argument("--episode-duration", type=float, default=45.0, help="Duration in sec per episode")
    parser.add_argument("--seed-base", type=int, default=42, help="Base seed offset")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/edge"), help="Output directory")
    args = parser.parse_args(argv)

    train_seeds = [args.seed_base + i for i in range(args.train_episodes)]
    val_seeds = [args.seed_base + 100 + i for i in range(args.val_episodes)]
    test_seeds = [args.seed_base + 200 + i for i in range(args.test_episodes)]

    print(f"Training Edge Random Forest baseline...", file=sys.stderr)
    res = train_and_evaluate_rf(
        train_seeds=train_seeds,
        val_seeds=val_seeds,
        test_seeds=test_seeds,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        episode_duration_s=args.episode_duration,
        output_dir=args.out_dir,
    )

    print(
        f"Completed training and evaluation:\n"
        f"  Test Accuracy: {res['test_metrics']['accuracy']:.4f} (vs Baseline: {res['majority_baseline']['accuracy']:.4f})\n"
        f"  Test Macro F1: {res['test_metrics']['macro_f1']:.4f} (vs Baseline: {res['majority_baseline']['macro_f1']:.4f})\n"
        f"  Artifacts saved to {args.out_dir}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

