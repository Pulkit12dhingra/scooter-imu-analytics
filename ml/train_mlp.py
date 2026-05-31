"""ml/train_mlp.py
Train a small dense neural network (MLP) for TensorRT deployment (Stage 11).

Architecture: 39 input features -> Dense(32) + ReLU -> Dense(16) + ReLU -> Dense(9)
Exports model to ONNX with pure MatMul/Relu operators compatible with NVIDIA TensorRT.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import onnx
import onnxruntime as ort
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType

from ml.features import feature_names
from ml.labeler import CLASS_NAMES, CLASS_TO_IDX
from ml.train_synthetic import generate_dataset_splits


def train_and_export_mlp(
    train_seeds: list[int],
    val_seeds: list[int],
    test_seeds: list[int],
    output_dir: Path = Path("artifacts/edge"),
    max_iter: int = 250,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train MLP on synthetic episode splits and export to ONNX."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = generate_dataset_splits(
        train_seeds=train_seeds,
        val_seeds=val_seeds,
        test_seeds=test_seeds,
        episode_duration_s=45.0,
    )

    X_train, y_train = dataset["train"]["X"], dataset["train"]["y"]
    X_test, y_test = dataset["test"]["X"], dataset["test"]["y"]

    # 1. Train MLP (39 -> 32 -> 16 -> 9)
    mlp = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=True,
    )
    mlp.fit(X_train, y_train)

    # 2. Evaluate on test episodes
    preds = mlp.predict(X_test)
    acc = float(accuracy_score(y_test, preds))
    f1 = float(f1_score(y_test, preds, average="macro", zero_division=0))

    # 3. Export to ONNX
    initial_type = [("float_input", FloatTensorType([None, 39]))]
    onnx_model = to_onnx(
        mlp,
        initial_types=initial_type,
        target_opset=15,
        options={type(mlp): {"zipmap": False}},
    )
    onnx.checker.check_model(onnx_model)

    onnx_path = output_dir / "maneuver_mlp.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # 4. Manifest
    manifest = {
        "model_name": onnx_path.name,
        "model_type": "MLPClassifier",
        "architecture": "Dense(32) -> ReLU -> Dense(16) -> ReLU -> Dense(9)",
        "num_features": 39,
        "feature_names": feature_names(),
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": {str(i): c for i, c in enumerate(CLASS_NAMES)},
        "test_metrics": {
            "accuracy": round(acc, 4),
            "macro_f1": round(f1, 4),
        },
        "onnx_file_size_kb": onnx_path.stat().st_size / 1024.0,
    }

    manifest_path = output_dir / "mlp_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/edge"))
    parser.add_argument("--seed-base", type=int, default=42)
    args = parser.parse_args(argv)

    train_seeds = [args.seed_base + i for i in range(6)]
    val_seeds = [args.seed_base + 100 + i for i in range(2)]
    test_seeds = [args.seed_base + 200 + i for i in range(2)]

    print("Training MLP Neural Network for TensorRT branch...", file=sys.stderr)
    res = train_and_export_mlp(train_seeds, val_seeds, test_seeds, output_dir=args.out_dir)
    print(
        f"MLP Training & ONNX Export Successful:\n"
        f"  Accuracy:  {res['test_metrics']['accuracy']:.4f}\n"
        f"  Macro-F1:  {res['test_metrics']['macro_f1']:.4f}\n"
        f"  ONNX Size: {res['onnx_file_size_kb']:.1f} KB\n"
        f"  Artifact:  {args.out_dir / 'maneuver_mlp.onnx'}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

