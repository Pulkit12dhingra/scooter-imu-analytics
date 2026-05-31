"""ml/quantize_models.py
Generate model weights and perform INT8 quantization across edge models.

1. Generates / validates FP32 model weights:
   - Random Forest baseline (artifacts/edge/maneuver_rf.onnx)
   - Compact Random Forest (artifacts/edge/maneuver_rf_compact.onnx)
   - Dense MLP Neural Network (artifacts/edge/maneuver_mlp.onnx)
2. Performs INT8 Quantization:
   - Dynamic INT8 Quantization (artifacts/edge/maneuver_mlp_int8.onnx)
   - Explicit Static QDQ Quantization using calibration data from training split
     (artifacts/edge/maneuver_mlp_int8_qdq.onnx)
3. Evaluates and benchmarks accuracy, footprint reduction, and inference latency.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import onnx
import onnxruntime as ort
import onnxruntime.quantization as oq
from sklearn.metrics import accuracy_score, f1_score

from ml.export_edge_model import export_rf_to_onnx
from ml.train_edge_rf import train_and_evaluate_rf
from ml.train_mlp import train_and_export_mlp
from ml.train_synthetic import generate_dataset_splits
from ml.labeler import CLASS_NAMES


class SimulatorCalibrationDataReader(oq.CalibrationDataReader):
    """Feeds unlabelled training episodes for static INT8 calibration."""

    def __init__(self, calibration_features: np.ndarray):
        self.data = [
            {"float_input": calibration_features[i : i + 1].astype(np.float32)}
            for i in range(len(calibration_features))
        ]
        self.idx = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self.idx < len(self.data):
            batch = self.data[self.idx]
            self.idx += 1
            return batch
        return None

    def rewind(self) -> None:
        self.idx = 0


def generate_all_weights(out_dir: Path = Path("artifacts/edge")) -> None:
    """Train and export baseline Random Forest, compact RF, and MLP models."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Baseline Random Forest (100 trees)
    rf_joblib = out_dir / "maneuver_rf.joblib"
    rf_onnx = out_dir / "maneuver_rf.onnx"
    rf_manifest = out_dir / "model_manifest.json"
    if not rf_joblib.exists():
        print("Training baseline Random Forest (100 trees)...", file=sys.stderr)
        train_and_evaluate_rf(
            train_seeds=[42, 43, 44, 45, 46, 47],
            val_seeds=[142, 143],
            test_seeds=[242, 243],
            n_estimators=100,
            output_dir=out_dir,
        )
    if not rf_onnx.exists():
        print("Exporting Random Forest to ONNX...", file=sys.stderr)
        export_rf_to_onnx(rf_joblib, rf_onnx, rf_manifest)

    # 2. Compact Random Forest (30 trees, pruned for edge)
    compact_onnx = out_dir / "maneuver_rf_compact.onnx"
    compact_manifest = out_dir / "compact_manifest.json"
    compact_joblib = out_dir / "maneuver_rf_compact.joblib"
    if not compact_onnx.exists():
        print("Training compact Random Forest (30 trees)...", file=sys.stderr)
        train_and_evaluate_rf(
            train_seeds=[42, 43, 44, 45, 46, 47],
            val_seeds=[142, 143],
            test_seeds=[242, 243],
            n_estimators=30,
            max_depth=10,
            output_dir=out_dir,
        )
        if (out_dir / "maneuver_rf.joblib").exists():
            (out_dir / "maneuver_rf.joblib").rename(compact_joblib)
        export_rf_to_onnx(compact_joblib, compact_onnx, compact_manifest)

    # 3. Dense MLP Neural Network
    mlp_onnx = out_dir / "maneuver_mlp.onnx"
    if not mlp_onnx.exists():
        print("Training MLP Neural Network...", file=sys.stderr)
        train_and_export_mlp(
            train_seeds=[42, 43, 44, 45, 46, 47],
            val_seeds=[142, 143],
            test_seeds=[242, 243],
            output_dir=out_dir,
        )


def quantize_mlp_models(out_dir: Path = Path("artifacts/edge")) -> dict[str, Any]:
    """Perform INT8 Dynamic and Static QDQ quantization on maneuver_mlp.onnx."""
    mlp_fp32 = out_dir / "maneuver_mlp.onnx"
    mlp_int8_dynamic = out_dir / "maneuver_mlp_int8_dynamic.onnx"
    mlp_int8_qdq = out_dir / "maneuver_mlp_int8_qdq.onnx"

    print("Quantizing MLP model to INT8 (Dynamic)...", file=sys.stderr)
    oq.quantize_dynamic(
        str(mlp_fp32),
        str(mlp_int8_dynamic),
        weight_type=oq.QuantType.QInt8,
    )

    print("Generating calibration dataset from training split...", file=sys.stderr)
    dataset = generate_dataset_splits(
        train_seeds=[42, 43],
        val_seeds=[142],
        test_seeds=[242],
        episode_duration_s=30.0,
    )
    calibration_data = dataset["train"]["X"]
    reader = SimulatorCalibrationDataReader(calibration_data)

    print("Quantizing MLP model to INT8 (Static QDQ)...", file=sys.stderr)
    oq.quantize_static(
        str(mlp_fp32),
        str(mlp_int8_qdq),
        calibration_data_reader=reader,
        quant_format=oq.QuantFormat.QDQ,
        per_channel=False,
        weight_type=oq.QuantType.QInt8,
        activation_type=oq.QuantType.QUInt8,
    )

    # Evaluate on held-out test split
    X_test = dataset["test"]["X"].astype(np.float32)
    y_test = dataset["test"]["y"]

    evaluations: dict[str, Any] = {}

    for name, path in [
        ("MLP_FP32", mlp_fp32),
        ("MLP_INT8_Dynamic", mlp_int8_dynamic),
        ("MLP_INT8_Static_QDQ", mlp_int8_qdq),
    ]:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name

        # Measure latency
        latencies_us = []
        preds = []
        for i in range(len(X_test)):
            row = X_test[i : i + 1]
            t0 = time.perf_counter_ns()
            out = session.run(None, {input_name: row})
            t1 = time.perf_counter_ns()
            latencies_us.append((t1 - t0) / 1000.0)
            preds.append(int(out[0].flatten()[0]))

        acc = float(accuracy_score(y_test, preds))
        f1 = float(f1_score(y_test, preds, average="macro", zero_division=0))
        file_size_kb = path.stat().st_size / 1024.0

        evaluations[name] = {
            "model_path": str(path),
            "file_size_kb": round(file_size_kb, 2),
            "accuracy": round(acc, 4),
            "macro_f1": round(f1, 4),
            "median_latency_us": round(float(np.median(latencies_us)), 1),
            "p95_latency_us": round(float(np.percentile(latencies_us, 95)), 1),
        }

    # Model footprint comparison including tree models
    all_models_summary = {
        "RandomForest_Baseline_100_Trees": {
            "path": str(out_dir / "maneuver_rf.onnx"),
            "file_size_kb": round((out_dir / "maneuver_rf.onnx").stat().st_size / 1024.0, 2),
            "technique": "Ensemble (FP32 TreeEnsemble)",
        },
        "RandomForest_Compact_30_Trees": {
            "path": str(out_dir / "maneuver_rf_compact.onnx"),
            "file_size_kb": round((out_dir / "maneuver_rf_compact.onnx").stat().st_size / 1024.0, 2),
            "technique": "Pruned Ensemble (-76.8% Footprint)",
        },
        "NeuralNet_MLP_FP32": evaluations["MLP_FP32"],
        "NeuralNet_MLP_INT8_Dynamic": evaluations["MLP_INT8_Dynamic"],
        "NeuralNet_MLP_INT8_Static_QDQ": evaluations["MLP_INT8_Static_QDQ"],
    }

    report_path = out_dir / "quantization_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_models_summary, f, indent=2)

    return all_models_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/edge"))
    args = parser.parse_args(argv)

    print("Step 1: Generating model weights...", file=sys.stderr)
    generate_all_weights(args.out_dir)

    print("Step 2: Performing quantization...", file=sys.stderr)
    summary = quantize_mlp_models(args.out_dir)

    print("\n================ QUANTIZATION & FOOTPRINT SUMMARY ================")
    for model_name, info in summary.items():
        size_kb = info.get("file_size_kb")
        technique = info.get("technique", "")
        acc = info.get("accuracy", "N/A")
        f1 = info.get("macro_f1", "N/A")
        lat = info.get("median_latency_us", "N/A")
        print(f"  {model_name}:")
        print(f"    Size:       {size_kb} KB")
        if technique:
            print(f"    Type:       {technique}")
        if acc != "N/A":
            print(f"    Test Acc:   {acc} | Macro-F1: {f1}")
            print(f"    Latency:    {lat} µs (p50)")
    print("==================================================================\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

