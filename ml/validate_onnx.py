"""ml/validate_onnx.py
Validate prediction parity between scikit-learn and ONNX Runtime CPU (Stage 5).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import onnxruntime as ort

# Make root and ml importable
ROOT_DIR = Path(__file__).resolve().parent.parent
ML_DIR = ROOT_DIR / "ml"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from ml.classifiers import ManeuverClassifier
from ml.labeler import CLASS_NAMES, CLASS_TO_IDX


def validate_onnx_parity(
    joblib_path: Path,
    onnx_path: Path,
    manifest_path: Path,
    n_test_samples: int = 100,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Validate prediction parity between joblib model and ONNX Runtime."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    mc: ManeuverClassifier = joblib.load(joblib_path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    input_name = manifest["input_name"]
    idx_to_class = {int(k): v for k, v in manifest["idx_to_class"].items()}

    # Generate test features covering varied value ranges
    rng = np.random.default_rng(random_seed)
    X_test = rng.standard_normal((n_test_samples, 39)).astype(np.float32)

    # Scikit-learn predictions
    sklearn_preds_str = mc.predict(X_test)
    sklearn_preds_idx = np.array([CLASS_TO_IDX[c] for c in sklearn_preds_str], dtype=np.int32)
    sklearn_probs = mc.clf.predict_proba(X_test)

    # ONNX Runtime inference
    ort_outputs = session.run(None, {input_name: X_test})
    # ONNX RF classifier output 0 is label (int64), output 1 is probabilities
    ort_preds_raw = ort_outputs[0].flatten()
    ort_probs = ort_outputs[1]

    # Decode ORT predictions
    ort_preds_str = [idx_to_class[int(idx)] for idx in ort_preds_raw]
    ort_preds_idx = np.array([int(idx) for idx in ort_preds_raw], dtype=np.int32)

    # Parity checks
    class_agreement_count = int(np.sum(sklearn_preds_idx == ort_preds_idx))
    agreement_rate = class_agreement_count / n_test_samples
    max_prob_diff = float(np.max(np.abs(sklearn_probs - ort_probs)))

    results = {
        "n_test_samples": n_test_samples,
        "class_agreement_count": class_agreement_count,
        "agreement_rate": round(agreement_rate, 4),
        "max_prob_difference": round(max_prob_diff, 6),
        "model_input_name": input_name,
        "onnx_output_names": [out.name for out in session.get_outputs()],
    }

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-in",
        type=Path,
        default=Path("artifacts/edge/maneuver_rf.joblib"),
        help="Input joblib model path",
    )
    parser.add_argument(
        "--onnx-in",
        type=Path,
        default=Path("artifacts/edge/maneuver_rf.onnx"),
        help="Input ONNX model path",
    )
    parser.add_argument(
        "--manifest-in",
        type=Path,
        default=Path("artifacts/edge/model_manifest.json"),
        help="Input model manifest path",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Number of evaluation samples",
    )
    args = parser.parse_args(argv)

    print(f"Validating ONNX Runtime CPU prediction parity...", file=sys.stderr)
    res = validate_onnx_parity(args.model_in, args.onnx_in, args.manifest_in, n_test_samples=args.samples)

    print(
        f"Validation Results:\n"
        f"  Total Samples:    {res['n_test_samples']}\n"
        f"  Class Agreement:  {res['class_agreement_count']}/{res['n_test_samples']} ({res['agreement_rate']*100:.1f}%)\n"
        f"  Max Prob Delta:   {res['max_prob_difference']:.6f}",
        file=sys.stderr,
    )

    if res["agreement_rate"] < 0.99:
        print("Error: Agreement rate below 99% threshold!", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

