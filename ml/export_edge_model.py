"""ml/export_edge_model.py
Export trained ManeuverClassifier Random Forest to ONNX and generate model manifest (Stage 5).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import sklearn
import skl2onnx
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType

# Make root and ml importable
ROOT_DIR = Path(__file__).resolve().parent.parent
ML_DIR = ROOT_DIR / "ml"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from ml.classifiers import ManeuverClassifier
from ml.features import feature_names
from ml.labeler import CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS


def export_rf_to_onnx(
    joblib_path: Path,
    output_onnx_path: Path,
    manifest_path: Path,
    target_opset: int = 15,
) -> dict[str, Any]:
    """Export fitted ManeuverClassifier to ONNX and write manifest."""
    if not joblib_path.exists():
        raise FileNotFoundError(f"Model file not found: {joblib_path}")

    mc: ManeuverClassifier = joblib.load(joblib_path)
    if not hasattr(mc, "clf") or not mc._fitted:
        raise ValueError("Provided model is not a fitted ManeuverClassifier")

    # Define float32 input shape: [batch_size, 39]
    initial_type = [("float_input", FloatTensorType([None, 39]))]

    # Convert scikit-learn Random Forest to ONNX
    onnx_model = to_onnx(
        mc.clf,
        initial_types=initial_type,
        target_opset=target_opset,
        options={type(mc.clf): {"zipmap": False}},  # zipmap=False outputs raw probability tensor
    )

    # Validate ONNX model consistency
    onnx.checker.check_model(onnx_model)

    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # Build model manifest
    input_tensor = onnx_model.graph.input[0]
    output_tensors = [out.name for out in onnx_model.graph.output]

    manifest: dict[str, Any] = {
        "model_name": output_onnx_path.name,
        "model_version": "1.0.0",
        "input_name": input_tensor.name,
        "input_dtype": "float32",
        "input_shape": [None, 39],
        "output_names": output_tensors,
        "num_features": 39,
        "feature_names": feature_names(),
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": {str(i): c for i, c in enumerate(CLASS_NAMES)},
        "opset": target_opset,
        "library_versions": {
            "onnx": onnx.__version__,
            "skl2onnx": skl2onnx.__version__,
            "onnxruntime": ort.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-in",
        type=Path,
        default=Path("artifacts/edge/maneuver_rf.joblib"),
        help="Input joblib model path",
    )
    parser.add_argument(
        "--onnx-out",
        type=Path,
        default=Path("artifacts/edge/maneuver_rf.onnx"),
        help="Output ONNX path",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=Path("artifacts/edge/model_manifest.json"),
        help="Output model manifest path",
    )
    args = parser.parse_args(argv)

    print(f"Exporting Random Forest to ONNX...", file=sys.stderr)
    manifest = export_rf_to_onnx(args.model_in, args.onnx_out, args.manifest_out)
    print(
        f"Export successful:\n"
        f"  ONNX Model: {args.onnx_out} ({args.onnx_out.stat().st_size / 1024:.1f} KB)\n"
        f"  Manifest:   {args.manifest_out}\n"
        f"  Input:      {manifest['input_name']} float32 [None, 39]\n"
        f"  Outputs:    {manifest['output_names']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

