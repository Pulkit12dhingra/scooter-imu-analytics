"""tests/test_onnx_export.py
Unit tests for Stage 5 ONNX export, checker validation, and parity.
"""
from pathlib import Path
import json
import onnx
import onnxruntime as ort
import pytest

from ml.export_edge_model import export_rf_to_onnx
from ml.validate_onnx import validate_onnx_parity


def test_export_and_validate_onnx(tmp_path):
    """Verify ONNX export, checker, and exact prediction parity with scikit-learn."""
    model_joblib = Path("artifacts/edge/maneuver_rf.joblib")
    if not model_joblib.exists():
        pytest.skip("Base joblib model does not exist, run training first")

    onnx_out = tmp_path / "test_model.onnx"
    manifest_out = tmp_path / "test_manifest.json"

    # 1. Export
    manifest = export_rf_to_onnx(model_joblib, onnx_out, manifest_out)
    assert onnx_out.exists()
    assert manifest_out.exists()

    # 2. Check ONNX graph
    model_proto = onnx.load(str(onnx_out))
    onnx.checker.check_model(model_proto)
    assert len(model_proto.graph.input) == 1
    assert model_proto.graph.input[0].type.tensor_type.shape.dim[1].dim_value == 39

    # 3. Check Manifest contents
    with open(manifest_out, "r") as f:
        loaded_manifest = json.load(f)
    assert loaded_manifest["num_features"] == 39
    assert loaded_manifest["num_classes"] == 9
    assert "idx_to_class" in loaded_manifest

    # 4. Parity test
    res = validate_onnx_parity(model_joblib, onnx_out, manifest_out, n_test_samples=100)
    assert res["class_agreement_count"] == 100
    assert res["agreement_rate"] == 1.0
    assert res["max_prob_difference"] < 1e-4

