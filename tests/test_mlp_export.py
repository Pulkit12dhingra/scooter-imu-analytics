"""tests/test_mlp_export.py
Unit tests for Stage 11 MLP neural network training and ONNX export.
"""
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from ml.labeler import CLASS_NAMES
from ml.train_mlp import train_and_export_mlp


def test_train_and_export_mlp(tmp_path):
    """Verify MLP trains, exports to ONNX, passes checker, and executes in ORT."""
    manifest = train_and_export_mlp(
        train_seeds=[1, 2],
        val_seeds=[10],
        test_seeds=[20],
        output_dir=tmp_path,
        max_iter=30,
    )

    onnx_path = tmp_path / "maneuver_mlp.onnx"
    assert onnx_path.exists()
    assert manifest["num_features"] == 39
    assert manifest["num_classes"] == 9

    # ONNX checker
    model_proto = onnx.load(str(onnx_path))
    onnx.checker.check_model(model_proto)

    # Inference via ONNX Runtime CPU
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    dummy_input = np.zeros((1, 39), dtype=np.float32)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: dummy_input})

    # Class ID output
    class_idx = int(outputs[0][0])
    assert 0 <= class_idx < len(CLASS_NAMES)

