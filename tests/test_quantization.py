"""tests/test_quantization.py
Unit tests verifying quantized ONNX models execute cleanly in ONNX Runtime CPU.
"""
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import pytest


def test_quantized_models_inference():
    """Verify quantized INT8 models execute cleanly on CPU."""
    dynamic_path = Path("artifacts/edge/maneuver_mlp_int8_dynamic.onnx")
    qdq_path = Path("artifacts/edge/maneuver_mlp_int8_qdq.onnx")

    assert dynamic_path.exists()
    assert qdq_path.exists()

    # Verify ONNX model structure
    onnx.checker.check_model(onnx.load(str(dynamic_path)))
    onnx.checker.check_model(onnx.load(str(qdq_path)))

    # Run inference through both
    x = np.random.randn(5, 39).astype(np.float32)

    sess_dyn = ort.InferenceSession(str(dynamic_path), providers=["CPUExecutionProvider"])
    out_dyn = sess_dyn.run(None, {"float_input": x})
    assert len(out_dyn[0]) == 5

    sess_qdq = ort.InferenceSession(str(qdq_path), providers=["CPUExecutionProvider"])
    out_qdq = sess_qdq.run(None, {"float_input": x})
    assert len(out_qdq[0]) == 5

