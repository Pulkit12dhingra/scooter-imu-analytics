# NVIDIA TensorRT Workflow & Remote GPU Guide

This document specifies the NVIDIA TensorRT optimization pipeline for deploying the EdgeScoot neural network model (`artifacts/edge/maneuver_mlp.onnx`) to compatible NVIDIA GPU targets, such as the NVIDIA Jetson Orin Nano or remote GPU environments (Google Colab / Kaggle).

> [!IMPORTANT] Host Environment Note
> The primary development environment is a MacBook Air Apple Silicon (`arm64`) host without NVIDIA CUDA or TensorRT hardware.
> Per Section 17 and Section 21 of the implementation plan, the core EdgeScoot pipeline executes entirely on host CPU. TensorRT engine compilation and inference require an NVIDIA GPU runtime and are executed in remote or target hardware environments.

## 1. Remote GPU Prerequisites (Google Colab / Kaggle)

In a free-tier Google Colab or Kaggle session with GPU accelerator enabled:

```bash
# 1. Verify NVIDIA GPU presence and driver
nvidia-smi

# 2. Check installed TensorRT version
python3 -c "import tensorrt as trt; print('TensorRT Version:', trt.__version__)"

# 3. Check trtexec CLI
which trtexec || find /usr -name trtexec 2>/dev/null
```

## 2. Compiling Engines via `trtexec`

Using the exported `maneuver_mlp.onnx` (39 inputs, Dense(32) -> ReLU -> Dense(16) -> ReLU -> Dense(9)):

### A. FP32 Baseline Engine
```bash
trtexec \
  --onnx=maneuver_mlp.onnx \
  --saveEngine=maneuver_mlp_fp32.engine \
  --avgRuns=100 \
  --duration=3
```

### B. FP16 Optimized Engine
```bash
trtexec \
  --onnx=maneuver_mlp.onnx \
  --saveEngine=maneuver_mlp_fp16.engine \
  --fp16 \
  --avgRuns=100 \
  --duration=3
```

## 3. TensorRT 11.x Explicit INT8 Quantization Workflow

In TensorRT 11.x, legacy `IInt8Calibrator` implicit calibration is deprecated in favor of **Explicit Quantization** with `QuantizeLinear` / `DequantizeLinear` (Q/DQ) nodes.

1. **Calibration Data:** Use only the training split episodes (`train_seeds`). Never calibrate on test data.
2. **Explicit Q/DQ Insertion:** Use NVIDIA Model Optimizer (`modelopt.torch.quantization` or `onnxruntime.quantization` with explicit Q/DQ format):
   ```python
   import onnxruntime.quantization as oq

   oq.quantize_static(
       model_input="maneuver_mlp.onnx",
       model_output="maneuver_mlp_int8_qdq.onnx",
       calibration_data_reader=calibration_reader,
       quant_format=oq.QuantFormat.QDQ,
       per_channel=False,
   )
   ```
3. **Build INT8 Engine:**
   ```bash
   trtexec \
     --onnx=maneuver_mlp_int8_qdq.onnx \
     --saveEngine=maneuver_mlp_int8.engine \
     --int8
   ```

## 4. Workload Characteristics & Limitations

For tiny models (e.g. 39 features, <10 KB parameter footprint):
- Kernel launch and PCIe transfer overheads often exceed compute duration.
- CPU inference latency (~7 µs in ONNX Runtime CPU) is typically lower than GPU round-trip launch latency.
- TensorRT is valuable primarily for high-dimensional sensor fusion (e.g. camera + IMU) or batch processing across large vehicle fleets.

