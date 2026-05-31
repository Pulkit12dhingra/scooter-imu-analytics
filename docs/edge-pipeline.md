# EdgeScoot: Embedded ML Pipeline Architecture

EdgeScoot transforms `scooter-imu-analytics` into an end-to-end, software-in-the-loop embedded ML pipeline targeting edge micro-mobility devices.

## 1. System Architecture

```text
               TRAINING PIPELINE (Batch Mode)
  
  scripts/live_imu_simulator.py (Batch Mode)
                 |
                 v
   50 Hz Synthetic Episodes (Train / Val / Test)
                 |
                 v
   Sliding Windows (1.0s window, 0.5s step)
                 |
                 v
   ml/window_labeling.py (80% Purity + Fall Event Rule)
                 |
                 v
   Canonical 39 Feature Vectors (ml/features.py)
                 |
                 +-----------------------+
                 |                       |
                 v                       v
      Random Forest Baseline    Small Dense Neural Net (MLP)
      (ml/train_edge_rf.py)          (ml/train_mlp.py)
                 |                       |
                 v                       v
      artifacts/edge/*.onnx     artifacts/edge/*.onnx
                 |                       |
                 |                       v
                 |           Optional Remote TensorRT GPU
                 |           (docs/tensorrt-setup.md)
                 |
                 v
===========================================================
             EMBEDDED INFERENCE RUNTIME (Live Mode)

  scripts/live_imu_simulator.py (Paced 50 Hz Live Stream)
                 |
                 | (Piped CSV stream: sample_idx, t, dt, gx, gy, gz)
                 v
     embedded/build/edge_scoot (C++17 Binary)
                 |
                 v
     csv_parser.cpp -> ImuSample struct
                 |
                 v
     window_buffer.cpp (50-sample Ring Buffer)
                 |
                 v
     feature_extractor.cpp (C++ Port of 39 Features)
                 |
                 v
     inference_engine.cpp (ONNX Runtime CPU Session)
                 |
                 v
     Live Predictions (PRED window=N t=Ts class=Name latency=Xµs)
```

## 2. Directory Structure Overview

- `embedded/`: Portable C++17 stream processor and inference engine.
  - `include/`: Clean modular headers (`imu_sample.hpp`, `window_buffer.hpp`, `feature_extractor.hpp`, `inference_engine.hpp`, `csv_parser.hpp`).
  - `src/`: Core implementations and `main.cpp` stream CLI.
  - `CMakeLists.txt`: Standard CMake build specification linking `onnxruntime`.
- `scripts/`:
  - `live_imu_simulator.py`: Online stateful procedural sensor simulator.
  - `benchmark_edge.py`: Profiling suite for latency and memory footprints.
- `ml/`:
  - `window_labeling.py`: Window purity and transition labeling policies.
  - `train_synthetic.py`: Episode-isolated dataset generator.
  - `train_edge_rf.py`: Random Forest edge model training CLI.
  - `export_edge_model.py`: ONNX exporter and manifest generator.
  - `validate_onnx.py`: Scikit-learn vs ONNX Runtime parity verifier.
  - `train_mlp.py`: Neural network branch for TensorRT experimentation.
- `artifacts/edge/`: Ignored directory storing exported ONNX models, manifests, and benchmark reports.
- `docs/`: Comprehensive technical guides for setup, simulation, benchmarking, TensorRT, and Jetson deployment.

## 3. End-to-End Execution Guide

```bash
# 1. Environment sync with uv
uv sync --locked --extra dev --extra ml --extra edge-ml

# 2. Train baseline model and export to ONNX
uv run python ml/train_edge_rf.py --out-dir artifacts/edge
uv run python ml/export_edge_model.py

# 3. Build C++ engine
cmake -S embedded -B embedded/build
cmake --build embedded/build

# 4. Stream real-time simulated sensors directly into C++ inference engine:
uv run python -u scripts/live_imu_simulator.py --mode live --rate 50 --format csv --duration 30 | \
  ./embedded/build/edge_scoot --model artifacts/edge/maneuver_rf.onnx
```

