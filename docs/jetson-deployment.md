# EdgeScoot: NVIDIA Jetson Orin Nano Deployment Guide

This guide details the physical deployment path for running the `edge_scoot` C++ embedded application on an NVIDIA Jetson Orin Nano developer kit running JetPack 6.x (Ubuntu 22.04 LTS).

## 1. Target Hardware Specifications

- **Target Device:** NVIDIA Jetson Orin Nano Developer Kit (4GB or 8GB module)
- **CPU:** 6-core Arm Cortex-A78AE v8.2 64-bit CPU
- **GPU:** 1024-core NVIDIA Ampere architecture GPU with 32 Tensor Cores
- **Target OS:** NVIDIA JetPack 6.0 / 6.1 (Ubuntu 22.04 LTS aarch64)

## 2. Emulation vs Physical Hardware Reality

To maintain strict engineering rigor, this project distinguishes between three tiers of execution:

1. **Software-in-the-Loop Simulation (Mac host):**
   - Procedural sensor streaming and local CPU ONNX Runtime inference.
   - Fully functional on host without hardware dependencies.
2. **Linux aarch64 User-space Emulation (Docker/QEMU):**
   - Validates Linux aarch64 ABI compatibility and binary compilation.
   - Does **not** simulate Jetson GPU, TensorRT CUDA kernels, physical buses (I2C/SPI), or thermal/power constraints.
3. **Physical Jetson Hardware:**
   - Real-world validation of latency, power draw (7W–15W modes), thermals, and physical sensor I/O.

## 3. Deployment Steps on Jetson Orin Nano

### Step 1: Install System Prerequisites
```bash
sudo apt update
sudo apt install -y build-essential cmake git libonnxruntime-dev
```

### Step 2: Clone Repository and Build C++ Engine
```bash
git clone https://github.com/Pulkit12dhingra/scooter-imu-analytics.git
cd scooter-imu-analytics

cmake -S embedded -B embedded/build -DCMAKE_BUILD_TYPE=Release
cmake --build embedded/build -j$(nproc)
```

### Step 3: Run Inference Pipeline
```bash
# Replay synthetic sensor stream into C++ engine:
./embedded/build/edge_scoot \
  --model artifacts/edge/maneuver_rf.onnx \
  --manifest artifacts/edge/model_manifest.json < test_ride.csv
```

## 4. Transitioning to Physical IMU Sensors

In production, the standard input CSV stream is replaced by a hardware driver reading a 6-axis IMU (such as TDK InvenSense ICM-42688-P or Bosch BMI088):

```text
[ Physical IMU (I2C / SPI) ]
             |
             v
 [ Linux IIO (/dev/iio:device0) or SPI driver ]
             |
             v
 [ ImuSample Struct: gyr_x, gyr_y, gyr_z ]
             |
             v
 [ 50-sample Ring Buffer (WindowBuffer) ]
             |
             v
 [ Feature Extraction (39 features) ]
             |
             v
 [ ONNX Runtime CPU / TensorRT Engine ]
             |
             v
 [ Maneuver Prediction (parked, turn, fall...) ]
```

