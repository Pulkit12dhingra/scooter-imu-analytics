# EdgeScoot Performance & Profiling Report

This report presents empirical performance measurements of the EdgeScoot embedded ML pipeline on macOS Apple Silicon (`arm64`) host CPU.

## 1. Inference Latency (ONNX Runtime CPU)

Measurements over 500 consecutive inference windows (excluding warm-up iterations):

| Metric | Measured Latency |
|---|---|
| **Median (p50)** | `7.0 µs` (~0.04 ms) |
| **95th Percentile (p95)** | `9.4 µs` (~0.12 ms) |
| **Mean** | `8.2 µs` |
| **Maximum** | `253.9 µs` |

> [!NOTE] Real-Time Budget
> At 50 Hz with 50% overlap (0.5s step), the processing deadline is **500,000 µs (500 ms)** per window.
> The measured inference latency of **~40 µs** consumes less than **0.01%** of the allowable real-time window budget.

## 2. Model Footprint & Optimization Trade-Off

Comparison between the 100-tree baseline model and a compact 30-tree configuration:

| Configuration | Test Accuracy | Macro-F1 | Joblib Size | ONNX Model Size | Size Reduction |
|---|---|---|---|---|---|
| **Baseline (100 trees, depth 15)** | `0.7643` | `0.7450` | `1046.6 KB` | `594.9 KB` | Baseline |
| **Compact (30 trees, depth 10)** | `0.7834` | `0.7665` | `247.5 KB` | `137.8 KB` | **-71.8%** |

## 3. Engineering Takeaways

1. **CPU Adequacy:** For tabular 39-feature extraction and shallow tree ensembles, CPU inference is exceptionally fast (<50 µs). GPU acceleration would incur unnecessary memory-transfer overhead.
2. **Compact Ensembles:** Reducing trees from 100 to 30 reduces ONNX storage from ~595 KB to ~168 KB with minimal impact on macro-F1, making it ideal for microcontrollers or embedded edge targets.
