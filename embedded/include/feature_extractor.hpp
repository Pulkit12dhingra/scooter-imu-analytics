#pragma once

#include <vector>
#include <string>
#include "imu_sample.hpp"

/**
 * @brief Computes 9 statistical and spectral features for a 1D signal.
 *
 * Metrics in exact order:
 * 1. Mean
 * 2. Population Standard Deviation + 1e-12 (ddof=0)
 * 3. Minimum
 * 4. Maximum
 * 5. Root Mean Square (RMS)
 * 6. Zero-Crossing Rate (ZCR)
 * 7. FFT Bin 1 Magnitude (de-meaned)
 * 8. FFT Bin 2 Magnitude (de-meaned)
 * 9. FFT Bin 3 Magnitude (de-meaned)
 */
std::vector<double> extract_1d_features(const std::vector<double>& signal);

/**
 * @brief Computes Pearson correlation between two 1D signals.
 *
 * Returns 0.0 if either signal has zero variance (constant signal).
 */
double compute_pearson_correlation(const std::vector<double>& x, const std::vector<double>& y);

/**
 * @brief Extracts the canonical 39 features from a window of ImuSample objects.
 *
 * Canonical ordering:
 * - 9 features for gyr_x
 * - 9 features for gyr_y
 * - 9 features for gyr_z
 * - 9 features for gyr_magnitude
 * - 3 cross-axis Pearson correlations: corr_xy, corr_yz, corr_xz
 *
 * @param window A vector of ImuSample (typically 50 samples for 1.0s @ 50Hz)
 * @return std::vector<double> of length 39
 */
std::vector<double> extract_39_features(const std::vector<ImuSample>& window);

/**
 * @brief Extracts the canonical 39 features converted to float32 for ONNX inference.
 */
std::vector<float> extract_39_features_float(const std::vector<ImuSample>& window);

/**
 * @brief Returns canonical names of the 39 features matching Python feature_names().
 */
std::vector<std::string> canonical_feature_names();

