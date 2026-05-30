#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>

/**
 * @brief Output container for a single window classification.
 */
struct InferenceResult {
    int64_t class_idx = -1;
    std::string class_name;
    double latency_us = 0.0;
    std::vector<float> probabilities;
};

/**
 * @brief ONNX Runtime CPU inference wrapper for the 39-feature Maneuver Classifier.
 */
class InferenceEngine {
public:
    /**
     * @brief Construct and initialize the ONNX Runtime CPU session.
     * @param model_path Path to the exported .onnx model file
     * @param manifest_path Optional path to model_manifest.json for class decoding
     */
    explicit InferenceEngine(const std::string& model_path, const std::string& manifest_path = "");
    ~InferenceEngine();

    // Prevent copying, allow moving
    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;
    InferenceEngine(InferenceEngine&&) noexcept;
    InferenceEngine& operator=(InferenceEngine&&) noexcept;

    /**
     * @brief Run inference on a 39-feature vector.
     * @param features float32 feature vector of length 39
     * @return InferenceResult containing predicted class index, name, and execution latency
     */
    InferenceResult predict(const std::vector<float>& features);

    const std::string& input_name() const { return input_name_; }
    const std::map<int, std::string>& idx_to_class() const { return idx_to_class_; }

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    std::string input_name_;
    std::map<int, std::string> idx_to_class_;
};

