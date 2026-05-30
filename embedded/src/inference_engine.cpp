#include "inference_engine.hpp"

#include <onnxruntime_cxx_api.h>

#include <chrono>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <array>
#include <regex>

namespace {
const std::vector<std::string> DEFAULT_CLASS_NAMES = {
    "parked", "straight", "left_turn", "right_turn",
    "lane_change", "sharp_turn", "fall", "bumpy_road", "aggressive"
};

std::map<int, std::string> parse_manifest(const std::string& manifest_path) {
    std::map<int, std::string> result;
    if (manifest_path.empty()) {
        for (size_t i = 0; i < DEFAULT_CLASS_NAMES.size(); ++i) {
            result[static_cast<int>(i)] = DEFAULT_CLASS_NAMES[i];
        }
        return result;
    }

    std::ifstream file(manifest_path);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open manifest at " << manifest_path << ", using default classes\n";
        for (size_t i = 0; i < DEFAULT_CLASS_NAMES.size(); ++i) {
            result[static_cast<int>(i)] = DEFAULT_CLASS_NAMES[i];
        }
        return result;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    // Regex match entries like "0": "parked"
    std::regex pair_regex(R"pattern("([0-9]+)"\s*:\s*"([^"]+)")pattern");
    auto words_begin = std::sregex_iterator(content.begin(), content.end(), pair_regex);
    auto words_end = std::sregex_iterator();

    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        std::smatch match = *i;
        int idx = std::stoi(match[1].str());
        std::string name = match[2].str();
        result[idx] = name;
    }

    if (result.empty()) {
        for (size_t i = 0; i < DEFAULT_CLASS_NAMES.size(); ++i) {
            result[static_cast<int>(i)] = DEFAULT_CLASS_NAMES[i];
        }
    }
    return result;
}
} // namespace

class InferenceEngine::Impl {
public:
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    std::string input_name;
    std::string output_name_label;
    std::string output_name_prob;

    explicit Impl(const std::string& model_path)
        : env(ORT_LOGGING_LEVEL_WARNING, "EdgeScoot") {
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        try {
            session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
        } catch (const Ort::Exception& e) {
            throw std::runtime_error("Failed to load ONNX model from " + model_path + ": " + e.what());
        }

        Ort::AllocatorWithDefaultOptions allocator;
        auto in_name_alloc = session->GetInputNameAllocated(0, allocator);
        input_name = in_name_alloc.get();

        auto out0_alloc = session->GetOutputNameAllocated(0, allocator);
        output_name_label = out0_alloc.get();

        if (session->GetOutputCount() > 1) {
            auto out1_alloc = session->GetOutputNameAllocated(1, allocator);
            output_name_prob = out1_alloc.get();
        }
    }
};

InferenceEngine::InferenceEngine(const std::string& model_path, const std::string& manifest_path)
    : impl_(std::make_unique<Impl>(model_path)),
      input_name_(impl_->input_name),
      idx_to_class_(parse_manifest(manifest_path)) {}

InferenceEngine::~InferenceEngine() = default;

InferenceEngine::InferenceEngine(InferenceEngine&&) noexcept = default;
InferenceEngine& InferenceEngine::operator=(InferenceEngine&&) noexcept = default;

InferenceResult InferenceEngine::predict(const std::vector<float>& features) {
    if (features.size() != 39) {
        throw std::invalid_argument(
            "Expected 39 features for inference, received " + std::to_string(features.size())
        );
    }

    InferenceResult res;

    // Create input tensor: shape [1, 39]
    std::array<int64_t, 2> input_shape = {1, 39};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Non-owning view of features memory
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(features.data()),
        features.size(),
        input_shape.data(),
        input_shape.size()
    );

    const char* input_names[] = {impl_->input_name.c_str()};
    std::vector<const char*> output_names;
    output_names.push_back(impl_->output_name_label.c_str());
    if (!impl_->output_name_prob.empty()) {
        output_names.push_back(impl_->output_name_prob.c_str());
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    auto output_tensors = impl_->session->Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names.data(),
        output_names.size()
    );

    auto end_time = std::chrono::high_resolution_clock::now();
    res.latency_us = std::chrono::duration<double, std::micro>(end_time - start_time).count();

    // Extract predicted class index from output tensor 0 (int64)
    int64_t class_idx = output_tensors[0].GetTensorData<int64_t>()[0];
    res.class_idx = class_idx;

    auto it = idx_to_class_.find(static_cast<int>(class_idx));
    if (it != idx_to_class_.end()) {
        res.class_name = it->second;
    } else {
        res.class_name = "unknown_" + std::to_string(class_idx);
    }

    // Extract probabilities if output 1 is available
    if (output_tensors.size() > 1 && output_tensors[1].IsTensor()) {
        auto prob_info = output_tensors[1].GetTensorTypeAndShapeInfo();
        size_t n_classes = prob_info.GetElementCount();
        const float* prob_data = output_tensors[1].GetTensorData<float>();
        res.probabilities.assign(prob_data, prob_data + n_classes);
    }

    return res;
}

