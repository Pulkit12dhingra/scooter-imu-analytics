#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <iomanip>

#include "imu_sample.hpp"
#include "csv_parser.hpp"
#include "window_buffer.hpp"
#include "feature_extractor.hpp"
#include "inference_engine.hpp"

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --model <path>      Path to ONNX model (default: artifacts/edge/maneuver_rf.onnx)\n"
              << "  --manifest <path>   Path to model manifest JSON (default: artifacts/edge/model_manifest.json)\n"
              << "  --help              Display this help message\n";
}

int main(int argc, char* argv[]) {
    std::string model_path = "artifacts/edge/maneuver_rf.onnx";
    std::string manifest_path = "artifacts/edge/model_manifest.json";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--manifest" && i + 1 < argc) {
            manifest_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 2;
        }
    }

    // Initialize inference engine
    std::unique_ptr<InferenceEngine> engine;
    try {
        engine = std::make_unique<InferenceEngine>(model_path, manifest_path);
    } catch (const std::exception& e) {
        std::cerr << "Error initializing InferenceEngine: " << e.what() << "\n";
        return 1;
    }

    std::cerr << "InferenceEngine initialized:\n"
              << "  Model:    " << model_path << "\n"
              << "  Input:    " << engine->input_name() << "\n"
              << "  Classes:  " << engine->idx_to_class().size() << "\n";

    WindowBuffer buffer(50, 25);
    std::string line;
    int64_t sample_count = 0;
    int64_t window_count = 0;
    bool is_first_line = true;

    while (std::getline(std::cin, line)) {
        if (line.empty() || line == "\r") {
            continue;
        }

        // Check and skip CSV header
        if (is_first_line) {
            is_first_line = false;
            if (line.rfind("sample_index", 0) == 0) {
                continue;
            }
        }

        ImuSample sample;
        std::string err;
        if (!parse_sensor_csv_line(line, sample, err)) {
            std::cerr << "Error parsing line " << (sample_count + 1) << ": " << err << "\n";
            std::cerr << "Offending row: " << line << "\n";
            return 1;
        }

        sample_count++;

        if (buffer.push_sample(sample)) {
            window_count++;
            const auto& win = buffer.current_window();
            double center_t = win[win.size() / 2].elapsed_s;

            // 1. Extract 39 float features
            auto feats_f = extract_39_features_float(win);

            // 2. Run ONNX Runtime CPU inference
            InferenceResult res = engine->predict(feats_f);

            // 3. Emit machine-readable prediction record
            std::cout << "PRED window=" << window_count
                      << " t=" << std::fixed << std::setprecision(3) << center_t << "s"
                      << " class=" << res.class_name
                      << " class_idx=" << res.class_idx
                      << " latency_us=" << std::setprecision(1) << res.latency_us
                      << "\n" << std::flush;
        }
    }

    std::cerr << "Processing completed. Total samples: " << sample_count
              << ", Predictions emitted: " << window_count << "\n";
    return 0;
}

