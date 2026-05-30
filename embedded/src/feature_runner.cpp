#include <iostream>
#include <string>
#include <vector>
#include <iomanip>

#include "imu_sample.hpp"
#include "csv_parser.hpp"
#include "feature_extractor.hpp"

int main() {
    std::string line;
    std::vector<ImuSample> window;
    bool is_first_line = true;

    while (std::getline(std::cin, line)) {
        if (line.empty() || line == "\r") continue;
        if (is_first_line) {
            is_first_line = false;
            if (line.rfind("sample_index", 0) == 0) continue;
        }

        ImuSample s;
        std::string err;
        if (!parse_sensor_csv_line(line, s, err)) {
            std::cerr << "Error: " << err << "\n";
            return 1;
        }
        window.push_back(s);
    }

    if (window.size() < 2) {
        std::cerr << "Error: Window must have at least 2 samples, got " << window.size() << "\n";
        return 2;
    }

    std::vector<double> feats = extract_39_features(window);
    std::cout << std::setprecision(12);
    for (size_t i = 0; i < feats.size(); ++i) {
        std::cout << feats[i] << (i + 1 < feats.size() ? "," : "\n");
    }

    return 0;
}

