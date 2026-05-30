#include "csv_parser.hpp"

#include <sstream>
#include <vector>
#include <cmath>
#include <stdexcept>

bool parse_sensor_csv_line(const std::string& line, ImuSample& sample, std::string& err) {
    if (line.empty()) {
        err = "Empty line";
        return false;
    }

    std::stringstream ss(line);
    std::string token;
    std::vector<std::string> tokens;

    while (std::getline(ss, token, ',')) {
        tokens.push_back(token);
    }

    if (tokens.size() != 6) {
        err = "Expected exactly 6 columns, got " + std::to_string(tokens.size());
        return false;
    }

    try {
        size_t idx_pos = 0;
        sample.sample_index = std::stoll(tokens[0], &idx_pos);
        if (idx_pos != tokens[0].size()) {
            err = "Invalid integer for sample_index: " + tokens[0];
            return false;
        }

        auto parse_double = [](const std::string& str, const std::string& col_name) -> double {
            size_t pos = 0;
            double val = std::stod(str, &pos);
            if (pos != str.size()) {
                throw std::invalid_argument("Extra trailing characters in " + col_name);
            }
            if (std::isnan(val) || std::isinf(val)) {
                throw std::invalid_argument("NaN or Inf detected in " + col_name);
            }
            return val;
        };

        sample.elapsed_s   = parse_double(tokens[1], "elapsed_s");
        sample.dt_s        = parse_double(tokens[2], "dt_s");
        sample.gyr_x_rad_s = parse_double(tokens[3], "gyr_x_rad_s");
        sample.gyr_y_rad_s = parse_double(tokens[4], "gyr_y_rad_s");
        sample.gyr_z_rad_s = parse_double(tokens[5], "gyr_z_rad_s");

    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }

    return true;
}

