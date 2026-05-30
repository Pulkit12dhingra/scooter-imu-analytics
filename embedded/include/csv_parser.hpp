#pragma once

#include <string>
#include "imu_sample.hpp"

/**
 * @brief Parse a single comma-separated CSV line into an ImuSample.
 *
 * Expected columns:
 *   sample_index,elapsed_s,dt_s,gyr_x_rad_s,gyr_y_rad_s,gyr_z_rad_s
 *
 * @param line Input line from CSV stream
 * @param sample Output parsed sample
 * @param err Error message if parsing fails
 * @return true if successfully parsed, false otherwise
 */
bool parse_sensor_csv_line(const std::string& line, ImuSample& sample, std::string& err);

