#pragma once

#include <cstdint>
#include <cmath>
#include <string>

/**
 * @brief Represents a single IMU gyroscope sensor reading.
 *
 * All angular velocities are in radians per second (rad/s).
 */
struct ImuSample {
    int64_t sample_index = 0;
    double elapsed_s = 0.0;
    double dt_s = 0.0;
    double gyr_x_rad_s = 0.0;
    double gyr_y_rad_s = 0.0;
    double gyr_z_rad_s = 0.0;

    /**
     * @brief Computes overall Euclidean angular velocity magnitude.
     */
    double magnitude() const {
        return std::sqrt(gyr_x_rad_s * gyr_x_rad_s +
                         gyr_y_rad_s * gyr_y_rad_s +
                         gyr_z_rad_s * gyr_z_rad_s);
    }
};

