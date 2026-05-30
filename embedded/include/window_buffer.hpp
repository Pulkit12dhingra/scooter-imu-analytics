#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include "imu_sample.hpp"

/**
 * @brief Sliding window buffer maintaining a fixed sample window with overlap.
 *
 * Default configuration for 50 Hz:
 * - window_size: 50 samples (1.0 second)
 * - step_size:   25 samples (0.5 second, 50% overlap)
 */
class WindowBuffer {
public:
    explicit WindowBuffer(size_t window_size = 50, size_t step_size = 25)
        : window_size_(window_size),
          step_size_(step_size),
          total_samples_(0),
          new_samples_since_window_(0) {
        buffer_.reserve(window_size_ + step_size_);
    }

    /**
     * @brief Append a sample to the buffer.
     * @return true if a new full window of size `window_size_` is ready for extraction.
     */
    bool push_sample(const ImuSample& sample) {
        buffer_.push_back(sample);
        total_samples_++;
        new_samples_since_window_++;

        if (total_samples_ < window_size_) {
            return false;
        }

        // First full window ready
        if (total_samples_ == window_size_) {
            new_samples_since_window_ = 0;
            return true;
        }

        // Subsequent windows ready every `step_size_` samples
        if (new_samples_since_window_ >= step_size_) {
            // Discard the oldest `step_size_` samples to keep memory strictly bounded
            if (buffer_.size() > window_size_) {
                size_t excess = buffer_.size() - window_size_;
                buffer_.erase(buffer_.begin(), buffer_.begin() + excess);
            }
            new_samples_since_window_ = 0;
            return true;
        }

        return false;
    }

    /**
     * @brief Access the current window samples (exactly `window_size` samples).
     */
    const std::vector<ImuSample>& current_window() const {
        return buffer_;
    }

    size_t total_samples() const {
        return total_samples_;
    }

    void reset() {
        buffer_.clear();
        total_samples_ = 0;
        new_samples_since_window_ = 0;
    }

private:
    size_t window_size_;
    size_t step_size_;
    size_t total_samples_;
    size_t new_samples_since_window_;
    std::vector<ImuSample> buffer_;
};

