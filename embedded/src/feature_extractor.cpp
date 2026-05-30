#include "feature_extractor.hpp"

#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>

namespace {
constexpr double PI = 3.14159265358979323846;

int sign_func(double val) {
    if (val > 0.0) return 1;
    if (val < 0.0) return -1;
    return 0;
}
} // namespace

std::vector<double> extract_1d_features(const std::vector<double>& signal) {
    const size_t N = signal.size();
    if (N < 2) {
        throw std::invalid_argument("Signal window must have at least 2 samples");
    }

    // 1. Mean
    double sum = 0.0;
    double sq_sum = 0.0;
    double min_val = signal[0];
    double max_val = signal[0];

    for (size_t i = 0; i < N; ++i) {
        double v = signal[i];
        sum += v;
        sq_sum += v * v;
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }
    double mean = sum / static_cast<double>(N);

    // 2. Population Standard Deviation (ddof=0) + 1e-12 epsilon
    double var_sum = 0.0;
    for (size_t i = 0; i < N; ++i) {
        double diff = signal[i] - mean;
        var_sum += diff * diff;
    }
    double pop_std = std::sqrt(var_sum / static_cast<double>(N)) + 1e-12;

    // 3. RMS
    double rms = std::sqrt(sq_sum / static_cast<double>(N));

    // 4. Zero-Crossing Rate: (diff(sign(w)) != 0).sum() / len(w)
    size_t zcr_count = 0;
    for (size_t i = 0; i < N - 1; ++i) {
        if (sign_func(signal[i + 1]) != sign_func(signal[i])) {
            zcr_count++;
        }
    }
    double zcr = static_cast<double>(zcr_count) / static_cast<double>(N);

    // 5. FFT Magnitudes for bins 1, 2, 3 of de-meaned signal
    // Corresponds to np.abs(np.fft.rfft(w - mean))[1:4]
    double fft1 = 0.0;
    double fft2 = 0.0;
    double fft3 = 0.0;

    auto compute_fft_bin = [&](int k) -> double {
        double re = 0.0;
        double im = 0.0;
        const double angle_step = 2.0 * PI * static_cast<double>(k) / static_cast<double>(N);
        for (size_t n = 0; n < N; ++n) {
            double angle = angle_step * static_cast<double>(n);
            double val = signal[n] - mean;
            re += val * std::cos(angle);
            im -= val * std::sin(angle);
        }
        return std::sqrt(re * re + im * im);
    };

    fft1 = compute_fft_bin(1);
    fft2 = compute_fft_bin(2);
    fft3 = compute_fft_bin(3);

    return {mean, pop_std, min_val, max_val, rms, zcr, fft1, fft2, fft3};
}

double compute_pearson_correlation(const std::vector<double>& x, const std::vector<double>& y) {
    const size_t N = x.size();
    if (N != y.size() || N < 2) {
        return 0.0;
    }

    double mean_x = std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(N);
    double mean_y = std::accumulate(y.begin(), y.end(), 0.0) / static_cast<double>(N);

    double cov = 0.0;
    double var_x = 0.0;
    double var_y = 0.0;

    for (size_t i = 0; i < N; ++i) {
        double dx = x[i] - mean_x;
        double dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if (var_x < 1e-15 || var_y < 1e-15) {
        // Zero variance (constant signal) returns 0.0 matching Python np.nan_to_num handling
        return 0.0;
    }

    double denom = std::sqrt(var_x * var_y);
    double r = cov / denom;

    if (std::isnan(r) || std::isinf(r)) {
        return 0.0;
    }
    return std::clamp(r, -1.0, 1.0);
}

std::vector<double> extract_39_features(const std::vector<ImuSample>& window) {
    const size_t N = window.size();
    if (N < 2) {
        throw std::invalid_argument("Window must contain at least 2 samples");
    }

    std::vector<double> gx(N), gy(N), gz(N), gmag(N);
    for (size_t i = 0; i < N; ++i) {
        gx[i] = window[i].gyr_x_rad_s;
        gy[i] = window[i].gyr_y_rad_s;
        gz[i] = window[i].gyr_z_rad_s;
        gmag[i] = window[i].magnitude();
    }

    std::vector<double> features;
    features.reserve(39);

    // 4 signals x 9 features = 36 features
    for (const auto& sig : {&gx, &gy, &gz, &gmag}) {
        auto feats_1d = extract_1d_features(*sig);
        features.insert(features.end(), feats_1d.begin(), feats_1d.end());
    }

    // 3 cross-axis Pearson correlations
    features.push_back(compute_pearson_correlation(gx, gy));
    features.push_back(compute_pearson_correlation(gy, gz));
    features.push_back(compute_pearson_correlation(gx, gz));

    return features;
}

std::vector<float> extract_39_features_float(const std::vector<ImuSample>& window) {
    auto d_feats = extract_39_features(window);
    std::vector<float> f_feats(d_feats.size());
    for (size_t i = 0; i < d_feats.size(); ++i) {
        f_feats[i] = static_cast<float>(d_feats[i]);
    }
    return f_feats;
}

std::vector<std::string> canonical_feature_names() {
    std::vector<std::string> stats = {"mean", "std", "min", "max", "rms", "zcr", "fft1", "fft2", "fft3"};
    std::vector<std::string> axes = {"gyrx", "gyry", "gyrz", "mag"};
    std::vector<std::string> names;
    names.reserve(39);

    for (const auto& ax : axes) {
        for (const auto& s : stats) {
            names.push_back(ax + "_" + s);
        }
    }
    names.push_back("corr_xy");
    names.push_back("corr_yz");
    names.push_back("corr_xz");
    return names;
}

