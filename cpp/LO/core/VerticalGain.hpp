#pragma once

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>

namespace lo {

struct VerticalSelection {
    Eigen::Matrix<double, 6, 1> step;
    bool gated = false;
    int action = 0;  // 0: full, 1: zero-z, 2: clipped-z
};

inline VerticalSelection SelectVerticalWithGainRetention(
    const Eigen::Matrix<double, 6, 6>& damped_hessian,
    const Eigen::Matrix<double, 6, 1>& gradient,
    const Eigen::Matrix<double, 6, 1>& full,
    double gate = 0.03, double clip = 0.05, double retention = 0.95) {
    VerticalSelection selected{full};
    if (!full.allFinite() || std::abs(full(2)) <= gate) return selected;
    selected.gated = true;
    const auto gain = [&](const Eigen::Matrix<double, 6, 1>& step) {
        return -gradient.dot(step) - 0.5 * step.dot(damped_hessian * step);
    };
    const double full_gain = gain(full);
    if (!std::isfinite(full_gain) || full_gain <= 0.0) return selected;
    const double minimum_gain = retention * full_gain;
    // Conditional quadratic minimizer under e_z^T step = desired_z.
    // Re-solving the coupled five-dimensional block is equivalent to this
    // rank-one Schur update; it minimizes the gain sacrificed by a z constraint.
    const auto factor = damped_hessian.ldlt();
    if (factor.info() != Eigen::Success || !factor.isPositive()) return selected;
    Eigen::Matrix<double, 6, 1> ez = Eigen::Matrix<double, 6, 1>::Zero();
    ez(2) = 1.;
    const Eigen::Matrix<double, 6, 1> inverse_column = factor.solve(ez);
    if (!inverse_column.allFinite() || inverse_column(2) <= 0.) return selected;
    const auto conditional = [&](double desired_z) {
        Eigen::Matrix<double, 6, 1> step =
            full + (desired_z - full(2)) / inverse_column(2) * inverse_column;
        step(2) = desired_z;
        return step;
    };
    auto candidate = conditional(0.0);
    const double zero_gain = gain(candidate);
    if (std::isfinite(zero_gain) && zero_gain >= minimum_gain) {
        selected.step = candidate;
        selected.action = 1;
        return selected;
    }
    candidate = conditional(std::clamp(full(2), -clip, clip));
    const double clipped_gain = gain(candidate);
    if (candidate(2) != full(2) && std::isfinite(clipped_gain) && clipped_gain >= minimum_gain) {
        selected.step = candidate;
        selected.action = 2;
    }
    return selected;
}

struct VerticalDiagnostics {
    size_t correspondences = 0;
    size_t geometry_matches = 0;
    int iterations = 0;
    int gates = 0;
    int zero_steps = 0;
    int clipped_steps = 0;
    int full_fallbacks = 0;
    bool stage_clipped = false;
};

}  // namespace lo
