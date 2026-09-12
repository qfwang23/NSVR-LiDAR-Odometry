#pragma once

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <vector>

namespace lo {

struct VerticalGeometry {
    Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();
    double confidence = 0.;
};

inline VerticalGeometry EstimateVerticalGeometry(
    const std::vector<Eigen::Vector3d>& points, double voxel_size) {
    VerticalGeometry result;
    if (points.size() < 6) return result;
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const auto& p : points) mean += p - points.front();
    mean /= static_cast<double>(points.size());
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    for (const auto& p : points) {
        const Eigen::Vector3d d = (p - points.front()) - mean;
        covariance.noalias() += d * d.transpose();
    }
    covariance /= static_cast<double>(points.size());
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(covariance);
    if (eigen.info() != Eigen::Success) return result;
    const Eigen::Vector3d values = eigen.eigenvalues().cwiseMax(0.);
    // Require 2-D support, low thickness, and a predominantly vertical normal.
    if (values(1) < 1e-4 * voxel_size * voxel_size ||
        values(0) > .1 * values(1)) return result;
    result.normal = eigen.eigenvectors().col(0);
    if (result.normal.z() * result.normal.z() < .5) return VerticalGeometry{};
    if (result.normal.z() < 0.) result.normal *= -1.;
    result.confidence = std::clamp((values(1)-values(0))/values(2), 0., 1.);
    return result;
}

// Trace=3 and PSD: eigenvalues 1-c, 1-c, 1+2c. The direction is observed,
// not imposed world z; isotropic point-to-point remains the unsupported fallback.
inline Eigen::Matrix3d VerticalMetric(const VerticalGeometry& geometry) {
    return (1.-geometry.confidence) * Eigen::Matrix3d::Identity() +
           3.*geometry.confidence * geometry.normal * geometry.normal.transpose();
}

}  // namespace lo
