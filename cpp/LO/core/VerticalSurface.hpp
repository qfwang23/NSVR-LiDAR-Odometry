#pragma once

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <array>
#include <vector>

namespace lo {

struct VerticalSurfaceDiagnostics {
    size_t source_patches = 0, reference_patches = 0, matched_patches = 0;
    size_t fit_rejected = 0, rank_rejected = 0, validation_rejected = 0;
    bool applied = false;
    double before = 0., after = 0., delta_z = 0., rotation_rad = 0., range_bias = 0.;
};

// Three points on a fitted source surface, restricted to the common observed
// footprint. Their residual covariance includes BOTH plane fits and roughness.
struct VerticalSurfaceFactor {
    std::array<Eigen::Vector3d, 3> points;
    Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();
    double offset = 0.;
    Eigen::Matrix3d information = Eigen::Matrix3d::Identity();
    // A per-pair range-dependent measurement nuisance, never a scan correction.
    Eigen::Vector3d range_basis = Eigen::Vector3d::Zero();
    Eigen::Vector2d lower_xy, upper_xy;
    int fold = 0;
};

struct VerticalSurfaceCorrection {
    Sophus::SO3d rotation;
    double z = 0., range_bias = 0.;
};

Eigen::Vector3d SurfaceResidual(const VerticalSurfaceFactor& factor,
                               const VerticalSurfaceCorrection& correction);
Eigen::Matrix3d SurfaceJacobian(const VerticalSurfaceFactor& factor,
                               const VerticalSurfaceCorrection& correction, double lever_scale);
VerticalSurfaceCorrection CrossValidateVerticalSurfaces(
    const std::vector<VerticalSurfaceFactor>& factors, VerticalSurfaceDiagnostics* diagnostics);

Sophus::SE3d RefineVerticalSurfaces(
    const std::vector<Eigen::Vector3d>& current,
    const std::vector<Eigen::Vector3d>& previous,
    const Sophus::SE3d& previous_pose, const Sophus::SE3d& base_pose,
    double voxel_size, VerticalSurfaceDiagnostics* diagnostics = nullptr);

}  // namespace lo
