#pragma once

#include <Eigen/Core>
#include <tuple>
#include <vector>

#include "LO/core/VoxelHashMap.hpp"
#include "LO/core/VerticalGain.hpp"
#include "LO/core/VerticalSurface.hpp"

namespace lo {

std::vector<Eigen::Vector3d> VoxelDownsample(const std::vector<Eigen::Vector3d>& frame,
                                             double voxel_size);

std::vector<Eigen::Vector3d> RemoveGroundPointsAdvanced(const std::vector<Eigen::Vector3d>& points,
                                                        double grid_resolution,
                                                        double delta_h1,
                                                        double delta_h2,
                                                        double normal_angle_thresh_deg,
                                                        double pca_radius);

struct AdaptiveThreshold {
    explicit AdaptiveThreshold(double initial_threshold, double min_motion_th, double max_range)
        : initial_threshold_(initial_threshold),
          min_motion_th_(min_motion_th),
          max_range_(max_range) {}

    inline void UpdateModelDeviation(const Sophus::SE3d& current_deviation) {
        model_deviation_ = current_deviation;
    }

    double ComputeThreshold();

private:
    double initial_threshold_;
    double min_motion_th_;
    double max_range_;

    double     model_error_sse2_ = 0;
    int        num_samples_      = 0;
    Sophus::SE3d model_deviation_ = Sophus::SE3d();
};

}  // namespace lo

namespace lo::pipeline {

struct LOConfig {
    double voxel_size           = 1.0;
    double max_range            = 100.0;
    double min_range            = 5.0;
    int    max_points_per_voxel = 20;

    double min_motion_th      = 0.1;
    double initial_threshold  = 2.0;

    bool deskew = false;

    // The defaults preserve the proposed two-stage registration pipeline.
    bool enable_f2f                 = true;
    bool enable_vertical_constraint = true;
};

struct RegistrationMetrics {
    VerticalSurfaceDiagnostics vertical_surface;
    double vertical_surface_ms = 0.;
    size_t surface_anchor_age_frames = 0;
    double surface_anchor_footprint_motion_m = 0.;
    bool surface_anchor_replaced = false;
    VerticalDiagnostics ff_vertical;
    VerticalDiagnostics lm_vertical;
    size_t frame_index       = 0;
    bool   f2f_executed      = false;
    double f2f_ms            = 0.0;
    double f2m_ms            = 0.0;
    double registration_ms   = 0.0;

    double ThroughputHz() const {
        return registration_ms > 0.0 ? 1000.0 / registration_ms : 0.0;
    }
};

class LO {
public:
    using Vector3dVector      = std::vector<Eigen::Vector3d>;
    using Vector3dVectorTuple = std::tuple<Vector3dVector, Vector3dVector>;

public:
    explicit LO(const LOConfig& config)
        : config_(config),
          local_map_(config.voxel_size, config.max_range, config.max_points_per_voxel),
          global_map_(config.voxel_size, config.max_range, config.max_points_per_voxel),
          last_frame(config.voxel_size, config.max_range, config.max_points_per_voxel),
          adaptive_frame(config.initial_threshold, config.min_motion_th, config.max_range),
          adaptive_local(config.initial_threshold, config.min_motion_th, config.max_range) {
        local_map_.enforce_spatial_coverage_ = true;
    }

    LO() : LO(LOConfig{}) {}

public:
    Vector3dVectorTuple RegisterFrame(const std::vector<Eigen::Vector3d>& frame,
                                     bool return_clouds = true);
    Vector3dVectorTuple Voxelize(const std::vector<Eigen::Vector3d>& frame) const;

    double GetAdaptive_frame();
    double GetAdaptive_local();

    const RegistrationMetrics& LastRegistrationMetrics() const {
        return last_registration_metrics_;
    }

    Sophus::SE3d GetPredictionModel() const;
    bool         HasMoved();

    std::vector<Eigen::Vector3d> Dedistortion(const std::vector<Eigen::Vector3d>& frame);

    std::vector<Eigen::Vector3d> Preprocess(const std::vector<Eigen::Vector3d>& frame,
                                            double max_range,
                                            double min_range);

public:
    std::vector<Eigen::Vector3d> LocalMap() const { return local_map_.Pointcloud(); }
    std::vector<Eigen::Vector3d> GlobalMap() const { return global_map_.Pointcloud(); }
    std::vector<Sophus::SE3d>    poses() const { return poses_; }
    const Sophus::SE3d& CurrentPose() const { return poses_.back(); }
    size_t LocalVoxelCount() const { return local_map_.map_.size(); }
    size_t LocalBucketCount() const { return local_map_.map_.bucket_count(); }
    size_t GlobalVoxelCount() const { return global_map_.map_.size(); }
    size_t GlobalBucketCount() const { return global_map_.map_.bucket_count(); }

private:
    std::vector<Sophus::SE3d> poses_;
    LOConfig                  config_;
    VoxelHashMap              local_map_;
    VoxelHashMap              global_map_;
    VoxelHashMap              last_frame;
    AdaptiveThreshold         adaptive_frame;
    AdaptiveThreshold         adaptive_local;
    RegistrationMetrics       last_registration_metrics_;
    bool                      has_moved = false;
    Vector3dVector            previous_surface_cloud_;
    Sophus::SE3d              previous_surface_pose_;
    // Unlike v7, this single-acquisition reference is retained until its
    // geometric footprint changes enough; it need not be the previous scan.
    size_t                    surface_anchor_frame_ = 0;
    double                    surface_anchor_lever_ = 0.;
};

}  // namespace lo::pipeline
