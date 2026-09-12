#include "VoxelHashMap.hpp"
#include "VerticalGeometry.hpp"
#include "VoxelGrid.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_for.h>

#include <Eigen/Core>
#include <algorithm>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

namespace {

struct ResultTuple {
    explicit ResultTuple(std::size_t n) {
        source.reserve(n);
        target.reserve(n);
    }
    std::vector<Eigen::Vector3d> source;
    std::vector<Eigen::Vector3d> target;
};

}  // namespace

namespace lo {

VoxelHashMap::Match VoxelHashMap::GetClosestNeighbor(const Eigen::Vector3d& point) const {
    Match match;
    if (!point.allFinite()) return match;
    const Voxel center = VoxelIndex(point, voxel_size_);
    double closest_distance2 = 1.0;  // Existing strict 1-m correspondence gate.
    const VoxelBlock* closest_block = nullptr;
    const auto lexicographic_less = [](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
        for (int axis = 0; axis < 3; ++axis) {
            if (a[axis] != b[axis]) return a[axis] < b[axis];
        }
        return false;
    };
    const auto visit = [&](const Voxel& key) {
        const auto bucket = map_.find(key);
        if (bucket == map_.end()) return;
        for (const auto& neighbor : bucket->second.points) {
            const double distance2 = (neighbor - point).squaredNorm();
            if (distance2 < closest_distance2 ||
                (closest_block && distance2 == closest_distance2 &&
                 lexicographic_less(neighbor, match.target))) {
                closest_distance2 = distance2;
                match.target = neighbor;
                closest_block = &bucket->second;
            }
        }
    };
    // Establish a tight upper bound before querying neighboring hash buckets.
    visit(center);
    double axis_distance2[3][3];
    for (int axis = 0; axis < 3; ++axis) {
        const double lower = static_cast<double>(center[axis]) * voxel_size_;
        const double to_lower = std::max(0., point[axis] - lower);
        const double to_upper = std::max(0., lower + voxel_size_ - point[axis]);
        axis_distance2[axis][0] = to_lower * to_lower;
        axis_distance2[axis][1] = 0.;
        axis_distance2[axis][2] = to_upper * to_upper;
    }
    // Conservative tolerance prevents pruning due solely to boundary roundoff.
    const double roundoff = 64. * std::numeric_limits<double>::epsilon() *
                            (1. + point.squaredNorm() + voxel_size_ * voxel_size_);
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                const double lower_bound2 = axis_distance2[0][dx + 1] +
                                            axis_distance2[1][dy + 1] +
                                            axis_distance2[2][dz + 1];
                if (lower_bound2 > closest_distance2 + roundoff) continue;
                visit(center + Voxel(dx, dy, dz));
            }
        }
    }
    // Exact within the same 27-cell domain, not an expanded radius search.
    if (closest_block) {
        match.found = true;
        match.normal = closest_block->vertical_normal;
        match.vertical_confidence = closest_block->vertical_confidence;
    }
    return match;
}

VoxelHashMap::Vector3dVectorTuple VoxelHashMap::GetCorrespondences(
    const Vector3dVector& points) const {

    using points_iterator = std::vector<Eigen::Vector3d>::const_iterator;

    auto result = tbb::parallel_reduce(
        tbb::blocked_range<points_iterator>{points.cbegin(), points.cend()},
        ResultTuple(points.size()),
        [this](const tbb::blocked_range<points_iterator>& r, ResultTuple res)
            -> ResultTuple {
            auto& [src, tgt] = res;
            src.reserve(r.size());
            tgt.reserve(r.size());
            for (const auto& point : r) {
                const auto match = GetClosestNeighbor(point);
                if (match.found) {
                    src.emplace_back(point);
                    tgt.emplace_back(match.target);
                }
            }
            return res;
        },
        [](ResultTuple a, const ResultTuple& b) -> ResultTuple {
            auto& [src, tgt]       = a;
            const auto& [srcp, tgtp] = b;
            src.insert(src.end(),
                       std::make_move_iterator(srcp.begin()),
                       std::make_move_iterator(srcp.end()));
            tgt.insert(tgt.end(),
                       std::make_move_iterator(tgtp.begin()),
                       std::make_move_iterator(tgtp.end()));
            return a;
        });

    return std::make_tuple(std::move(result.source), std::move(result.target));
}

std::vector<Eigen::Vector3d> VoxelHashMap::Pointcloud() const {
    std::vector<Eigen::Vector3d> points;
    points.reserve(max_points_per_voxel_ * map_.size());
    for (const auto& [voxel, voxel_block] : map_) {
        (void)voxel;
        for (const auto& point : voxel_block.points) {
            points.push_back(point);
        }
    }
    return points;
}

void VoxelHashMap::Update(const Vector3dVector& points, const Eigen::Vector3d& origin) {
    AddPoints(points);
    RemovePointsFarFromLocation(origin);
}

void VoxelHashMap::Update(const Vector3dVector& points) {
    AddPoints(points);
}

void VoxelHashMap::Update(const Vector3dVector& points,
                          const Sophus::SE3d& pose,
                          const bool is_global) {
    Vector3dVector points_transformed(points.size());
    std::transform(points.cbegin(), points.cend(), points_transformed.begin(),
                   [&](const auto& point) { return pose * point; });

    const Eigen::Vector3d& origin = pose.translation();
    if (is_global == true) {
        Update(points_transformed);
    } else {
        Update(points_transformed, origin);
    }
}

void VoxelHashMap::Update(const Vector3dVector& points,
                          const Sophus::SE3d& pose,
                          const bool is_global,
                          const bool is_last_frame) {
    Vector3dVector points_transformed(points.size());
    std::transform(points.cbegin(), points.cend(), points_transformed.begin(),
                   [&](const auto& point) { return pose * point; });

    if (is_global == false && is_last_frame == true) {
        map_.clear();
        Update(points_transformed);
    }
}

void VoxelHashMap::AddPoints(const std::vector<Eigen::Vector3d>& points) {
    std::vector<Voxel> dirty;
    // Surface samples occupy a 2-D manifold: derive spacing from bucket area
    // and capacity, rather than adding a dataset/sequence-specific threshold.
    const double min_separation2 = enforce_spatial_coverage_
        ? voxel_size_ * voxel_size_ / max_points_per_voxel_ : 0.;
    std::for_each(points.cbegin(), points.cend(), [&](const auto& point) {
        if (!point.allFinite()) return;
        const auto voxel = VoxelIndex(point, voxel_size_);
        auto search = map_.find(voxel);
        if (search != map_.end()) {
            auto& voxel_block = search.value();
            const bool inserted = voxel_block.AddPoint(point, min_separation2);
            if (inserted && enable_vertical_geometry_ && !voxel_block.geometry_dirty) {
                dirty.push_back(voxel);
                voxel_block.geometry_dirty = true;
            }
        } else {
            map_.insert({voxel, VoxelBlock{{point}, max_points_per_voxel_}});
            if (enable_vertical_geometry_) {
                map_.at(voxel).geometry_dirty = true;
                dirty.push_back(voxel);
            }
        }
    });
    // No map insertion/rehashing while workers update distinct dirty buckets.
    tbb::parallel_for(size_t(0), dirty.size(), [&](size_t i) {
        auto& block = map_.at(dirty[i]);
        const auto geometry = EstimateVerticalGeometry(block.points, voxel_size_);
        block.vertical_normal = geometry.normal;
        block.vertical_confidence = geometry.confidence;
        block.geometry_dirty = false;
    });
}

void VoxelHashMap::RemovePointsFarFromLocation(const Eigen::Vector3d& origin) {
    const auto max_d2 = max_distance_ * max_distance_;
    for (auto it = map_.begin(); it != map_.end();) {
        const auto& pt = it->second.points.front();
        if ((pt - origin).squaredNorm() > max_d2) {
            it = map_.erase(it);
        } else {
            ++it;
        }
    }
}

}  // namespace lo
