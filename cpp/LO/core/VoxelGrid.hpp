#pragma once

#include <Eigen/Core>
#include <cmath>

namespace lo {

// Uniform half-open cells [k*v, (k+1)*v), also on the negative axes.
// Shared by downsampling, insertion and correspondence search.
inline Eigen::Vector3i VoxelIndex(const Eigen::Vector3d& point, double size) {
    return Eigen::Vector3i(static_cast<int>(std::floor(point.x() / size)),
                           static_cast<int>(std::floor(point.y() / size)),
                           static_cast<int>(std::floor(point.z() / size)));
}

}  // namespace lo
