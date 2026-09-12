#pragma once

#include <Eigen/Core>
#include <sophus/se3.hpp>

namespace lo {

// World-axis translation at the sensor center, and rotation about that center.
// Unlike a world-origin SE(3) twist, step.head<3>() is the EXACT sensor displacement.
inline Sophus::SE3d CenteredIncrement(
    const Eigen::Vector3d& origin, const Eigen::Matrix<double, 6, 1>& step) {
    const auto rotation = Sophus::SO3d::exp(step.tail<3>());
    return Sophus::SE3d(rotation, origin + step.head<3>() - rotation * origin);
}

inline Eigen::Matrix<double, 3, 6> CenteredJacobian(
    const Eigen::Vector3d& point, const Eigen::Vector3d& origin) {
    Eigen::Matrix<double, 3, 6> jacobian;
    jacobian.leftCols<3>().setIdentity();
    jacobian.rightCols<3>() = -Sophus::SO3d::hat(point - origin);
    return jacobian;
}

}  // namespace lo
