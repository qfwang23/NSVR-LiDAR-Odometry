#pragma once

#include <algorithm>
#include <cmath>
#include <vector>
#include <sophus/se3.hpp>

namespace lo {

// A representative lever arm, not an externally specified sensor angle.
inline double SurfaceAnchorLever(const std::vector<Eigen::Vector3d>& cloud) {
    if (cloud.empty()) return 0.;
    std::vector<double> radii;
    radii.reserve(cloud.size());
    for (const auto& p:cloud) radii.push_back(p.norm());
    const auto middle=radii.begin()+radii.size()/2;
    std::nth_element(radii.begin(),middle,radii.end());
    return *middle;
}

inline double SurfaceAnchorFootprintMotion(const Sophus::SE3d& anchor,
                                           const Sophus::SE3d& current,double lever) {
    const double angle=(anchor.so3().inverse()*current.so3()).log().norm();
    return (current.translation()-anchor.translation()).norm()+
           2.*lever*std::sin(.5*angle);
}

// Refresh after movement of one surface-patch width, or loss of enough
// matched support for the existing two-fold solver. No clock/speed/GT gate.
inline bool ReplaceSurfaceAnchor(bool empty,double footprint_motion,
                                 size_t matched_patches,double voxel_size) {
    return empty || !std::isfinite(footprint_motion) || footprint_motion>=3.*voxel_size ||
           matched_patches<6;
}
}  // namespace lo
