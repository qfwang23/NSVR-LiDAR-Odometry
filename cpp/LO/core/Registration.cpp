#include "Registration.hpp"
#include "CenteredUpdate.hpp"

#include <tbb/parallel_for.h>
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <vector>

namespace {
using Matrix6 = Eigen::Matrix<double, 6, 6>;
using Vector6 = Eigen::Matrix<double, 6, 1>;

struct LinearSystem {
    Matrix6 h = Matrix6::Zero();
    Vector6 g = Vector6::Zero();
    size_t count = 0;
    size_t vertical_count = 0;
    void Add(const LinearSystem& other) {
        h += other.h; g += other.g;
        count += other.count; vertical_count += other.vertical_count;
    }
};

void TransformPointsInPlace(const Sophus::SE3d& pose, std::vector<Eigen::Vector3d>& points) {
    for (auto& point : points) point = pose * point;
}

LinearSystem BuildLinearSystem(const std::vector<Eigen::Vector3d>& source,
                               const lo::VoxelHashMap& map,
                               const Eigen::Vector3d& origin,
                               double kernel, bool local_stage, bool vertical) {
    // Fixed chunks and serial reduction make the sum independent of thread scheduling.
    // Fused matching/linearization eliminates two correspondence vectors and their merges.
    constexpr size_t chunk_size = 256;
    std::vector<LinearSystem> chunks((source.size()+chunk_size-1)/chunk_size);
    tbb::parallel_for(size_t(0), chunks.size(), [&](size_t chunk) {
        auto& system = chunks[chunk];
        const size_t end = std::min(source.size(), (chunk+1)*chunk_size);
        for (size_t i=chunk*chunk_size; i<end; ++i) {
            const auto match = map.GetClosestNeighbor(source[i]);
            if (!match.found) continue;
            ++system.count;
            const Eigen::Vector3d r = source[i] - match.target;
            const auto j = lo::CenteredJacobian(source[i], origin);
            const double r2 = r.squaredNorm();
            double w = 1.;
            if (r2 >= kernel*kernel) {
                // Preserve the historical two stage-specific robust weights.
                w = local_stage ? kernel/(kernel+std::sqrt(r2)) :
                    kernel*kernel/((kernel+r2)*(kernel+r2));
            }
            const double c = vertical ? match.vertical_confidence : 0.;
            system.h.noalias() += (w*(1.-c)) * j.transpose() * j;
            system.g.noalias() += (w*(1.-c)) * j.transpose() * r;
            if (c>0.) {
                ++system.vertical_count;
                const Vector6 jn = j.transpose()*match.normal;
                system.h.noalias() += (3.*w*c) * jn * jn.transpose();
                system.g.noalias() += (3.*w*c*match.normal.dot(r)) * jn;
            }
        }
    });
    LinearSystem result;
    for (const auto& chunk : chunks) result.Add(chunk);
    return result;
}
}  // namespace

namespace lo {
Sophus::SE3d RegisterFrame(const std::vector<Eigen::Vector3d>& frame,
                          const VoxelHashMap& voxel_map,
                          Sophus::SE3d initial_guess, double kernel,
                          bool use_robust_loss, bool enable_vertical_constraint,
                          VerticalDiagnostics* diagnostics) {
    if (diagnostics) *diagnostics = {};
    if (voxel_map.Empty()) return initial_guess;
    std::vector<Eigen::Vector3d> source = frame;
    TransformPointsInPlace(initial_guess, source);
    Sophus::SE3d accumulated;
    for (int iteration=0; iteration<500; ++iteration) {
        if (diagnostics) ++diagnostics->iterations;
        const Eigen::Vector3d origin = (accumulated*initial_guess).translation();
        auto system = BuildLinearSystem(source, voxel_map, origin, kernel,
                                        use_robust_loss,
                                        enable_vertical_constraint && !use_robust_loss);
        if (diagnostics) {
            diagnostics->correspondences += system.count;
            diagnostics->geometry_matches += system.vertical_count;
        }
        if (!system.count) break;
        system.h.diagonal().array() += 1e-6;
        const Vector6 step = system.h.ldlt().solve(-system.g);
        if (!step.allFinite()) break;
        const auto update = CenteredIncrement(origin, step);
        TransformPointsInPlace(update, source);
        accumulated = update*accumulated;
        if (step.norm()<1e-5) break;
    }
    // Geometry supplies a measured vertical direction; no height clipping is applied.
    return accumulated*initial_guess;
}
}  // namespace lo
