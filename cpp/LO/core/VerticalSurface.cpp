#include "VerticalSurface.hpp"
#include "VoxelGrid.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <tbb/parallel_for.h>

namespace lo {
namespace {
using Key = std::array<int, 3>;
using Points = std::vector<Eigen::Vector3d>;
constexpr double eps = std::numeric_limits<double>::epsilon();

struct PlaneFit {
    Eigen::Vector3d theta = Eigen::Vector3d::Zero();
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    double variance = 0.;
    bool valid = false;
};
struct Patch {
    Eigen::Vector3d center;
    Eigen::Vector2d lower, upper;
    PlaneFit fit;
};

double Median(std::vector<double> values) {
    const size_t middle = values.size()/2;
    std::nth_element(values.begin(), values.begin()+middle, values.end());
    return values[middle];
}

Eigen::Vector3d Design(const Eigen::Vector2d& xy, const Eigen::Vector3d& center, double size) {
    return Eigen::Vector3d(1., (xy.x()-center.x())/size, (xy.y()-center.y())/size);
}

PlaneFit Fit(const Points& points, const Eigen::Vector3d& center, double size) {
    PlaneFit fit;
    if (points.size()<6) return fit;
    std::vector<double> weights(points.size(), 1.);
    Eigen::Matrix3d h;
    Eigen::Vector3d g;
    for (int iteration=0; iteration<6; ++iteration) {
        h.setZero(); g.setZero();
        for (size_t i=0; i<points.size(); ++i) {
            const auto x = Design(points[i].head<2>(), center, size);
            h.noalias() += weights[i]*x*x.transpose();
            g.noalias() += weights[i]*x*(points[i].z()-center.z());
        }
        const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> spectrum(h);
        if (spectrum.info()!=Eigen::Success || spectrum.eigenvalues().minCoeff() <
            1e-6*spectrum.eigenvalues().maxCoeff()) return PlaneFit{};
        fit.theta = h.ldlt().solve(g);
        if (!fit.theta.allFinite()) return PlaneFit{};
        std::vector<double> residuals, deviations;
        residuals.reserve(points.size()); deviations.reserve(points.size());
        for (const auto& p:points)
            residuals.push_back(p.z()-center.z()-Design(p.head<2>(),center,size).dot(fit.theta));
        const double median = Median(residuals);
        for (double r:residuals) deviations.push_back(std::abs(r-median));
        const double scale = std::max(1.4826022185*Median(deviations), std::sqrt(128.*eps)*size);
        double square_error = 0., weight_sum = 0.;
        for (size_t i=0; i<points.size(); ++i) {
            const double magnitude = std::abs(residuals[i]);
            const double next_weight = magnitude<=1.345*scale ? 1. : 1.345*scale/magnitude;
            square_error += weights[i]*residuals[i]*residuals[i];
            weight_sum += weights[i];
            weights[i] = next_weight;
        }
        fit.variance = std::max(square_error/std::max(1.,weight_sum-3.),128.*eps*size*size);
    }
    // Conditional regression covariance, not a calibrated ICP pose covariance.
    fit.covariance = fit.variance*h.ldlt().solve(Eigen::Matrix3d::Identity());
    const double slope2 = fit.theta.tail<2>().squaredNorm()/(size*size);
    fit.valid = slope2<=1. && fit.variance<=.01*size*size && fit.covariance.allFinite();
    return fit;
}

std::map<Key, Patch> Extract(const Points& input, const Sophus::SE3d& pose,
                             const Eigen::Vector3d& origin, double size,
                             VerticalSurfaceDiagnostics* diagnostics) {
    std::map<Key, Points> groups;
    for (const auto& p:input) {
        const Eigen::Vector3d q=pose.so3()*p+(pose.translation()-origin);
        if (!q.allFinite()) continue;
        const auto k=VoxelIndex(q,size);
        groups[{k.x(),k.y(),k.z()}].push_back(q);
    }
    std::vector<std::pair<Key,const Points*>> work;
    for (const auto& entry:groups) {
        if (entry.second.size()>=12) work.emplace_back(entry.first,&entry.second);
    }
    std::vector<Patch> patches(work.size());
    std::vector<size_t> rejected(work.size(),0);
    // Independent fits run in parallel; input point order and final key order
    // are unchanged. No floating-point reduction is shared between workers.
    tbb::parallel_for(size_t(0),work.size(),[&](size_t index) {
        const auto& key=work[index].first;
        const auto& points=*work[index].second;
        auto& patch=patches[index];
        patch.center=size*(Eigen::Vector3d(key[0],key[1],key[2])+Eigen::Vector3d::Constant(.5));
        Points halves[2];
        patch.lower=Eigen::Vector2d::Constant(std::numeric_limits<double>::infinity());
        patch.upper=-patch.lower;
        for (size_t i=0; i<points.size(); ++i) {
            halves[i%2].push_back(points[i]);
            patch.lower=patch.lower.cwiseMin(points[i].head<2>());
            patch.upper=patch.upper.cwiseMax(points[i].head<2>());
        }
        const PlaneFit a=Fit(halves[0],patch.center,size), b=Fit(halves[1],patch.center,size);
        if (!a.valid || !b.valid) { rejected[index]=1; return; }
        // Independent subsets must predict compatible local plane parameters.
        // 11.345 is the 99% chi-square(3) reference, used as a model check;
        // correlated LiDAR samples prevent interpreting it as a calibrated probability.
        const Eigen::Vector3d difference=a.theta-b.theta;
        const Eigen::Matrix3d uncertainty=a.covariance+b.covariance;
        const double disagreement=difference.dot(uncertainty.ldlt().solve(difference));
        if (!std::isfinite(disagreement) || disagreement>11.345) {
            rejected[index]=1; return;
        }
        patch.fit=Fit(points,patch.center,size);
    });
    std::map<Key, Patch> output;
    for (size_t i=0;i<work.size();++i) {
        diagnostics->fit_rejected+=rejected[i];
        if (patches[i].fit.valid) output.emplace(work[i].first,std::move(patches[i]));
    }
    return output;
}

std::vector<VerticalSurfaceFactor> MatchPatches(const std::map<Key,Patch>& source,
                                                const std::map<Key,Patch>& reference,
                                                double size,
                                                const Sophus::SE3d& previous_pose,
                                                const Sophus::SE3d& base_pose) {
    std::vector<VerticalSurfaceFactor> factors;
    for (const auto& entry:source) {
        const auto& key=entry.first;
        const auto& src=entry.second;
        double best=1.;
        VerticalSurfaceFactor selected;
        bool found=false;
        // Same XY footprint; inspect adjacent height cells for boundary crossing.
        for (int dz=-1; dz<=1; ++dz) {
            const auto it=reference.find({key[0],key[1],key[2]+dz});
            if (it==reference.end()) continue;
            const auto& ref=it->second;
            VerticalSurfaceFactor factor;
            factor.lower_xy=src.lower.cwiseMax(ref.lower);
            factor.upper_xy=src.upper.cwiseMin(ref.upper);
            const Eigen::Vector2d extent=factor.upper_xy-factor.lower_xy;
            if (extent.minCoeff()<size/6.) continue;
            const Eigen::Vector2d center=.5*(factor.lower_xy+factor.upper_xy);
            factor.normal=Eigen::Vector3d(-ref.fit.theta[1]/size,-ref.fit.theta[2]/size,1.).normalized();
            const double reference_height=ref.center.z()+Design(center,ref.center,size).dot(ref.fit.theta);
            factor.offset=-factor.normal.dot(Eigen::Vector3d(center.x(),center.y(),reference_height));
            const Eigen::Vector3d source_normal(-src.fit.theta[1]/size,-src.fit.theta[2]/size,1.);
            if (source_normal.normalized().dot(factor.normal)<std::sqrt(.5)) continue;
            Eigen::Matrix3d xs,xt;
            for (int j=0; j<3; ++j) {
                Eigen::Vector2d xy=center;
                if (j) xy[j-1]+=.25*extent[j-1];
                xs.row(j)=Design(xy,src.center,size).transpose();
                xt.row(j)=Design(xy,ref.center,size).transpose();
                factor.points[j]=Eigen::Vector3d(xy.x(),xy.y(),src.center.z()+xs.row(j).dot(src.fit.theta));
                const Eigen::Vector3d source_local=base_pose.so3().inverse()*factor.points[j];
                const Eigen::Vector3d reference_point(xy.x(),xy.y(),
                    ref.center.z()+xt.row(j).dot(ref.fit.theta));
                const Eigen::Vector3d reference_local=previous_pose.so3().inverse()*
                    (reference_point+(base_pose.translation()-previous_pose.translation()));
                factor.range_basis[j]=
                    factor.normal.dot(base_pose.so3()*Eigen::Vector3d::UnitZ())*source_local.head<2>().norm()-
                    factor.normal.dot(previous_pose.so3()*Eigen::Vector3d::UnitZ())*reference_local.head<2>().norm();
            }
            // Preserve cross-correlation among the three virtual evaluations;
            // retain roughness variance so dense patches do not become noiseless.
            Eigen::Matrix3d covariance=xs*src.fit.covariance*xs.transpose()+xt*ref.fit.covariance*xt.transpose();
            covariance.diagonal().array()+=src.fit.variance+ref.fit.variance;
            covariance*=factor.normal.z()*factor.normal.z();
            factor.information=covariance.ldlt().solve(Eigen::Matrix3d::Identity());
            const double distance=SurfaceResidual(factor,{}).cwiseAbs().maxCoeff();
            if (distance<best && factor.information.allFinite()) {
                best=distance; selected=factor; found=true;
            }
        }
        if (found) {
            // Spatial patch folds, separate from the within-patch point split.
            selected.fold=static_cast<unsigned>(key[0]+key[1])&1U;
            factors.push_back(selected);
        }
    }
    return factors;
}

double Cost(const std::vector<VerticalSurfaceFactor>& factors,
            const VerticalSurfaceCorrection& correction, int fold) {
    double cost=0.;
    for (const auto& factor:factors) {
        if (fold>=0 && factor.fold!=fold) continue;
        const auto residual=SurfaceResidual(factor,correction);
        cost+=3.*std::log1p(std::max(0.,residual.dot(factor.information*residual))/3.);
    }
    return cost;
}

bool AssociationValid(const std::vector<VerticalSurfaceFactor>& factors,
                      const VerticalSurfaceCorrection& correction) {
    for (const auto& factor:factors) for (const auto& p:factor.points) {
        const Eigen::Vector3d q=correction.rotation*p+Eigen::Vector3d(0.,0.,correction.z);
        if ((q-p).norm()>=1. || (q.head<2>().array()<factor.lower_xy.array()).any() ||
            (q.head<2>().array()>factor.upper_xy.array()).any()) return false;
    }
    return true;
}

bool Solve(const std::vector<VerticalSurfaceFactor>& factors, int fold, double lever_scale,
           VerticalSurfaceCorrection& correction) {
    size_t count=0;
    for (const auto& f:factors) if (fold<0 || f.fold==fold) ++count;
    if (count<3) return false;
    for (int iteration=0; iteration<8; ++iteration) {
        Eigen::Matrix3d h=Eigen::Matrix3d::Zero();
        Eigen::Vector3d g=Eigen::Vector3d::Zero();
        Eigen::Vector3d cross=Eigen::Vector3d::Zero();
        double nuisance_h=0., nuisance_g=0.;
        for (const auto& factor:factors) {
            if (fold>=0 && factor.fold!=fold) continue;
            const auto residual=SurfaceResidual(factor,correction);
            const auto j=SurfaceJacobian(factor,correction,lever_scale);
            const double weight=1./(1.+std::max(0.,residual.dot(factor.information*residual))/3.);
            h.noalias()+=weight*j.transpose()*factor.information*j;
            g.noalias()+=weight*j.transpose()*factor.information*residual;
            const Eigen::Vector3d nuisance_j=-factor.range_basis;
            cross.noalias()+=weight*j.transpose()*factor.information*nuisance_j;
            nuisance_h+=weight*nuisance_j.dot(factor.information*nuisance_j);
            nuisance_g+=weight*nuisance_j.dot(factor.information*residual);
        }
        // Profile out the nuisance with a Schur complement. An angular/range
        // residual pattern cannot be spent again as height/tilt information.
        const bool has_nuisance=nuisance_h>128.*eps*std::max(1.,h.trace());
        if (has_nuisance) {
            h.noalias()-=cross*cross.transpose()/nuisance_h;
            g.noalias()-=cross*(nuisance_g/nuisance_h);
        }
        const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(h);
        if (eig.info()!=Eigen::Success || eig.eigenvalues().minCoeff()<=1e-6*eig.eigenvalues().maxCoeff())
            return false;
        const Eigen::Vector3d step=h.ldlt().solve(-g);
        if (!step.allFinite()) return false;
        const double nuisance_step=has_nuisance ? -(nuisance_g+cross.dot(step))/nuisance_h : 0.;
        if (!std::isfinite(nuisance_step)) return false;
        if (step.norm()<1e-8 && std::abs(nuisance_step)<1e-8) return true;
        const double old_cost=Cost(factors,correction,fold);
        bool accepted=false;
        for (double scale=1.; scale>=1./32.; scale*=.5) {
            VerticalSurfaceCorrection proposal=correction;
            proposal.z+=scale*step[0];
            proposal.range_bias+=scale*nuisance_step;
            proposal.rotation=Sophus::SO3d::exp(Eigen::Vector3d(scale*step[1]/lever_scale,
                                                               scale*step[2]/lever_scale,0.))*correction.rotation;
            if (Cost(factors,proposal,fold)<old_cost) {
                correction=proposal; accepted=true; break;
            }
        }
        if (!accepted) return true;
    }
    return true;
}

// A fair null model also fits its nuisance. Merely explaining a range pattern
// is not evidence that the pose needs correction.
VerticalSurfaceCorrection FitNuisanceOnly(const std::vector<VerticalSurfaceFactor>& factors,
                                         int fold) {
    VerticalSurfaceCorrection correction;
    for (int iteration=0;iteration<8;++iteration) {
        double h=0.,g=0.;
        for (const auto& f:factors) {
            if (fold>=0 && f.fold!=fold) continue;
            const auto r=SurfaceResidual(f,correction);
            const double weight=1./(1.+std::max(0.,r.dot(f.information*r))/3.);
            h+=weight*f.range_basis.dot(f.information*f.range_basis);
            g-=weight*f.range_basis.dot(f.information*r);
        }
        if (h<=128.*eps) break;
        const double step=-g/h;
        if (!std::isfinite(step) || std::abs(step)<1e-8) break;
        const double old=Cost(factors,correction,fold);
        bool accepted=false;
        for (double scale=1.;scale>=1./32.;scale*=.5) {
            auto proposed=correction;
            proposed.range_bias+=scale*step;
            if (Cost(factors,proposed,fold)<old) {correction=proposed;accepted=true;break;}
        }
        if (!accepted) break;
    }
    return correction;
}
}  // namespace

Eigen::Vector3d SurfaceResidual(const VerticalSurfaceFactor& factor,
                               const VerticalSurfaceCorrection& correction) {
    Eigen::Vector3d residual;
    for (int i=0;i<3;++i)
        residual[i]=factor.normal.dot(correction.rotation*factor.points[i])+factor.normal.z()*correction.z+
                    factor.offset-correction.range_bias*factor.range_basis[i];
    return residual;
}

Eigen::Matrix3d SurfaceJacobian(const VerticalSurfaceFactor& factor,
                               const VerticalSurfaceCorrection& correction, double lever_scale) {
    Eigen::Matrix3d jacobian;
    for (int i=0;i<3;++i) {
        const Eigen::Vector3d lever=correction.rotation*factor.points[i];
        const Eigen::Vector3d angular=lever.cross(factor.normal);
        jacobian.row(i)<<factor.normal.z(),angular.x()/lever_scale,angular.y()/lever_scale;
    }
    return jacobian;
}

VerticalSurfaceCorrection CrossValidateVerticalSurfaces(
    const std::vector<VerticalSurfaceFactor>& factors, VerticalSurfaceDiagnostics* diagnostics) {
    VerticalSurfaceDiagnostics fallback;
    if (!diagnostics) diagnostics=&fallback;
    diagnostics->applied=false;
    diagnostics->before=diagnostics->after=diagnostics->delta_z=diagnostics->rotation_rad=diagnostics->range_bias=0.;
    diagnostics->rank_rejected=diagnostics->validation_rejected=0;
    diagnostics->matched_patches=factors.size();
    if (factors.size()<6) return {};
    std::vector<double> levers;
    for (const auto& f:factors) levers.push_back(f.points[0].head<2>().norm());
    const double lever_scale=std::max(Median(levers),1.);
    const double before[2]={Cost(factors,FitNuisanceOnly(factors,0),0),
                            Cost(factors,FitNuisanceOnly(factors,1),1)};
    diagnostics->before=before[0]+before[1]; diagnostics->after=diagnostics->before;
    for (int fold=0;fold<2;++fold) {
        VerticalSurfaceCorrection correction;
        if (!Solve(factors,fold,lever_scale,correction)) { ++diagnostics->rank_rejected; return {}; }
        if (!AssociationValid(factors,correction) ||
            !(Cost(factors,correction,1-fold)<before[1-fold]-1e-8)) {
            ++diagnostics->validation_rejected; return {};
        }
    }
    VerticalSurfaceCorrection combined;
    if (!Solve(factors,-1,lever_scale,combined)) { ++diagnostics->rank_rejected; return {}; }
    if (!AssociationValid(factors,combined) || !(Cost(factors,combined,0)<before[0]-1e-8) ||
        !(Cost(factors,combined,1)<before[1]-1e-8)) {
        ++diagnostics->validation_rejected; return {};
    }
    diagnostics->applied=true;
    diagnostics->after=Cost(factors,combined,-1);
    diagnostics->delta_z=combined.z;
    diagnostics->rotation_rad=combined.rotation.log().norm();
    diagnostics->range_bias=combined.range_bias;
    return combined;
}

Sophus::SE3d RefineVerticalSurfaces(const Points& current, const Points& previous,
                                  const Sophus::SE3d& previous_pose, const Sophus::SE3d& base_pose,
                                  double voxel_size, VerticalSurfaceDiagnostics* diagnostics) {
    VerticalSurfaceDiagnostics fallback;
    if (!diagnostics) diagnostics=&fallback;
    *diagnostics={};
    if (previous.empty() || current.empty()) return base_pose;
    const double size=3.*voxel_size;
    const auto source=Extract(current,base_pose,base_pose.translation(),size,diagnostics);
    const auto reference=Extract(previous,previous_pose,base_pose.translation(),size,diagnostics);
    diagnostics->source_patches=source.size(); diagnostics->reference_patches=reference.size();
    const auto factors=MatchPatches(source,reference,size,previous_pose,base_pose);
    const auto correction=CrossValidateVerticalSurfaces(factors,diagnostics);
    if (!diagnostics->applied) return base_pose;
    Eigen::Vector3d translation=base_pose.translation();
    translation.z()+=correction.z;
    return Sophus::SE3d(correction.rotation*base_pose.so3(),translation);
}
}  // namespace lo
