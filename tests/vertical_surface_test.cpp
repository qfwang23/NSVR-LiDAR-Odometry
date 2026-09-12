#include "LO/core/VerticalSurface.hpp"
#include "LO/pipeline/LO.hpp"
#include "LO/core/TemporalSurfaceAnchor.hpp"
#include <tbb/global_control.h>
#include <iostream>
#include <random>
#include <stdexcept>

void Require(bool value,const char* message) { if(!value) throw std::runtime_error(message); }

std::vector<lo::VerticalSurfaceFactor> Factors(bool contradictory=false) {
    std::vector<lo::VerticalSurfaceFactor> factors;
    for(int i=-3;i<=3;++i) for(int j=-3;j<=3;++j) {
        lo::VerticalSurfaceFactor f;
        f.fold=static_cast<unsigned>(i+j)&1U;
        const double z=contradictory && f.fold ? -.08:.08;
        f.points={Eigen::Vector3d(i*2.,j*2.,z),Eigen::Vector3d(i*2.+.3,j*2.,z),
                  Eigen::Vector3d(i*2.,j*2.+.3,z)};
        f.lower_xy=Eigen::Vector2d(i*2.-.8,j*2.-.8);
        f.upper_xy=Eigen::Vector2d(i*2.+.8,j*2.+.8);
        f.information*=10000.;
        factors.push_back(f);
    }
    return factors;
}

int main() {
    try {
        Require(lo::ReplaceSurfaceAnchor(true,0.,100,1.),"empty reference not initialized");
        Require(!lo::ReplaceSurfaceAnchor(false,0.,100,1.),"stationary reference replaced");
        Require(!lo::ReplaceSurfaceAnchor(false,2.9,6,1.),"short baseline not retained");
        Require(lo::ReplaceSurfaceAnchor(false,3.,6,1.),"one-patch movement not refreshed");
        Require(lo::ReplaceSurfaceAnchor(false,0.,5,1.),"lost support not refreshed");
        const Sophus::SE3d turn(Sophus::SO3d::exp(Eigen::Vector3d(0.,0.,.5)),Eigen::Vector3d::Zero());
        Require(lo::SurfaceAnchorFootprintMotion(Sophus::SE3d(),turn,10.)>3.,"rotation did not refresh footprint");
        const Sophus::SE3d shift(Sophus::SO3d(),Eigen::Vector3d(0.,0.,.35));
        Require(std::abs(lo::SurfaceAnchorFootprintMotion(Sophus::SE3d(),shift,10.)-.35)<1e-12,
                "real height change miscomputed as a footprint motion");
        auto factors=Factors();
        lo::VerticalSurfaceCorrection state;
        state.rotation=Sophus::SO3d::exp(Eigen::Vector3d(.01,-.02,.03)); state.z=.12;
        const auto jacobian=lo::SurfaceJacobian(factors[0],state,10.);
        for(int j=0;j<3;++j) {
            auto a=state,b=state;
            const double epsilon=1e-6;
            if(j==0) { a.z+=epsilon; b.z-=epsilon; }
            else {
                Eigen::Vector3d axis=Eigen::Vector3d::Zero(); axis[j-1]=epsilon/10.;
                a.rotation=Sophus::SO3d::exp(axis)*state.rotation;
                b.rotation=Sophus::SO3d::exp(-axis)*state.rotation;
            }
            const Eigen::Vector3d numeric=(lo::SurfaceResidual(factors[0],a)-lo::SurfaceResidual(factors[0],b))/(2.*epsilon);
            Require((numeric-jacobian.col(j)).norm()<1e-8,"surface Jacobian failed finite differences");
        }
        lo::VerticalSurfaceDiagnostics d;
        const auto correction=lo::CrossValidateVerticalSurfaces(factors,&d);
        Require(d.applied && std::abs(correction.z+.08)<1e-7,"consistent height evidence not recovered");
        Require(d.after<d.before,"accepted correction did not improve both folds");
        auto biased=factors;
        for (auto& f:biased) for (int j=0;j<3;++j) {
            const auto& p=f.points[j];
            const double basis=p.head<2>().norm()-(p.head<2>()-Eigen::Vector2d(2.,0.)).norm();
            f.range_basis[j]=basis;
            f.points[j].z()=.08+.01*basis;
        }
        lo::VerticalSurfaceDiagnostics profiled_d;
        const auto profiled=lo::CrossValidateVerticalSurfaces(biased,&profiled_d);
        Require(profiled_d.applied && std::abs(profiled.z+.08)<1e-6,
                "range nuisance was mistaken for height");
        Require(std::abs(profiled.range_bias-.01)<1e-6,"range nuisance was not recovered");
        for (auto& f:biased) for (int j=0;j<3;++j) f.points[j].z()=.01*f.range_basis[j];
        lo::VerticalSurfaceDiagnostics null_d;
        lo::CrossValidateVerticalSurfaces(biased,&null_d);
        Require(!null_d.applied,"nuisance-only improvement was credited to the pose");
        for (auto& f:biased) f.range_basis.setOnes();
        lo::VerticalSurfaceDiagnostics confounded;
        lo::CrossValidateVerticalSurfaces(biased,&confounded);
        Require(!confounded.applied && confounded.rank_rejected>0,
                "height-nuisance ambiguity was not rejected");
        lo::VerticalSurfaceDiagnostics rejected;
        lo::CrossValidateVerticalSurfaces(Factors(true),&rejected);
        Require(!rejected.applied && rejected.validation_rejected>0,"contradictory held-out planes accepted");
        auto line=factors;
        for(auto& f:line) for(auto& p:f.points) p.x()=0.;
        lo::VerticalSurfaceDiagnostics rank;
        lo::CrossValidateVerticalSurfaces(line,&rank);
        Require(!rank.applied && rank.rank_rejected>0,"unobservable inclination was solved");

        // A sloping static surface viewed after a REAL 35-cm height change.
        // Correct an additional small pose error; do not force height to zero.
        std::vector<Eigen::Vector3d> reference,current;
        const Sophus::SE3d truth(Sophus::SO3d::exp(Eigen::Vector3d(.025,-.015,.01)),
                                Eigen::Vector3d(.5,.2,.35));
        std::mt19937 rng(461);
        std::normal_distribution<double> noise(0.,.002);
        for(int i=-48;i<=48;++i) for(int j=-48;j<=48;++j) {
            const Eigen::Vector3d world(i*.25,j*.25,-1.5+.12*i*.25-.06*j*.25);
            reference.push_back(world+Eigen::Vector3d(0.,0.,noise(rng)));
            current.push_back(truth.inverse()*(world+Eigen::Vector3d(0.,0.,noise(rng))));
        }
        const Sophus::SE3d base(Sophus::SO3d::exp(Eigen::Vector3d(.002,-.003,0.))*truth.so3(),
                               truth.translation()+Eigen::Vector3d(0.,0.,.04));
        lo::VerticalSurfaceDiagnostics measured;
        const auto refined=lo::RefineVerticalSurfaces(current,reference,Sophus::SE3d(),base,1.,&measured);
        std::cout<<"synthetic patches="<<measured.matched_patches<<" applied="<<measured.applied
                 <<" z="<<refined.translation().z()<<" rank_rejected="<<measured.rank_rejected
                 <<" validation_rejected="<<measured.validation_rejected<<'\n';
        Require(measured.applied,"valid overlapping sloped patches were not used");
        Require(std::abs(refined.translation().z()-.35)<.003,"genuine elevation was clipped or correction biased");
        Require((refined.so3().inverse()*truth.so3()).log().norm()<.001,"tilt correction failed");
        Require(refined.translation().head<2>()==base.translation().head<2>(),"horizontal translation changed");
        Sophus::SE3d serial_pose, parallel_pose;
        {
            tbb::global_control workers(tbb::global_control::max_allowed_parallelism,1);
            serial_pose=lo::RefineVerticalSurfaces(current,reference,Sophus::SE3d(),base,1.);
        }
        {
            tbb::global_control workers(tbb::global_control::max_allowed_parallelism,20);
            parallel_pose=lo::RefineVerticalSurfaces(current,reference,Sophus::SE3d(),base,1.);
        }
        Require(serial_pose.matrix()==parallel_pose.matrix(),"parallel plane fitting changed pose");
        const auto empty=lo::RefineVerticalSurfaces({},reference,Sophus::SE3d(),base,1.);
        Require(empty.matrix()==base.matrix(),"unsupported refinement did not retain base pose exactly");
        const Eigen::Vector3d offset(1e5,-2e4,3e3);
        const Sophus::SE3d shifted_previous(Sophus::SO3d(),offset);
        const Sophus::SE3d shifted_base(base.so3(),base.translation()+offset);
        const auto shifted=lo::RefineVerticalSurfaces(current,reference,shifted_previous,shifted_base,1.);
        Require((shifted.translation()-offset-refined.translation()).norm()<1e-7,"result depends on world origin");
        Require((shifted.so3().inverse()*refined.so3()).log().norm()<1e-8,"tilt depends on world origin");

        lo::pipeline::LOConfig config;
        config.enable_vertical_constraint=false;
        lo::pipeline::LO pipeline(config);
        pipeline.RegisterFrame(reference,false);
        pipeline.RegisterFrame(current,false);
        Require(!pipeline.LastRegistrationMetrics().vertical_surface.applied &&
                pipeline.LastRegistrationMetrics().vertical_surface.matched_patches==0 &&
                pipeline.LastRegistrationMetrics().vertical_surface_ms==0.,"off control executed surface module");
        Require(pipeline.LastRegistrationMetrics().surface_anchor_age_frames==0 &&
                !pipeline.LastRegistrationMetrics().surface_anchor_replaced,"off control maintained an anchor");
        config.enable_vertical_constraint=true;
        lo::pipeline::LO anchored(config);
        anchored.RegisterFrame(reference,false);
        Require(anchored.LastRegistrationMetrics().surface_anchor_replaced,"first cloud did not initialize anchor");
        anchored.RegisterFrame(reference,false);
        Require(anchored.LastRegistrationMetrics().surface_anchor_age_frames==1 &&
                !anchored.LastRegistrationMetrics().surface_anchor_replaced,"static anchor replaced on second frame");
        anchored.RegisterFrame(reference,false);
        Require(anchored.LastRegistrationMetrics().surface_anchor_age_frames==2 &&
                !anchored.LastRegistrationMetrics().surface_anchor_replaced,"reference was not retained across frames");
        std::cout<<"Cross-validated vertical surface tests passed\n";
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
