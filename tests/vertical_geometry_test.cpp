#include "LO/core/VerticalGeometry.hpp"
#include "LO/core/Registration.hpp"
#include <tbb/global_control.h>
#include <iostream>
#include <stdexcept>

void require(bool b, const char* m) { if (!b) throw std::runtime_error(m); }
int main() {
    try {
        std::vector<Eigen::Vector3d> plane, wall, line;
        for (int x=0;x<4;++x) for(int y=0;y<4;++y) {
            plane.emplace_back(.1+.2*x,.1+.2*y,-1.2+.04*x);
            wall.emplace_back(.1,.1+.2*x,.1+.2*y);
            line.emplace_back(.1+.2*x,0.,0.);
        }
        const auto geometry=lo::EstimateVerticalGeometry(plane,1.);
        require(geometry.confidence>.8,"sloped support plane was not found");
        require(std::abs(geometry.normal.dot(Eigen::Vector3d(-.2,0.,1.).normalized()))>1.-1e-9,"wrong plane normal");
        require(lo::EstimateVerticalGeometry(wall,1.).confidence==0.,"wall treated as vertical support");
        require(lo::EstimateVerticalGeometry(line,1.).confidence==0.,"collinear points treated as plane");
        require(lo::EstimateVerticalGeometry({},1.).confidence==0.,"empty set has geometry");
        const auto metric=lo::VerticalMetric(geometry);
        require(std::abs(metric.trace()-3.)<1e-12,"information trace changed");
        require(Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(metric).eigenvalues().minCoeff()>-1e-12,"metric is not PSD");
        for(auto& p:plane) p+=Eigen::Vector3d(1e5,-2e4,3e3);
        require((lo::EstimateVerticalGeometry(plane,1.).normal-geometry.normal).norm()<1e-9,"geometry depends on origin");

        // Actual height changes greater than the old 8-cm clamp must remain estimable.
        lo::VoxelHashMap map(1.,100.,15);
        map.enable_vertical_geometry_=true;
        std::vector<Eigen::Vector3d> source, target;
        for(int bx=-4;bx<4;++bx) for(int by=-4;by<4;++by)
            for(int x=0;x<4;++x) for(int y=0;y<4;++y) {
                source.emplace_back(bx+.1+.2*x,by+.1+.2*y,-1.2);
                target.push_back(source.back()+Eigen::Vector3d(0.,0.,.15));
            }
        map.AddPoints(target);
        lo::VerticalDiagnostics d;
        Sophus::SE3d serial_pose;
        {
            tbb::global_control serial(tbb::global_control::max_allowed_parallelism,1);
            serial_pose=lo::RegisterFrame(source,map,Sophus::SE3d(),2.,false,true,&d);
        }
        require(d.geometry_matches>0,"geometry factor never applied");
        require(std::abs(serial_pose.translation().z()-.15)<1e-6,"real vertical displacement was clipped");
        const auto parallel_pose=lo::RegisterFrame(source,map,Sophus::SE3d(),2.,false,true,&d);
        require((parallel_pose.matrix()-serial_pose.matrix()).norm()==0.,"fixed reduction is not repeatable");
        const auto off=lo::RegisterFrame(source,map,Sophus::SE3d(),2.,false,false,&d);
        require(d.geometry_matches==0,"disabled control used geometry");
        require(std::abs(off.translation().z()-.15)<1e-6,"point-only control failed known translation");
        lo::RegisterFrame(source,map,Sophus::SE3d(),2.,true,true,&d);
        require(d.geometry_matches==0,"v4 geometry leaked into local-map stage");
        std::cout << "Vertical geometry, fallback, displacement and determinism tests passed\n";
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
