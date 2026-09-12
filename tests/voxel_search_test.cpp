#include "LO/core/VoxelHashMap.hpp"
#include "LO/core/VoxelGrid.hpp"
#include <tbb/global_control.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <random>

void require(bool b, const char* message) { if (!b) throw std::runtime_error(message); }

bool lex_less(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    for (int axis=0; axis<3; ++axis)
        if (a[axis]!=b[axis]) return a[axis]<b[axis];
    return false;
}

lo::VoxelHashMap::Match exhaustive(const lo::VoxelHashMap& map, const Eigen::Vector3d& p) {
    lo::VoxelHashMap::Match match;
    const auto center=lo::VoxelIndex(p,map.voxel_size_);
    double best=1.;
    for(int x=-1;x<=1;++x) for(int y=-1;y<=1;++y) for(int z=-1;z<=1;++z) {
        const auto it=map.map_.find(center+Eigen::Vector3i(x,y,z));
        if(it==map.map_.end()) continue;
        for(const auto& q:it->second.points) {
            const double d=(p-q).squaredNorm();
            if(d<best || (match.found && d==best && lex_less(q,match.target))) {
                match.found=true; match.target=q; best=d;
            }
        }
    }
    return match;
}

int main() {
    try {
        tbb::global_control serial(tbb::global_control::max_allowed_parallelism, 1);
        lo::VoxelHashMap map(1., 5., 15);
        const std::vector<Eigen::Vector3d> queries{{0,0,0}, {10,0,0}};
        require(std::get<0>(map.GetCorrespondences(queries)).empty(), "empty map returned correspondence");
        map.AddPoints({{0.1,0.1,0.1},{0.2,0.1,0.1}});
        const auto corr = map.GetCorrespondences(queries);
        require(std::get<0>(corr).size()==1, "missing-neighbor query was accepted");
        require((std::get<1>(corr)[0]-Eigen::Vector3d(.1,.1,.1)).norm()==0, "wrong nearest neighbor");
        map.Clear();
        std::vector<Eigen::Vector3d> cloud;
        for(int x=-12;x<=12;++x) for(int y=-12;y<=12;++y) cloud.emplace_back(x+.15,y+.23,.1);
        map.AddPoints(cloud);
        const auto before = map.map_.size();
        std::size_t expected=0;
        for(const auto& bucket:map.map_) if(bucket.second.points.front().squaredNorm()<=25) ++expected;
        map.RemovePointsFarFromLocation(Eigen::Vector3d::Zero());
        require(map.map_.size()==expected && expected<before, "pruning skipped entries");
        map.RemovePointsFarFromLocation(Eigen::Vector3d(100,100,100));
        require(map.Empty(), "pruning did not remove all far buckets");
        require(lo::VoxelIndex(Eigen::Vector3d(-.01,0.,.99),1.)==Eigen::Vector3i(-1,0,0),
                "negative coordinates do not use uniform floor cells");
        map.AddPoints({{-1.,0.,0.},{1.,0.,0.}});
        require(!map.GetClosestNeighbor(Eigen::Vector3d::Zero()).found,"strict 1-m gate changed");
        map.Clear();
        map.AddPoints({{.25,0.,0.},{-.25,0.,0.}});
        require(map.GetClosestNeighbor(Eigen::Vector3d::Zero()).target.x()==-.25,
                "center-first traversal changed deterministic tie rule");
        map.Clear();
        map.enforce_spatial_coverage_=true;
        for(int i=0;i<100;++i) map.AddPoints({{.1+1e-5*i,.1,.1}});
        map.AddPoints({{.7,.1,.1},{.1,.7,.1},{.7,.7,.1}});
        require(map.Pointcloud().size()==4,"duplicate hits filled the bounded coverage bucket");
        for(const auto& entry:map.map_) {
            const auto& pts=entry.second.points;
            for(size_t i=0;i<pts.size();++i) for(size_t j=i+1;j<pts.size();++j)
                require((pts[i]-pts[j]).squaredNorm()>=1./15.,"within-bucket spacing violated");
        }
        map.Clear();
        map.enforce_spatial_coverage_=false;
        for(int i=0;i<100;++i) map.AddPoints({{.1,.1,.1}});
        require(map.Pointcloud().size()==15,"disabled coverage changed last-frame bucket behavior");

        std::mt19937 rng(91);
        std::uniform_real_distribution<double> xyz(-3.,3.);
        for(const double v:{1.,.4}) for(const double shift:{0.,1e5}) {
            lo::VoxelHashMap random_map(v,100.,15);
            const Eigen::Vector3d origin(shift,-shift,.3*shift);
            std::vector<Eigen::Vector3d> samples;
            for(int i=0;i<6000;++i)
                samples.push_back(origin+Eigen::Vector3d(xyz(rng),xyz(rng),xyz(rng)));
            random_map.AddPoints(samples);
            for(int i=0;i<6000;++i) {
                Eigen::Vector3d query=origin+Eigen::Vector3d(xyz(rng),xyz(rng),xyz(rng));
                if(i%3==0) {
                    query=lo::VoxelIndex(query,v).cast<double>()*v;
                    if(i%2==0) query.x()=std::nextafter(query.x(),-std::numeric_limits<double>::infinity());
                }
                const auto actual=random_map.GetClosestNeighbor(query);
                const auto expected_match=exhaustive(random_map,query);
                require(actual.found==expected_match.found,"AABB pruning changed acceptance");
                if(actual.found) require(actual.target==expected_match.target,"AABB pruning changed exact neighbor");
            }
        }
        std::cout << "Voxel search and pruning tests passed\n";
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
