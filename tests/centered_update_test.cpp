#include "LO/core/CenteredUpdate.hpp"
#include <iostream>
#include <stdexcept>

int main() {
    try {
        const Eigen::Vector3d origin(1200., -80., 17.);
        const Eigen::Vector3d point = origin + Eigen::Vector3d(20., 7., -1.5);
        const auto jacobian = lo::CenteredJacobian(point, origin);
        for (int i=0; i<6; ++i) {
            Eigen::Matrix<double, 6, 1> d = Eigen::Matrix<double, 6, 1>::Zero();
            d(i) = 1e-5;
            const Eigen::Vector3d numeric =
                (lo::CenteredIncrement(origin, d)*point - lo::CenteredIncrement(origin, -d)*point)/2e-5;
            if ((numeric-jacobian.col(i)).norm() > 1e-7)
                throw std::runtime_error("centered retraction Jacobian mismatch");
        }
        Eigen::Matrix<double, 6, 1> d;
        d << .1, -.2, .04, .02, -.01, .03;
        const auto update = lo::CenteredIncrement(origin, d);
        if ((update*origin-origin-d.head<3>()).norm() > 1e-10)
            throw std::runtime_error("translation is not actual sensor displacement");
        const Eigen::Vector3d shift(10000., -2000., 300.);
        if ((lo::CenteredIncrement(origin+shift, d)*(point+shift)-shift-update*point).norm()>1e-9)
            throw std::runtime_error("world translation invariance failed");
        if ((lo::CenteredJacobian(point+shift,origin+shift)-jacobian).norm()>1e-9)
            throw std::runtime_error("Jacobian depends on world translation");
        std::cout << "Centered update invariance/Jacobian tests passed\n";
    } catch(const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
