#include "LO/core/VerticalGain.hpp"
#include <Eigen/Cholesky>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

using Matrix = Eigen::Matrix<double, 6, 6>;
using Vector = Eigen::Matrix<double, 6, 1>;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    try {
        const Matrix h = Matrix::Identity();
        Vector full = Vector::Zero();
        full(0) = 1.; full(2) = .02;
        require(lo::SelectVerticalWithGainRetention(h, -full, full).action == 0, "below gate");
        full(2) = .04;
        const auto zero = lo::SelectVerticalWithGainRetention(h, -full, full);
        require(zero.action == 1 && zero.step(2) == 0., "weak vertical contribution should be zeroed");
        full(2) = .23;
        const auto clip = lo::SelectVerticalWithGainRetention(h, -full, full);
        require(clip.action == 2 && clip.step(2) == .05, "gain-admissible clipped candidate");
        full.setZero(); full(2) = 1.;
        require(lo::SelectVerticalWithGainRetention(h, -full, full).action == 0, "preserve indispensable height descent");
        std::mt19937 rng(37);
        std::normal_distribution<double> normal;
        for (int n = 0; n < 1000; ++n) {
            Matrix a;
            for (int i=0; i<36; ++i) a.data()[i] = normal(rng);
            const Matrix H = a.transpose()*a + .1*Matrix::Identity();
            Vector g;
            for (int i=0; i<6; ++i) g(i) = normal(rng);
            const Vector f = H.ldlt().solve(-g);
            const auto s = lo::SelectVerticalWithGainRetention(H,g,f);
            const double gf = -g.dot(f)-.5*f.dot(H*f);
            const double gs = -g.dot(s.step)-.5*s.step.dot(H*s.step);
            require(gs + 1e-10 >= .95*gf, "gain-retention bound violated");
            require(std::abs(s.step(2)) <= std::abs(f(2)), "vertical magnitude increased");
            if (s.action != 0) {
                const Vector residual_gradient = H * s.step + g;
                for (int i=0;i<6;++i) if(i!=2)
                    require(std::abs(residual_gradient(i)) < 1e-9, "conditional optimum not stationary");
                Vector deleted = f;
                deleted(2) = s.step(2);
                const double naive_gain = -g.dot(deleted)-.5*deleted.dot(H*deleted);
                require(gs + 1e-10 >= naive_gain, "conditional solve worse than component deletion");
            }
            const auto mirror = lo::SelectVerticalWithGainRetention(H,-g,-f);
            require((mirror.step+s.step).norm()<1e-10, "asymmetric sign handling");
        }
        std::cout << "Vertical gain-retention tests passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
