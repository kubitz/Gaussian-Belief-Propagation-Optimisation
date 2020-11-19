#pragma once
#include <gaussian.h>
namespace utils {
Gaussian make_prior(double sigma) {
    Eigen::VectorXd eta(2);
    eta << 0, 0;
    Eigen::MatrixXd lam(2, 2);
    lam << 1 / sigma, 0,
           0, 1 / sigma;
    return Gaussian(eta, lam);
}

Gaussian make_measurement(double x, double y, double sigma_x, double sigma_y) {
    Eigen::VectorXd mu(2);
    mu << x, y;
    Eigen::MatrixXd lam(2, 2);
    lam << 1 / sigma_x, 0,
           0, 1 / sigma_y;
    return Gaussian(lam * mu, lam);
}
};
