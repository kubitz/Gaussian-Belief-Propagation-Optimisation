#pragma once
#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>

class Gaussian {
    /* N-dimensional Gaussian Distribution
     * Uses canonical form rather than standard form
     * i.e. normally Gaussian distribution is represented as
     * P(x) ~ N(mu, sig), where mu is the mean, and sig is the covariance.
     * In information form, the Gaussian distribution is parameterised using
     * eta = sig ^ -1 * mu
     * lam = sig ^ -1
     * */
private:
    Eigen::VectorXd eta_;
    Eigen::MatrixXd lam_;
public:
    Gaussian() {}
    Gaussian(const Eigen::VectorXd &eta, const Eigen::MatrixXd &lam) :
        eta_(eta), lam_(lam) {}

    /* Getters/Setters */
    const Eigen::VectorXd &eta() const { return eta_; }
    const Eigen::MatrixXd &lam() const { return lam_; }
    Eigen::VectorXd &eta() { return eta_; }
    Eigen::MatrixXd &lam() { return lam_; }

    /* Helpers to convert between canonical form and standard form */
    Eigen::VectorXd mu() const { return lam_.inverse() * eta_; }
    Eigen::MatrixXd sig() const { return lam_.inverse(); }

    /* Marginalization */
    Gaussian marginalize(uint32_t i, uint32_t j) const {
        uint32_t k = eta_.size();
        /* Indices excluding [i, j] */
        std::vector<int> N;
        for (int n = 0; n < k; ++n) { if (n < i || n > j) { N.push_back(n); } }
        Eigen::VectorXd eta_a = eta_(Eigen::seq(i, j));
        Eigen::VectorXd eta_b = eta_(N);

        Eigen::MatrixXd lam_aa = lam_(Eigen::seq(i, j), Eigen::seq(i, j));
        Eigen::MatrixXd lam_ab = lam_(Eigen::seq(i, j), N);
        Eigen::MatrixXd lam_ba = lam_(N, Eigen::seq(i, j));
        Eigen::MatrixXd lam_bb = lam_(N, N);

        Eigen::MatrixXd lam_bb_inv = lam_bb.inverse();

        Eigen::VectorXd eta = eta_a - lam_ab * lam_bb_inv * eta_b;
        Eigen::MatrixXd lam = lam_aa - lam_ab * lam_bb_inv * lam_ba;
        return Gaussian(eta, lam);
    }
};
