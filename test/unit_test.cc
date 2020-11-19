#define CATCH_CONFIG_MAIN
#include <catch2/catch2.hpp>
#include <gaussian.h>
#include <factor.h>
#include <variable.h>
#include <factor_graph.h>
#include <utils.h>

template <typename T, typename U>
bool isApprox(T x, U y) {
    return (x - y).norm() < 1e-8;
}

TEST_CASE("Conversion between canonical and standard form", "[Gaussian]" ) {
    for (int i = 3; i < 10; ++i) {
        Eigen::VectorXd eta = Eigen::VectorXd::Random(i);
        Eigen::MatrixXd sqrt_lam = Eigen::MatrixXd::Random(i, i);
        Eigen::MatrixXd lam = sqrt_lam * sqrt_lam.transpose() + Eigen::MatrixXd::Identity(i, i);
        Gaussian G = Gaussian(eta, lam);
        REQUIRE(isApprox(G.eta(), eta));
        REQUIRE(isApprox(G.lam(), lam));
        REQUIRE(isApprox(G.mu(), lam.inverse() * eta));
        REQUIRE(isApprox(G.sig(), lam.inverse()));
    }
}

TEST_CASE("Marginalization", "[Gaussian]" ) {
    Eigen::VectorXd eta = Eigen::VectorXd::Random(10);
    Eigen::MatrixXd sqrt_lam = Eigen::MatrixXd::Random(10, 10);
    Eigen::MatrixXd lam = sqrt_lam * sqrt_lam.transpose() + Eigen::MatrixXd::Identity(10, 10);
    Eigen::VectorXd mu = lam.inverse() * eta;
    Eigen::MatrixXd sig = lam.inverse();

    Gaussian G = Gaussian(eta, lam);

    Gaussian M_0_3 = G.marginalize(0, 3);
    Gaussian M_4_9 = G.marginalize(4, 9);

    REQUIRE(isApprox(M_0_3.mu(), mu(Eigen::seq(0, 3))));
    REQUIRE(isApprox(M_0_3.sig(), sig(Eigen::seq(0, 3), Eigen::seq(0, 3))));
    REQUIRE(isApprox(M_4_9.mu(), mu(Eigen::seq(4, 9))));
    REQUIRE(isApprox(M_4_9.sig(), sig(Eigen::seq(4, 9), Eigen::seq(4, 9))));
}


TEST_CASE("Simple Tree", "[GBP]") {
    Gaussian strong_prior = utils::make_prior(1e-8);
    Gaussian weak_prior = utils::make_prior(1e8);
    FactorGraph G;
    Variable *v1 = G.add_variable("v1");
    Variable *v2 = G.add_variable("v2");
    Variable *v3 = G.add_variable("v3");

    v1->set_prior(strong_prior);
    v2->set_prior(weak_prior);
    v3->set_prior(weak_prior);

    Factor *f1 = G.add_factor("f1");
    Factor *f2 = G.add_factor("f2");

    f1->add_neighbor(v1);
    f1->add_neighbor(v2);

    f2->add_neighbor(v2);
    f2->add_neighbor(v3);

    f1->set_measurement(utils::make_measurement(1, 0, 0.1, 0.1));
    f2->set_measurement(utils::make_measurement(1, 1, 0.1, 0.1));

    for (int i = 0; i < 10; ++i) {
        G.iteration();
    }

    REQUIRE(isApprox(v1->belief().mu(), (Eigen::VectorXd(2) << 0.f, 0.f).finished()));
    REQUIRE(isApprox(v2->belief().mu(), (Eigen::VectorXd(2) << 1.f, 0.f).finished()));
    REQUIRE(isApprox(v3->belief().mu(), (Eigen::VectorXd(2) << 2.f, 1.f).finished()));

}
