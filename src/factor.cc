#include <factor.h>
#include <variable.h>
// #include <iostream>

Factor::Factor(const std::pair<int,int>& id) : id_(id) {}

const std::pair<int,int>& Factor::id() const { return id_; }

void Factor::add_message(const int& from, const Gaussian &message) { inbox_[from] = message; }

void Factor::add_neighbor(Variable *v) {
    neighbors_.push_back(v);
    v->add_neighbor(this);
}

void Factor::set_measurement(const Gaussian &measurement) {
    measurement_ = measurement;
}

void Factor::update_state() {
    state_ = {neighbors_[0]->belief().mu(), neighbors_[1]->belief().mu()};
}

void Factor::update_factor() {
    update_state();
    Eigen::MatrixXd J = jacobian(state_);
    Eigen::VectorXd h = predict_measurement(state_);
    Eigen::VectorXd x0 = flatten(state_);
    Eigen::VectorXd eta = J.transpose() * ( measurement_.lam() * (J * x0 - h) +   measurement_.eta() );
    Eigen::MatrixXd lam = J.transpose() * measurement_.lam() * J;
    factor_ = Gaussian(eta, lam);
}

void Factor::send_messages() {
    Eigen::VectorXd eta_all = factor_.eta();
    Eigen::MatrixXd lam_all = factor_.lam();

    Eigen::VectorXd eta_all_copy = factor_.eta();
    Eigen::MatrixXd lam_all_copy = factor_.lam();

    const Gaussian& msg = inbox_[neighbors_[0]->id()];

    eta_all(Eigen::seq(0, 1)) += msg.eta();
    lam_all(Eigen::seq(0, 1), Eigen::seq(0, 1)) += msg.lam();

    neighbors_[1]->add_message(id_, Gaussian(eta_all, lam_all).marginalize(2, 3));
    
    const Gaussian& msg2 = inbox_[neighbors_[1]->id()];

    eta_all_copy(Eigen::seq(2, 3)) += msg2.eta();
    lam_all_copy(Eigen::seq(2, 3), Eigen::seq(2, 3)) += msg2.lam();

    neighbors_[0]->add_message(id_, Gaussian(eta_all_copy, lam_all_copy).marginalize(0, 1));
}

double Factor::residual() const {
    return (predict_measurement(state_) - measurement_.mu()).norm();
}

Eigen::MatrixXd Factor::jacobian(const std::vector<Eigen::VectorXd> &state) const {
    /* For linear factor, state argugment is unused */
    Eigen::MatrixXd J(2, 4);
    J << -1, 0, 1, 0,
         0, -1, 0, 1;
    return J;
}

Eigen::VectorXd Factor::predict_measurement(const std::vector<Eigen::VectorXd> &state) const {
    /* For this factor we assume that is only connects two variable nodes */
    assert(state.size() == 2);
    Eigen::VectorXd x1 = state[0];
    Eigen::VectorXd x2 = state[1];
    return x2 - x1;
}


Eigen::VectorXd Factor::flatten(const std::vector<Eigen::VectorXd> &xs) {
    size_t size = 0;
    for (const Eigen::VectorXd &x : xs) { size += x.size(); }
    Eigen::VectorXd X(size);
    int i = 0;
    for (const Eigen::VectorXd &x : xs) {
        int j = i + x.size() - 1;
        X(Eigen::seq(i, j)) = x;
        i = j + 1;
    }
    return X;
}