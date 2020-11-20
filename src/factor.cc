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
    std::vector<Eigen::VectorXd> state;
    for (Variable *neighbor : neighbors_) {
        state.push_back(neighbor->belief().mu());
    }
    state_ = state;
}

void Factor::update_factor() {
    update_state();
    Eigen::MatrixXd J = jacobian(state_);
    Eigen::VectorXd h = predict_measurement(state_);
    Eigen::VectorXd x0 = flatten(state_);
    Eigen::VectorXd eta = J.transpose() * measurement_.lam() * (J * x0 + measurement_.mu() - h);
    Eigen::MatrixXd lam = J.transpose() * measurement_.lam() * J;
    factor_ = Gaussian(eta, lam);
}
int count = 0;

void Factor::send_messages() {
    Eigen::VectorXd eta_all = factor_.eta();
    Eigen::MatrixXd lam_all = factor_.lam();
    // count += 1;
    // my attempt ---------------------------------------------------------------------------
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
    // end attempt --------------------------------------------------------------------------

    // second attempt -----------------------------------------------------------------------
    // Eigen::VectorXd eta_all_copy = factor_.eta();
    // Eigen::MatrixXd lam_all_copy = factor_.lam();
    // int i = 0;

    // for (int ind = 0; ind < neighbors_.size(); ++ind) {
    //     Variable *v = neighbors_[ind];
    //     Gaussian msg = inbox_[v->id()];
    //     int j = i + msg.eta().size() - 1;

    //     if (ind == 0) {
    //         eta_all(Eigen::seq(i, j)) += msg.eta();
    //         lam_all(Eigen::seq(i, j), Eigen::seq(i, j)) += msg.lam();
    //     }
        
    //     if (ind == 1) {
    //         eta_all_copy(Eigen::seq(i, j)) += msg.eta();
    //         lam_all_copy(Eigen::seq(i, j), Eigen::seq(i, j)) += msg.lam();
    //         neighbors_[ind]->add_message(id_, Gaussian(eta_all, lam_all).marginalize(i, j));
    //         neighbors_[ind - 1]->add_message(id_, Gaussian(eta_all_copy, lam_all_copy).marginalize(i, j));
    //     }
    // }
    // end attempt --------------------------------------------------------------------------
    // std::cout << "SEND MESSAGES -------------------------------------------------------------" << count << std::endl;
    // int i = 0;
    // // std::cout << "How many neighbours?: " << neighbors_.size() << std::endl; 
    // // std::cout << neighbors_.at(2) << std::endl;
    // for (Variable *v : neighbors_) {
    //     // std::cout << "START ----------------------------------" << std::endl;
    //     Gaussian msg = inbox_[v->id()];
    //     int j = i + msg.eta().size() - 1;
    //     // std::cout << "i: " << i << ", j: " << j << std::endl;
    //     // std::cout << "Eta.size(): " << msg.eta().size() << std::endl;
    //     // if (once == false) {
    //         // std::cout << "Eta before \n" << eta_all << std::endl;
    //         // std::cout << "Lam before \n" << lam_all << std::endl;
    //     // }
    //     eta_all(Eigen::seq(i, j)) += msg.eta();
    //     lam_all(Eigen::seq(i, j), Eigen::seq(i, j)) += msg.lam();
    //     // if (once == false) {
    //         // std::cout << "Eta adding this \n" << msg.eta() << std::endl;
    //         // std::cout << "Eta after \n" << eta_all << std::endl;
    //         // std::cout << "Lam adding this \n" << msg.lam() << std::endl;
    //         // std::cout << "Lam after \n" << lam_all << std::endl;
    //     // }
    //     // once = true;
    //     i = j + 1;
    //     // std::cout << "END ------------------------------------" << std::endl;
    // }
    // i = 0;
    // for (Variable *v : neighbors_) {
    //     Gaussian msg = inbox_[v->id()];
    //     int j = i + msg.eta().size() - 1;
    //     Eigen::VectorXd eta = eta_all;
    //     Eigen::MatrixXd lam = lam_all;
    //     // std::cout << "Eta before \n" << eta << std::endl;
    //     eta(Eigen::seq(i, j)) -= msg.eta();
    //     // std::cout << "Eta after \n" << eta << std::endl;
    //     lam(Eigen::seq(i, j), Eigen::seq(i, j)) -= msg.lam();
    //     v->add_message(id_, Gaussian(eta, lam).marginalize(i, j));
    //     i = j + 1;
    // }

    // std::cout << "Iteration done" << std::endl;
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