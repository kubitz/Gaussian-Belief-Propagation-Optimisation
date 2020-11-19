#include <factor.h>
#include <variable.h>

Factor::Factor(const std::string &id) : id_(id) {}

const std::string Factor::id() const { return id_; }

void Factor::add_message(const std::string &from, const Gaussian &message) { inbox_[from] = message; }

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

void Factor::send_messages() {
    Eigen::VectorXd eta_all = factor_.eta();
    Eigen::MatrixXd lam_all = factor_.lam();

    int i = 0;
    for (Variable *v : neighbors_) {
        Gaussian msg = inbox_[v->id()];
        int j = i + msg.eta().size() - 1;
        eta_all(Eigen::seq(i, j)) += msg.eta();
        lam_all(Eigen::seq(i, j), Eigen::seq(i, j)) += msg.lam();
        i = j + 1;
    }
    i = 0;
    for (Variable *v : neighbors_) {
        Gaussian msg = inbox_[v->id()];
        int j = i + msg.eta().size() - 1;
        Eigen::VectorXd eta = eta_all;
        Eigen::MatrixXd lam = lam_all;
        eta(Eigen::seq(i, j)) -= msg.eta();
        lam(Eigen::seq(i, j), Eigen::seq(i, j)) -= msg.lam();
        v->add_message(id_, Gaussian(eta, lam).marginalize(i, j));
        i = j + 1;
    }
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
