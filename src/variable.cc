#include <variable.h>
#include <factor.h>

Variable::Variable() {}

void Variable::add_neighbor(Factor *f) { neighbors_.push_back(f); }
void Variable::set_prior(const Gaussian &prior) { prior_ = prior; }
void Variable::add_message(Factor* from, const Gaussian &message) { inbox_[from] = message; }
const Gaussian &Variable::belief() const { return belief_; }

void Variable::update_belief() {
    Eigen::VectorXd eta = prior_.eta();
    Eigen::MatrixXd lam = prior_.lam();
    for (Factor *f : neighbors_) {
        if (inbox_.count(f)) {
            eta += inbox_[f].eta();
            lam += inbox_[f].lam();
        }
    }
    belief_ = Gaussian(eta, lam);
}

void Variable::send_messages() {
    for (Factor *f : neighbors_) {
        Gaussian msg = belief_;
        if (inbox_.count(f)) {
            msg.eta() -= inbox_[f].eta();
            msg.lam() -= inbox_[f].lam();
        }
        f->add_message(this, msg);
    }
}
