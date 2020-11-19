#include <variable.h>
#include <factor.h>
#include <utility> 

Variable::Variable(const int& id) : id_(id) {}

const int Variable::id() const { return id_; }

void Variable::add_neighbor(Factor *f) { neighbors_.push_back(f); }
void Variable::set_prior(const Gaussian &prior) { prior_ = prior; }
void Variable::add_message(const std::pair<int,int>& from, const Gaussian &message) { inbox_[from] = message; }
const Gaussian &Variable::belief() const { return belief_; }

void Variable::update_belief() {
    Eigen::VectorXd eta = prior_.eta();
    Eigen::MatrixXd lam = prior_.lam();
    for (Factor *f : neighbors_) {
        const std::pair<int,int>& id = f->id();
        if (inbox_.count(id)) {
            const Gaussian& g = inbox_[id];
            eta += g.eta();
            lam += g.lam();
        }
    }
    belief_ = Gaussian(eta, lam);
}

void Variable::send_messages() {
    for (Factor *f : neighbors_) {
        Gaussian msg = belief_;
        const std::pair<int,int>& id = f->id();
        if (inbox_.count(id)) {
            const Gaussian& g = inbox_[id];
            msg.eta() -= g.eta();
            msg.lam() -= g.lam();
        }
        f->add_message(id_, msg);
    }
}
