#include <variable.h>
#include <factor.h>
#include <utility> 

Variable::Variable() {}

//const int Variable::id() const { return id_; }

void Variable::add_neighbor(Factor *f) { neighbors_.push_back(f); }
void Variable::set_prior(const Gaussian &prior) { prior_ = prior; }
void Variable::add_message(Factor* from, const Gaussian &message) { inbox_[from] = message; }
const Gaussian &Variable::belief() const { return belief_; }

void Variable::update_belief1() {
    //Eigen::VectorXd eta = prior_.eta();
    //Eigen::MatrixXd lam = prior_.lam();
    //for (Factor *f : neighbors_) {
    //    const std::pair<int,int>& id = f->id();
    //}
    //belief_ = Gaussian(eta, lam);
    belief_ = prior_;
}

void Variable::send_messages1() {
    for (Factor *f : neighbors_) {
        Gaussian msg = belief_;
        //const std::pair<int,int>& id = f->id();
        f->add_message(this, msg);
    }
}

void Variable::update_belief() {
    Eigen::VectorXd eta = prior_.eta();
    Eigen::MatrixXd lam = prior_.lam();
    //if(id_==0){
    //std::cout<<neighbors_.size()<<std::endl;
    //}
    for (Factor *f : neighbors_) {
        const Gaussian& g = inbox_[f];
        eta += g.eta();
        lam += g.lam();
    }
    belief_ = Gaussian(eta, lam);
}


void Variable::send_messages() {
    for (Factor *f : neighbors_) {
        Gaussian msg = belief_;
        const Gaussian& g = inbox_[f];
        msg.eta() -= g.eta();
        msg.lam() -= g.lam();
        f->add_message(this, msg);
    }
}
