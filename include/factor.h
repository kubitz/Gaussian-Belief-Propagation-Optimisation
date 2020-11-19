#pragma once
#include <unordered_map>
#include <gaussian.h>
class Variable;

class Factor {
private:
    std::string id_;
    Gaussian factor_;
    Gaussian measurement_;
    std::vector<Eigen::VectorXd> state_;
    std::unordered_map<std::string, Gaussian> inbox_;
    std::vector<Variable *> neighbors_;
    Eigen::VectorXd flatten(const std::vector<Eigen::VectorXd> &xs);

public:
    Factor(const std::string &id);
    const std::string id() const;
    void add_message(const std::string &from, const Gaussian &message);
    void add_neighbor(Variable *v);
    void set_measurement(const Gaussian &measurement);
    void send_messages();
    void update_state();
    void update_factor();
    double residual() const;
    Eigen::MatrixXd jacobian(const std::vector<Eigen::VectorXd> &state) const;
    Eigen::VectorXd predict_measurement(const std::vector<Eigen::VectorXd> &state) const;
};
