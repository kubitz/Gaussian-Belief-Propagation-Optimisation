#pragma once
#include <vector>
#include <memory>
#include <factor.h>
#include <variable.h>
class FactorGraph {
private:
    std::vector<std::unique_ptr<Factor>> factors_;
    std::vector<std::unique_ptr<Variable>> variables_;
public:

    FactorGraph() {}
    Variable *add_variable(const std::string &variable_id) {
        variables_.emplace_back(std::unique_ptr<Variable>(new Variable(variable_id)));
        return variables_.back().get();
    }
    Factor *add_factor(const std::string &factor_id) {
        factors_.emplace_back(std::unique_ptr<Factor>(new Factor(factor_id)));
        return factors_.back().get();
    }
    void connect(Factor *factor, std::initializer_list<size_t> variables) {
        for (size_t v : variables) {
            factor->add_neighbor(variables_[v].get());
        }
    }
    void iteration() {
        for (auto &variable : variables_) {
            variable->update_belief();
        }
        for (auto &factor: factors_) {
            factor->update_factor();
        }
        for (auto &variable : variables_) {
            variable->send_messages();
        }
        for (auto &factor: factors_) {
            factor->send_messages();
        }
    }
    double ARE() const {
        double residual_error = 0.f;
        for (auto &factor: factors_) {
            residual_error += factor->residual();
        }
        return residual_error;

    }

    const Variable *v(size_t i) const { return variables_[i].get(); }
    const Factor *f(size_t i) const { return factors_[i].get(); }
};
