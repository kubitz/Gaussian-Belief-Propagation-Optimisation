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
        variables_.emplace_back(variable_id);
        return variables_.back().get();
    }
    Factor *add_factor(const std::string &factor_id) {
        factors_.emplace_back(factor_id);
        return factors_.back().get();
    }
    void connect(size_t f, std::initializer_list<size_t> variables) {
        for (size_t v : variables) {
            factors_[f]->add_neighbor(variables_[v].get());
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
};
