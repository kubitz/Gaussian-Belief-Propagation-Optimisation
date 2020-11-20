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
    Variable *add_variable(const int &variable_id) {
        variables_.emplace_back(std::unique_ptr<Variable>(new Variable(variable_id)));
        return variables_.back().get();
    }
    Factor *add_factor(const std::pair<int,int> &factor_id) {
        factors_.emplace_back(std::unique_ptr<Factor>(new Factor(factor_id)));
        return factors_.back().get();
    }
    void connect(Factor *factor, std::initializer_list<size_t> variables) {
        for (size_t v : variables) {
            factor->add_neighbor(variables_[v].get());
        }
    }
    void iteration() {
    #pragma omp parallel for
        for (auto &variable : variables_) {
            variable->update_belief();
        #pragma omp critical
        {  
            variable->send_messages();
        }
        }
     #pragma omp parallel for
        for (auto &factor: factors_) {
            factor->update_factor();
        #pragma omp critical
        {  
            factor->send_messages();
        }        }
        //for (auto &variable : variables_) {
        //}
        //for (auto &factor: factors_) {
        //}
    }
    double ARE() const {
        double residual_error = 0.f;
    #pragma omp parallel for reduction(+:residual_error)
        for (auto &factor: factors_) {
            residual_error += factor->residual();
        }
        return residual_error;

    }

    const Variable *v(size_t i) const { return variables_[i].get(); }
    const Factor *f(size_t i) const { return factors_[i].get(); }
};
