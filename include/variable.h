#pragma once
#include <unordered_map>
#include <gaussian.h>
class Factor;

class Variable {
private:
    Gaussian prior_;
    Gaussian belief_;
    std::unordered_map<Factor*, Gaussian> inbox_;
    std::vector<Factor *> neighbors_;

public:
    Variable();
    const Gaussian &belief() const;
    void add_neighbor(Factor *f);
    void add_message(Factor* from, const Gaussian &message);
    void set_prior(const Gaussian &prior);
    void update_belief();
    void send_messages();
};
