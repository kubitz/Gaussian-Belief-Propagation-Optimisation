#pragma once
#include <map>
#include <gaussian.h>
#include <utility>

class Factor;

class Variable {
private:
    //int id_;
    Gaussian prior_;
    Gaussian belief_;
    std::map<Factor*, Gaussian> inbox_;
    std::vector<Factor*> neighbors_;

public:
    Variable();
    //const int id() const;
    const Gaussian &belief() const;
    void add_neighbor(Factor *f);
    void add_message(Factor* from, const Gaussian &message);
    void set_prior(const Gaussian &prior);
    void update_belief();
    void send_messages();
    void update_belief1();
    void send_messages1();
};
