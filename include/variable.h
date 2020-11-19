#pragma once
#include <map>
#include <gaussian.h>
#include <utility>

class Factor;

class Variable {
private:
    int id_;
    Gaussian prior_;
    Gaussian belief_;
    std::map<std::pair<int,int>, Gaussian> inbox_;
    std::vector<Factor*> neighbors_;

public:
    Variable(const int &id);
    const int id() const;
    const Gaussian &belief() const;
    void add_neighbor(Factor *f);
    void add_message(const std::pair<int,int>& from, const Gaussian &message);
    void set_prior(const Gaussian &prior);
    void update_belief();
    void send_messages();
};
