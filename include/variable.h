#pragma once
#include <map>
#include <gaussian.h>
class Factor;

class Variable {
private:
    std::string id_;
    Gaussian prior_;
    Gaussian belief_;
    std::map<std::string, Gaussian> inbox_;
    std::vector<Factor *> neighbors_;

public:
    Variable(const std::string &id);
    const std::string id() const;
    const Gaussian &belief() const;
    void add_neighbor(Factor *f);
    void add_message(const std::string &from, const Gaussian &message);
    void set_prior(const Gaussian &prior);
    void update_belief();
    void send_messages();
};
