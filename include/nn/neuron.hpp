#pragma once
#include "module.hpp"
#include "../autograd/node.hpp"
#include "../autograd/operations.hpp"
#include "../autograd/activations.hpp"
#include <vector>
#include <random>
#include <memory>

class Neuron:public Module{
private:
    std::vector<std::shared_ptr<Node>> weight;
    std::shared_ptr<Node> bias;
    bool nonlin; 
public:
    // const
    Neuron(int nin, bool nonlin = true);

    // forward pass
    std::shared_ptr<Node> operator()(const std::vector<std::shared_ptr<Node>>& x);

    // return parameters
    std::vector<std::shared_ptr<Node>> parameters() const override;
};