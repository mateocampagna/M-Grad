#include "../../include/nn/neuron.hpp"
#include "../../include/autograd/node.hpp"
#include "../../include/autograd/activations.hpp"
#include <random>
#include <stdexcept>

Neuron::Neuron(int nin) {
    for (int i = 0; i < nin; ++i) {
      float random_val = ((float)rand()/RAND_MAX) * 2.0f - 1.0f;
      w.push_back(std::make_shared<Node>(random_val));
    }
    b = std::make_shared<Node>(0.0f);
}

