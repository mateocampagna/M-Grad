#include "../../include/nn/neuron.hpp"
#include "../../include/autograd/node.hpp"
#include "../../include/autograd/activations.hpp"
#include <random>
#include <stdexcept>

Neuron::Neuron(int nin, bool nonlin) : nonlin(nonlin){
    // random ini
    for (int i = 0; i < nin; ++i){
      float random_val = ((float)rand()/RAND_MAX) * 2.0f - 1.0f;
      weight.push_back(std::make_shared<Node>(random_val));
    }
    // bias = 0
    bias = std::make_shared<Node>(0.0f);
}

std::shared_ptr<Node> Neuron::operator()(const std::vector<std::shared_ptr<Node>> &x){
  if(x.size() != weight.size()) throw std::invalid_argument("la dimension de los pesos debe ser la misma que la del input");
  
  auto out = bias;
  for(size_t i=0; i<x.size(); i++){
    out = out + x[i] * weight[i];
  }

  return relu(out);
}


std::vector<std::shared_ptr<Node>> Neuron::parameters() const {
  std::vector<std::shared_ptr<Node>> params;
  for(const auto& w : weight){
    params.push_back(w);
  }
  params.push_back(bias);
  
  return params;
}


