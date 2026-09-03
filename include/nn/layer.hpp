#pragma once
#include "module.hpp"
#include "neuron.hpp"
#include <vector>
#include <memory>

class Layer : public Module {
private:
  std::vector<Neuron> neurons; 

public:
  Layer(int nin, int nout, bool nonlin = true);

  // Forward Pass
  std::vector<std::shared_ptr<Node>> operator()(const std::vector<std::shared_ptr<Node>>& x);

  // recolect params (composite)
  std::vector<std::shared_ptr<Node>> parameters() const override;
};