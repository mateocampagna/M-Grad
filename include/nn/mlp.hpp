#pragma once
#include "module.hpp"
#include "layer.hpp"
#include <vector>
#include <memory>

class MLP : public Module{
private:
  std::vector<Layer> layers;
public:
    MLP(int nin, const std::vector<int>& layer_sizes);
    
    // Forward Pass
    std::vector<std::shared_ptr<Node>> operator()(const std::vector<std::shared_ptr<Node>>& x);
    
    // recolect params (composite)
    std::vector<std::shared_ptr<Node>> parameters() const override;
};