#pragma once

#include <memory>
#include "../include/autograd/node.hpp"


// relu 
std::shared_ptr<Node> relu(std::shared_ptr<Node> node){
  // x > 0 -> 1,  x <= 0 -> 0
  
  // forward
  float value = node->value > 0 ? node->value : 0.0f;
  std::vector<std::shared_ptr<Node>> father = {node};

  // backward
  // op = 'R' relu
  std::shared_ptr<Node> out = std::make_shared<Node>(value, father, 'R');
  Node* ans = out.get();
  out->_backward = [node, ans](){
    float local_derivative = (node->value > 0) ? 1.0f: 0.0f;
    node->gradient += ans->gradient * local_derivative;
  };

  return out;
}