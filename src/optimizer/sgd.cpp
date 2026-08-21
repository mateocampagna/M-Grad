#pragma once

#include <memory>
#include <vector>
#include "../../include/autograd/node.hpp"


void gradient_descent(std::vector<std::shared_ptr<Node>>& parameters, float learning_rate){
  for(auto& p:parameters){
    p->value = p->value - (p->gradient * learning_rate);
    p->gradient = 0.0f;
  }
}
