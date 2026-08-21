#pragma once
#include <vector>
#include <memory>
#include "../autograd/node.hpp"

void gradient_descent(std::vector<std::shared_ptr<Node>>& parameters, float learning_rate);