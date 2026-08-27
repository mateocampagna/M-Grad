#include "../include/autograd/engine.hpp" 
#include "../include/optimizer/optimizer.hpp"   
#include "../include/nn/neuron.hpp"
#include <iostream>
#include <vector>

int main() {
  std::cout << " ---- neuron test ---- " << std::endl;
  Neuron n(2);

  // Input
  std::vector<std::shared_ptr<Node>> x = {
    std::make_shared<Node>(2.0f),
    std::make_shared<Node>(3.0f)
  };

  // Weights before training
  auto parameters = n.parameters();
  std::cout << "Weights BEFORE training:" << std::endl;
  for (size_t i = 0; i < parameters.size(); ++i) {
    std::cout << "Parameter " << i << ": " << parameters[i]->value << std::endl;
  }

  // FORWARD PASS
  auto pred = n(x);
  std::cout << "\n Initial prediction (Forward): " << pred->value << std::endl;

  // BACKWARD PASS
  pred->backward_opt();

  std::cout << "\n Gradients calculated by the Engine:" << std::endl;
  for (size_t i = 0; i < parameters.size(); ++i) {
    std::cout << "Parameter gradient " << i << ": " << parameters[i]->gradient << std::endl;
  }

  // OPTIMIZATION (Gradient Descent)
  gradient_descent(parameters, 0.05f);

  std::cout << "\n Weights AFTER applying the Optimizer:" << std::endl;
  for (size_t i = 0; i < parameters.size(); ++i) {
    std::cout << "Parameter " << i << ": " << parameters[i]->value << std::endl;
  }

  return 0;
}