#include "include/autograd/engine.hpp"
#include <iostream>
#include <vector>

int main() {
  auto x = std::make_shared<Node>(2.0);       // input
  auto target = std::make_shared<Node>(10.0);  // target output

  auto w = std::make_shared<Node>(-3.0);  // initial value
  std::vector<std::shared_ptr<Node>> parameters = {w};  // variables the optimizer can modify (weight)
  float learning_rate = 0.05f;  // learning rate

  std::cout << "Starting training... initial weight W = " << w->value << std::endl;

  // Training loop
  for (int epoch = 0; epoch < 1000; epoch++) {
    
    // FORWARD PASS
    auto pred = x * w;
    
    auto error = pred - target;
    auto loss = error * error;

    // BACKWARD PASS
    loss->backward_opt();

    // OPTIMIZER STEP
    gradient_descent(parameters, learning_rate);

    if ((epoch) % 10 == 0) {
      std::cout << "Epoch " << epoch 
                << " | Loss: " << loss->value 
                << " | Prediction: " << pred->value 
                << " | Weight value: " << w->value << std::endl;
    }
  }

  std::cout << "\nTraining finished. The ideal value of W was 5.0 " << std::endl;
  std::cout << "The network learned that W should be: " << w->value << std::endl;

  return 0;
}