#include "../include/autograd/engine.hpp" 
#include "../include/optimizer/optimizer.hpp"   
#include "../include/nn/mlp.hpp"
#include <iostream>
#include <vector>

int main() {
    MLP model(2, {8, 8, 1});
    auto parameters = model.parameters();

    std::vector<std::shared_ptr<Node>> x = {
        std::make_shared<Node>(2.0f),
        std::make_shared<Node>(-1.0f)
    };
    
    auto target = std::make_shared<Node>(1.0f); 

    std::cout << "Starting training...\n" << std::endl;

    for(int epoch = 0; epoch < 30; epoch++) {
        // Forward pass
        auto pred = model(x);

        // Cost function (MSE)
        auto diff = pred[0] - target;
        auto loss = diff * diff; 

        std::cout << "Epoch " << epoch << " | Prediction: " << pred[0]->value << " | Loss: " << loss->value << std::endl;

        // Backward pass
        loss->backward();

        // Optimization
        gradient_descent(parameters, 0.05f); 
    }

    return 0;
}