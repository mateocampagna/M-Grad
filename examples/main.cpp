#include "../include/autograd/engine.hpp" 
#include "../include/optimizer/optimizer.hpp"   
#include "../include/nn/mlp.hpp"
#include <iostream>
#include <vector>

int main() {
    // Arquitectura: 2 entradas -> capa oculta de 4 -> oculta de 4 -> salida de 1
    MLP model(2, {8, 8, 1});
    auto parameters = model.parameters();

    // Input de prueba
    std::vector<std::shared_ptr<Node>> x = {
        std::make_shared<Node>(2.0f),
        std::make_shared<Node>(-1.0f)
    };
    
    // Valor que queremos que la red aprenda a predecir
    auto target = std::make_shared<Node>(1.0f); 

    std::cout << "Iniciando entrenamiento...\n" << std::endl;

    // Entrenamos durante 20 iteraciones
    for(int epoch = 0; epoch < 30; epoch++) {
        
        // 1. FORWARD PASS
        auto pred = model(x);

        // 2. FUNCION DE COSTO (MSE simplificado: error^2)
        // Esto agrega dinámicamente nuevos nodos a tu grafo computacional
        auto diff = pred[0] - target;
        auto loss = diff * diff; 

        std::cout << "Epoch " << epoch << " | Prediccion: " << pred[0]->value << " | Error (Loss): " << loss->value << std::endl;

        // 3. BACKWARD PASS
        // Propagamos los gradientes empezando desde el nodo final de error
        loss->backward_opt();

        // 4. OPTIMIZACION
        // Actualiza los pesos y resetea los gradientes a 0 gracias a tu lógica en sgd.cpp
        gradient_descent(parameters, 0.05f); 
    }

    return 0;
}