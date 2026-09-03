#include "../../include/nn/layer.hpp"

Layer::Layer(int nin, int nout, bool nonlin){
    neurons.reserve(nout);
    size_t output = nout;
    for(size_t i=0; i<output; i++){
        neurons.emplace_back(nin, nonlin);
    }
}

// forward pass
std::vector<std::shared_ptr<Node>> Layer::operator()(const std::vector<std::shared_ptr<Node>>& x){
    std::vector<std::shared_ptr<Node>> output;
    output.reserve(neurons.size());
    
    for(auto& n : neurons){ output.push_back(n(x)); }

    return output;
}

// recolect params
std::vector<std::shared_ptr<Node>> Layer::parameters() const {
    std::vector<std::shared_ptr<Node>> params;
    for(const auto& n : neurons){
        auto n_params = n.parameters();
        params.insert(params.end(), n_params.begin(), n_params.end());
    }
    return params;
}