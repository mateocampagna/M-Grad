#include "../../include/nn/layer.hpp"
#include "../../include/nn/mlp.hpp"


MLP::MLP(int nin, const std::vector<int>& layer_sizes){
    std::vector<int> input_sizes = {nin};
    input_sizes.push_back(nin);
    input_sizes.insert(input_sizes.end(), layer_sizes.begin(), layer_sizes.end());

    for(size_t i=0; i<input_sizes.size()-1; i++){
        bool last_layer = (i == input_sizes.size()-2);
        layers.emplace_back(input_sizes[i], input_sizes[i+1], !last_layer);
    }
}

std::vector<std::shared_ptr<Node>> MLP::operator()(const std::vector<std::shared_ptr<Node>>& x){
    std::vector<std::shared_ptr<Node>> output = x;
    for(auto& l:layers){
        output = l(output);
    }
    return output;
}

std::vector<std::shared_ptr<Node>> MLP::parameters() const{
    std::vector<std::shared_ptr<Node>> params;
    for(const auto& l:layers){
        auto l_params = l.parameters();
        params.insert(params.end(), l_params.begin(), l_params.end());
    }
    return params;
}

