#pragma once
#include <memory>
#include <vector>
#include "../autograd/node.hpp"

class Module{
public: 
    virtual ~Module() = default;

    virtual std::vector<std::shared_ptr<Node>> parameters() const = 0;
};