#pragma once
#include <memory>
#include "node.hpp"

std::shared_ptr<Node> operator+(std::shared_ptr<Node> a, std::shared_ptr<Node> b);
std::shared_ptr<Node> operator*(std::shared_ptr<Node> a, std::shared_ptr<Node> b);
std::shared_ptr<Node> operator-(std::shared_ptr<Node> a, std::shared_ptr<Node> b);

