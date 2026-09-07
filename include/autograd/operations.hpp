#pragma once
#include <memory>
#include "node.hpp"

std::shared_ptr<Node> operator+(std::shared_ptr<Node> a, std::shared_ptr<Node> b);
std::shared_ptr<Node> operator+(std::shared_ptr<Node> a, float b);
std::shared_ptr<Node> operator+(float a, std::shared_ptr<Node> b);

    
std::shared_ptr<Node> operator*(std::shared_ptr<Node> a, std::shared_ptr<Node> b);
std::shared_ptr<Node> operator*(std::shared_ptr<Node> a, float b);
std::shared_ptr<Node> operator*(float a, std::shared_ptr<Node> b);


std::shared_ptr<Node> operator-(std::shared_ptr<Node> a, std::shared_ptr<Node> b);
std::shared_ptr<Node> operator-(std::shared_ptr<Node> a, float b);
std::shared_ptr<Node> operator-(float a, std::shared_ptr<Node> b);


std::shared_ptr<Node> operator/(std::shared_ptr<Node> a, std::shared_ptr<Node> b);
std::shared_ptr<Node> operator/(std::shared_ptr<Node> a, float b);
std::shared_ptr<Node> operator/(float a, std::shared_ptr<Node> b);


std::shared_ptr<Node> pow(std::shared_ptr<Node> a, std::shared_ptr<Node> b);
std::shared_ptr<Node> pow(std::shared_ptr<Node> a, float b);
std::shared_ptr<Node> pow(float a, std::shared_ptr<Node> b);


std::shared_ptr<Node> exp(std::shared_ptr<Node> a);


std::shared_ptr<Node> log(std::shared_ptr<Node> a);