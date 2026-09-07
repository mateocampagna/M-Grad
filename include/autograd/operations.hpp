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


std::shared_ptr<Node> operatorpow(std::shared_ptr<Node> a, std::shared_ptr<Node> b);
std::shared_ptr<Node> operatorpow(std::shared_ptr<Node> a, float b);
std::shared_ptr<Node> operatorpow(float a, std::shared_ptr<Node> b);


std::shared_ptr<Node> operatorexp(std::shared_ptr<Node> a);


std::shared_ptr<Node> operatorlog(std::shared_ptr<Node> a);