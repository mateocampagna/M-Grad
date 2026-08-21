#pragma once

#include <vector>
#include <set>
#include <memory>
#include <functional>

struct Node{
  float value;
  float gradient;
  char op;
  std::vector<std::shared_ptr<Node>> fathers;
  std::function<void()> _backward; // update fathers gradients

  // leaf node
  Node(float v);

  // internal node
  Node(float v, std::vector<std::shared_ptr<Node>> f, char p);
    
  void backward();
  void backward_opt();
};

// operators
std::shared_ptr<Node> operator+(std::shared_ptr<Node> a, std::shared_ptr<Node> b);
std::shared_ptr<Node> operator*(std::shared_ptr<Node> a, std::shared_ptr<Node> b);
std::shared_ptr<Node> operator-(std::shared_ptr<Node> a, std::shared_ptr<Node> b);


// activation funcitons
std::shared_ptr<Node> relu(std::shared_ptr<Node> node);

// optimizers
void gradient_descent(std::vector<std::shared_ptr<Node>>& parameters, float learning_rate);