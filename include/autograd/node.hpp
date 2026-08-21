#pragma once
#include <vector>
#include <functional>
#include <memory>

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