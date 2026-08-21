#pragma once

#include <set>
#include <unordered_set>
#include "../include/autograd/node.hpp"

Node::Node(float v):
  value(v),
  gradient(0.0f),
  op(' '),
  _backward([](){})
{}

Node::Node(float v, std::vector<std::shared_ptr<Node>> f, char p):
  value(v),
  gradient(0.0f),
  op(p),
  fathers(f),
  _backward([](){})
{}

void Node::backward(){
  std::vector<Node*> topo;
  std::set<Node*> visited;

  std::function<void(Node*)> build_topo = [&](Node* v){
    if(visited.find(v) == visited.end()){
      visited.insert(v);
      for (auto ch : v->fathers) build_topo(ch.get());
      topo.push_back(v);
    }
  };

  build_topo(this);
  this->gradient = 1.0f;
  for (auto it=topo.rbegin(); it!=topo.rend(); ++it)
      (*it)->_backward();
}

void Node::backward_opt(){
  std::vector<Node*> topo;
  std::unordered_set<Node*> visited;

  std::vector<std::pair<Node*, bool>> stack;
  stack.push_back({this, false});

  while (!stack.empty()) {
    auto [v, processed] = stack.back();
    stack.pop_back();

    if(processed){
      topo.push_back(v);
      continue;
    }

    if(visited.find(v) != visited.end()) continue;
    
    visited.insert(v);
    stack.push_back({v, true});

    for(auto it = v->fathers.rbegin(); it != v->fathers.rend(); ++it) stack.push_back({it->get(), false});
  }

  this->gradient = 1.0f;
  for (auto it=topo.rbegin(); it!=topo.rend(); ++it) (*it)->_backward();
}