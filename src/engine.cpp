#include "../include/autograd/engine.hpp"
#include <unordered_set>

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

std::shared_ptr<Node> operator+(std::shared_ptr<Node> a, std::shared_ptr<Node> b){
  float value = a->value + b->value;
  std::vector<std::shared_ptr<Node>> fathers = {a, b};
  
  std::shared_ptr<Node> c = std::make_shared<Node>(value, fathers, '+');
  // auto c = std::make_shared<Node>(value, fathers, '+');

  Node* ans = c.get();
  c->_backward = [a, b, ans](){
    a->gradient += 1.0f * ans->gradient;
    b->gradient += 1.0f * ans->gradient;
  };

  return c;
}

std::shared_ptr<Node> operator*(std::shared_ptr<Node> a, std::shared_ptr<Node> b){
  float value = a->value * b->value;
  std::vector<std::shared_ptr<Node>> fathers = {a, b};
  
  std::shared_ptr<Node> c = std::make_shared<Node>(value, fathers, '*');
  // auto c = std::make_shared<Node>(value, fathers, '+');

  Node* ans = c.get();
  c->_backward = [a, b, ans](){
    a->gradient += b->value * ans->gradient;
    b->gradient += a->value * ans->gradient;
  };

  return c;
}

std::shared_ptr<Node> operator-(std::shared_ptr<Node> a, std::shared_ptr<Node> b){
  float value = a->value - b->value;
  std::vector<std::shared_ptr<Node>> fathers = {a, b};
  
  std::shared_ptr<Node> c = std::make_shared<Node>(value, fathers, '-');
  // auto c = std::make_shared<Node>(value, fathers, '+');

  Node* ans = c.get();
  c->_backward = [a, b, ans](){
    a->gradient += 1.0f * ans->gradient;
    b->gradient += -1.0f * ans->gradient;
  };

  return c;
}


// relu 
std::shared_ptr<Node> relu(std::shared_ptr<Node> node){
  // x > 0 -> 1,  x <= 0 -> 0
  
  // forward
  float value = node->value > 0 ? node->value : 0.0f;
  std::vector<std::shared_ptr<Node>> father = {node};

  // backward
  // op = 'R' relu
  std::shared_ptr<Node> out = std::make_shared<Node>(value, father, 'R');
  Node* ans = out.get();
  out->_backward = [node, ans](){
    float local_derivative = (node->value > 0) ? 1.0f: 0.0f;
    node->gradient += ans->gradient * local_derivative;
  };

  return out;
}

void gradient_descent(std::vector<std::shared_ptr<Node>>& parameters, float learning_rate){
  for(auto& p:parameters){
    p->value = p->value - (p->gradient * learning_rate);
    p->gradient = 0.0f;
  }
}
