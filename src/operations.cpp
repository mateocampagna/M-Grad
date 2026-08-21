
#include <memory>
#include "../include/autograd/node.hpp"


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
