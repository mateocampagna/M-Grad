
#include <memory>
#include "../include/autograd/node.hpp"
#include <cmath>

//------------------------- OPERATOR+
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

std::shared_ptr<Node> operator+(std::shared_ptr<Node> a, float b){
  float value = a->value + b;
  std::vector<std::shared_ptr<Node>> fathers = {a};
  
  std::shared_ptr<Node> c = std::make_shared<Node>(value, fathers, '+');
  // auto c = std::make_shared<Node>(value, fathers, '+');

  Node* ans = c.get();
  c->_backward = [a, ans](){
    a->gradient += 1.0f * ans->gradient;
  };

  return c;
}

std::shared_ptr<Node> operator+(float a, std::shared_ptr<Node> b){
  float value = a + b->value;
  std::vector<std::shared_ptr<Node>> fathers = {b};
  
  std::shared_ptr<Node> c = std::make_shared<Node>(value, fathers, '+');
  // auto c = std::make_shared<Node>(value, fathers, '+');

  Node* ans = c.get();
  c->_backward = [b, ans](){
    b->gradient += 1.0f * ans->gradient;
  };

  return c;
}

// ------------------------- OPERATOR-
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

std::shared_ptr<Node> operator-(std::shared_ptr<Node> a, float b){
  float value = a->value - b;
  std::vector<std::shared_ptr<Node>> fathers = {a};
  
  std::shared_ptr<Node> c = std::make_shared<Node>(value, fathers, '-');
  // auto c = std::make_shared<Node>(value, fathers, '+');

  Node* ans = c.get();
  c->_backward = [a, ans](){
    a->gradient += 1.0f * ans->gradient;
  };

  return c;
}


std::shared_ptr<Node> operator-(float a, std::shared_ptr<Node> b){
  float value = a - b->value;
  std::vector<std::shared_ptr<Node>> fathers = {b};
  
  std::shared_ptr<Node> c = std::make_shared<Node>(value, fathers, '-');
  // auto c = std::make_shared<Node>(value, fathers, '+');

  Node* ans = c.get();
  c->_backward = [b, ans](){
    b->gradient += -1.0f * ans->gradient;
  };

  return c;
}



//------------------------- OPERATOR*
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

std::shared_ptr<Node> operator*(std::shared_ptr<Node> a, float b){
  float value = a->value * b;
  std::vector<std::shared_ptr<Node>> fathers = {a};
  
  std::shared_ptr<Node> c = std::make_shared<Node>(value, fathers, '*');
  // auto c = std::make_shared<Node>(value, fathers, '+');

  Node* ans = c.get();
  c->_backward = [a, b, ans](){
    a->gradient += b*ans->gradient;
  };

  return c;
}



std::shared_ptr<Node> operator*(float a, std::shared_ptr<Node> b){
  float value = a * b->value;
  std::vector<std::shared_ptr<Node>> fathers = {b};
  
  std::shared_ptr<Node> c = std::make_shared<Node>(value, fathers, '*');
  // auto c = std::make_shared<Node>(value, fathers, '+');

  Node* ans = c.get();
  c->_backward = [a, b, ans](){
    b->gradient += a * ans->gradient;
  };

  return c;
}


//------------------------- OPERATOR/
std::shared_ptr<Node> operator/(std::shared_ptr<Node> a, std::shared_ptr<Node> b){
  float value = a->value / b->value;
  std::vector<std::shared_ptr<Node>> fathers = {a, b};
  
  std::shared_ptr<Node> c = std::make_shared<Node>(value, fathers, '/');
  // auto c = std::make_shared<Node>(value, fathers, '+');

  Node* ans = c.get();
  c->_backward = [a, b, ans](){
    a->gradient += (1.0f / b->value) * ans->gradient;
    b->gradient += (-a->value / (b->value * b->value)) * ans->gradient;
  };

  return c;
}

std::shared_ptr<Node> operator/(std::shared_ptr<Node> a, float b){
  float value = a->value / b;
  std::vector<std::shared_ptr<Node>> fathers = {a};
  
  std::shared_ptr<Node> c = std::make_shared<Node>(value, fathers, '/');
  // auto c = std::make_shared<Node>(value, fathers, '+');

  Node* ans = c.get();
  c->_backward = [a, b, ans](){
    a->gradient += (1.0f / b) * ans->gradient;
  };

  return c;
}


std::shared_ptr<Node> operator/(float a, std::shared_ptr<Node> b){
  float value = a / b->value;
  std::vector<std::shared_ptr<Node>> fathers = {b};
  
  std::shared_ptr<Node> c = std::make_shared<Node>(value, fathers, '/');
  // auto c = std::make_shared<Node>(value, fathers, '+');

  Node* ans = c.get();
  c->_backward = [a, b, ans](){
    b->gradient += (-a / (b->value * b->value)) * ans->gradient;
  };

  return c;
}


//------------------------- OPERATORPOW 
std::shared_ptr<Node> operatorpow(std::shared_ptr<Node> a, std::shared_ptr<Node> b){
  float value = std::pow(a->value, b->value);
  std::vector<std::shared_ptr<Node>> fathers = {a, b};
  
  std::shared_ptr<Node> c = std::make_shared<Node>(value, fathers, '^');
  Node* ans = c.get();
  
  c->_backward = [a, b, ans](){
    a->gradient += (b->value * std::pow(a->value, b->value - 1.0f)) * ans->gradient;
    b->gradient += (ans->value * std::log(a->value)) * ans->gradient;
  };
  return c;
}

std::shared_ptr<Node> operatorpow(std::shared_ptr<Node> a, float b){
  float value = std::pow(a->value, b);
  std::vector<std::shared_ptr<Node>> fathers = {a};
  
  std::shared_ptr<Node> c = std::make_shared<Node>(value, fathers, '^');
  Node* ans = c.get();
  
  c->_backward = [a, b, ans](){
    a->gradient += (b * std::pow(a->value, b - 1.0f)) * ans->gradient;
  };
  return c;
}

std::shared_ptr<Node> operatorpow(float a, std::shared_ptr<Node> b){
  float value = std::pow(a, b->value);
  std::vector<std::shared_ptr<Node>> fathers = {b};
  
  std::shared_ptr<Node> c = std::make_shared<Node>(value, fathers, '^');
  Node* ans = c.get();
  
  c->_backward = [a, b, ans](){
    b->gradient += (ans->value * std::log(a)) * ans->gradient;
  };
  return c;
}

//------------------------- OPERATOREXP
std::shared_ptr<Node> operatorexp(std::shared_ptr<Node> a){
  float value = std::exp(a->value);
  std::vector<std::shared_ptr<Node>> fathers = {a};
  
  std::shared_ptr<Node> c = std::make_shared<Node>(value, fathers, 'e');
  Node* ans = c.get();
  
  c->_backward = [a, ans](){
    a->gradient += ans->value * ans->gradient; 
  };
  return c;
}

//------------------------- OPERATORLOG
std::shared_ptr<Node> operatorlog(std::shared_ptr<Node> a){
  float value = std::log(a->value);
  std::vector<std::shared_ptr<Node>> fathers = {a};
  
  std::shared_ptr<Node> c = std::make_shared<Node>(value, fathers, 'l');
  Node* ans = c.get();
  
  c->_backward = [a, ans](){
    a->gradient += (1.0f / a->value) * ans->gradient;
  };
  return c;
}