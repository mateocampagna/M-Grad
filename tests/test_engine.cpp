#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../include/autograd/engine.hpp"
#include <cmath>


TEST_CASE("Gradient Accumulation in Basic Addition") {
    auto x = std::make_shared<Node>(3.0f);
    
    // y = x + x + x
    auto y = x + x + x; 
    y->backward();
    
    CHECK(y->value == doctest::Approx(9.0f));
    // dx/dy = 1 + 1 + 1 = 3
    CHECK(x->gradient == doctest::Approx(3.0f));
}

TEST_CASE("Multiplication and Subtraction Chain") {
    auto a = std::make_shared<Node>(4.0f);
    auto b = std::make_shared<Node>(2.0f);
    
    // c = (a * b) - b
    auto c = (a * b) - b; 
    c->backward();
    
    CHECK(c->value == doctest::Approx(6.0f));
    // dc/da = b = 2
    CHECK(a->gradient == doctest::Approx(2.0f)); 
    // dc/db = a - 1 = 3
    CHECK(b->gradient == doctest::Approx(3.0f)); 
}

TEST_CASE("Division and Fractional Gradients") {
    auto a = std::make_shared<Node>(10.0f);
    auto b = std::make_shared<Node>(2.0f);
    
    // c = a / b
    auto c = a / b;
    c->backward();
    
    CHECK(c->value == doctest::Approx(5.0f));
    // dc/da = 1/b = 0.5
    CHECK(a->gradient == doctest::Approx(0.5f)); 
    // dc/db = -a / b^2 = -10 / 4 = -2.5
    CHECK(b->gradient == doctest::Approx(-2.5f)); 
}

TEST_CASE("Power Operation (Variables and Constants)") {
    auto x = std::make_shared<Node>(2.0f);
    auto y = std::make_shared<Node>(3.0f);
    
    // z = x^y
    auto z = pow(x, y); 
    z->backward();
    
    CHECK(z->value == doctest::Approx(8.0f));
    // dz/dx = y * x^(y-1) = 3 * 2^2 = 12
    CHECK(x->gradient == doctest::Approx(12.0f)); 
    // dz/dy = x^y * ln(x) = 8 * ln(2)
    CHECK(y->gradient == doctest::Approx(8.0f * std::log(2.0f))); 
}

TEST_CASE("Exponential and Natural Logarithm") {
    auto x = std::make_shared<Node>(2.0f);
    auto y = exp(x);
    y->backward();
    
    CHECK(y->value == doctest::Approx(std::exp(2.0f)));
    // dy/dx = e^x
    CHECK(x->gradient == doctest::Approx(std::exp(2.0f)));

    auto a = std::make_shared<Node>(std::exp(1.0f));
    auto b = log(a);
    b->backward();
    
    CHECK(b->value == doctest::Approx(1.0f));
    // db/da = 1/a = 1/e
    CHECK(a->gradient == doctest::Approx(1.0f / std::exp(1.0f)));
}

TEST_CASE("ReLU Activation States") {
    // Active Node (x > 0)
    auto x_active = std::make_shared<Node>(5.0f);
    auto y1 = relu(x_active);
    y1->backward();
    
    CHECK(y1->value == doctest::Approx(5.0f));
    CHECK(x_active->gradient == doctest::Approx(1.0f));

    // Dead Node (x <= 0)
    auto x_dead = std::make_shared<Node>(-5.0f);
    auto y2 = relu(x_dead);
    y2->backward();
    
    CHECK(y2->value == doctest::Approx(0.0f));
    CHECK(x_dead->gradient == doctest::Approx(0.0f));
}

TEST_CASE("Complex Analytical Topology (Chain Rule)") {
    auto x = std::make_shared<Node>(1.5f);
    auto y = std::make_shared<Node>(0.5f);
    
    // f(x,y) = (x^2 + y) * e^y
    auto x_sq = pow(x, 2.0f);
    auto sum = x_sq + y;
    auto e_y = exp(y);
    auto out = sum * e_y;
    
    out->backward();
    
    // Analytical check:
    // f(1.5, 0.5) = (2.25 + 0.5) * e^0.5 = 2.75 * e^0.5
    float expected_val = 2.75f * std::exp(0.5f);
    // df/dx = 2x * e^y = 3 * e^0.5
    float expected_dx = 3.0f * std::exp(0.5f);
    // df/dy = e^y + (x^2 + y)*e^y = e^y * (1 + 1.5^2 + 0.5) = e^0.5 * 3.75
    float expected_dy = 3.75f * std::exp(0.5f);
    
    CHECK(out->value == doctest::Approx(expected_val));
    CHECK(x->gradient == doctest::Approx(expected_dx));
    CHECK(y->gradient == doctest::Approx(expected_dy));
}