# M-Grad

**M-Grad** is a scalar automatic differentiation (autograd) engine, written from scratch in **C++17**, with no dependencies beyond the STL.

The goal of this project is to **take apart the "magic" behind frameworks like PyTorch or TensorFlow**: instead of treating `loss.backward()` as a black box, this repo implements the dynamic computational graph, chain-rule gradient accumulation, and a small neural network (`Neuron` → `Layer` → `MLP`) trained with gradient descent on top of that engine, all by hand. It's an educational project meant to understand, at the level of pointers and memory, how backpropagation actually works.

## Table of contents

- [Architecture decisions](#architecture-decisions)
- [Quick Start](#quick-start)
- [Build and testing](#build-and-testing)
- [License and author](#license-and-author)

## Architecture decisions

### Dynamic graph with `std::shared_ptr`

Every `Node` is a node in the computational graph (either a leaf or the result of an operation), and it's handled exclusively through `std::shared_ptr<Node>`. This isn't an arbitrary choice: in an autodiff graph **the same node can be the parent of more than one child node** (e.g. `y = x + x + x` reuses `x` three times), so the structure isn't a tree with a single owner but a **DAG with shared ownership**. `shared_ptr` solves this with reference counting: each node stays alive exactly as long as something still references it (another node in the graph, or the user), with no manual `delete` calls, no dangling pointers, and no use-after-free.

One non-trivial detail: each node's `_backward` closure captures its **parents by `shared_ptr`** (extending their lifetime for as long as gradients still need to be propagated), but captures **its own node (`ans`) by raw pointer** instead of `shared_ptr`:

```cpp
Node* ans = c.get();
c->_backward = [a, b, ans](){
    a->gradient += b->value * ans->gradient;
    b->gradient += a->value * ans->gradient;
};
```

If `c->_backward` captured a `shared_ptr` to itself (`c`), it would create a **reference cycle** (`c` → `_backward` → `shared_ptr<c>` → `c`), and the node would never free its memory. Using a raw pointer for the self-reference breaks the cycle without sacrificing safety, because the node never needs to keep itself alive, its parents are the ones keeping it alive.

### Iterative topological sort (avoiding stack overflow)

`backward()` needs to visit nodes in **reverse topological order**: a node's gradient is only complete once *every* node that uses it as a parent has already propagated its own. The classic way to build that order is a recursive post-order DFS, but that ties the depth of the graph to the depth of the OS call stack, for a network with many chained layers (or any computation with a long chain of operations), that can end in a stack overflow.

To avoid that, `Node::backward()` implements the DFS with an **explicit heap-allocated stack**, manually simulating the recursion's call frames:

```cpp
std::vector<std::pair<Node*, bool>> stack;
stack.push_back({this, false});

while (!stack.empty()) {
    auto [v, processed] = stack.back();
    stack.pop_back();

    if (processed) { topo.push_back(v); continue; }
    if (visited.count(v)) continue;

    visited.insert(v);
    stack.push_back({v, true});                 // re-queued to be processed "on the way back"
    for (auto it = v->fathers.rbegin(); it != v->fathers.rend(); ++it)
        stack.push_back({it->get(), false});    // parents get queued
}
```

Each node is pushed onto the stack twice: the first time (`processed = false`) marks it as visited and queues its parents; the second time (`processed = true`) corresponds to the point in a recursive function *after* all of its children's calls have returned, and that's when it's actually appended to the topological order. The result is exactly the same post-order DFS, but with a stack that grows in dynamic memory (the heap) instead of the execution stack bounded only by available RAM instead of the thread's fixed stack size. It's the same technique interpreters and compilers use to walk arbitrarily deep syntax trees without native recursion.

## Quick Start

```cpp
#include "autograd/engine.hpp"
#include "nn/mlp.hpp"
#include <iostream>
#include <vector>

int main() {
    // Network with 2 inputs, two hidden layers of 8 neurons, and 1 output
    MLP model(2, {8, 8, 1});

    std::vector<std::shared_ptr<Node>> x = {
        std::make_shared<Node>(2.0f),
        std::make_shared<Node>(-1.0f)
    };
    auto target = std::make_shared<Node>(1.0f);

    // Forward pass through the MLP
    auto pred = model(x);

    // Squared error (Mean Squared Error for a single output)
    auto diff = pred[0] - target;
    auto loss = diff * diff;

    // Backward pass: builds the topological order and propagates gradients
    loss->backward();

    std::cout << "Loss: " << loss->value << "\n";
    std::cout << "Gradient of the first parameter: "
              << model.parameters()[0]->gradient << "\n";
}
```

A full training loop (forward → loss → backward → step) lives in `examples/main.cpp`.

## Build and testing

The project builds with `g++` and `-std=c++17` via `Makefile`, with nothing to install.

```bash
# Build the "engine" executable (examples/main.cpp + the full engine)
make
./engine

# Build the engine together with the test suite (doctest) and run it
make test
```

`make test` validates with exact mathematical assertions (`doctest::Approx`) that the gradients computed by the engine match the expected analytical derivative for addition, subtraction, multiplication, division, power, `exp`, `log`, ReLU, and chained compositions of the above (chain rule).

## License and author

Distributed under the **MIT** license. See [`LICENSE`](./LICENSE).

**Mateo Campagna** 