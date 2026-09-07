# The compiler
CXX = g++

# Flags: Enable C++17, warnings, and specify WHERE the .hpp files are (-Iinclude)
CXXFLAGS = -Wall -Wextra -std=c++17 -Iinclude

# The final executable
TARGET = engine

# Automatically find all source files using wildcards
SRCS = examples/main.cpp \
       $(wildcard src/*.cpp) \
       $(wildcard src/optimizer/*.cpp) \
       $(wildcard src/nn/*.cpp)

# Default main rule
all: $(TARGET)

# How to build the executable
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

# Cleanup
clean:
	rm -f $(TARGET)

# Exclude the original main, compile the engine together with the tests, and run them
test: $(filter-out examples/main.cpp, $(SRCS)) tests/test_engine.cpp
	$(CXX) $(CXXFLAGS) $^ -o test_engine
	./test_engine