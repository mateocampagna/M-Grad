CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++17

TARGET = engine

# source files
SRCS = main.cpp engine.cpp

# Regla principal
all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

clean:
	rm -f $(TARGET)