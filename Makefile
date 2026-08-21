# El compilador
CXX = g++

# Banderas: Activamos C++17, warnings, y le decimos DÓNDE están los .hpp (-Iinclude)
CXXFLAGS = -Wall -Wextra -std=c++17 -Iinclude

# El ejecutable final
TARGET = engine

# Encontramos automáticamente todos los archivos fuente usando wildcards
SRCS = examples/main.cpp \
       $(wildcard src/*.cpp) \
       $(wildcard src/optimizer/*.cpp) \
       $(wildcard src/nn/*.cpp)

# Regla principal por defecto
all: $(TARGET)

# Cómo construir el ejecutable
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

# Limpieza
clean:
	rm -f $(TARGET)