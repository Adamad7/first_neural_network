# Compiler
CXX := g++       
# Compiler flags
CXXFLAGS := -std=c++14 -Iinclude
# Linker flags
SRCDIR := src
BUILDDIR := build

SOURCES := $(wildcard $(SRCDIR)/*.cpp) 
OBJECTS := $(patsubst $(SRCDIR)/%.cpp,$(BUILDDIR)/%.o,$(SOURCES)) 
TARGET := $(BUILDDIR)/neural_network

.PHONY: all clean test

all: $(TARGET)
	.\$(TARGET).exe

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ 

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<