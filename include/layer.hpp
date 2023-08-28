#pragma once
#include "neuron.hpp"
#include <cstdint>
#include <random>

enum LayerType
{
    INPUT,
    HIDDEN,
    OUTPUT
};

class Layer
{
public:
    Layer(uint8_t numberOfNeurons, ActivationFunction &activationFunction);
    // Layer(LayerType type, Layer *previousLayer);
    Layer(LayerType type, Layer *previousLayer, uint8_t numberOfNeurons, ActivationFunction &activationFunction);

    ~Layer();
    // void addNeuron(Neuron *neuron);
    void setNextLayer(Layer *nextLayer);

    std::vector<Neuron *> &getNeurons();

private:
    std::vector<Neuron *> neurons;
    Layer *previousLayer = nullptr;
    Layer *nextLayer = nullptr;
    LayerType type;
    ActivationFunction &activationFunction;
    std::random_device randomDevice;
    std::uniform_real_distribution<double> distribution = std::uniform_real_distribution<double>(-2, 2);
};