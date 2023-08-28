#pragma once
#include "neuron.hpp"

class Neuron;

class Edge
{
public:
    Edge(Neuron *inputNeuron, Neuron *outputNeuron, double weight);
    void setWeight(double weight);
    double getWeight();
    // void setDeltaWeight(double deltaWeight);
    // double getDeltaWeight();
    Neuron *getInputNeuron();
    Neuron *getOutputNeuron();

private:
    double weight;
    double deltaWeight;
    Neuron *inputNeuron = nullptr;
    Neuron *outputNeuron = nullptr;
};