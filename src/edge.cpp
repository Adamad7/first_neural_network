#include <../include/edge.hpp>

Edge::Edge(Neuron *inputNeuron, Neuron *outputNeuron, double weight)
    : inputNeuron(inputNeuron), outputNeuron(outputNeuron), weight(weight)
{
}

void Edge::setWeight(double weight)
{
    this->weight = weight;
}

double Edge::getWeight()
{
    return weight;
}

Neuron *Edge::getInputNeuron()
{
    return inputNeuron;
}

Neuron *Edge::getOutputNeuron()
{
    return outputNeuron;
}