#include <../include/neuron.hpp>

Neuron::Neuron(ActivationFunction &activationFunction, double bias)
    : activationFunction(activationFunction), bias(bias)
{
}

void Neuron::addInputEdge(Edge *edge)
{
    inputEdges.push_back(edge);
}

void Neuron::addOutputEdge(Edge *edge)
{
    outputEdges.push_back(edge);
}

void Neuron::setBiasValue(double biasValue)
{
    bias = biasValue;
}

double Neuron::getBiasValue()
{
    return bias;
}

void Neuron::calculateValue()
{
    double sum = 0.0;
    for (auto edge : inputEdges)
    {
        sum += edge->getInputNeuron()->getValue() * edge->getWeight();
    }
    sum += bias;
    value = activationFunction(sum);
}

void Neuron::setValue(double value)
{
    this->value = value;
}

double Neuron::getValue()
{
    return value;
}

std::vector<Edge *> Neuron::getInputEdges()
{
    return inputEdges;
}

std::vector<Edge *> Neuron::getOutputEdges()
{
    return outputEdges;
}

Neuron::~Neuron()
{
    for (auto edge : inputEdges)
    {
        delete edge;
    }
    for (auto edge : outputEdges)
    {
        delete edge;
    }
}