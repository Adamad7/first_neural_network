#pragma once
#include "edge.hpp"
#include "activation_functions.hpp"
#include <vector>
class Edge;

class Neuron
{
public:
    Neuron(ActivationFunction &activationFunction, double bias);
    void addInputEdge(Edge *edge);
    void addOutputEdge(Edge *edge);
    // void setBiasEdge(Edge *edge);
    void calculateValue();
    // void calculateError(double targetValue);
    // void calculateError();
    // void updateWeights(double learningRate, double momentum);
    double getValue();
    void setValue(double value);
    // double getError();
    std::vector<Edge *> getInputEdges();
    std::vector<Edge *> getOutputEdges();
    // Edge *getBiasEdge();
    void setBiasValue(double biasValue);
    double getBiasValue();

    ~Neuron();

private:
    std::vector<Edge *> inputEdges;
    std::vector<Edge *> outputEdges;
    ActivationFunction &activationFunction;
    double value;
    // double error;
    // Edge *biasEdge;
    double bias;
};