#include <edge.hpp>
#include <activation_functions.hpp>
#include <vector>
class Neuron
{
private:
    std::vector<Edge *> inputEdges;
    std::vector<Edge *> outputEdges;
    ActivationFunction &activationFunction;
    double value;
    double error;
    Edge *biasEdge;
};