#include <../include/layer.hpp>

Layer::Layer(uint8_t numberOfNeurons, ActivationFunction &activationFunction)
    : activationFunction(activationFunction)
{
    type = LayerType::INPUT;
    for (uint8_t i = 0; i < numberOfNeurons; i++)
    {
        neurons.push_back(new Neuron(activationFunction, 0.0));
    }
}

Layer::Layer(LayerType type, Layer *previousLayer, uint8_t numberOfNeurons, ActivationFunction &activationFunction)
    : type(type), previousLayer(previousLayer), activationFunction(activationFunction)
{
    for (uint8_t i = 0; i < numberOfNeurons; i++)
    {
        neurons.push_back(new Neuron(activationFunction, 0.0));
        for (auto previousNeuron : previousLayer->getNeurons())
        {
            Edge *edge = new Edge(previousNeuron, neurons.back(), distribution(randomDevice));
            previousNeuron->addOutputEdge(edge);
            neurons.back()->addInputEdge(edge);
        }
    }
}

Layer::~Layer()
{
    for (auto neuron : neurons)
    {
        delete neuron;
    }
}

void Layer::setNextLayer(Layer *nextLayer)
{
    this->nextLayer = nextLayer;
}

std::vector<Neuron *> &Layer::getNeurons()
{
    return neurons;
}