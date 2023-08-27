#include <neuron.hpp>
#include <cstdint>

enum LayerType
{
    INPUT,
    HIDDEN,
    OUTPUT
};

class Layer
{
public:
    Layer();
    // Layer(LayerType type, Layer *previousLayer);
    Layer(LayerType type, Layer *previousLayer, uint8_t numberOfNeurons);

    ~Layer();
    // void addNeuron(Neuron *neuron);
    void setNextLayer(Layer *nextLayer);

private:
    std::vector<Neuron *> neurons;
    Layer *previousLayer;
    Layer *nextLayer;
    LayerType type;
};