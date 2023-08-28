#include "../include/neural_network.hpp"
#include <iostream>

NeuralNetwork::NeuralNetwork(std::vector<uint8_t> topology, ActivationFunction &activationFunction, std::vector<ExpectedOutput> expectedOutputs)
    : activationFunction(activationFunction), expectedOutputs(expectedOutputs)
{

    layers.push_back(new Layer(topology[0], activationFunction));
    for (uint8_t i = 1; i < topology.size() - 1; i++)
    {
        layers.push_back(new Layer(LayerType::HIDDEN, layers.back(), topology[i], activationFunction));
    }

    layers.push_back(new Layer(LayerType::OUTPUT, layers.back(), topology.back(), activationFunction));

    // for (uint8_t i = 0; i < layers.size() - 1; i++)
    // {
    //     for (uint8_t j = 0; j < layers[i].getNeurons().size(); j++)
    //     {
    //         for (uint8_t k = 0; k < layers[i + 1].getNeurons().size(); k++)
    //         {
    //             layers[i].getNeurons()[j].addOutputEdge(&layers[i + 1].getNeurons()[k]);
    //             layers[i + 1].getNeurons()[k].addInputEdge(&layers[i].getNeurons()[j]);
    //         }
    //     }
    // }
}

void NeuralNetwork::train(uint16_t populationSize, uint8_t bestPopulationSize, uint16_t numberOfEpochs, double mutationRate)
{
    this->populationSize = populationSize;
    this->bestPopulationSize = bestPopulationSize;
    this->numberOfEpochs = numberOfEpochs;
    this->mutationRate = mutationRate;
    this->mutationDistribution = std::uniform_real_distribution<double>(1 - mutationRate, 1 + mutationRate);

    bestDistribution = std::uniform_int_distribution<int>(0, bestPopulationSize - 1);

    for (uint16_t i = 0; i < populationSize; i++)
    {
        weigths.push_back(getNeuralNetData());
        randomizeWeights();
    }

    std::nth_element(weigths.begin(), weigths.begin() + bestPopulationSize, weigths.end(), [](NeuralNetData &a, NeuralNetData &b)
                     { return a.error < b.error; });

    bestWeights = std::vector<NeuralNetData>(weigths.begin(), weigths.begin() + bestPopulationSize);

    for (uint16_t epoch = 1; epoch < numberOfEpochs; epoch++)
    {
        for (uint16_t i = 0; i < populationSize; i++)
        {
            shuffleAndMutateWeights();
            weigths[i] = getNeuralNetData();
        }

        std::nth_element(weigths.begin(), weigths.begin() + bestPopulationSize, weigths.end(), [](NeuralNetData &a, NeuralNetData &b)
                         { return a.error < b.error; });

        bestWeights = std::vector<NeuralNetData>(weigths.begin(), weigths.begin() + bestPopulationSize);
        std::cout << "Epoch: " << epoch << " Error: " << bestWeights[0].error << std::endl;
    }
    std::nth_element(weigths.begin(), weigths.begin(), weigths.end(), [](NeuralNetData &a, NeuralNetData &b)
                     { return a.error < b.error; });
    assignWeigthtsAndBiases(weigths[0]);
}

void NeuralNetwork::propagateForward(std::vector<double> input)
{
    assert(layers[0]->getNeurons().size() == input.size());
    for (uint8_t i = 0; i < layers[0]->getNeurons().size(); i++)
    {
        layers[0]->getNeurons()[i]->setValue(input[i]);
    }
    for (uint8_t i = 1; i < layers.size(); i++)
    {
        for (auto neuron : layers[i]->getNeurons())
        {
            neuron->calculateValue();
        }
    }
}

double NeuralNetwork::calculateError()
{
    double sum = 0.0;
    for (uint8_t i = 0; i < expectedOutputs.size(); i++)
    {
        assert(layers.back()->getNeurons().size() == expectedOutputs[i].expectedOutput.size());
        propagateForward(expectedOutputs[i].input);
        for (uint8_t j = 0; j < layers.back()->getNeurons().size(); j++)
        {
            sum += pow(layers.back()->getNeurons()[j]->getValue() - expectedOutputs[i].expectedOutput[j], 2);
        }
    }

    return sum / expectedOutputs.size();
}

NeuralNetData NeuralNetwork::getNeuralNetData()
{
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> biases;
    for (uint8_t i = 1; i < layers.size(); i++)
    {
        std::vector<std::vector<double>> layerWeights;
        std::vector<double> layerBiases;
        for (auto neuron : layers[i]->getNeurons())
        {
            std::vector<double> neuronWeights;
            layerBiases.push_back(neuron->getBiasValue());
            for (auto edge : neuron->getInputEdges())
            {
                neuronWeights.push_back(edge->getWeight());
            }
            layerWeights.push_back(neuronWeights);
        }
        weights.push_back(layerWeights);
        biases.push_back(layerBiases);
    }
    return NeuralNetData(weights, biases, calculateError());
}

void NeuralNetwork::randomizeWeights()
{
    for (uint8_t i = 1; i < layers.size(); i++)
    {
        for (auto neuron : layers[i]->getNeurons())
        {
            neuron->setBiasValue(distribution(randomDevice));
            for (auto edge : neuron->getInputEdges())
            {
                edge->setWeight(distribution(randomDevice));
            }
        }
    }
}

void NeuralNetwork::shuffleAndMutateWeights()
{
    for (uint8_t i = 1; i < layers.size(); i++)
    {
        uint16_t neuronIndex = 0;
        for (auto neuron : layers[i]->getNeurons())
        {
            neuron->setBiasValue(bestWeights[bestDistribution(randomDevice)].biases[i - 1][neuronIndex] * mutationDistribution(randomDevice));
            uint16_t edgeIndex = 0;
            for (auto edge : neuron->getInputEdges())
            {
                edge->setWeight(bestWeights[bestDistribution(randomDevice)].weights[i - 1][neuronIndex][edgeIndex]);
                edgeIndex++;
            }
            neuronIndex++;
        }
    }
}

void NeuralNetwork::assignWeigthtsAndBiases(NeuralNetData &neuralNetData)
{
    for (uint8_t i = 1; i < layers.size(); i++)
    {
        uint16_t neuronIndex = 0;
        for (auto neuron : layers[i]->getNeurons())
        {
            neuron->setBiasValue(neuralNetData.biases[i - 1][neuronIndex]);
            uint16_t edgeIndex = 0;
            for (auto edge : neuron->getInputEdges())
            {
                edge->setWeight(neuralNetData.weights[i - 1][neuronIndex][edgeIndex]);
                edgeIndex++;
            }
            neuronIndex++;
        }
    }
}

std::vector<double> NeuralNetwork::predict(std::vector<double> input)
{
    assert(layers[0]->getNeurons().size() == input.size());
    propagateForward(input);
    std::vector<double> output;
    for (auto neuron : layers.back()->getNeurons())
    {
        output.push_back(neuron->getValue());
    }
    return output;
}