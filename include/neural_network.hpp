#pragma once
#include "layer.hpp"
#include "weights_and_biases.hpp"
#include <unordered_map>
#include <cassert>
#include <cmath>
#include <algorithm>

class ExpectedOutput
{
public:
    ExpectedOutput(std::vector<double> input, std::vector<double> expectedOutput)
    {
        this->input = input;
        this->expectedOutput = expectedOutput;
    }

    std::vector<double> input;
    std::vector<double> expectedOutput;
};
class NeuralNetwork
{
public:
    NeuralNetwork(std::vector<uint8_t> topology, ActivationFunction &activationFunction, std::vector<ExpectedOutput> expectedOutputs);
    void train(uint16_t populationSize = 100, uint8_t bestPopulationSize = 20, uint16_t numberOfEpochs = 1000, double mutationRate = 0.1);
    std::vector<double> predict(std::vector<double> input);

private:
    std::vector<ExpectedOutput> expectedOutputs;
    ActivationFunction &activationFunction;
    std::vector<Layer *> layers;
    void propagateForward(std::vector<double> input);
    double calculateError();
    NeuralNetData getNeuralNetData();
    void randomizeWeights();
    void shuffleAndMutateWeights();
    void assignWeigthtsAndBiases(NeuralNetData &neuralNetData);
    std::vector<NeuralNetData> weigths;
    std::vector<NeuralNetData> bestWeights;
    uint8_t bestPopulationSize;
    uint16_t populationSize;
    uint16_t numberOfEpochs;
    double mutationRate;
    std::random_device randomDevice;
    std::uniform_real_distribution<double> distribution = std::uniform_real_distribution<double>(-2, 2);
    std::uniform_int_distribution<int> bestDistribution;
    std::uniform_real_distribution<double> mutationDistribution;
};