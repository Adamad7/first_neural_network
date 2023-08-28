#include "../include/weights_and_biases.hpp"

NeuralNetData::NeuralNetData(std::vector<std::vector<std::vector<double>>> weights, std::vector<std::vector<double>> biases, double error)
    : weights(weights), biases(biases), error(error)
{
}
