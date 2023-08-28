#pragma once
#include <vector>

class NeuralNetData
{
public:
    NeuralNetData(std::vector<std::vector<std::vector<double>>> weights, std::vector<std::vector<double>> biases, double error);
    double error;
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> biases;
};