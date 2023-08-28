#include <iostream>
#include "../include/neural_network.hpp"
#include "../include/activation_functions.hpp"

int main()
{
    ReLU function = ReLU();
    NeuralNetwork network({2, 3, 2}, function, {ExpectedOutput({0, 0}, {0, 1}), ExpectedOutput({0, 1}, {1, 0}), ExpectedOutput({1, 0}, {1, 0}), ExpectedOutput({1, 1}, {0, 1})});
    network.train(500, 30, 1000, 0.1);

    std::vector<double> input;
    std::vector<double> output;
    while (true)
    {
        input.clear();
        std::cout << "A: ";
        double x;
        std::cin >> x;
        input.push_back(x);
        std::cout << "B: ";
        std::cin >> x;
        input.push_back(x);
        output = network.predict(input);
        std::cout << "Output: ";
        if (output[0] > output[1])
        {
            std::cout << "1";
        }
        else if (output[0] < output[1])
        {
            std::cout << "0";
        }
        else
        {
            std::cout << "I don't know";
        }
        std::cout << std::endl;
        std::cout << std::endl;
    }
    return 0;
}