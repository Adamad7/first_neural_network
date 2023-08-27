#include <../include/activation_functions.hpp>

double Sigmoid::operator()(double x) const
{
    return 1.0 / (1.0 + std::exp(-x));
}

double Sigmoid::derivative(double x) const
{
    return (*this)(x) * (1.0 - (*this)(x));
}

double Tanh::operator()(double x) const
{
    return std::tanh(x);
}

double Tanh::derivative(double x) const
{
    return 1.0 - std::pow((*this)(x), 2);
}

double ReLU::operator()(double x) const
{
    return std::max(0.0, x);
}

double ReLU::derivative(double x) const
{
    return x > 0.0 ? 1.0 : 0.0;
}

double LeakyReLU::operator()(double x) const
{
    return x > 0.0 ? x : 0.01 * x;
}

double LeakyReLU::derivative(double x) const
{
    return x > 0.0 ? 1.0 : 0.01;
}

double Softmax::operator()(double x) const
{
    return std::exp(x);
}

double Softmax::derivative(double x) const
{
    return (*this)(x) * (1.0 - (*this)(x));
}
