#include <cmath>
#include <algorithm>

class ActivationFunction
{

public:
    virtual double operator()(double x) const = 0;
    virtual double derivative(double x) const = 0;
    virtual ~ActivationFunction() = default;
};

class Sigmoid : public ActivationFunction
{
public:
    double operator()(double x) const override;
    double derivative(double x) const override;
};

class Tanh : public ActivationFunction
{
public:
    double operator()(double x) const override;
    double derivative(double x) const override;
};

class ReLU : public ActivationFunction
{
public:
    double operator()(double x) const override;
    double derivative(double x) const override;
};

class LeakyReLU : public ActivationFunction
{
public:
    double operator()(double x) const override;
    double derivative(double x) const override;
};

class Softmax : public ActivationFunction
{
public:
    double operator()(double x) const override;
    double derivative(double x) const override;
};
