#pragma once

#include <torch/torch.h>
#include <vector>
#include <iostream>


class ModelOutput
{
    friend std::ostream& operator<<(std::ostream& os, const ModelOutput& output);
public:
    ModelOutput(std::vector<torch::Tensor> probPolicy, torch::Tensor value)
        : m_ProbPolicy(std::move(probPolicy)), m_Value(value)
    {
    }

    // Move Constructor
    ModelOutput(ModelOutput&& other)
        : m_ProbPolicy(std::move(other.m_ProbPolicy)), m_Value(other.m_Value)
    {
    }

    const std::vector<torch::Tensor>& GetProbPolicy() const { return m_ProbPolicy; }
    const torch::Tensor& GetValue() const { return m_Value; }

private:
    std::vector<torch::Tensor> m_ProbPolicy;
    torch::Tensor m_Value;
};

class A2CAgent : public torch::nn::Module
{
    using FCLayer = torch::nn::Linear;

public:
    A2CAgent(int stateSize, int hiddenSize, int actionSize, float discountFactor);
    ModelOutput Forward(std::vector<float>& state);
    const std::vector<int> GetAction(std::vector<float>& state);
    void SetStateNormalizeValue(float average, float std) { m_StateAverage = average; m_StateSTD = std; }
    void SetDevice(torch::Device device) { this->to(device); m_Device = device; }
    void TrainModel(std::vector<float>& state, std::vector<int>& action, float reward, std::vector<float>& nextState);

private:
    int m_StateSize, m_HiddenSize, m_ActionSize;
    float m_StateAverage, m_StateSTD;
    float m_DiscountFactor;
    torch::Device m_Device;

    FCLayer m_FCCommon;
    FCLayer m_FCPolicy0, m_FCPolicy1, m_FCPolicy2, m_FCPolicy3, m_FCPolicy4;
    FCLayer m_FCValue;
};

