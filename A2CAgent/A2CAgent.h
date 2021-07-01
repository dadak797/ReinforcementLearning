#pragma once

#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <memory>


class ModelOutput
{
    friend std::ostream& operator<<(std::ostream& os, const ModelOutput& output);

public:
    ModelOutput(torch::Tensor probPolicy, torch::Tensor value)
        : m_ProbPolicy(std::move(probPolicy)), m_Value(value)
    {
    }

    // Move Constructor
    ModelOutput(ModelOutput&& other)
        : m_ProbPolicy(std::move(other.m_ProbPolicy)), m_Value(other.m_Value)
    {
    }

    const torch::Tensor& GetProbPolicy() const { return m_ProbPolicy; }
    const torch::Tensor& GetValue() const { return m_Value; }

private:
    torch::Tensor m_ProbPolicy;
    torch::Tensor m_Value;
};

class A2CAgentImpl : public torch::nn::Module
{
    using FCLayer = torch::nn::Linear;

public:
    A2CAgentImpl(int stateSize, int hiddenSize, int actionSize, float discountFactor, float lr, float beta1, float beta2);
    ModelOutput Forward(std::vector<float>& state);
    const std::vector<int> GetAction(std::vector<float>& state);
    void SetStateNormalizeValue(float average, float std) { m_StateAverage = average; m_StateSTD = std; }
    void SetDevice(torch::Device device) { this->to(device); m_Device = device; }
    float TrainModel(std::vector<float>& state, std::vector<int>& action, float reward, std::vector<float>& nextState);

private:
    int m_StateSize, m_HiddenSize, m_ActionSize;
    float m_StateAverage, m_StateSTD;
    float m_DiscountFactor;
    float m_Alpha;
    torch::Device m_Device;
    const float m_SmallValue;

    float m_LearningRate, m_Beta1, m_Beta2;
    std::unique_ptr<torch::optim::Adam> m_Optimizer;

    FCLayer m_FCCommon;
    FCLayer m_FCPolicy0, m_FCPolicy1, m_FCPolicy2, m_FCPolicy3, m_FCPolicy4;
    FCLayer m_FCValue;
};

TORCH_MODULE(A2CAgent);