#include <iostream>
#include <vector>
#include <fstream>
#include "A2CAgent.h"

class TestNetImpl : public torch::nn::Module
{
    using FCLayer = torch::nn::Linear;

public:
    TestNetImpl(int64_t inputSize, int64_t hiddenSize, int64_t outputSize, float lr, float beta1, float beta2);
    float GetLoss(torch::Tensor input, torch::Tensor trueOutput, bool bAdvantage = false);

private:
    int64_t m_InputSize, m_HiddenSize, m_OutputSize;
    float m_LearningRate, m_Beta1, m_Beta2;
    std::unique_ptr<torch::optim::Adam> m_Optimizer;
    torch::Tensor m_Advantage;

    FCLayer m_FCCommon, m_FCPolicy, m_FCValue;
};

TestNetImpl::TestNetImpl(int64_t inputSize, int64_t hiddenSize, int64_t outputSize, float lr, float beta1, float beta2)
    : m_InputSize(inputSize), m_HiddenSize(hiddenSize), m_OutputSize(outputSize), m_LearningRate(lr), m_Beta1(beta1), m_Beta2(beta2)
    , m_FCCommon(nullptr), m_FCPolicy(nullptr), m_FCValue(nullptr)
{
    m_FCCommon = register_module("m_FCCommon", FCLayer(m_InputSize, m_HiddenSize));
    m_FCPolicy = register_module("m_FCPolicy", FCLayer(m_HiddenSize, m_OutputSize));
    m_FCValue = register_module("m_FCValue", FCLayer(m_HiddenSize, 1));

    m_Optimizer = std::make_unique<torch::optim::Adam>(this->parameters(), torch::optim::AdamOptions().lr(m_LearningRate).betas({ m_Beta1, m_Beta2 }));
}

float TestNetImpl::GetLoss(torch::Tensor input, torch::Tensor trueOutput, bool bAdvantage)
{
    torch::Tensor y = m_FCCommon->forward(input);
    torch::Tensor x = torch::softmax(m_FCPolicy->forward(y), 1);
    torch::Tensor actionProb = (trueOutput * x).sum({ 1 });

    if (bAdvantage)
    {
        torch::Tensor value = m_FCValue->forward(y);
        m_Advantage = value.detach();
    }

    torch::Tensor loss = -torch::log(actionProb + 1.e-5).mean({ 0 }) * m_Advantage;
    std::cout << "m_Advantage:\n" << m_Advantage << std::endl;
    std::cout << "loss:\n" << loss << std::endl;

    if (bAdvantage)
        loss.backward();

    return loss.item().toFloat();
}

TORCH_MODULE(TestNet);

int main()
{
    torch::manual_seed(1);

    TestNet network(3, 5, 2, 0.01f, 0.9f, 0.999f);
    float dx = 0.001f;

    std::vector<float> intputVector({ 1.0f, 2.0f, 3.0f });
    torch::Tensor input = torch::from_blob(intputVector.data(), { 1, 3 });

    std::vector<float> outputVector({ 1.0f, 0.0f });
    torch::Tensor output = torch::from_blob(outputVector.data(), { 1, 2 });

    float loss1 = network->GetLoss(input, output, true);

    std::cout << "input:\n" << input << std::endl;
    std::cout << "output:\n" << output << std::endl;
    std::cout << "loss1: " << loss1 << std::endl;
    
    for (auto& kv : network->named_parameters())
    {
        if (kv.key() == "m_FCCommon.weight")
        {
            std::cout << kv.value() << std::endl;
            (kv.value())[0][0] = (kv.value())[0][0] + dx;
            std::cout << kv.value() << std::endl;

            std::cout << "gradient:\n" << kv.value().grad() << std::endl;
        }
    }

    float loss2 = network->GetLoss(input, output, false);

    std::cout << "loss2: " << loss2 << std::endl;

    std::cout << "numerical_gradient: " << (loss2 - loss1) / dx << std::endl;
}
