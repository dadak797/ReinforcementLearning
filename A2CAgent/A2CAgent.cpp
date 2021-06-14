#include "A2CAgent.h"


std::ostream& operator<<(std::ostream& os, const ModelOutput& output)
{
    std::cout << "Action Probability\n";
    int idx = 0;
    for (const auto& prob : output.m_ProbPolicy)
    {
        std::cout << "[" << idx << "]: " << prob << std::endl;
        idx++;
    }

    std::cout << "Value Function: " << output.m_Value;

    return os;
}

A2CAgent::A2CAgent(int stateSize, int hiddenSize, int actionSize, float discountFactor)
    : m_StateSize(stateSize), m_HiddenSize(hiddenSize), m_ActionSize(actionSize)
    , m_FCCommon(nullptr), m_FCPolicy0(nullptr), m_FCPolicy1(nullptr), m_FCPolicy2(nullptr), m_FCPolicy3(nullptr), m_FCPolicy4(nullptr)
    , m_FCValue(nullptr), m_StateAverage(0.0f), m_StateSTD(1.0f), m_Device(torch::kCPU)
    , m_DiscountFactor(discountFactor)
{
    m_FCCommon = register_module("m_FCCommon", FCLayer(m_StateSize, m_HiddenSize));
    m_FCPolicy0 = register_module("m_FCPolicy0", FCLayer(m_HiddenSize, m_ActionSize));
    m_FCPolicy1 = register_module("m_FCPolicy1", FCLayer(m_HiddenSize, m_ActionSize));
    m_FCPolicy2 = register_module("m_FCPolicy2", FCLayer(m_HiddenSize, m_ActionSize));
    m_FCPolicy3 = register_module("m_FCPolicy3", FCLayer(m_HiddenSize, m_ActionSize));
    m_FCPolicy4 = register_module("m_FCPolicy4", FCLayer(m_HiddenSize, m_ActionSize));
    m_FCValue = register_module("m_FCValue", FCLayer(m_HiddenSize, 1));
}

ModelOutput A2CAgent::Forward(std::vector<float>& state)
{
    namespace Func = torch::nn::functional;

    // state to torch::Tensor
    torch::Tensor stateTensor = torch::from_blob(state.data(), { 1, m_StateSize }).to(m_Device);

    // Normalize stateTensor
    stateTensor = (stateTensor - m_StateAverage) / m_StateSTD;

    // Forward Propagation
    torch::Tensor x = torch::tanh(m_FCCommon->forward(stateTensor));
    
    torch::Tensor prob0 = Func::softmax(m_FCPolicy0->forward(x), Func::SoftmaxFuncOptions(1));
    torch::Tensor prob1 = Func::softmax(m_FCPolicy1->forward(x), Func::SoftmaxFuncOptions(1));
    torch::Tensor prob2 = Func::softmax(m_FCPolicy2->forward(x), Func::SoftmaxFuncOptions(1));
    torch::Tensor prob3 = Func::softmax(m_FCPolicy3->forward(x), Func::SoftmaxFuncOptions(1));
    torch::Tensor prob4 = Func::softmax(m_FCPolicy4->forward(x), Func::SoftmaxFuncOptions(1));

    torch::Tensor value = m_FCValue->forward(x);

    return std::move(ModelOutput(std::vector<torch::Tensor>({ prob0, prob1, prob2, prob3, prob4 }), value));
}

const std::vector<int> A2CAgent::GetAction(std::vector<float>& state)
{
    namespace Func = torch::nn::functional;

    ModelOutput output = Forward(state);

    std::vector<int> actions;
    for (auto policy : output.GetProbPolicy())
    {
        auto action = policy.multinomial(1, true).to(torch::kInt32);  // Choose Action with Probability
        std::cout << action << std::endl;
        //auto onehotAction = Func::one_hot(action, m_ActionSize);
        //std::cout << onehotAction << std::endl;
        actions.push_back(action[0][0].item().toInt());  // Convert to int and push_back
    }
        
    return std::move(actions);
}

void A2CAgent::TrainModel(std::vector<float>& state, std::vector<int>& action, float reward, std::vector<float>& nextState)
{
    namespace Func = torch::nn::functional;

    ModelOutput output = Forward(state);
    ModelOutput nextOutput = Forward(nextState);

    std::cout << nextOutput << std::endl;

    float df[] = { m_DiscountFactor };
    float rwd[] = { reward };

    torch::Tensor torchDF = torch::from_blob(&df, { 1 }).to(m_Device);
    torch::Tensor torchReward = torch::from_blob(&rwd, { 1 }).to(m_Device);
    
    torch::Tensor torchTarget = torchReward + torchDF * nextOutput.GetValue();

    torch::Tensor torchAction = torch::from_blob(action.data(), { 1, m_ActionSize }).to(m_Device);

    //std::cout << torchAction[0] << std::endl;

    //torch::nn::functional::one_hot()
    //auto actionProb = torchAction[0] * output.GetProbPolicy()[0];

    //std::cout << actionProb << std::endl;

}
