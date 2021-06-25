#include "A2CAgent.h"


std::ostream& operator<<(std::ostream& os, const ModelOutput& output)
{
    std::cout << "Action Probability\n" << output.m_ProbPolicy;
    std::cout << "\nValue Function\n" << output.m_Value;

    return os;
}

A2CAgentImpl::A2CAgentImpl(int stateSize, int hiddenSize, int actionSize, float discountFactor, float lr, float beta1, float beta2)
    : m_StateSize(stateSize), m_HiddenSize(hiddenSize), m_ActionSize(actionSize)
    , m_FCCommon(nullptr), m_FCPolicy0(nullptr), m_FCPolicy1(nullptr), m_FCPolicy2(nullptr), m_FCPolicy3(nullptr), m_FCPolicy4(nullptr)
    , m_FCValue(nullptr), m_StateAverage(0.0f), m_StateSTD(1.0f), m_Device(torch::kCPU)
    , m_DiscountFactor(discountFactor), m_SmallValue(1.e-5), m_Alpha(0.5f), m_LearningRate(lr), m_Beta1(beta1), m_Beta2(beta2)
{
    m_FCCommon = register_module("m_FCCommon", FCLayer(m_StateSize, m_HiddenSize));
    m_FCPolicy0 = register_module("m_FCPolicy0", FCLayer(m_HiddenSize, m_ActionSize));
    m_FCPolicy1 = register_module("m_FCPolicy1", FCLayer(m_HiddenSize, m_ActionSize));
    m_FCPolicy2 = register_module("m_FCPolicy2", FCLayer(m_HiddenSize, m_ActionSize));
    m_FCPolicy3 = register_module("m_FCPolicy3", FCLayer(m_HiddenSize, m_ActionSize));
    m_FCPolicy4 = register_module("m_FCPolicy4", FCLayer(m_HiddenSize, m_ActionSize));
    m_FCValue = register_module("m_FCValue", FCLayer(m_HiddenSize, 1));

    m_Optimizer = std::make_unique<torch::optim::Adam>(this->parameters(), torch::optim::AdamOptions().lr(m_LearningRate).betas({ m_Beta1, m_Beta2 }));
}

ModelOutput A2CAgentImpl::Forward(std::vector<float>& state)
{
    namespace Func = torch::nn::functional;

    // state to torch::Tensor
    torch::Tensor stateTensor = torch::from_blob(state.data(), { 1, m_StateSize }).to(m_Device);

    // Normalize stateTensor
    stateTensor = (stateTensor - m_StateAverage) / m_StateSTD;

    // Forward Propagation
    torch::Tensor x = torch::tanh(m_FCCommon->forward(stateTensor));
    
    torch::Tensor prob0 = torch::softmax(m_FCPolicy0->forward(x), 1);
    torch::Tensor prob1 = torch::softmax(m_FCPolicy1->forward(x), 1);
    torch::Tensor prob2 = torch::softmax(m_FCPolicy2->forward(x), 1);
    torch::Tensor prob3 = torch::softmax(m_FCPolicy3->forward(x), 1);
    torch::Tensor prob4 = torch::softmax(m_FCPolicy4->forward(x), 1);

    torch::Tensor value = m_FCValue->forward(x);

    return std::move(ModelOutput(torch::cat({ prob0, prob1, prob2, prob3, prob4 }, 0), value.squeeze()));  // Concatenate probs to row(0)
}

const std::vector<int> A2CAgentImpl::GetAction(std::vector<float>& state)
{
    ModelOutput output = Forward(state);

    torch::Tensor action = output.GetProbPolicy().multinomial(1, true).to(torch::kInt32);
    std::vector<int> actions(action.data_ptr<int>(), action.data_ptr<int>() + action.numel());

    return std::move(std::vector<int>(action.data_ptr<int>(), action.data_ptr<int>() + action.numel()));  // torch::Tensor to std::vector<int>
}

float A2CAgentImpl::TrainModel(std::vector<float>& state, std::vector<int>& action, float reward, std::vector<float>& nextState)
{
    namespace Func = torch::nn::functional;

    ModelOutput output = Forward(state);
    ModelOutput nextOutput = Forward(nextState);

    torch::Tensor torchAction = torch::from_blob(action.data(), { static_cast<int>(action.size()) }, torch::kInt32).to(m_Device); // { NumberOfActions(5) }
    torch::Tensor oneHotAction = Func::one_hot(torchAction.to(torch::kInt64), m_ActionSize).to(torch::kInt32);  // { NumberOfActions(5), ActionSize }
    torch::Tensor actionProb = (oneHotAction * output.GetProbPolicy()).sum({ 1 });  // { NumberOfActions(5) }

    torch::Tensor torchTarget = reward + m_DiscountFactor * nextOutput.GetValue();
    torch::Tensor torchAdvantage = (torchTarget - output.GetValue()).detach();

    torch::Tensor actorLoss = -torch::log(actionProb + m_SmallValue).mean({ 0 }) * torchAdvantage;

    torch::Tensor criticLoss = 0.5 * torch::square(torchTarget.detach() - output.GetValue());

    torch::Tensor loss = m_Alpha * actorLoss + (1.0f - m_Alpha) * criticLoss;

    m_Optimizer->zero_grad();  // Clear Gradients and Initialize Them to Zeros
    loss.backward();  // Backpropagation
    m_Optimizer->step();  // Update Parameters

    return loss.item().toFloat();
}
