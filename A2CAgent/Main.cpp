#include <iostream>
#include <vector>

#include "A2CAgent.h"

void PrintVector(const std::vector<int>& vec)
{
    for (const auto& value : vec)
    {
        std::cout << value << " ";
    }
    std::cout << std::endl;
}

int main()
{
    A2CAgent agent(5, 6, 5, 0.99f);

    int act[] = { 3 };
    const torch::Tensor torchAct = torch::from_blob(&act, { 1 }, torch::kInt32);
    //torchAct.
    torch::Tensor target = torch::randint(0, 10, { 10, });
    std::cout << target << std::endl;
    //std::cout << torchAct << std::endl;
    auto onehotAction = torch::nn::functional::one_hot(target, 10);
    std::cout << onehotAction << std::endl;

    //std::cout << onehotAction << std::endl;

    std::vector<float> state({ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f });
    agent.SetStateNormalizeValue(3.0f, 2.0f);

    //ModelOutput output = agent.Forward(state);
    //std::cout << output << std::endl;
    std::vector<int> action = agent.GetAction(state);
    //std::cout << "===== Action =====\n" << action << std::endl;

    //std::vector<float> nextState({ 5.0f, 4.0f, 3.0f, 2.0f, 1.0f });

    //float reward = 1.5f;
    //agent.TrainModel(state, action, reward, nextState);
    
    //for (const auto& policy : output.GetProbPolicy())
    //{
    //    std::cout << "policy:\n";
    //    std::cout << policy << std::endl;
    //}
    //std::cout << "value: " << output.GetValue();

    //std::vector<int> action = agent.GetAction(state);

    //PrintVector(action);

    std::cin.get();
}
