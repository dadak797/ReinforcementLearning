#include <iostream>
#include <vector>
#include <fstream>
#include "A2CAgent.h"


int main()
{
    torch::manual_seed(1);

    A2CAgent agent(5, 6, 3, /*df*/0.99f, /*lr*/0.01f, /*beta1*/0.9f, /*beta2*/0.999f);
    int maxIter = 100;

    std::vector<float> lossList;

    std::ifstream fin("A2CAgent.pt");
    if (fin.is_open())
    {
        torch::load(agent, "A2CAgent.pt");
        std::cout << "File Exist" << std::endl;
    }

    //std::ifstream fopti("Optimizer.pt");
    //if (fopti.is_open())
    //{
    //    torch::load(*(agent->GetOptimizer()), "Optimizer.pt");
    //    std::cout << "Optimizer File Exist" << std::endl;            
    //}

    for (int i = 0; i < maxIter; i++)
    {
        torch::manual_seed(1);

        std::vector<float> state({ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f });
        agent->SetStateNormalizeValue(3.0f, 2.0f);

        std::vector<int> action = agent->GetAction(state);

        std::vector<float> nextState({ 5.0f, 4.0f, 3.0f, 2.0f, 1.0f });

        float reward = 1.5f;
        lossList.push_back(agent->TrainModel(state, action, reward, nextState));
    }

    torch::save(agent, "A2CAgent.pt");
    //torch::save(*(agent->GetOptimizer()), "Optimizer.pt");

    for (int i = 0; i < maxIter; i++)
    {
        std::cout << i << "-th Iteration, Loss: " << lossList[i] << std::endl;
    }
}
