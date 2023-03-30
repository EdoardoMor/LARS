#include "NeuralNetwork.h"
#include <iostream>


int main()
{
    NeuralNetwork nn{1, 2};
    /* testing forward()
    for (float i=0;i<10; ++i){
        std::cout << nn.forward({i/10}) << std::endl;
    }
    return 0;
    */

    /* testing addTrainingData()
    nn.addTrainingData({0.1}, {2.2, 3.5});
    nn.addTrainingData({0.8}, {1.2, 5.5});
    nn.addTrainingData({2.8}, {3.4, 1.6});
    */

    for (float i=0;i<10; ++i){
        nn.addTrainingData({i/10}, {i/5, i/3});
    }
    nn.runTraining(10);
    return 0;

}
