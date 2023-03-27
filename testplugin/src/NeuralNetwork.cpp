
#include "NeuralNetwork.h"

#include <iostream>





NeuralNetwork::NeuralNetwork(int64_t _n_inputs, int64_t _n_outputs)
: n_inputs{_n_inputs}, n_outputs{_n_outputs}
{
    linear1 = register_module( //function inherited from Module class, useful to keep track of the layers
        "linear1",
        torch::nn::Linear(n_inputs, n_outputs)
    );

    linear2 = register_module(
        "linear2", 
        torch::nn::Linear(n_outputs, n_outputs)
    );

    //sig1 = register_module("sig1", torch::nn::Sigmoid());


    softmax = register_module(
        "softmax",
        torch::nn::Softmax(1)
    );

    optimiser = std::make_unique<torch::optim::SGD>(
        this->parameters(), .01);// params, learning rate

}

std::vector<float> NeuralNetwork::forward(
        const std::vector<float>& inputs)
{
    // copy input data into a tensor
    torch::Tensor in_t = torch::empty({1, n_inputs});
    for (long i=0; i<n_inputs; ++i){
        in_t[0][i] = inputs[i];
    }
    // pass through the network:
    torch::Tensor out_t = forward(in_t);
    // copy output back out to a vector
    std::vector<float> outputs(n_outputs);// initialise size to n_outputs
    for (long i=0; i<n_outputs; ++i){
        outputs[i] = out_t[0][i].item<float>();
    }
    return outputs;
}

void NeuralNetwork::addTrainingData(
        std::vector<float>inputs,
        std::vector<float>outputs)
{

    torch::Tensor inT = torch::from_blob( (float*)(inputs.data()), inputs.size() ).clone();
    torch::Tensor outT = torch::from_blob( (float*)(outputs.data()), outputs.size() ).clone();
    trainInputs.push_back(inT);
    trainOutputs.push_back(outT);

    /* test 
    std::cout << trainInputs[0][0] << std::endl;
    std::cout << trainOutputs[0][0] << std::endl;
    */

}

void NeuralNetwork::runTraining(int epochs)
{
    // push inputs to one big tensor
    torch::Tensor inputs = torch::cat(trainInputs).reshape({(signed long) trainInputs.size(), trainInputs[0].sizes()[0]});
    // push outputs to one big tensor
    torch::Tensor outputs = torch::cat(trainOutputs).reshape({(signed long) trainOutputs.size(), trainOutputs[0].sizes()[0]});
    float loss{0}, pLoss{1000}, dLoss{1000};
    // run the training loop
    for (int i=0;i<epochs; ++i){
        // clear out the optimiser
        this->optimiser->zero_grad();
        auto loss_result = torch::mse_loss(forward(inputs), outputs);
        float loss = loss_result.item<float>();
        std::cout << "iter: " << i << "loss " << loss << std::endl;
        // Backward pass
        loss_result.backward();
        // Apply gradients
        this->optimiser->step();
        // now compute the change in loss
        // and exit if it is too low
        dLoss = pLoss - loss; 
        pLoss = loss; // prev loss
        if (i > 0){// only after first iteration
            if (dLoss < 0.000001) {
                std::cout << "dLoss very low, exiting at iter " << i << std::endl;
                break;
            }
        }
    }


}
torch::Tensor NeuralNetwork::forward(const torch::Tensor& input)
{
    //std::cout << "forward input " << input << std::endl;
    
    torch::Tensor out = linear1(input);
    //std::cout << "forward after linear1 " << out << std::endl;
    
    out = softmax(out);
    //std::cout << "forward after softmax " << out << std::endl;

    //out = sig1(out);

    out = linear2(out);

    
    return out;

}
