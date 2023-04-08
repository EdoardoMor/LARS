
#include <cmath>
#include <torch/torch.h>
#include <torch/script.h>

//N.B. forward functions pass the arguments by reference! --> I am not allowed to modify the input data
class NeuralNetwork : torch::nn::Module { //inheriting from nn::Module
    public:
        NeuralNetwork(int64_t n_inputs, int64_t n_outputs); //constructor
        std::vector<float> forward(const std::vector<float>& inputs); //will use the private forward function, works with standard vectors
        void addTrainingData( //add training ecamples (inputs, outputs)
            std::vector<float>inputs,
            std::vector<float>outputs);
        void runTraining(int epochs); //train the NN against the training examples
    private:
        int64_t n_inputs;
        int64_t n_outputs;
        std::vector<torch::Tensor> trainInputs; 
        std::vector<torch::Tensor> trainOutputs;
        
        torch::nn::Linear linear1{nullptr}; //linear network layer
        torch::nn::Linear linear2{nullptr}; //second linear network layer
        torch::nn::Sigmoid sig1{nullptr}; //sigmoid layer, not used
        torch::nn::Softmax softmax{nullptr}; //softmax layer
        torch::Tensor forward(const torch::Tensor& input); //will be used by the public forward function, uses torch Tensors

        // unique ptr so we can initialise it
        // in the constructor after building the model
        std::unique_ptr<torch::optim::SGD> optimiser; // istanziato senza aver specificato gli argomenti del costruttore, cosa che verrà fatta quando creerò il modello



};
