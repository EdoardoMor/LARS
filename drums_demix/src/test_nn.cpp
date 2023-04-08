#include "NeuralNetwork.h"
#include <iostream>


int main()
{
    
    //NeuralNetwork nn{1, 2};
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

   /*
    for (float i=0;i<10; ++i){
        nn.addTrainingData({i/10}, {i/5, i/3});
    }
    nn.runTraining(10);
    return 0;
    */

   /*
    int n_fft_test = 512;
    int hop_length_test = 256;
    int win_length_test = 512;

    at::Tensor x = torch::randn({ 1377648 });

    std::cout << "test" << x.sizes() << std::endl;

    torch::Tensor hann_win = torch::hann_window(win_length_test, true, at::TensorOptions().dtype(at::kFloat).requires_grad(false));



    // reflect padding


    int length = x.sizes()[0];
    int pad = length / 2;

    torch::Tensor left_pad = x.slice(0, 1, pad + 1).flip(0);
    torch::Tensor right_pad = x.slice(0, length - pad - 1, length - 1).flip(0);
    torch::Tensor x2 = torch::cat({ left_pad, x, right_pad }, 0);

    torch::Tensor y = torch::stft(x, n_fft_test, hop_length_test, win_length_test, hann_win, true, "reflect", false, false, true);


    std::cout << "STFT calculated" << y.sizes()  << std::endl;

    std::cout << "dims" << " " << y.sizes()[0] << " " << y.sizes()[1] << " " << y.sizes()[2] << " " << y.sizes()[3] << std::endl;

    */

    
    torch::Tensor testTensor = torch::randn({ 1,2,10,20 });

    std::cout << "testTensor sizes" << testTensor.sizes() << std::endl;

    //std::cout << "testTensor dims" << " " << testTensor.sizes()[0] << " " << testTensor.sizes()[1] << " " << testTensor.sizes()[2] << " " << testTensor.sizes()[3] << std::endl;

    c10::IntArrayRef test_shape = testTensor.sizes();

    std::cout << "test_shape: " << test_shape << std::endl;


    c10::IntArrayRef sliced_test_shape1 = test_shape.slice(0, test_shape.size() -1);

    std::cout << "sliced_test_shape1: " << sliced_test_shape1 << std::endl;
    
    c10::IntArrayRef sliced_test_shape2 = test_shape.slice(test_shape.size() - 2, 2);

    std::cout << "sliced_test_shape2: " << sliced_test_shape2 << std::endl;

        std::cout << "sliced_test_shape2 size: " << sliced_test_shape2.size() << std::endl;

    //torch::Tensor y = torch::stft(x2, n_fft_test, hop_length_test, win_length_test);

}
