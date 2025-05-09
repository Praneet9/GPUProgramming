#include <torch/torch.h>
#include <iostream>

void launch_histogram_kernel(int* input_ptr, int* output_ptr, int length);  // declare it

int main() {
    int length = 1024;

    at::Tensor input_tensor = torch::randint(0, 1000, {length}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    at::Tensor output_tensor = torch::zeros({10}, torch::dtype(torch::kInt32).device(torch::kCUDA));

    launch_histogram_kernel(input_tensor.data_ptr<int>(), output_tensor.data_ptr<int>(), input_tensor.size(0));

    std::cout << input_tensor.cpu() << std::endl;
}