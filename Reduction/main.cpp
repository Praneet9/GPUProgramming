#include <torch/torch.h>
#include <iostream>

void launch_reduction_kernel(int* input_ptr, int* output_ptr, int length);

int main() {

	int length = 1024;

	at::Tensor input_tensor = torch::randint(0, 100, {length}, torch::dtype(torch::kInt32).device(torch::kCUDA));
	at::Tensor output = torch::tensor(0, torch::dtype(torch::kInt32).device(torch::kCUDA));

	launch_reduction_kernel(input_tensor.data_ptr<int>(), output.data_ptr<int>(), length);

	std::cout << "CUDA Kernel output:" << output.cpu() << std::endl;
	std::cout << "Torch Answer: " << input_tensor.sum().cpu() << std::endl;
}