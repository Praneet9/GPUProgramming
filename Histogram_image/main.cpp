#include <torch/torch.h>
#include <iostream>

void launch_hist_image_kernel(int* input_ptr, int* output_ptr, int width, int height, int channels, int min_score, int max_score, int n_bins);

int main() {

	int width = 256;
	int height = 512;
	int channels = 3;
	int min_score = 0;
	int max_score = 255;
	int n_bins = 255 + 1;

	at::Tensor input_tensor = torch::randint(min_score, max_score, {height, width, channels}, torch::dtype(torch::kInt32).device(torch::kCUDA));
	at::Tensor output_tensor = torch::zeros({n_bins * channels}, torch::dtype(torch::kInt32).device(torch::kCUDA));

	launch_hist_image_kernel(input_tensor.data_ptr<int>(), output_tensor.data_ptr<int>(), width, height, channels, min_score, max_score, n_bins);

	std::cout << "CUDA Kernel output:" << output_tensor.cpu() << std::endl;
}