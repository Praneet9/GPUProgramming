#include <torch/torch.h>
#include <iostream>

void launch_histogram_kernel(int* input_ptr, int* output_ptr, int length, int min_score, int max_score, int n_bins);  // declare it

int main() {
    int length = 1024;
    int min_score = 0;
    int max_score = 1000;
    int n_bins = 10;

    at::Tensor input_tensor = torch::randint(min_score, max_score, {length}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    at::Tensor output_tensor = torch::zeros({n_bins}, torch::dtype(torch::kInt32).device(torch::kCUDA));

    launch_histogram_kernel(input_tensor.data_ptr<int>(), output_tensor.data_ptr<int>(), input_tensor.size(0), min_score, max_score, n_bins);

    std::cout << "CUDA Kernel output:\n" << output_tensor.cpu() << std::endl;
    auto output = torch::histogram(input_tensor.toType(torch::kFloat32).cpu(), n_bins);

    at::Tensor bin_counts = std::get<0>(output);
    at::Tensor bin_edges  = std::get<1>(output);

    std::cout << "\nTorch output: \n" << "Bin counts:\n" << bin_counts << "\n";
    std::cout << "Bin edges:\n"  << bin_edges  << "\n";
}