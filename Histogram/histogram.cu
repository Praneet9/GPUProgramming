#include <cuda_runtime.h>

__global__ void histogramKernel(int *input, int* output, int length, int min_score, int max_score, int n_bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= length) return;

    // Formula to bin any range into n_bins:
    // int bin = min((value - min_value) * num_bins / (max_value - min_val + 1), num_bins - 1)
    // Note: bin is converted to int from float, so 2.99 becomes 2
    int bin = min((input[idx] - min_score) * n_bins / (max_score - min_score + 1), n_bins - 1);

    atomicAdd(&(output[bin]), 1);
}

void launch_histogram_kernel(int* input_ptr, int* output_ptr, int length, int min_score, int max_score, int n_bins) {

    int threads = 256;
    int blocks = (length + threads - 1) / threads;

    histogramKernel<<<blocks, threads>>>(input_ptr, output_ptr, length, min_score, max_score, n_bins);
}