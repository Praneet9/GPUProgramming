#include <cuda_runtime.h>

__global__ void histogramKernel(int *input, int* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx = idx + 2;
    // Your kernel logic here
}

void launch_histogram_kernel(int* input_ptr, int* output_ptr, int length) {

    int threads = 256;
    int blocks = (length + threads - 1) / threads;

    histogramKernel<<<blocks, threads>>>(input_ptr, output_ptr);
}