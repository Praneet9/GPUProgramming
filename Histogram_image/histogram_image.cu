#include <cuda_runtime.h>

__global__ void hist_image_kernel(int* input, int* output, int width, int height, int channels, int min_score, int max_score, int n_bins) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int channel = blockIdx.z;

	if (row >= height || col >= width || channel >= channels) return;
	int idx = (row * width + col) * channels + channel;

	int bin = min((input[idx] - min_score) * n_bins / (max_score - min_score + 1), n_bins - 1);
	atomicAdd(&(output[n_bins * channel + bin]), 1);
}

void launch_hist_image_kernel(int* input_ptr, int* output_ptr, int width, int height, int channels, int min_score, int max_score, int n_bins) {

	dim3 threads = {16, 16, 1};
	dim3 blocks = {(width + 16 - 1)/16, (height + 16 - 1)/16, channels};

	hist_image_kernel<<<blocks, threads>>>(input_ptr, output_ptr, width, height, channels, min_score, max_score, n_bins);
}