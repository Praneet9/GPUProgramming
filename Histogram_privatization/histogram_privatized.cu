#include <cuda_runtime.h>

__global__ void hist_image_privatized_kernel(int* input, int* output, int width, int height, int channels, int min_score, int max_score, int n_bins) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int channel = blockIdx.z;

	extern __shared__ int sharedHist[];
	int *localHist = sharedHist + (n_bins * channels);

	for(int i=threadIdx.y * blockDim.x + threadIdx.x; i < n_bins; i+=(blockDim.x * blockDim.y)) {
		localHist[i] = 0;
	}
	__syncthreads();

	if (row >= height || col >= width || channel >= channels) return;
	int idx = (row * width + col) * channels + channel;

	int bin = min((input[idx] - min_score) * n_bins / (max_score - min_score + 1), n_bins - 1);
	atomicAdd(&(localHist[bin]), 1);
	__syncthreads();

	for (int i=threadIdx.y * blockDim.x + threadIdx.x; i < n_bins; i+=blockDim.x * blockDim.y) {
		atomicAdd(&(output[channel * n_bins + i]), localHist[i]);
	}
}

void launch_hist_image_privatized_kernel(int* input_ptr, int* output_ptr, int width, int height, int channels, int min_score, int max_score, int n_bins) {

	dim3 threads = {16, 16, 1};
	dim3 blocks = {(width + 16 - 1)/16, (height + 16 - 1)/16, channels};
	int shared_mem_bytes = n_bins * channels * sizeof(int);

	hist_image_privatized_kernel<<<blocks, threads, shared_mem_bytes>>>(input_ptr, output_ptr, width, height, channels, min_score, max_score, n_bins);
}