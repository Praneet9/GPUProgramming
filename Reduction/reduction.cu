#include <cuda_runtime.h>

__global__ void reduction_kernel(int* input, int* output, int length) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	for(int stride=1; stride <= blockDim.x; stride*=2) {
		if (idx % stride == 0 && idx + stride < length) {
			input[idx] += input[idx + stride];
		}
		__syncthreads();
	}

	if (idx == 0) {
		*output = input[0];
	}
}

void launch_reduction_kernel(int* input_ptr, int* output_ptr, int length) {

	int threads = 256;
	dim3 blocks = {(length + threads - 1) / threads};

	reduction_kernel<<<blocks, threads>>>(input_ptr, output_ptr, length);
}