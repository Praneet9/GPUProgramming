#include <cuda_runtime.h>
#define BLOCK_SIZE 256

__global__ void reduction_kernel(int* input, int* output, int length) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__ int sdata[];

	sdata[threadIdx.x] = (idx < length) ? input[idx] : 0;
    __syncthreads();

	for(int stride=blockDim.x / 2; stride > 0; stride=stride/2) {
		if (threadIdx.x < stride) {
			sdata[threadIdx.x] += sdata[threadIdx.x + stride];
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		output[blockIdx.x] = sdata[0];
	}
}

void launch_reduction_kernel(int* input_ptr, int* output_ptr, int length) {

	int threads = 1024;
	dim3 blocks = {(length + threads - 1) / threads};

	reduction_kernel<<<blocks, threads, threads * sizeof(int)>>>(input_ptr, output_ptr, length);
}