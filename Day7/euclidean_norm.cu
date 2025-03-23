#include "cuda_runtime.h"
#include <stdio.h>
#include <iostream>
#include <cmath>

void generateRandomArray(float *arr, int size) {
	for (int i=0 ; i < size ; i++) {
		arr[i] = (float)rand() / (float)(RAND_MAX / 100); // Random float between 0 and 100
	}
}

__global__ void euclideanNorm(float *A, float *B, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int localIdx = threadIdx.x;
	extern __shared__ float sharedArray[];

	if (idx < N) {
		sharedArray[localIdx] = A[idx] * A[idx];
	} else {
		sharedArray[localIdx] = 0.0f;
	}

	__syncthreads();

	for (int stride = blockDim.x / 2 ; stride > 0 ; stride /= 2) {
		if (localIdx < stride) {
			sharedArray[localIdx] += sharedArray[localIdx + stride];
		}
		__syncthreads();
	}

	if (localIdx == 0) {
		B[blockIdx.x] = sharedArray[0];
	}
}

__global__ void reduction(float *B, int N) {
	int idx =  blockIdx.x * blockDim.x + threadIdx.x;
	int localIdx = threadIdx.x;
	extern __shared__ float sharedArray[];

	if (idx < N) {
		sharedArray[localIdx] = B[idx];
	} else {
		sharedArray[localIdx] = 0;
	}

	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
		sharedArray[localIdx] += sharedArray[localIdx + stride];
	}
	__syncthreads();

	if (localIdx == 0) {
		B[blockDim.x] = sharedArray[0];
	}
}

int main () {

	// Initialize host arrays and constants
	int SIZE = 256;
	float A_h[SIZE], B_h[SIZE];
	size_t size = SIZE * sizeof(float);

	// Generate random data for arrays
	generateRandomArray(A_h, SIZE);

	// Initialize device arrays
	// B_d doesn't require the same size array
	// Naming conventions can be improved
	float *A_d, *B_d;

	// Cudamalloc arrays
	cudaMalloc((void **)&A_d, size);
	cudaMalloc((void **)&B_d, size);

	// Cudamemcopy host array to device array
	cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);

	// Invoke kernel
	uint threads = 16;
	dim3 threadsPerBlock = {threads, 1, 1};
	uint blocks = (SIZE + threads - 1) / threads;
	uint newBlocks = blocks;
	dim3 blocksPerGrid = {blocks, 1, 1};

	euclideanNorm<<<blocksPerGrid, threadsPerBlock, threads * size>>>(A_d, B_d, SIZE);

	while (blocks > 1) {
		newBlocks = (blocks + threads - 1) / threads;
		reduction<<<newBlocks, threadsPerBlock, threads * size>>>(B_d, SIZE);
		blocks = newBlocks;
	}

	// Cudamemcpy device array to host
	cudaMemcpy(B_h, B_d, size, cudaMemcpyDeviceToHost);

	float norm = sqrt(B_h[0]);
	// Verify / Print output array
	std::cout << "Norm: " << norm << std::endl;

	// Cuda free arrays
	cudaFree(A_d);
	cudaFree(B_d);
}