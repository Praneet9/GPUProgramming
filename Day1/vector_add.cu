#include "cuda_runtime.h"
#include <stdio.h>
#include <iostream>

void generateRandomArray(float *arr, int size) {
	for (int i=0 ; i < size ; i++) {
		arr[i] = (float)rand() / (float)(RAND_MAX / 100); // Random float between 0 and 100
	}
}

__global__ void vectorAdd(float *A, float *B, float *C, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N) {
		C[idx] = A[idx] + B[idx];
	}
}

int main() {
	int N = 1000;
	size_t size = N * sizeof(float);

	float *A_h = (float *)malloc(size);
	float *B_h = (float *)malloc(size);
	float *C_h = (float *)malloc(size);

	generateRandomArray(A_h, N);
	generateRandomArray(B_h, N);

	float *A_d, *B_d, *C_d;
	cudaMalloc((void **)&A_d, size);
	cudaMalloc((void **)&B_d, size);
	cudaMalloc((void **)&C_d, size);
	
	cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
	vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, N);

	cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    // Verify results
	std::cout << "A values: ";
	for (int i = 0; i < 5; i++) { // Print first 5 results
        std::cout << A_h[i] << " ";
    }
	std::cout << std::endl;
	std::cout << "B values: ";
	for (int i = 0; i < 5; i++) { // Print first 5 results
        std::cout << B_h[i] << " ";
    }
	std::cout << std::endl;
	std::cout << "C values: ";
	for (int i = 0; i < 5; i++) { // Print first 5 results
        std::cout << C_h[i] << " ";
    }
	std::cout << std::endl;

	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);

	return 0;
}