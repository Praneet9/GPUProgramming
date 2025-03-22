#include "cuda_runtime.h"
#include <stdio.h>
#include <iostream>

void generateRandomArray(float *arr, int size) {
	for (int i=0 ; i < size ; i++) {
		arr[i] = (float)rand() / (float)(RAND_MAX / 100); // Random float between 0 and 100
	}
}

__global__ void euclideanNorm(float *A, float *B, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int localIdx = threadIdx.x;
	__shared__ float sharedArray[256];

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

__global void reduction(float *B, int N) {
	int idx =  blockIdx.x * blockDim.x + threadIdx.x;
	int localIdx = threadIdx.x;
	__shared__ float sharedArray[256];


}
