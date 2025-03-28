#include <stdio.h>
#include "cuda_runtime.h"
#include <iostream>

#define TILE_SIZE 16

void generateRandomValues(int *arr, int elements) {
    for (int i=0 ; i < elements ; i++) {
		arr[i] = (int)(1 + (float)rand() / (float)(RAND_MAX / 255));
	}
}

int main () {
	int N = 256;
	size_t size = N * sizeof(int);

	int A_h[N];

	generateRandomValues(A_h, N);

	int *A_d;

	cudaMalloc((void **)&A_d, size);

	cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);


}