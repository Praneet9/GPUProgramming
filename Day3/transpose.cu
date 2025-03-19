#include "cuda_runtime.h"
#include <stdio.h>
#include <iostream>

void generateRandomValues(float *arr, int rows, int cols) {
    for (int i=0 ; i < rows ; i++) {
        for (int j=0 ; j < cols ; j++) {
            arr[i * cols + j] = 1 + (float)rand() / (float)(RAND_MAX / 9); // Random float between 0 and 10
        }
	}
}

__global__ void transposeKernel(float *A, float *B, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ( i < cols && j < rows ) {
        B[i * rows + j] = A[j * cols + i];
    }
}

int main() {

    int M = 3;
    int N = 4;

    size_t size = M * N * sizeof(float);
    float A_h[M][N], B_h[N][M];

    generateRandomValues((float *)A_h, M, N);

    float *A_d, *B_d;

    cudaMalloc((void **)&A_d, size);
    cudaMalloc((void **)&B_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
    
    u_int threads = 32;
    dim3 threadsPerBlock = {threads, threads, 1};
    dim3 blocksPerGrid = {(M + threads - 1)/threads, (N + threads - 1) / threads, 1};

    transposeKernel<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, M, N);

    cudaMemcpy(B_h, B_d, size, cudaMemcpyDeviceToHost);

    // Verify results
	std::cout << "A matrix: \n";
	for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j ++) {
            std::cout << A_h[i][j] << " ";
        }
        std::cout << "\n";
    }
	std::cout << std::endl;

	std::cout << "B matrix: \n";
	for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j ++) {
            std::cout << B_h[i][j] << " ";
        }
        std::cout << "\n";
    }
	std::cout << std::endl;

    return 0;
}