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

__global__ void matrixMul(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        float total = 0;
        for (int i=0 ; i < N ; i++) {
            total += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = total;
    }
}

int main() {
    int M = 3;
    int N = 4;
    int K = 5;

    size_t size_A = M * N * sizeof(float);
    size_t size_B = N * K * sizeof(float);
    size_t size_C = M * K * sizeof(float);

    // Create matrices on Host
    float A_h[M][N], B_h[N][K], C_h[M][K];

    generateRandomValues((float *)A_h, M, N);
    generateRandomValues((float *)B_h, N, K);

    float *A_d, *B_d, *C_d;

    cudaMalloc((void **)&A_d, size_A);
    cudaMalloc((void **)&B_d, size_B);
    cudaMalloc((void **)&C_d, size_C);

    cudaMemcpy(A_d, A_h, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size_B, cudaMemcpyHostToDevice);

    u_int threads = 16;
    dim3 threadsPerBlock = {threads, threads, 1};
    dim3 blocksPerGrid = {(K + threads - 1) / threads, (M + threads - 1) / threads};
    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, M, N, K);

    cudaMemcpy(C_h, C_d, size_C, cudaMemcpyDeviceToHost);

    // Verify results
	std::cout << "A first row values: ";
	for (int i = 0; i < N; i++) {
        std::cout << A_h[0][i] << " ";
    }
	std::cout << std::endl;
	std::cout << "B first column values: ";
	for (int i = 0; i < N; i++) {
        std::cout << B_h[i][0] << " ";
    }
	std::cout << std::endl;
	std::cout << "C first element: " << C_h[0][0] << " ";
	std::cout << std::endl;

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return 0;
}