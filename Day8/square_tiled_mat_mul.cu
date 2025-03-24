#include <cuda_runtime.h>
#include <iostream>

#define TILE_SIZE 16

__global__ void matrixMulShared(float* A, float* B, float* C, int N) {
    __shared__ float Asub[TILE_SIZE][TILE_SIZE];
    __shared__ float Bsub[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

}

void generateRandomValues(float *arr, int rows, int cols) {
    for (int i=0 ; i < rows ; i++) {
        for (int j=0 ; j < cols ; j++) {
            arr[i * cols + j] = 1 + (float)rand() / (float)(RAND_MAX / 9); // Random float between 0 and 10
        }
	}
}

int main() {
    int N = 512;  // Matrix size (N x N)
    size_t sizeMat = N * N * sizeof(float);

    // Create matrices on Host
    float A_h[N][N], B_h[N][N], C_h[N][N];

    generateRandomValues((float *)A_h, N, N);
    generateRandomValues((float *)B_h, N, N);

    float *A_d, *B_d, *C_d;
    cudaMalloc(&A_d, sizeMat);
    cudaMalloc(&B_d, sizeMat);
    cudaMalloc(&C_d, sizeMat);

    cudaMemcpy(A_d, A_h, sizeMat, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, sizeMat, cudaMemcpyHostToDevice);

}
