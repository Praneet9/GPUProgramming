#include <cuda_runtime.h>
#include <iostream>

#define TILE_SIZE 16

__global__ void matrixMulShared(float* A, float* B, float* C, int N) {
    __shared__ float Asub[TILE_SIZE][TILE_SIZE];
    __shared__ float Bsub[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int tile = 0; tile < N / TILE_SIZE; tile++) {
        Asub[threadIdx.y][threadIdx.x] = A[row * N + tile * TILE_SIZE + threadIdx.x];
        Bsub[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE + threadIdx.y) * N + col];
        
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += Asub[threadIdx.y][k] * Bsub[k][threadIdx.x];
        }
        
        __syncthreads();
    }

    C[row * N + col] = sum;

}

void generateRandomValues(float *arr, int rows, int cols) {
    for (int i=0 ; i < rows ; i++) {
        for (int j=0 ; j < cols ; j++) {
            arr[i * cols + j] = 1 + (float)rand() / (float)(RAND_MAX / 9); // Random float between 0 and 10
        }
	}
}

int main() {
    int N = 128;  // Matrix size (N x N)
    size_t sizeMat = N * N * sizeof(float);

    // Create matrices on Host
    float *A_h, *B_h, *C_h;
    A_h = (float*)malloc(sizeMat);
    B_h = (float*)malloc(sizeMat);
    C_h = (float*)malloc(sizeMat);

    float *A_d, *B_d, *C_d;
    cudaMalloc((void **)&A_d, sizeMat);
    cudaMalloc((void **)&B_d, sizeMat);
    cudaMalloc((void **)&C_d, sizeMat);

    cudaMemcpy(A_d, A_h, sizeMat, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, sizeMat, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid(N / TILE_SIZE, N / TILE_SIZE);

    matrixMulShared<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, N);
    cudaMemcpy(C_h, C_d, sizeMat, cudaMemcpyDeviceToHost);

    std::cout << "Sample output (first 5 elements):\n";
    for (int i = 0; i < 5; i++) {
        std::cout << C_h[i] << " ";
    }
    std::cout << std::endl;

    free(A_h); free(B_h); free(C_h);
    cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);

    return 0;
}
