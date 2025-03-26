#include <stdio.h>
#include "cuda_runtime.h"
#include <iostream>

#define TILE_SIZE 16

void generateRandomValues(int *arr, int rows, int cols) {
    for (int i=0 ; i < rows ; i++) {
        for (int j=0 ; j < cols ; j++) {
            arr[i * cols + j] = (int)(1 + (float)rand() / (float)(RAND_MAX / 255)); // Random float between 0 and 10
        }
	}
}

__global__ void sharpenImage(int *image, int *sharpened, int height, int width) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    __shared__ int imageShared[TILE_SIZE + 2][TILE_SIZE + 2];

    imageShared[threadIdx.y][threadIdx.x] = image[row * width + col];

    __syncthreads();

    if (threadIdx.x > 0 && threadIdx.y > 0 && threadIdx.x < TILE_SIZE - 1 && threadIdx.y < TILE_SIZE - 1) {
        int sum = 5 * imageShared[threadIdx.y][threadIdx.x]
                    - imageShared[threadIdx.y - 1][threadIdx.x]
                    - imageShared[threadIdx.y - 1][threadIdx.x - 1]
                    - imageShared[threadIdx.y][threadIdx.x]
                    - imageShared[threadIdx.y][threadIdx.x - 1];
        
        sharpened[row * width + col] = sum;
    }
}

int main() {
    u_int height = 256;
    u_int width = 256;
    size_t size = height * width * sizeof(int);

    int image_h[height][width], sharpened_h[height][width];

    generateRandomValues((int *)image_h, height, width);

    int *image_d, *sharpened_d;

    cudaMalloc((void **)&image_d, size);
    cudaMalloc((void **)&sharpened_d, size);
    
    cudaMemcpy(image_d, image_h, size, cudaMemcpyHostToDevice);

    // Kernel
    dim3 threadsPerBlock = {TILE_SIZE, TILE_SIZE, 1};
    dim3 blocksPerGrid = {width / TILE_SIZE, height / TILE_SIZE, 1};
    sharpenImage<<<blocksPerGrid, threadsPerBlock>>>(image_d, sharpened_d, height, width);

    // Verify
    cudaMemcpy(sharpened_h, sharpened_d, size, cudaMemcpyDeviceToHost);

    std::cout << "Image input (first 5 elements):\n";
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            std::cout << image_h[i][j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;

    std::cout << "Image output (first 5 elements):\n";
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            std::cout << sharpened_h[i][j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
    
    cudaFree(image_d); cudaFree(sharpened_d);

    return 0;
}