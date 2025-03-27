#include <stdio.h>
#include "cuda_runtime.h"
#include <iostream>
#include <math.h>

#define TILE_SIZE 16

void generateRandomValues(int *arr, int rows, int cols) {
    for (int i=0 ; i < rows ; i++) {
        for (int j=0 ; j < cols ; j++) {
            arr[i * cols + j] = (int)(1 + (float)rand() / (float)(RAND_MAX / 255)); // Random float between 0 and 10
        }
	}
}

__global__ void detectEdges(int *image, int *edge, int height, int width) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    __shared__ int sharedMem[TILE_SIZE + 2][TILE_SIZE + 2];

    if (row < height && col < width) {
        sharedMem[threadIdx.y][threadIdx.x] = image[row * width + col];
    } else {
        sharedMem[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();

    if (threadIdx.x > 0 && threadIdx.y > 0 && threadIdx.x < TILE_SIZE - 1 && threadIdx.y < TILE_SIZE - 1) {
        // Top right + Center right * 2 + Bottom right - Top left - Center left * 2 - Bottom left
        int horizontalEdge = sharedMem[threadIdx.y - 1][threadIdx.x + 1] + \
                            sharedMem[threadIdx.y][threadIdx.x + 1] * 2 + \
                            sharedMem[threadIdx.y + 1][threadIdx.x + 1] - \
                            sharedMem[threadIdx.y - 1][threadIdx.x - 1] - \
                            sharedMem[threadIdx.y][threadIdx.x - 1] * 2 - \
                            sharedMem[threadIdx.y + 1][threadIdx.x - 1];
        
        // Bottom left + Bottom center * 2 + Bottom right - Top left - Top center * 2 - Top right
        int verticalEdge = sharedMem[threadIdx.y + 1][threadIdx.x - 1] + \
                            sharedMem[threadIdx.y + 1][threadIdx.x] * 2 + \
                            sharedMem[threadIdx.y + 1][threadIdx.x + 1] - \
                            sharedMem[threadIdx.y - 1][threadIdx.x - 1] - \
                            sharedMem[threadIdx.y - 1][threadIdx.x] * 2 - \
                            sharedMem[threadIdx.y - 1][threadIdx.x + 1];
        // G = (Gx^2 + Gy^2)^(1/2) --> Gx = Horizontal filter, Gy = Vertical filter
        int total = sqrtf((horizontalEdge * horizontalEdge) + (verticalEdge * verticalEdge));

        edge[row * width + col] = total;
    }

}

int main() {

    uint height = 256;
    uint width = 256;
    size_t size = height * width * sizeof(int);

    int *image_h[height][width], *edge_h[height][width];

    generateRandomValues((int *)image_h, height, width);

    int *image_d, *edge_d;

    cudaMalloc((void **)&image_d, size);
    cudaMalloc((void **)&edge_d, size);

    cudaMemcpy(image_d, image_h, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock = {TILE_SIZE, TILE_SIZE, 1};
    dim3 blocksPerGrid = {(width + TILE_SIZE - 1)/TILE_SIZE, (height + TILE_SIZE - 1)/TILE_SIZE, 1};

    detectEdges<<<blocksPerGrid, threadsPerBlock>>>(image_d, edge_d, height, width);

    cudaMemcpy(edge_h, edge_d, size, cudaMemcpyDeviceToHost);

    std::cout << "Image input (first 2 elements):\n";
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            std::cout << image_h[i][j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;

    std::cout << "Image output (first 2 elements):\n";
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            std::cout << edge_h[i][j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
    
    cudaFree(image_d); cudaFree(edge_d);

    return 0;
}