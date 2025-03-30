#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 16

__global__ void dilateKernel(unsigned char *input, unsigned char *output, int width, int height) {
    
}

__global__ void erodeKernel(unsigned char *input, unsigned char *output, int width, int height) {
    
}

__global__ void binarizeKernel(unsigned char *input, unsigned char *output, int threshold) {
    
}

void imageDilation(unsigned char *input, unsigned char *output, int width, int height) {
    unsigned char *d_input, *d_output;
    size_t imageSize = width * height * sizeof(unsigned char);

    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_output, imageSize);

    cudaMemcpy(d_input, input, imageSize, cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
}