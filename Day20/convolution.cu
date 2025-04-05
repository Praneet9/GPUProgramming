#include "cuda_runtime.h"
#include <iostream>
#include <stdlib.h>

#define BLOCK_SIZE 16
#define CHANNELS 3

__global__ void convolution(int *input, int *output, int *kernel, int height, int width, int channels, int kernel_dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= height || col >= width) return;

    int half_k = kernel_dim / 2;
    int sum = 0;

    // Convolution loop
    for (int i = -half_k; i <= half_k; ++i) {
        for (int j = -half_k; j <= half_k; ++j) {
            int cur_row = row + i;
            int cur_col = col + j;

            if (cur_row >= 0 && cur_row < height && cur_col >= 0 && cur_col < width) {
                for (int c = 0; c < channels; ++c) {
                    int image_idx = (cur_row * width + cur_col) * channels + c;
                    int kernel_idx = (i + half_k) * kernel_dim + (j + half_k);
                    sum += input[image_idx] * kernel[kernel_idx];
                }
            }
        }
    }

    output[row * width + col] = sum;
}

void generateRandomValues(int *arr, int rows, int cols, int channels) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            for (int k = 0; k < channels; k++)
                arr[(i * cols + j) * channels + k] = rand() % 256;
}

int main() {
    int height = 256;
    int width = 256;
    int channels = CHANNELS;
    int kernel_dim = 3;

    size_t image_size = height * width * channels * sizeof(int);
    size_t output_size = height * width * sizeof(int);
    size_t kernel_size = kernel_dim * kernel_dim * sizeof(int);

    // Host memory
    int *image_h = (int *)malloc(image_size);
    int *output_h = (int *)malloc(output_size);
    int *kernel_h = (int *)malloc(kernel_size);

    generateRandomValues(image_h, height, width, channels);
    generateRandomValues(output_h, height, width, 1);
    generateRandomValues(kernel_h, kernel_dim, kernel_dim, 1);

    // Device memory
    int *image_d, *output_d, *kernel_d;
    cudaMalloc((void **)&image_d, image_size);
    cudaMalloc((void **)&output_d, output_size);
    cudaMalloc((void **)&kernel_d, kernel_size);

    // Copy to device
    cudaMemcpy(image_d, image_h, image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_d, kernel_h, kernel_size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    convolution<<<gridSize, blockSize>>>(image_d, output_d, kernel_d, height, width, channels, kernel_dim);

    // Copy result back
    cudaMemcpy(output_h, output_d, output_size, cudaMemcpyDeviceToHost);

    // Example verification (print some output values)
    for (int i = 0; i < 10; i++) {
        std::cout << output_h[i] << " ";
    }
    std::cout << std::endl;

    // Free memory
    cudaFree(image_d);
    cudaFree(output_d);
    cudaFree(kernel_d);
    free(image_h);
    free(output_h);
    free(kernel_h);

    return 0;
}
