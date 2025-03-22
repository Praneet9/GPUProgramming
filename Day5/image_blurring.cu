#include "cuda_runtime.h"
#include <stdio.h>
#include <iostream>

void generateRandomValues(int *arr, int height, int width) {
    for (int i=0 ; i < height ; i++) {
        for (int j=0 ; j < width ; j++) {
            arr[i * width + j] = rand() % 256; // Random float between 0 and 255
        }
	}
}

__global__ void blurringKernel(int *img, int *src, int height, int width, int kernel_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int adj_idx = 0;

    if (i < height && j < width) {
        int idx = i * width + j;
        // Assuming 3 channels here
        int sum = 0;
        int count = 0;
        for (int w = -1; w < kernel_size - 1 ; w++) {
            for (int h = -1; h < kernel_size - 1 ; h++) {
                // conditional statement to check
                adj_idx = idx + w * kernel_size + h * kernel_size;
                if (adj_idx >= 0 && adj_idx <= width && adj_idx <= height) {
                    count++;
                    sum += src[idx + w * kernel_size + h * kernel_size];
                }
            }
        }
        img[idx] = (int)(sum / count);
    }
}

int main() {

    int width = 256;
    int height = 256;
    int kernel_size = 3;
    int grayscale_h[height][width], src_h[height][width];

    size_t size = width * height * sizeof(int);

    generateRandomValues((int *)src_h, height, width);

    int *grayscale_d, *src_d;

    cudaMalloc((void **)&grayscale_d, size);
    cudaMalloc((void **)&src_d, size);

    cudaMemcpy(src_d, src_h, size, cudaMemcpyHostToDevice);

    u_int threads = 32;
    dim3 threadsPerBlock = {threads, threads, 1};
    dim3 blocksPerGrid = {(height + threads - 1)/threads, (width + threads - 1)/threads, 1};

    blurringKernel<<<blocksPerGrid, threadsPerBlock>>>(grayscale_d, src_d, height, width, kernel_size);

    cudaMemcpy(grayscale_h, grayscale_d, size, cudaMemcpyDeviceToHost);

    // Verify results
    std::cout << "Original (0,0): " << src_h[0][0] << std::endl;
    std::cout << "Blurred (0,0): " << grayscale_h[0][0] << std::endl;

    cudaFree(grayscale_d);
    cudaFree(src_d);

    return 0;
}