#include "cuda_runtime.h"
#include <stdio.h>
#include <iostream>

void generateRandomValues(int *arr, int height, int width, int channels) {
    for (int i=0 ; i < height ; i++) {
        for (int j=0 ; j < width ; j++) {
            for (int k=0 ; k < channels ; k++) {
                arr[(i * width + j) * channels + k] = rand() % 256; // Random float between 0 and 255
            }
        }
	}
}

__global__ void grayscaleKernel(int *img, int *src, int height, int width, int channels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < height && j < width) {
        int idx = (i * width + j) * channels;
        // Assuming 3 channels here
        // R * 0.2989 + G * 0.5870 + B * 0.1140
        img[i * width + j] = (int)(src[idx + 0] * 0.2989 + src[idx + 1] * 0.5870 + src[idx + 2] * 0.1140);
    }
}

int main() {

    int width = 256;
    int height = 256;
    int channels = 3;
    int grayscale_h[height][width], src_h[height][width][channels];

    size_t size = width * height * channels * sizeof(int);

    generateRandomValues((int *)src_h, height, width, channels);

    int *grayscale_d, *src_d;

    cudaMalloc((void **)&grayscale_d, size / channels);
    cudaMalloc((void **)&src_d, size);

    cudaMemcpy(src_d, src_h, size, cudaMemcpyHostToDevice);

    u_int threads = 32;
    dim3 threadsPerBlock = {threads, threads, 1};
    dim3 blocksPerGrid = {(height + threads - 1)/threads, (width + threads - 1)/threads, 1};

    grayscaleKernel<<<blocksPerGrid, threadsPerBlock>>>(grayscale_d, src_d, height, width, channels);

    cudaMemcpy(grayscale_h, grayscale_d, size / channels, cudaMemcpyDeviceToHost);

    // Verify results
	std::cout << "A first index: ";
	for (int i = 0; i < channels; i++) {
        std::cout << src_h[0][0][i] << " ";
    }
	std::cout << std::endl;

	std::cout << "Grayscale first index: ";
	std::cout << grayscale_h[0][0] << " ";
	std::cout << std::endl;

    cudaFree(grayscale_d);
    cudaFree(src_d);

    return 0;
}