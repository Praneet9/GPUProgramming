#include "cuda_runtime.h"
#include <iostream>
#include <stdlib.h>

__global__ void convolution(int *input, int *output, int height, int width, int channels) {
    // Compute global threadIdx and blockIdx

    // Define shared memory for the operation (3 channel array?)

    // Copy data to shared memory array

    // 3 loops (nested) for the convolution operation

    // Sum across channels on convolution

    // Copy computed value back to output
}

void generateRandomValues(int *arr, int rows, int cols, int channels) {
    for (int i=0 ; i < rows ; i++) {
        for (int j=0 ; j < cols ; j++) {
            for (int k=0 ; k < channels ; k++) {
                arr[i * cols + j + k] = (int)(1 + (float)rand() / (float)(RAND_MAX / 256)); // Random float between 0 and 10
            }
        }
	}
}

int main() {
    // Define configs/variables
    int height = 256;
    int width = 256;
    int channels = 3;
    int kernel_dim = 3;

    // Initial image and kernel
    // (Kernel should be of same number of channels as input)
    // Output channels will be 1 as we are using only one kernel
    int image_h[height][width][channels], output_h[height][width];
    int kernel_h[kernel_dim][kernel_dim];

    // Generate random values
    generateRandomValues((int *)image_h, height, width, channels);
    generateRandomValues((int *)output_h, height, width, 1);
    generateRandomValues((int *)kernel_h, kernel_dim, kernel_dim, 1);

    // TODO - Read about Memory Coalescing - Storing the image in channel first format for better memory access efficiency

    // Define device variables
    int *image_d, *output_d, *kernel_d;

    // Allocate memory for device variables
    cudaMalloc((void **)&image_d, height * width * channels * sizeof(int));
    cudaMalloc((void **)&output_d, height * width * channels * sizeof(int));
    cudaMalloc((void **)&kernel_d, kernel_dim * kernel_dim * sizeof(int));

    // Copy memory from host to device for image and kernel
    cudaMemcpy(image_d, image_h, height * width * channels * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(output_d, output_h, height * width * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_d, kernel_h, kernel_dim * kernel_dim * sizeof(int), cudaMemcpyHostToDevice);

    // Define threads and blocks size

    // Execute kernel

    // Copy output from device to host

    // Verify output

    // Free resources
}