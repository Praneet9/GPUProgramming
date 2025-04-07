#include "cuda_runtime.h"
#include <iostream>
#include <stdlib.h>

#define BLOCK_SIZE 16
#define CHANNELS 3

__global__ void convolutionShared(float *input, float *output, float *kernel, int height, int width, int channels, int kernelDim, int kernelCenter) {
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int localRow = threadIdx.y;
    int localCol = threadIdx.x;

	__shared__ float sharedMem[3][3];
    __shared__ float tile[3][BLOCK_SIZE + 2][BLOCK_SIZE + 2];

	if (threadIdx.x < kernelDim && threadIdx.y < kernelDim) {
		sharedMem[threadIdx.y][threadIdx.x] = kernel[threadIdx.y * kernelDim + threadIdx.x];
		__syncthreads();
	}

    for (int c=0; c < channels; c++) {
        int sharedRow = localRow - kernelCenter;
        int sharedCol = localCol - kernelCenter;

        if (sharedRow >= 0 && sharedRow < height && sharedCol >= 0 && sharedCol < width) {
            tile[c][localRow][localCol] = input[(sharedRow * width + sharedCol) * channels + c];
        } else {
            tile[c][localRow][localCol] = 0.0f;
        }
    }
    __syncthreads();

	float sum = 0;

	for(int i=0; i<=kernelCenter*2; i++) {
		for(int j=0; j<=kernelCenter*2; j++) {
			int rowIdx = localRow + j;
			int colIdx = localCol + i;
			// int kernelIdx = (kernelCenter + i) * kernelDim + (kernelCenter + j);

			for (int c=0; c < channels; c++) {
				if (rowIdx >= 0 && rowIdx < height && colIdx >= 0 && colIdx < width) {
					// int imageIdx = (rowIdx * width + colIdx) * channels + c;
					sum += tile[c][rowIdx][colIdx] * sharedMem[j][i];
				}
			}

		}
	}

	if (row < height && col < width) {
		output[row * width + col] = sum;
	}
}

void generateRandomValues(float *arr, int rows, int cols, int channels) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            for (int k = 0; k < channels; k++)
                arr[(i * cols + j) * channels + k] = (float)rand() / RAND_MAX;
}

int main() {
    int height = 256;
    int width = 256;
    int channels = CHANNELS;
    int kernelDim = 3;
	int kernelCenter = (kernelDim - 1)/2;

    size_t imageSize = height * width * channels * sizeof(float);
    size_t outputSize = height * width * sizeof(float);
    size_t kernelSize = kernelDim * kernelDim * sizeof(float);

    // Host memory
    float *image_h = (float *)malloc(imageSize);
    float *output_h = (float *)malloc(outputSize);
    float *kernel_h = (float *)malloc(kernelSize);

    generateRandomValues((float *)image_h, height, width, channels);
    generateRandomValues((float *)output_h, height, width, 1);
    generateRandomValues((float *)kernel_h, kernelDim, kernelDim, 1);

    // Device memory
    float *image_d, *output_d, *kernel_d;
    cudaMalloc((void **)&image_d, imageSize);
    cudaMalloc((void **)&output_d, outputSize);
    cudaMalloc((void **)&kernel_d, kernelSize);

    // Copy to device
    cudaMemcpy(image_d, image_h, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_d, kernel_h, kernelSize, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    convolutionShared<<<gridSize, blockSize>>>(image_d, output_d, kernel_d, height, width, channels, kernelDim, kernelCenter);

    // Copy result back
    cudaMemcpy(output_h, output_d, outputSize, cudaMemcpyDeviceToHost);

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
