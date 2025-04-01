#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#define BLOCK_SIZE 16

// Assumed kernel size = 3

void generateRandomValues(int *arr, int rows, int cols) {
    for (int i=0 ; i < rows ; i++) {
        for (int j=0 ; j < cols ; j++) {
            arr[i * cols + j] = (int)(1 + (float)rand() / (float)(RAND_MAX / 255)); // Random float between 0 and 10
        }
	}
}

__global__ void dilateKernel(int *input, int *output, uint width, uint height) {
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int localRow = threadIdx.y + 1;
    int localCol = threadIdx.x + 1;

    __shared__ int sharedMem[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

    if (row < height && col < width) {
        sharedMem[localRow][localCol] = input[row * width + col];

        // Left column
        if (threadIdx.x == 0) {
            sharedMem[localRow][0] = 0;
        }

        // Top row
        if (threadIdx.y == 0) {
            sharedMem[0][localCol] = 0;
        }

        // Right column
        if (threadIdx.x == BLOCK_SIZE - 1) {
            sharedMem[localRow][BLOCK_SIZE + 1] = 0;
        }

        // Bottom row
        if (threadIdx.y == BLOCK_SIZE - 1) {
            sharedMem[BLOCK_SIZE + 1][localCol] = 0;
        }

        // Top left
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            sharedMem[0][0] = 0;
        }

        // Top right
        if (threadIdx.x == BLOCK_SIZE - 1 && threadIdx.y == 0) {
            sharedMem[0][BLOCK_SIZE + 1] = 0;
        }

        // Bottom left
        if (threadIdx.x == 0 && threadIdx.y == BLOCK_SIZE - 1) {
            sharedMem[BLOCK_SIZE - 1][BLOCK_SIZE - 1] = 0;
        }

        // Bottom right
        if (threadIdx.x == BLOCK_SIZE - 1 && threadIdx.y == BLOCK_SIZE - 1) {
            sharedMem[BLOCK_SIZE + 1][BLOCK_SIZE + 1] = 0;
        }
    }

    __syncthreads();

    int maxValue = 0;

    for (int i=-1; i < 2; i++) {
        for (int j=-1; j < 2; j++) {
            maxValue = max(maxValue, sharedMem[localRow + i][localCol + j]);
        }
    }

    if (row < height && col < width) {
        output[row * width + col] = maxValue;
    }
}

__global__ void erodeKernel(int *input, int *output, uint width, uint height) {
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int localRow = threadIdx.y + 1;
    int localCol = threadIdx.x + 1;

    __shared__ int sharedMem[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

    if (row < height && col < width) {
        sharedMem[localRow][localCol] = input[row * width + col];

        // Left Column
        if (threadIdx.x == 0) {
            sharedMem[localRow][0] = 0;
        }

        // Top Row
        if (threadIdx.y == 0) {
            sharedMem[0][localCol] = 0;
        }

        // Right column
        if (threadIdx.x == BLOCK_SIZE - 1) {
            sharedMem[localRow][BLOCK_SIZE + 1] = 0;
        }

        // Bottom row
        if (threadIdx.y == BLOCK_SIZE - 1) {
            sharedMem[BLOCK_SIZE + 1][localCol] = 0;
        }

        // Top left
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            sharedMem[0][0] = 0;
        }

        // Top right
        if (threadIdx.x == BLOCK_SIZE - 1 && threadIdx.y == 0) {
            sharedMem[0][BLOCK_SIZE + 1] = 0;
        }

        // Bottom right
        if (threadIdx.y == BLOCK_SIZE - 1 && threadIdx.x == BLOCK_SIZE - 1) {
            sharedMem[BLOCK_SIZE + 1][BLOCK_SIZE + 1] = 0;
        }

        // Bottom left
        if (threadIdx.x == 0 && threadIdx.y == BLOCK_SIZE - 1) {
            sharedMem[BLOCK_SIZE + 1][0] = 0;
        }
    }

    __syncthreads();

    int minValue = 255;
    for (int i=-1; i < 2; i++) {
        for (int j=-1; j < 2; j++) {
            minValue = min(minValue, sharedMem[localRow + i][localCol + j]);
        }
    }

    if (row < height && col < width) {
        output[row * width + col] = minValue;
    }

}

__global__ void binarizeKernel(int *input, int *output, uint width, uint height, int threshold) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        output[row * width + col] = input[row * width + col] >= threshold ? 255 : 0;
    }
}

int main() {
    uint height = 256;
    uint width = 256;
    int threshold = 128;
    int imageInput_h[height][width], binaryOutput_h[height][width], dilationOutput_h[height][width], erosionOutput_h[height][width];
    size_t size = height * width * sizeof(int);

    generateRandomValues((int *)imageInput_h, height, width);

    int *imageInput_d, *binaryOutput_d, *erosionOutput_d, *dilationOutput_d;

    cudaMalloc((void **)&imageInput_d, size);
    cudaMalloc((void **)&binaryOutput_d, size);
    cudaMalloc((void **)&erosionOutput_d, size);
    cudaMalloc((void **)&dilationOutput_d, size);

    cudaMemcpy(imageInput_d, imageInput_h, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock = {BLOCK_SIZE, BLOCK_SIZE, 1};
    dim3 blocksPerGrid = {(width + BLOCK_SIZE - 1)/BLOCK_SIZE, (height + BLOCK_SIZE - 1)/BLOCK_SIZE, 1};

    binarizeKernel<<<blocksPerGrid, threadsPerBlock>>>(imageInput_d, binaryOutput_d, width, height, threshold);
    dilateKernel<<<blocksPerGrid, threadsPerBlock>>>(binaryOutput_d, dilationOutput_d, width, height);
    erodeKernel<<<blocksPerGrid, threadsPerBlock>>>(dilationOutput_d, erosionOutput_d, width, height);

    cudaMemcpy(binaryOutput_h, binaryOutput_d, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(dilationOutput_h, dilationOutput_d, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(erosionOutput_h, erosionOutput_d, size, cudaMemcpyDeviceToHost);

    // Print sample verification (first 5x5 section)
    printf("Input Image (5x5):\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++)
            printf("%3d ", imageInput_h[i][j]);
        printf("\n");
    }

    printf("\nBinarized Image (5x5):\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++)
            printf("%3d ", binaryOutput_h[i][j]);
        printf("\n");
    }

    printf("\nDilated Image (5x5):\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++)
            printf("%3d ", dilationOutput_h[i][j]);
        printf("\n");
    }

    printf("\nEroded Image (5x5):\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++)
            printf("%3d ", erosionOutput_h[i][j]);
        printf("\n");
    }

    cudaFree(imageInput_d); cudaFree(binaryOutput_d);
    cudaFree(erosionOutput_d); cudaFree(dilationOutput_d);
}