#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 16

// Assumed kernel size = 3

void generateRandomValues(int *arr, int rows, int cols) {
    for (int i=0 ; i < rows ; i++) {
        for (int j=0 ; j < cols ; j++) {
            arr[i * cols + j] = (int)(1 + (float)rand() / (float)(RAND_MAX / 255)); // Random float between 0 and 10
        }
	}
}

__global__ void dilateKernel(unsigned char *input, unsigned char *output, int width, int height) {
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int localRow = row + 1;
    int localCol = col + 1;

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

__global__ void erodeKernel(unsigned char *input, unsigned char *output, int width, int height) {
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int localRow = row + 1;
    int localCol = col + 1;

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

    int minValue = 0;
    for (int i=-1; i < 2; i++) {
        for (int j=-1; j < 2; j++) {
            minValue = min(minValue, sharedMem[localRow + i][localCol + j]);
        }
    }

    if (row < height && col < width) {
        output[row * width + col] = minValue;
    }

}

__global__ void binarizeKernel(unsigned char *input, unsigned char *output, int width, int height, int threshold) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        output[row * width + col] = input[row * width + col] >= threshold ? 255 : 0;
    }
}

int main() {
    int height = 256;
    int width = 256;
    int threshold = 128;
    int imageInput_h[height][width], imageOutput_h[height][width];
    size_t size = height * width * sizeof(int);

    generateRandomValues((int *)imageInput_h, height, width);

    int *imageInput_d, imageOutput_d;

    cudaMalloc((void **)&imageInput_d, size);
    cudaMalloc((void **)&imageOutput_d, size);

    cudaMemcpy(imageInput_d, imageInput_h, size, cudaMemcpyHostToDevice);


}