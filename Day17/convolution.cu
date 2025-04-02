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

int main() {
    // Define configs/variables

    // Initial image and kernel
    // (Kernel should be of same number of channels as input)
    // Output channels will be 1 as we are using only one kernel

    // TODO - Read about Memory Coalescing - Storing the image in channel first format for better memory access efficiency

    // Define device variables

    // Allocate memory for device variables

    // Copy memory from host to device for image and kernel

    // Define threads and blocks size

    // Execute kernel

    // Copy output from device to host

    // Verify output

    // Free resources
}