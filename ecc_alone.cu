#include <iostream>
#include <cuda_runtime.h>
#include "ecc.cuh"

#include "kernel.cu"
#include "kernel2.cu"

void checkCudaError(cudaError_t error, const char *message) {
    if (error != cudaSuccess) {
        std::cerr << message << ": " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void BFSGraph(int argc, char** argv);

int main(int argc, char** argv) {
    BFSGraph(argc, argv);
    uint8_t *h_data, *d_data, *d_encoded_data, *d_corrected_data;

    // Adjust for 8GB of memory
    size_t total_rows = TOTAL_ROWS;

    // Allocate host memory for data generation
    h_data = new uint8_t[DATA_LEN * total_rows];

    // Allocate device memory for multiple rows
    checkCudaError(cudaMalloc((void **)&d_data, DATA_LEN * total_rows), "Failed to allocate device memory for data");
    checkCudaError(cudaMalloc((void **)&d_encoded_data, TOTAL_LEN * total_rows), "Failed to allocate device memory for encoded data");
    checkCudaError(cudaMalloc((void **)&d_corrected_data, DATA_LEN * total_rows), "Failed to allocate device memory for corrected data");

    // Generate data
    generate_data(h_data, total_rows);

    // Copy generated data to device memory
    checkCudaError(cudaMemcpy(d_data, h_data, DATA_LEN * total_rows, cudaMemcpyHostToDevice), "Failed to copy data to device");

    // Encode data
    encode_data(d_data, d_encoded_data, total_rows);

    // Decode data
    decode_data(d_encoded_data, d_corrected_data, total_rows);


    // Free device memory
    checkCudaError(cudaFree(d_data), "Failed to free device memory for data");
    checkCudaError(cudaFree(d_encoded_data), "Failed to free device memory for encoded data");
    checkCudaError(cudaFree(d_corrected_data), "Failed to free device memory for corrected data");

    // Free host memory
    delete[] h_data;

    return 0;
}
