#include <iostream>
#include <cuda_runtime.h>
#include "ecc.cuh"
#include "bfs_new.cuh"

void checkCudaError(cudaError_t error, const char *message) {
    if (error != cudaSuccess) {
        std::cerr << message << ": " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {

    BFSGraph(argc, argv);

    uint8_t *h_data, *d_data, *d_encoded_data, *d_corrected_data;

    size_t total_rows = TOTAL_ROWS; // 1GB / 256 bytes per row

    h_data = new uint8_t[DATA_LEN * total_rows];

    cudaMalloc((void **)&d_data, DATA_LEN * total_rows);
    cudaMalloc((void **)&d_encoded_data, TOTAL_LEN * total_rows);
    cudaMalloc((void **)&d_corrected_data, DATA_LEN * total_rows);

    generate_data(h_data, total_rows);

    cudaMemcpy(d_data, h_data, DATA_LEN * total_rows, cudaMemcpyHostToDevice);

    cudaStream_t stream1, stream2;
    int leastPriority, greatestPriority;
    cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);

    cudaStreamCreateWithPriority(&stream1, cudaStreamDefault, greatestPriority);
    cudaStreamCreateWithPriority(&stream2, cudaStreamDefault, leastPriority);

    int threads_per_block = 256;
    size_t num_blocks = (total_rows + threads_per_block - 1) / threads_per_block;
    rs_encode<<<num_blocks, threads_per_block, 0, stream2>>>(d_data, d_encoded_data, total_rows);

    bool stop = false;
    do {
        stop = false;
        cudaMemcpyAsync(d_over, &stop, sizeof(bool), cudaMemcpyHostToDevice, stream1);

        Kernel<<<num_of_blocks, MAX_THREADS_PER_BLOCK, 0, stream1>>>(d_graph_nodes, d_graph_edges, d_graph_mask, d_updating_graph_mask, d_graph_visited, d_cost, no_of_nodes);
        Kernel2<<<num_of_blocks, MAX_THREADS_PER_BLOCK, 0, stream1>>>(d_graph_mask, d_updating_graph_mask, d_graph_visited, d_over, no_of_nodes);

        cudaMemcpyAsync(&stop, d_over, sizeof(bool), cudaMemcpyDeviceToHost, stream1);
        cudaStreamSynchronize(stream1);
    } while(stop);


    rs_decode<<<num_blocks, threads_per_block, 0, stream2>>>(d_encoded_data, d_corrected_data, total_rows);

    cudaStreamSynchronize(stream2);
    cudaStreamSynchronize(stream1);


    checkCudaError(cudaFree(d_data), "Failed to free device memory for data");
    checkCudaError(cudaFree(d_encoded_data), "Failed to free device memory for encoded data");
    checkCudaError(cudaFree(d_corrected_data), "Failed to free device memory for corrected data");

    delete[] h_data;

    checkCudaError(cudaStreamDestroy(stream1), "Failed to destroy stream1");
    checkCudaError(cudaStreamDestroy(stream2), "Failed to destroy stream2");

    return 0;
}
