#include "ecc.cuh"
#include "gf256.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>

__global__ void rs_encode(uint8_t *data, uint8_t *encoded_data, size_t rows) {
    size_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_idx < rows) {
        size_t data_offset = row_idx * DATA_LEN;
        size_t encoded_offset = row_idx * TOTAL_LEN;

        uint8_t parity[ECC_LEN] = {0};

        // Copy data to encoded array and calculate parity
        for (int i = 0; i < DATA_LEN; ++i) {
            encoded_data[encoded_offset + i] = data[data_offset + i];
            uint8_t feedback = data[data_offset + i] ^ parity[0];
            for (int j = 1; j < ECC_LEN; ++j) {
                parity[j - 1] = parity[j] ^ gf256_mul(feedback, gf256_exp[j]);
            }
            parity[ECC_LEN - 1] = gf256_mul(feedback, gf256_exp[ECC_LEN]);
        }

        // Copy parity bytes to encoded array
        for (int i = 0; i < ECC_LEN; ++i) {
            encoded_data[encoded_offset + DATA_LEN + i] = parity[i];
        }
    }
}

__global__ void rs_decode(uint8_t *encoded_data, uint8_t *corrected_data, size_t rows) {
    size_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_idx < rows) {
        size_t encoded_offset = row_idx * TOTAL_LEN;
        size_t corrected_offset = row_idx * DATA_LEN;

        uint8_t syndromes[ECC_LEN] = {0};
        uint8_t error_locations[ECC_LEN] = {0};
        __shared__ int error_count;

        // Initialize shared memory
        if (threadIdx.x == 0) {
            error_count = 0;
        }
        __syncthreads();

        // Calculate syndromes
        for (int i = 0; i < ECC_LEN; ++i) {
            for (int j = 0; j < TOTAL_LEN; ++j) {
                syndromes[i] = gf256_add(syndromes[i], gf256_mul(encoded_data[encoded_offset + j], gf256_exp[(i + 1) * j % 255]));
            }
        }

        // Check if syndromes are all zero (no errors)
        bool all_zero = true;
        for (int i = 0; i < ECC_LEN; ++i) {
            if (syndromes[i] != 0) {
                all_zero = false;
                break;
            }
        }

        if (all_zero) {
            // No errors, copy data directly
            for (int i = 0; i < DATA_LEN; ++i) {
                corrected_data[corrected_offset + i] = encoded_data[encoded_offset + i];
            }
            return;
        }

        // Find error locations using Berlekamp-Massey algorithm
        uint8_t sigma[ECC_LEN + 1] = {1};
        uint8_t b[ECC_LEN + 1] = {1};
        uint8_t t[ECC_LEN + 1];
        uint8_t l = 0;

        for (int i = 0; i < ECC_LEN; ++i) {
            uint8_t discrepancy = syndromes[i];
            for (int j = 1; j <= l; ++j) {
                discrepancy ^= gf256_mul(sigma[j], syndromes[i - j]);
            }
            if (discrepancy != 0) {
                for (int j = 0; j <= ECC_LEN; ++j) {
                    t[j] = sigma[j];
                }
                for (int j = 0; j <= ECC_LEN; ++j) {
                    if (b[j] != 0) {
                        sigma[j] ^= gf256_mul(discrepancy, b[j]);
                    }
                }
                if (2 * l <= i) {
                    l = i + 1 - l;
                    for (int j = 0; j <= ECC_LEN; ++j) {
                        b[j] = gf256_div(t[j], discrepancy);
                    }
                }
            }
            for (int j = ECC_LEN; j > 0; --j) {
                b[j] = b[j - 1];
            }
            b[0] = 0;
        }

        // Find roots of the error locator polynomial using Chien search
        for (int i = 0; i < TOTAL_LEN; ++i) {
            uint8_t sum = 0;
            for (int j = 0; j <= l; ++j) {
                sum ^= gf256_mul(sigma[j], gf256_exp[(i * j) % 255]);
            }
            if (sum == 0) {
                int loc = atomicAdd(&error_count, 1);
                if (loc < ECC_LEN) {
                    error_locations[loc] = i;
                }
            }
        }
        __syncthreads();

        // Correct errors
        if (error_count > 0) {
            for (int i = 0; i < error_count; ++i) {
                if (encoded_data[encoded_offset + error_locations[i]] != 0) {
                    uint8_t error_val = syndromes[0];
                    for (int j = 1; j < ECC_LEN; ++j) {
                        error_val ^= gf256_mul(syndromes[j], gf256_exp[(j * error_locations[i]) % 255]);
                    }
                    encoded_data[encoded_offset + error_locations[i]] ^= gf256_div(error_val, gf256_exp[(255 - error_locations[i]) % 255]);
                }
            }
        }

        // Copy corrected data to output
        for (int i = 0; i < DATA_LEN; ++i) {
            corrected_data[corrected_offset + i] = encoded_data[encoded_offset + i];
        }
    }
}

void generate_data(uint8_t *data, size_t rows) {
    for (size_t i = 0; i < DATA_LEN * rows; ++i) {
        data[i] = static_cast<uint8_t>(i % 256);
    }
}

void encode_data(uint8_t *d_data, uint8_t *d_encoded_data, size_t rows) {
    int threads_per_block = 256;
    size_t num_blocks = (rows + threads_per_block - 1) / threads_per_block;
    rs_encode<<<num_blocks, threads_per_block>>>(d_data, d_encoded_data, rows);
    cudaDeviceSynchronize(); // Ensure all encoding operations are complete
}

void decode_data(uint8_t *d_encoded_data, uint8_t *d_corrected_data, size_t rows) {
    int threads_per_block = 256;
    size_t num_blocks = (rows + threads_per_block - 1) / threads_per_block;
    rs_decode<<<num_blocks, threads_per_block>>>(d_encoded_data, d_corrected_data, rows);
    cudaDeviceSynchronize(); // Ensure all decoding operations are complete
}
