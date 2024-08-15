#ifndef ECC_CUH
#define ECC_CUH

#include <stdint.h>

#define DATA_LEN 256
#define ECC_LEN 2
#define TOTAL_LEN (DATA_LEN + ECC_LEN) // 258 bytes
#define TOTAL_MEMORY (1ULL * 1024 * 1024 * 1024 / 64) // 1GB in bytes
#define TOTAL_ROWS (TOTAL_MEMORY / DATA_LEN) // Total rows for 1GB memory
void generate_data(uint8_t *data, size_t rows);
void encode_data(uint8_t *d_data, uint8_t *d_encoded_data, size_t rows);
void decode_data(uint8_t *d_encoded_data, uint8_t *d_corrected_data, size_t rows);


// Kernel declarations
__global__ void rs_encode(uint8_t *data, uint8_t *encoded_data, size_t rows);
__global__ void rs_decode(uint8_t *encoded_data, uint8_t *corrected_data, size_t rows);
#endif // ECC_CUH
