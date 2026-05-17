#pragma once
#include <cstdint>
#include <cuda_fp16.h>

/**
 * ECC_KV_Block: 128-byte L2-aligned memory structure for one token, one head.
 *
 * Layout (128 bytes = 1 L2 cache line on Ampere/Hopper):
 *   [0:64]   int4_data    — 128 INT4 values, packed 2-per-byte
 *   [64:80]  ecc_syndrome — 128 Rademacher sign bits (1 bit per dimension)
 *   [80:82]  scale        — Lloyd-Max scale (FP16)
 *   [82:84]  zero_point   — Lloyd-Max zero point (FP16)
 *   [84:86]  alpha        — ECC compensation E[|epsilon|] (FP16)
 *   [86:128] padding      — 42 bytes to hit 128-byte boundary
 *
 * Compression: 256 bytes (FP16) → 128 bytes = 2.0x per block
 * System ratio at 128k tokens: 3.2x (model weights unchanged)
 *
 * Memory transaction: loading one block = 1 L2 cache line = 1 HBM transaction.
 * This is the core hardware alignment that enables near-peak HBM utilization.
 */
struct alignas(128) ECC_KV_Block {
    uint8_t  int4_data[64];    // 128 INT4 values packed: byte[i] = (val[2i+1]<<4)|val[2i]
    uint16_t ecc_syndrome[8];  // 128 sign bits: bit[j] = 1 if epsilon[j] >= 0
    __half   scale;            // Lloyd-Max range scale
    __half   zero_point;       // Lloyd-Max zero offset
    __half   alpha;            // Mean absolute residual E[|k_rot - k_tilde|]
    uint8_t  _pad[42];         // Explicit padding to 128 bytes
};

static_assert(sizeof(ECC_KV_Block) == 128,
    "ECC_KV_Block must be exactly 128 bytes (one L2 cache line)");

/**
 * Inline helpers for INT4 packing/unpacking.
 * INT4 values are in range [0, 15], packed as:
 *   byte = (hi_nibble << 4) | lo_nibble
 *   lo_nibble = even index, hi_nibble = odd index
 */
__host__ __device__ __forceinline__
uint8_t pack_int4(int lo, int hi) {
    return (uint8_t)(((hi & 0xF) << 4) | (lo & 0xF));
}

__host__ __device__ __forceinline__
void unpack_int4(uint8_t packed, int& lo, int& hi) {
    lo = packed & 0xF;
    hi = (packed >> 4) & 0xF;
}

/**
 * Rademacher syndrome helpers.
 * syndrome[j] = 1 if epsilon[j] >= 0, else 0
 * s_float[j]  = +1.0 if syndrome[j]=1, else -1.0
 */
__host__ __device__ __forceinline__
bool get_syndrome_bit(const uint16_t* syndrome, int j) {
    return (syndrome[j / 16] >> (j % 16)) & 1;
}

__host__ __device__ __forceinline__
void set_syndrome_bit(uint16_t* syndrome, int j) {
    syndrome[j / 16] |= (uint16_t)(1 << (j % 16));
}
