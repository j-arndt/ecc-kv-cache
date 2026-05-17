/**
 * hadamard.cu — Block-diagonal Walsh-Hadamard Transform (WHT) kernel.
 *
 * Applies the normalized WHT to key/value activation vectors to diffuse
 * outlier energy across all dimensions before INT4 quantization.
 *
 * Mathematical guarantee:
 *   H_n @ H_n^T = I  (orthogonal → inner products preserved)
 *   q · k = q_rot · k_rot  (attention scores unchanged)
 *   WHT applied AFTER RoPE (positional encoding preserved)
 *
 * Reference: KVLinC (arXiv:2510.05373), QuaRot (arXiv:2404.00456)
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include "ecc_ops.h"

/**
 * hadamard_inplace_f16
 *
 * In-place normalized WHT on FP16 input via Cooley-Tukey butterfly.
 * Each block processes one (token, head) pair independently.
 *
 * Grid:  (num_tokens * num_heads,)   — one block per (token, head)
 * Block: (D,) = (128,)               — one thread per dimension
 * Shared: D floats                   — staging for butterfly passes
 *
 * @param x     [num_tokens * num_heads * D] FP16, modified in-place
 * @param D     head dimension (must be power of 2, typically 128)
 */
__global__ void hadamard_inplace_f16(
    __half* __restrict__ x,
    const int D
) {
    // Shared memory: one float per thread (D floats total per block)
    extern __shared__ float sdata[];

    const int block_offset = blockIdx.x * D;
    const int tid = threadIdx.x;

    // Load from global memory (FP16 → FP32 for butterfly precision)
    sdata[tid] = __half2float(x[block_offset + tid]);
    __syncthreads();

    // Cooley-Tukey WHT butterfly: log2(D) = 7 passes for D=128
    // Each pass halves the stride.
    for (int stride = D / 2; stride >= 1; stride >>= 1) {
        // Pair threads: (tid, tid ^ stride)
        int partner = tid ^ stride;
        if (tid < partner) {
            float a = sdata[tid];
            float b = sdata[partner];
            sdata[tid]     = a + b;
            sdata[partner] = a - b;
        }
        __syncthreads();
    }

    // Normalize by 1/sqrt(D) to make H orthogonal (H @ H^T = I)
    const float inv_sqrt_D = rsqrtf((float)D);
    x[block_offset + tid] = __float2half(sdata[tid] * inv_sqrt_D);
}

/**
 * hadamard_inplace_f32
 *
 * Same as above but for FP32 inputs (used during calibration).
 */
__global__ void hadamard_inplace_f32(
    float* __restrict__ x,
    const int D
) {
    extern __shared__ float sdata[];

    const int block_offset = blockIdx.x * D;
    const int tid = threadIdx.x;

    sdata[tid] = x[block_offset + tid];
    __syncthreads();

    for (int stride = D / 2; stride >= 1; stride >>= 1) {
        int partner = tid ^ stride;
        if (tid < partner) {
            float a = sdata[tid];
            float b = sdata[partner];
            sdata[tid]     = a + b;
            sdata[partner] = a - b;
        }
        __syncthreads();
    }

    const float inv_sqrt_D = rsqrtf((float)D);
    x[block_offset + tid] = sdata[tid] * inv_sqrt_D;
}

// ─── C++ launcher functions (called from ecc_ops.cpp) ─────────────────────

/**
 * Launch hadamard rotation for a batch of (token, head) pairs.
 *
 * @param x_ptr   CUDA pointer to FP16 tensor [N, D] where N = tokens*heads
 * @param N       total number of (token, head) pairs
 * @param D       head dimension (128)
 * @param stream  CUDA stream for async execution
 */
void launch_hadamard_f16(
    __half* x_ptr,
    int N,
    int D,
    cudaStream_t stream
) {
    dim3 grid(N);
    dim3 block(D);
    size_t shared_bytes = D * sizeof(float);
    hadamard_inplace_f16<<<grid, block, shared_bytes, stream>>>(x_ptr, D);
}

void launch_hadamard_f32(
    float* x_ptr,
    int N,
    int D,
    cudaStream_t stream
) {
    dim3 grid(N);
    dim3 block(D);
    size_t shared_bytes = D * sizeof(float);
    hadamard_inplace_f32<<<grid, block, shared_bytes, stream>>>(x_ptr, D);
}
