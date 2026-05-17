/**
 * compress_ecc.cu — Lloyd-Max INT4 quantization + Rademacher ECC syndrome.
 *
 * This kernel takes Hadamard-rotated key/value vectors and compresses them
 * into the 128-byte ECC_KV_Block format:
 *   1. Quantize each dimension to INT4 using Lloyd-Max centroids
 *   2. Compute quantization residual: epsilon = x_rot - dequantize(q)
 *   3. Extract 1-bit Rademacher syndrome: sign(epsilon)
 *   4. Compute scalar compensation: alpha = mean(|epsilon|)
 *   5. Write all fields into the 128-byte aligned ECC_KV_Block
 *
 * Reference: QJL (arXiv:2406.03482), TurboQuant (dejan.ai/blog/turboquant)
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include "ecc_ops.h"

// ─── Lloyd-Max centroid lookup ─────────────────────────────────────────────
// Pre-computed centroids for 16-level Lloyd-Max quantizer
// fitted to N(0,1) distribution (scaled by runtime sigma).
// These are the decision boundaries for quantile-based INT4.
// See: calibration.py::lloyd_max_gaussian() for the full computation.
__constant__ float LM_CENTROIDS_NORM[16] = {
    -2.4008f, -1.8435f, -1.4371f, -1.0993f,
    -0.7996f, -0.5224f, -0.2582f,  0.0000f,
     0.2582f,  0.5224f,  0.7996f,  1.0993f,
     1.4371f,  1.8435f,  2.4008f,  3.1584f,
};

/**
 * lloyd_max_quantize_device
 *
 * Find the nearest centroid index for a given value using binary search.
 * Centroids are assumed sorted (they are for Lloyd-Max).
 *
 * @param val     value to quantize (after Hadamard rotation, zero-mean assumed)
 * @param scale   per-block scale factor (fitted from calibration)
 * @param zero    per-block zero offset
 * @return        INT4 index in [0, 15]
 */
__device__ __forceinline__
int lloyd_max_quantize_device(float val, float scale, float zero) {
    // Normalize to N(0,1) space
    float v_norm = (val - zero) / (scale + 1e-8f);

    // Binary search for nearest centroid
    int lo = 0, hi = 15;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        float boundary = (LM_CENTROIDS_NORM[mid] + LM_CENTROIDS_NORM[mid + 1]) * 0.5f;
        if (v_norm <= boundary) hi = mid;
        else lo = mid + 1;
    }
    return lo;
}

__device__ __forceinline__
float lloyd_max_dequantize_device(int q, float scale, float zero) {
    return LM_CENTROIDS_NORM[q] * scale + zero;
}

// ─── Main compression kernel ───────────────────────────────────────────────

/**
 * compress_ecc_kernel
 *
 * Grid:  (N,)    where N = num_tokens * num_heads
 * Block: (D,)    where D = head_dim = 128
 *
 * One block per (token, head) pair processes all D dimensions.
 *
 * @param x_rot      [N, D] FP16 — Hadamard-rotated input
 * @param cache      [N]    ECC_KV_Block pool — output (pre-allocated)
 * @param N          total (token, head) count
 * @param D          head dimension = 128
 */
__global__ void compress_ecc_kernel(
    const __half* __restrict__ x_rot,
    ECC_KV_Block* __restrict__ cache,
    const int N,
    const int D
) {
    extern __shared__ float sdata[];  // [D] floats for this block

    const int block_id = blockIdx.x;
    const int tid = threadIdx.x;     // 0 .. D-1

    if (block_id >= N) return;

    // ── Step 1: Load into shared memory ─────────────────────────────────
    float val = __half2float(x_rot[block_id * D + tid]);
    sdata[tid] = val;
    __syncthreads();

    // ── Step 2: Compute per-block statistics for scale/zero ──────────────
    // Thread 0 computes mean and std for this block
    __shared__ float blk_mean, blk_std;
    if (tid == 0) {
        float sum = 0.0f, sum_sq = 0.0f;
        for (int i = 0; i < D; i++) {
            sum    += sdata[i];
            sum_sq += sdata[i] * sdata[i];
        }
        blk_mean = sum / D;
        blk_std  = sqrtf(sum_sq / D - blk_mean * blk_mean + 1e-8f);
    }
    __syncthreads();

    // Use std as scale, mean as zero for Lloyd-Max in N(mean, std) space
    float scale = blk_std;
    float zero  = blk_mean;

    // ── Step 3: Quantize each dimension ──────────────────────────────────
    int q_idx = lloyd_max_quantize_device(val, scale, zero);
    float k_tilde = lloyd_max_dequantize_device(q_idx, scale, zero);
    float epsilon = val - k_tilde;

    // ── Step 4: Pack INT4 (2 values per byte) ────────────────────────────
    // Threads work in pairs: even tid packs with odd tid
    if (tid % 2 == 0) {
        // Wait for odd partner's q_idx
        __syncthreads();  // both threads have computed their q_idx
        // Note: we need the odd thread's q_idx — use shared memory
    }

    // Store q_idx to shared for packing
    __shared__ int q_shared[128];
    q_shared[tid] = q_idx;
    __syncthreads();

    // Only even threads write one packed byte
    if (tid % 2 == 0) {
        int q_lo = q_shared[tid];
        int q_hi = q_shared[tid + 1];
        cache[block_id].int4_data[tid / 2] = pack_int4(q_lo, q_hi);
    }

    // ── Step 5: Rademacher syndrome (1 bit per dimension) ────────────────
    // bit = 1 if epsilon >= 0, else 0
    bool sign_bit = (epsilon >= 0.0f);
    int word_idx = tid / 16;
    int bit_idx  = tid % 16;

    // Atomic OR to set bits (multiple threads write same uint16)
    if (sign_bit) {
        atomicOr(
            (unsigned int*)&cache[block_id].ecc_syndrome[word_idx & ~1],
            (unsigned int)((1u << bit_idx) << (16 * (word_idx & 1)))
        );
    }

    // ── Step 6: Compute alpha = mean(|epsilon|) ──────────────────────────
    __shared__ float epsilon_abs_sum;
    __shared__ float alpha_val;
    if (tid == 0) epsilon_abs_sum = 0.0f;
    __syncthreads();
    atomicAdd(&epsilon_abs_sum, fabsf(epsilon));
    __syncthreads();

    // ── Step 7: Write metadata (thread 0 only) ───────────────────────────
    if (tid == 0) {
        alpha_val = epsilon_abs_sum / (float)D;
        cache[block_id].scale       = __float2half(scale);
        cache[block_id].zero_point  = __float2half(zero);
        cache[block_id].alpha       = __float2half(alpha_val);
        // Clear syndrome first (may have garbage from allocation)
        for (int i = 0; i < 8; i++) cache[block_id].ecc_syndrome[i] = 0;
    }
    __syncthreads();

    // Re-run syndrome write after clear
    if (sign_bit) {
        atomicOr(
            (unsigned int*)&cache[block_id].ecc_syndrome[word_idx & ~1],
            (unsigned int)((1u << bit_idx) << (16 * (word_idx & 1)))
        );
    }
}

// ─── C++ launcher ─────────────────────────────────────────────────────────

void launch_compress_ecc(
    const __half* x_rot_ptr,
    ECC_KV_Block* cache_ptr,
    int N,
    int D,
    cudaStream_t stream
) {
    dim3 grid(N);
    dim3 block(D);
    size_t shared_bytes = D * sizeof(float);
    compress_ecc_kernel<<<grid, block, shared_bytes, stream>>>(
        x_rot_ptr, cache_ptr, N, D);
}
