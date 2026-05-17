/**
 * ecc_ops.cpp — pybind11 bindings for custom_ecc_cuda extension.
 *
 * Exposes PyTorch-compatible functions that wrap the CUDA/Triton kernels.
 * Called from custom_kv/cache.py via `import custom_ecc_cuda`.
 */
#include <torch/extension.h>
#include <cuda_fp16.h>
#include "ecc_ops.h"

// ─── Forward declarations (defined in .cu files) ──────────────────────────
void launch_hadamard_f16(__half* x_ptr, int N, int D, cudaStream_t stream);
void launch_hadamard_f32(float* x_ptr, int N, int D, cudaStream_t stream);
void launch_compress_ecc(const __half* x_rot_ptr, ECC_KV_Block* cache_ptr,
                          int N, int D, cudaStream_t stream);

// ─── Macro checks ─────────────────────────────────────────────────────────
#define CHECK_CUDA(x)    TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)   CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// ─── Hadamard rotation ────────────────────────────────────────────────────

/**
 * hadamard_rotate
 *
 * Apply normalized WHT to input tensor in-place.
 * Input shape: [N, D] where N = batch*heads*tokens, D = head_dim.
 * Modifies tensor in-place and returns it.
 */
torch::Tensor hadamard_rotate(torch::Tensor x) {
    CHECK_INPUT(x);
    TORCH_CHECK(x.dim() == 2, "Expected 2D tensor [N, D]");

    int N = x.size(0);
    int D = x.size(1);
    TORCH_CHECK((D & (D - 1)) == 0, "D must be a power of 2");
    TORCH_CHECK(D <= 1024, "D must be <= 1024 (shared memory limit)");

    auto stream = c10::cuda::getCurrentCUDAStream();

    if (x.scalar_type() == torch::kFloat16) {
        launch_hadamard_f16(
            reinterpret_cast<__half*>(x.data_ptr<at::Half>()),
            N, D, stream);
    } else if (x.scalar_type() == torch::kFloat32) {
        launch_hadamard_f32(x.data_ptr<float>(), N, D, stream);
    } else {
        TORCH_CHECK(false, "hadamard_rotate only supports float16 and float32");
    }

    return x;  // in-place, returns same tensor
}

// ─── Compress + ECC syndrome ──────────────────────────────────────────────

/**
 * compress_and_store
 *
 * Compress a Hadamard-rotated key/value tensor into the ECC_KV_Block cache.
 *
 * @param x_rot      [N, D] FP16 — already Hadamard-rotated
 * @param cache_raw  [N, 128] uint8 — pre-allocated ECC_KV_Block pool (as bytes)
 * @param N          total (token, head) count
 * @param D          head dimension = 128
 */
void compress_and_store(
    torch::Tensor x_rot,      // [N, D] FP16
    torch::Tensor cache_raw   // [N, 128] uint8 — ECC_KV_Block pool
) {
    CHECK_INPUT(x_rot);
    CHECK_INPUT(cache_raw);
    TORCH_CHECK(x_rot.scalar_type() == torch::kFloat16);
    TORCH_CHECK(cache_raw.scalar_type() == torch::kUInt8);

    int N = x_rot.size(0);
    int D = x_rot.size(1);
    TORCH_CHECK(D == 128, "Only D=128 supported (hardcoded kernel)");
    TORCH_CHECK(cache_raw.size(1) == 128, "Cache must be [N, 128] bytes");

    auto stream = c10::cuda::getCurrentCUDAStream();

    launch_compress_ecc(
        reinterpret_cast<const __half*>(x_rot.data_ptr<at::Half>()),
        reinterpret_cast<ECC_KV_Block*>(cache_raw.data_ptr<uint8_t>()),
        N, D, stream);
}

// ─── Metadata accessors ───────────────────────────────────────────────────

/**
 * get_block_metadata
 *
 * Read (scale, zero_point, alpha) from an ECC_KV_Block at given index.
 * Returns a [3] FP32 tensor. Used for debugging.
 */
torch::Tensor get_block_metadata(torch::Tensor cache_raw, int block_idx) {
    CHECK_INPUT(cache_raw);
    auto out = torch::empty({3}, torch::TensorOptions().dtype(torch::kFloat32));

    const ECC_KV_Block* block =
        reinterpret_cast<const ECC_KV_Block*>(
            cache_raw.data_ptr<uint8_t>()) + block_idx;

    // Copy via cudaMemcpy (small, acceptable overhead for debug)
    __half h[3];
    cudaMemcpy(h, &block->scale, 3 * sizeof(__half), cudaMemcpyDeviceToHost);

    out[0] = __half2float(h[0]);  // scale
    out[1] = __half2float(h[1]);  // zero_point
    out[2] = __half2float(h[2]);  // alpha

    return out;
}

// ─── pybind11 module registration ─────────────────────────────────────────

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Error-Corrected KV Cache CUDA extension";

    m.def("hadamard_rotate",
          &hadamard_rotate,
          "In-place normalized WHT rotation for outlier diffusion",
          py::arg("x"));

    m.def("compress_and_store",
          &compress_and_store,
          "Compress rotated KV tensor into ECC_KV_Block cache pool",
          py::arg("x_rot"),
          py::arg("cache_raw"));

    m.def("get_block_metadata",
          &get_block_metadata,
          "Read (scale, zero_point, alpha) from block (debug utility)",
          py::arg("cache_raw"),
          py::arg("block_idx"));
}
