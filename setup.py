"""
setup.py — builds the custom_ecc_cuda C++/CUDA extension.
Run: python setup.py build_ext --inplace
"""
import os
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import setup, find_packages

# Detect GPU architecture from current device (fallback to common targets)
def get_cuda_arch_flags():
    if not torch.cuda.is_available():
        # CPU-only build: no CUDA arch needed (for unit tests)
        return []
    cc = torch.cuda.get_device_capability()
    arch = f"sm_{cc[0]}{cc[1]}"
    flags = [f"-arch={arch}", "--use_fast_math", "-O3"]
    print(f"[setup.py] Building for GPU arch: {arch}")
    return flags

cuda_sources = [
    "csrc/hadamard.cu",
    "csrc/compress_ecc.cu",
    "csrc/ecc_ops.cpp",
]

nvcc_flags = get_cuda_arch_flags() + [
    "-DBLOCK_D=128",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
]

cxx_flags = ["-O3", "-std=c++17"]

setup(
    name="ecc-kv-cache",
    version="0.1.0",
    description="Error-Corrected Ultra-Low Precision KV Cache for Llama-3",
    author="Your Name",
    url="https://github.com/YOUR_USERNAME/ecc-kv-cache",
    license="MIT",
    packages=find_packages(exclude=["tests*", "benchmarks*", "scripts*"]),
    ext_modules=[
        CUDAExtension(
            name="custom_ecc_cuda",
            sources=cuda_sources,
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
            },
            include_dirs=["csrc"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "triton>=2.3.0",
    ],
    python_requires=">=3.10",
)
