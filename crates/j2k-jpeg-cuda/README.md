# j2k-jpeg-cuda

NVIDIA CUDA GPU adapter for J2K JPEG decode surfaces.

Supported CUDA paths use J2K-owned CUDA kernels and CUDA device memory
outputs. Explicit CUDA requests are strict; unsupported JPEG shapes return
structured errors instead of silently falling back to CPU.
