# signinum-jpeg-cuda

CUDA adapter for Signinum JPEG decode surfaces.

Supported CUDA paths use Signinum-owned CUDA kernels and CUDA device memory
outputs. Explicit CUDA requests are strict; unsupported JPEG shapes return
structured errors instead of silently falling back to CPU.
