# j2k-cuda

CUDA adapter for J2K J2K / HTJ2K paths.

The crate provides strict CUDA device-memory decode and encode-stage integration
for supported HTJ2K/J2K workloads using J2K-owned CUDA kernels.
Unsupported explicit CUDA requests return structured errors.

NVIDIA performance claims require self-hosted benchmark evidence.
