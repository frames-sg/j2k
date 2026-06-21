# j2k-cuda

CUDA adapter for JPEG 2000 / HTJ2K decode and encode-stage paths.

The crate provides strict CUDA device-memory decode and encode-stage integration
for supported HTJ2K/J2K workloads using J2K-owned CUDA kernels.
Unsupported explicit CUDA requests return structured errors.

NVIDIA performance claims require self-hosted benchmark evidence.
