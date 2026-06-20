# j2k-cuda-runtime

CUDA codec engine and Driver API runtime crate for J2K CUDA adapters.

It owns J2K CUDA kernel modules and the host-side launch logic for CUDA
codec stages, plus allocation, copy, stream, timing, and pooled resource
helpers used by the adapter crates.

This crate is the shared CUDA engine layer, but not proof of NVIDIA performance.
CUDA benchmark claims require self-hosted benchmark evidence.
