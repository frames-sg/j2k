# j2k-cuda-runtime

CUDA codec engine and Driver API runtime crate for J2K CUDA adapters.

It owns J2K CUDA kernel modules and the host-side launch logic for CUDA
codec stages, plus allocation, copy, stream, timing, and pooled resource
helpers used by the adapter crates.

This crate is the shared CUDA engine layer, but not proof of NVIDIA performance.
CUDA benchmark claims require self-hosted benchmark evidence.

## Launch geometry policy

Host launch geometry is validated before the CUDA Driver API is called. Grid
axes must be nonzero and no larger than `2^31 - 1` for x or `65,535` for y/z;
block axes must be nonzero and no larger than `1,024` for x/y or `64` for z,
with at most `1,024` threads per block. Safe operations also preflight
caller-derived geometry before upload or output allocation where possible.

These are the documented limits for the modern CUDA compute capabilities this
crate supports. Device-specific acceptance still belongs to the Driver API and
must be verified on NVIDIA hardware. See NVIDIA's [compute-capability
limits](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
and [Driver API device
attributes](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html).

## Resource completion policy

Successful CUDA submission is not treated as completion. Pooled resources used
by asynchronous work stay behind a reuse guard until a proven same-context
completion point. A safe API that returns an initialized pooled buffer after an
otherwise unobserved asynchronous memset synchronizes before returning it.

## Links

- API docs: <https://docs.rs/j2k-cuda-runtime>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
