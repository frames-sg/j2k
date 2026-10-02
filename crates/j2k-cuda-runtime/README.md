# j2k-cuda-runtime

CUDA codec engine and Driver API runtime crate for J2K CUDA adapters.

It owns J2K CUDA kernel modules and the host-side launch logic for CUDA
codec stages, plus allocation, copy, stream, timing, and pooled resource
helpers used by the adapter crates.

CUDA performance numbers come from self-hosted benchmark runs, not from this
crate's tests.

## Launch geometry policy

Launch geometry is checked on the host before calling the CUDA Driver API. Grid
axes must be nonzero and no larger than `2^31 - 1` for x or `65,535` for y/z;
block axes must be nonzero and no larger than `1,024` for x/y or `64` for z,
with at most `1,024` threads per block. Safe operations also check geometry
derived from caller input before uploading or allocating output, where
possible.

These are NVIDIA's documented limits for the compute capabilities this crate
supports. The Driver API can still reject a launch on a specific device. See NVIDIA's [compute-capability
limits](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
and [Driver API device
attributes](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html).

## Resource completion policy

Submitting CUDA work is not treated as completing it. Pooled resources used by
asynchronous work are not reused until the work is known to have finished in
the same context. A safe API that returns a pooled buffer initialized by an
asynchronous memset synchronizes before returning it.

## Links

- API docs: <https://docs.rs/j2k-cuda-runtime>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
