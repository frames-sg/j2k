# j2k-jpeg-cuda

CUDA adapter for J2K JPEG decode surfaces.

Supported CUDA paths use J2K-owned CUDA kernels and CUDA device memory
outputs. Explicit CUDA requests are strict; unsupported JPEG shapes return
structured errors instead of silently falling back to CPU.

Enable the `cuda-runtime` feature for CUDA Driver API dispatch. Default builds
expose the API surface but do not require a CUDA runtime.

`cuda-runtime` is not proof that every CUDA Oxide kernel was built on the local
host. Product PTX is generated only on supported Linux cuda-oxide build hosts;
other builds may embed placeholder PTX. Set `J2K_REQUIRE_CUDA_OXIDE_BUILD=1`
on CUDA validation and benchmark hosts to fail the build when PTX is missing.
Runtime errors for placeholder kernels state that CUDA Oxide PTX was not built.

## Links

- API docs: <https://docs.rs/j2k-jpeg-cuda>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
