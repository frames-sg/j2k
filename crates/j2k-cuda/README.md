# j2k-cuda

CUDA adapter for JPEG 2000 / HTJ2K decode and encode-stage paths.

The crate provides strict CUDA device-memory decode and encode-stage integration
for supported HTJ2K/J2K workloads using J2K-owned CUDA kernels.
Unsupported explicit CUDA requests return structured errors.

Enable the `cuda-runtime` feature for CUDA Driver API dispatch. Default builds
expose the API surface but do not require a CUDA runtime.

`cuda-runtime` is not proof that every CUDA Oxide kernel was built on the local
host. Product PTX is generated only on supported Linux cuda-oxide build hosts;
other builds may embed placeholder PTX. Set `J2K_REQUIRE_CUDA_OXIDE_BUILD=1`
on CUDA validation and benchmark hosts to fail the build when PTX is missing.
Runtime errors for placeholder kernels state that CUDA Oxide PTX was not built.

NVIDIA performance claims require self-hosted benchmark evidence.

## Links

- API docs: <https://docs.rs/j2k-cuda>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
