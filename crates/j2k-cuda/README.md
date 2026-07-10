# j2k-cuda

CUDA adapter for JPEG 2000 / HTJ2K decode, resident encode, and shared
encode-stage paths.

CPU and Auto surface requests may return host-backed surfaces. Strict
CUDA-resident decode and CUDA-buffer encode use J2K-owned kernels and currently
support HTJ2K codestreams: classic J2K subband plans and classic block coding
are rejected by those resident paths. Separately, the shared encode-stage
adapter can accelerate supported stages without widening the strict resident
codec contract. Unsupported explicit CUDA requests return structured errors.

Host-backed fallbacks and the shared adapter/session types compile in default
builds. Enable `cuda-runtime` for CUDA Driver API dispatch, constructible
CUDA-resident surface and buffer types, and the CUDA-buffer encode APIs. Without
that feature, strict CUDA requests cannot dispatch and return `CudaUnavailable`
or the corresponding structured unsupported-request error.

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
