# j2k-jpeg-cuda

CUDA adapter for J2K baseline JPEG decode and encode surfaces.

Supported CUDA paths use J2K-owned CUDA kernels and CUDA device memory
decode outputs. Baseline encode accepts Gray8 or Rgb8 CUDA input buffers and
returns host `EncodedJpeg` output, for both single images and batches. Explicit
CUDA requests are strict; unsupported JPEG shapes return structured errors
instead of silently falling back to CPU.

Adapter, session, error, and runtime-free stub types compile in default builds.
Enable `cuda-runtime` for CUDA Driver API dispatch, constructible CUDA-buffer
and output-tile types, and actual decode or encode execution. Without that
feature, strict CUDA operations return `CudaUnavailable`.

`cuda-runtime` is not proof that every CUDA Oxide kernel was built on the local
host. Product PTX is generated only on supported Linux cuda-oxide build hosts;
other builds may embed placeholder PTX. Set `J2K_REQUIRE_CUDA_OXIDE_BUILD=1`
on CUDA validation and benchmark hosts to fail the build when PTX is missing.
Runtime errors for placeholder kernels state that CUDA Oxide PTX was not built.

## Links

- API docs: <https://docs.rs/j2k-jpeg-cuda>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
