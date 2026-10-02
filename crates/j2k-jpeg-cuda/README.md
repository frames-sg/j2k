# j2k-jpeg-cuda

CUDA adapter for J2K baseline JPEG decode and encode surfaces.

The CUDA paths use J2K's own kernels and decode into CUDA device memory.
Baseline encode takes Gray8 or Rgb8 CUDA buffers and returns host
`EncodedJpeg` output, for single images and batches. Explicit CUDA requests
return an error for unsupported JPEGs; they do not fall back to the CPU.

The adapter, session, and error types build by default. Enable `cuda-runtime`
for CUDA Driver API dispatch, the CUDA buffer and output-tile types, and actual
decode or encode. Without it, explicit CUDA operations return `CudaUnavailable`.

Enabling `cuda-runtime` does not guarantee that the CUDA Oxide kernels were
built. PTX is only generated on Linux hosts with the cuda-oxide toolchain;
other builds may embed placeholder PTX. Set `J2K_REQUIRE_CUDA_OXIDE_BUILD=1` on
CUDA test and benchmark hosts to make missing PTX a build error. Calling a
placeholder kernel returns an error saying the PTX was not built.

Scaled and region-scaled decodes run on the CPU, follow the CPU JPEG sampling
rules, and report the scaled output dimensions. Scaled JPEG decoding does not
run on the GPU.

## Links

- API docs: <https://docs.rs/j2k-jpeg-cuda>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
