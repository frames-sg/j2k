# j2k-transcode-cuda

CUDA acceleration for J2K's JPEG-to-HTJ2K coefficient-domain transcode.

This crate runs the supported transform and code-block preparation stages on
CUDA. The transcode API itself is in `j2k-transcode`.

Enable the `cuda-runtime` feature for CUDA Driver API dispatch. Default builds
have the full API but do not need a CUDA runtime.

Enabling `cuda-runtime` does not guarantee that the CUDA Oxide kernels were
built. PTX is only generated on Linux hosts with the cuda-oxide toolchain;
other builds may embed placeholder PTX. Set `J2K_REQUIRE_CUDA_OXIDE_BUILD=1` on
CUDA test and benchmark hosts to make missing PTX a build error. Calling a
placeholder kernel returns an error saying the PTX was not built.

`Auto` sends a single transform job to CUDA only from `224 * 224` component
samples, and same-shape reversible 5/3 and 9/7 batches from 32 jobs and
`224 * 224 * 32` samples (the same as the Metal adapter). Lower the batch
thresholds with `with_auto_reversible_batch_thresholds` or
`with_auto_dwt97_batch_thresholds` if your own measurements justify it.
Meeting a threshold does not guarantee a speedup.

## Links

- API docs: <https://docs.rs/j2k-transcode-cuda>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
