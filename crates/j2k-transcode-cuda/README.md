# j2k-transcode-cuda

CUDA acceleration adapter for supported stages in J2K's JPEG-to-HTJ2K
coefficient-domain transcode pipeline.

This crate accelerates supported transform and code-block preparation stages.
It does not replace the transcode API in `j2k-transcode`.

Enable the `cuda-runtime` feature for CUDA Driver API dispatch. Default builds
expose the API surface but do not require a CUDA runtime.

`cuda-runtime` is not proof that every CUDA Oxide kernel was built on the local
host. Product PTX is generated only on supported Linux cuda-oxide build hosts;
other builds may embed placeholder PTX. Set `J2K_REQUIRE_CUDA_OXIDE_BUILD=1`
on CUDA validation and benchmark hosts to fail the build when PTX is missing.
Runtime errors for placeholder kernels state that CUDA Oxide PTX was not built.

Auto routing is conservative. Single transform jobs are offered only after the
shared `224 * 224` component-sample floor, while same-shape reversible 5/3 and
9/7 batches use the same 32-job / `224 * 224 * 32` floors as the Metal adapter.
Callers with local evidence can lower the batch floors with
`with_auto_reversible_batch_thresholds` or `with_auto_dwt97_batch_thresholds`;
the defaults are routing policy, not a speedup promise.

## Links

- API docs: <https://docs.rs/j2k-transcode-cuda>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
