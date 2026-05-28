# signinum-j2k-cuda

CUDA-facing device-output adapter for `signinum-j2k`.

Install this crate when a pipeline needs strict CUDA-resident HTJ2K decode
contracts or an explicitly named CPU-staged upload path:

```sh
cargo add signinum-j2k-cuda --features cuda-runtime
```

`BackendRequest::Cpu` and `BackendRequest::Auto` return host-backed CPU
surfaces. `BackendRequest::Cuda` is reserved for CUDA-resident HTJ2K codestream
decode and must not silently CPU-decode and upload pixels. Classic JPEG 2000 is
unsupported by the strict CUDA codestream path.

The current strict CUDA implementation covers full-frame, ROI,
reduced-resolution, and ROI+scaled HTJ2K `Gray8`, `Gray16`, `Rgb8`, `Rgba8`,
`Rgb16`, and `Rgba16` decode through CUDA HT entropy, IDWT, inverse MCT, and
store kernels. Color ROI and color ROI+scaled are resident-correct today by
running MCT over the full decoded color plane and compacting the final store;
ROI-pruned color MCT remains an optimization target. Decode uploads static HT
tables and the flattened compressed payload once per surface decode through
pinned host staging, then reuses those device resources across component and
sub-band dispatches. The 5/3 and 9/7 inverse DWT paths use separate CUDA
entrypoints.

`encode_j2k_lossless_with_cuda` exposes the strict CUDA encode-stage adapter for
HTJ2K lossless output. The CUDA-named encode APIs treat every backend preference
as `RequireDevice`; they return an unsupported error instead of a CPU codestream
whenever a required encode stage or packet layout is not CUDA-dispatched. CUDA
encode now covers forward RCT/ICT, forward 5/3 and 9/7 DWT, sub-band
quantization, batched HT cleanup code-block encode with cooperative
per-code-block magnitude reduction, first-inclusion packetization with HT
refinement pass headers, cooperative packet payload assembly, later-layer packet
contributions for code blocks already included in prior packets, and deferred
first inclusion after empty or non-empty prior packets through flattened
persistent tag-tree state.

Use `decode_to_cpu_staged_cuda_surface_with_session` and its region/scale
variants when a caller intentionally wants CPU decode followed by CUDA upload.
Those surfaces report `SurfaceResidency::CpuStagedCudaUpload`; strict CUDA
decode surfaces report `SurfaceResidency::CudaResidentDecode`.

`build_cuda_htj2k_grayscale_plan_with_profile` exposes the current CPU
parse/validation/direct-plan flattening boundary and returns a
`CudaHtj2kProfileReport`. `SIGNINUM_J2K_PROFILE_STAGES=1` emits profile rows,
and `SIGNINUM_J2K_CUDA_TRACE=/path/trace.json` writes Chrome-trace-compatible
stage spans for the plan/dispatch schema. Build with `cuda-profiling` to add
NVTX ranges around CUDA HTJ2K stages for Nsight Systems/Compute; NVTX is loaded
dynamically and is not required by normal builds.

The stable CPU decode API lives in `signinum-j2k`.
