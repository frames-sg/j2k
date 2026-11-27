# J2K Transcode CUDA

> CUDA acceleration for [J2K](https://github.com/frames-sg/j2k)'s JPEG-to-HTJ2K transcode

[![Current Crates.io Version](https://img.shields.io/crates/v/j2k-transcode-cuda.svg)](https://crates.io/crates/j2k-transcode-cuda)
[![Documentation](https://docs.rs/j2k-transcode-cuda/badge.svg)](https://docs.rs/j2k-transcode-cuda)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/frames-sg/j2k#license)

Run the transform stages of
[`j2k-transcode`](https://crates.io/crates/j2k-transcode)'s JPEG-to-HTJ2K
conversion on an NVIDIA GPU: the DCT-to-wavelet transforms, quantization, and
HTJ2K code-block preparation. `CudaDctToWaveletStageAccelerator` plugs into
the `j2k-transcode` API, which still assembles the codestream, and the output
is byte-identical to the CPU and Metal backends. `for_auto()` sends only large
jobs and batches to CUDA and runs the rest on the CPU; `new_explicit()`
returns an error for anything CUDA can't run.

CUDA needs Linux, an NVIDIA driver, and the `cuda-runtime` feature. The
kernels are compiled to PTX at build time with CUDA-Oxide (`cargo oxide`), for
compute capability 8.0 and newer unless `J2K_CUDA_OXIDE_ARCH` says otherwise.
Without CUDA-Oxide the crate still builds, and the kernels return an error
when called. The API is experimental, so a minor release can change it (see
[API stability](https://github.com/frames-sg/j2k/blob/main/book/stability.md)).

## Usage

```rust
use j2k_transcode::{JpegToHtj2kOptions, JpegToHtj2kTranscoder};
use j2k_transcode_cuda::CudaDctToWaveletStageAccelerator;

let jpeg = std::fs::read("tile.jpg")?;
let mut cuda = CudaDctToWaveletStageAccelerator::for_auto();
let encoded = JpegToHtj2kTranscoder::default().transcode_with_accelerator(
    &jpeg,
    &JpegToHtj2kOptions::lossless_53(),
    &mut cuda,
)?;
```

The [GPU acceleration](https://github.com/frames-sg/j2k/blob/main/book/transcoding.md#gpu-acceleration)
section of the J2K Book covers batches, the `Auto` thresholds, and how to
change them.

## Feature flags

- `cuda-runtime`: CUDA dispatch and the kernels. Without it, the API still
  builds, `for_auto()` runs everything on the CPU, and `new_explicit()`
  returns an error.
- `cuda-profiling`: marks CUDA stages with NVTX ranges for Nsight Systems.
  Implies `cuda-runtime`.

<!-- j2k-crate-footer -->

---

Part of [J2K](https://github.com/frames-sg/j2k), a JPEG 2000 and HTJ2K codec
for Rust. See the [J2K Book](https://github.com/frames-sg/j2k/tree/main/book)
and the [API documentation](https://docs.rs/j2k).
