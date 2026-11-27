# J2K CUDA

> CUDA decode and encode for [J2K](https://github.com/frames-sg/j2k)

[![Current Crates.io Version](https://img.shields.io/crates/v/j2k-cuda.svg)](https://crates.io/crates/j2k-cuda)
[![Documentation](https://docs.rs/j2k-cuda/badge.svg)](https://docs.rs/j2k-cuda)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/frames-sg/j2k#license)

Decode JPEG 2000 and HTJ2K on an NVIDIA GPU, and encode lossless HTJ2K there.
Decoding covers classic JPEG 2000 and HTJ2K with reversible 5/3 or
irreversible 9/7 transforms, as whole images, regions, reduced resolutions,
or batches, into CUDA device memory. CUDA does not decode components deeper
than 16 bits, and encoding from CUDA buffers writes HTJ2K only.

You choose the backend per call with `BackendRequest`. `Auto` uses CUDA only
for the image types and sizes where it was measured to be faster than the CPU
and to give identical output, and runs everything else on the CPU. `Cuda`
always uses CUDA and returns an error for an image it can't handle; a CUDA
error is never retried on the CPU.

CUDA needs Linux, an NVIDIA driver, and the `cuda-runtime` feature. The
kernels are compiled to PTX at build time with CUDA-Oxide (`cargo oxide`), for
compute capability 8.0 and newer unless `J2K_CUDA_OXIDE_ARCH` says otherwise.
Without CUDA-Oxide the crate still builds, and the kernels return an error
when called. The API is experimental, so a minor release can change it (see
[API stability](https://github.com/frames-sg/j2k/blob/main/book/stability.md)).

## Usage

```rust
use j2k_core::{BackendRequest, DeviceSurface, ImageDecodeDevice, PixelFormat};
use j2k_cuda::J2kDecoder;

let bytes = std::fs::read("image.jph")?;
let mut decoder = J2kDecoder::new(&bytes)?;
let surface = decoder.decode_to_device(PixelFormat::Rgb8, BackendRequest::Auto)?;
println!("{:?} {:?}", surface.residency(), surface.dimensions());
```

The [GPU chapter](https://github.com/frames-sg/j2k/blob/main/book/gpu.md) of
the J2K Book covers batches, routing, encoding, and reading device memory.

## Feature flags

- `cuda-runtime`: CUDA dispatch, the kernels, and the device-memory types and
  CUDA-buffer encode functions. Without it, the API still builds, `Auto` runs
  on the CPU, and explicit CUDA requests return an error.
- `cuda-profiling`: marks CUDA stages with NVTX ranges for Nsight Systems.
  Implies `cuda-runtime`.

<!-- j2k-crate-footer -->

---

Part of [J2K](https://github.com/frames-sg/j2k), a JPEG 2000 and HTJ2K codec
for Rust. See the [J2K Book](https://github.com/frames-sg/j2k/tree/main/book)
and the [API documentation](https://docs.rs/j2k).
