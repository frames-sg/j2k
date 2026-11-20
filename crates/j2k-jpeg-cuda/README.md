# J2K JPEG CUDA

> CUDA JPEG decode and encode for [J2K](https://github.com/frames-sg/j2k)

[![Current Crates.io Version](https://img.shields.io/crates/v/j2k-jpeg-cuda.svg)](https://crates.io/crates/j2k-jpeg-cuda)
[![Documentation](https://docs.rs/j2k-jpeg-cuda/badge.svg)](https://docs.rs/j2k-jpeg-cuda)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/frames-sg/j2k#license)

Decode baseline JPEG into CUDA device memory, and encode baseline and lossless
JPEG from CUDA buffers, with J2K's own kernels. CUDA decodes full images of
single-scan 8-bit YCbCr JPEG at 4:2:0, 4:2:2, or 4:4:4 to `Rgb8`. Baseline
encode takes `Gray8` or `Rgb8` buffers, one image or a batch. Lossless encode
takes `Gray8`, `Gray16`, `Rgb8`, or `Rgb16` and writes the same bytes as
`j2k_jpeg::encode_jpeg_lossless`. Encoded JPEGs are returned in host memory.

`BackendRequest::Auto` always decodes on the CPU with
[`j2k-jpeg`](https://crates.io/crates/j2k-jpeg). `Cuda` returns an error for
any JPEG or request outside the scope above, including regions and scaled
output; it never falls back to the CPU.

CUDA needs Linux, an NVIDIA driver, and the `cuda-runtime` feature. The
kernels are compiled to PTX at build time with CUDA-Oxide (`cargo oxide`), for
compute capability 8.0 and newer unless `J2K_CUDA_OXIDE_ARCH` says otherwise.
Without CUDA-Oxide the crate still builds, and the kernels return an error
when called. The API is experimental, so a minor release can change it (see
[API stability](https://github.com/frames-sg/j2k/blob/main/book/stability.md)).

## Usage

```rust
use j2k_core::{BackendRequest, DeviceSurface, ImageDecodeDevice, PixelFormat};
use j2k_jpeg_cuda::Decoder;

let bytes = std::fs::read("photo.jpg")?;
let mut decoder = Decoder::new(&bytes)?;
let surface = decoder.decode_to_device(PixelFormat::Rgb8, BackendRequest::Cuda)?;
println!("{:?} {:?}", surface.residency(), surface.dimensions());
```

The [GPU JPEG](https://github.com/frames-sg/j2k/blob/main/book/jpeg.md#gpu-jpeg)
section of the J2K Book covers batch encode and its memory limits.

## Feature flags

- `cuda-runtime`: CUDA dispatch, the kernels, and the CUDA buffer and
  output-tile types. Without it, the API still builds, and CUDA requests
  return `Error::CudaUnavailable`.

<!-- j2k-crate-footer -->

---

Part of [J2K](https://github.com/frames-sg/j2k), a JPEG 2000 and HTJ2K codec
for Rust. See the [J2K Book](https://github.com/frames-sg/j2k/tree/main/book)
and the [API documentation](https://docs.rs/j2k).
