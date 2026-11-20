# J2K ML

> [Burn](https://github.com/tracel-ai/burn) tensors from [J2K](https://github.com/frames-sg/j2k)'s batch decoders

[![Current Crates.io Version](https://img.shields.io/crates/v/j2k-ml.svg)](https://crates.io/crates/j2k-ml)
[![Documentation](https://docs.rs/j2k-ml/badge.svg)](https://docs.rs/j2k-ml)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/frames-sg/j2k#license)

Decode batches of JPEG, JPEG 2000, and HTJ2K images into Burn 0.21 tensors.
Each group of images with the same shape and sample type becomes one rank-4
integer tensor (`U8`, `U16`, or `I16`, in NCHW or NHWC order), and its
`source_indices()` map the tensor's rows back to your inputs. Casting,
normalization, and augmentation stay in ordinary Burn code. This crate is
maintained by the J2K project; it is not an official Tracel or Burn crate.

`CpuBurnDecoder` decodes on the CPU for any Burn backend, and takes JPEG and
JPEG 2000 in one batch. For JPEG 2000 and HTJ2K on a GPU,
`CudaDirectBurnDecoder` decodes straight into Burn-owned CUDA memory, and
`CudaUploadBurnDecoder` and `MetalUploadBurnDecoder` decode on the GPU and
upload the pixels through host memory. The API is experimental, so a minor
release can change it (see
[API stability](https://github.com/frames-sg/j2k/blob/main/book/stability.md)).

## Usage

```rust
use std::sync::Arc;

use burn::tensor::backend::Backend;
use j2k::{BatchDecodeOptions, EncodedImage};
use j2k_ml::{BurnDecodeError, CpuBurnDecoder};

fn decode<B: Backend>(device: B::Device, files: Vec<Arc<[u8]>>) -> Result<(), BurnDecodeError> {
    let mut decoder = CpuBurnDecoder::<B>::new(device, BatchDecodeOptions::default());
    let batch = decoder.decode(files.into_iter().map(EncodedImage::full).collect())?;
    for group in batch.groups() {
        println!("{:?} {:?}", group.source_indices(), group.tensor().tensor().dims());
    }
    Ok(())
}
```

Images that fail to decode are listed in `batch.errors()`, and groups that
fail in `batch.group_errors()`. The
[Burn section](https://github.com/frames-sg/j2k/blob/main/book/machine-learning.md#burn)
of the J2K Book has a full `Dataset` and `Batcher` example and covers the GPU
routes and error handling.

## Feature flags

- `cpu`: `CpuBurnDecoder`.
- `cuda`: `CudaDirectBurnDecoder` and `CudaUploadBurnDecoder`, for Burn's CUDA
  backend. It builds `j2k-cuda` with `cuda-runtime`, so it needs Linux, an
  NVIDIA driver, and CUDA-Oxide (see
  [`j2k-cuda`](https://crates.io/crates/j2k-cuda)).
- `metal`: `MetalUploadBurnDecoder`, for Burn's wgpu backend on macOS.

<!-- j2k-crate-footer -->

---

Part of [J2K](https://github.com/frames-sg/j2k), a JPEG 2000 and HTJ2K codec
for Rust. See the [J2K Book](https://github.com/frames-sg/j2k/tree/main/book)
and the [API documentation](https://docs.rs/j2k).
