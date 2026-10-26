# j2k-ml

`j2k-ml` is an independent integration for Burn 0.21 that adapts JPEG, JPEG
2000, and HTJ2K batches. It is maintained by the `j2k` project and is not an official Tracel or Burn crate.

[GPU image decoding for AI pipelines](https://frames-sg.com/gpu-image-decoding-ai/)
provides the project-level overview of the tensor and resident-device routes.

## Quick start

The portable route decodes with the CPU codec and creates the resulting tensor
on the selected Burn backend:

```bash
cargo add j2k-ml --features cpu
```

For a complete `Dataset` -> `DataLoader` -> `Batcher` example:

```bash
cargo run -p j2k-ml --example training_batcher --features cpu
```

The example keeps a `CpuBurnDecoder` behind a mutex because Burn's `Batcher`
receives `&self`, preserves label order through each codec group's
`source_indices`, then casts and normalizes with ordinary Burn tensor
operations.

With the `cpu` feature, `CpuBurnDecoder::decode_batch(&[ImageInput])` accepts
JPEG and JPEG 2000 or HTJ2K together. `ImageInput::Jpeg` carries a
`j2k_jpeg::JpegBatchInput`; `ImageInput::Jpeg2000` borrows an `EncodedImage`.
The common `BurnBatchDecode` exposes groups, indexed codec-tagged errors and
group failures through accessors. Group order is unspecified; `source_indices()`
is authoritative. Each group exposes returned sample metadata, typed warnings,
`recovered()`, and optional `j2k_info()` for JPEG 2000 details. Session strictness
controls both codec families while retaining JPEG color hints. The existing
J2K `decode` and prepared paths remain available. JPEG region, reduction and GPU
codec execution remain unsupported; a GPU Burn destination uses an ordinary upload.

JPEG tensor output remains RGB uint8 by default, with grayscale expanded to three identical channels. Grayscale JPEG 2000 and HTJ2K remain one channel. Output metadata describes the returned tensor. A later explicit `channels="native"` option may preserve grayscale JPEG as one channel without changing existing defaults. Source component counts, when exposed, are per input row and remain aligned with source_indices.

The Rust session defaults to strict decoding with the same pixel-integrity
contract as Python: complete required image data, no damage concealment,
missing-data substitution or unverified completion. Leniency permits supported
recovery, records it per row, and never also reports that row as an error.
Cosmetic warnings do not set `recovered()`. Unknown intent at progressive EOF
is rejected by default. No channels option is currently exposed.

CUDA can decode directly into Burn-owned device storage:

```bash
cargo run -p j2k-ml --example cuda_direct --features cuda
```

`CudaDirectBurnDecoder` allocates the destination through Burn/CubeCL, decodes
into that allocation, validates synchronous codec completion, and then returns
the same allocation as an ordinary fused `burn_cuda::Cuda` tensor. Decoded
pixels do not pass through host memory or an intermediate device copy.

Explicit staged accelerator-decode-and-upload examples are:

```bash
cargo run -p j2k-ml --example cuda_upload --features cuda
cargo run -p j2k-ml --example metal_upload --features metal
```

`CudaUploadBurnDecoder` and `MetalUploadBurnDecoder` run codec decoding on the
GPU, copy the decoded pixels to host memory, and then upload them with Burn's
ordinary tensor API.

## Who owns batching?

| Responsibility | Owner |
| --- | --- |
| Dataset reads, labels, sampling, prefetch, resizing, augmentation, and training batches | The Burn application and its `Dataset`/`DataLoader`/`Batcher` |
| Parsing, compatible-image grouping, preparation reuse, and JPEG 2000/HTJ2K execution | The `j2k` codec crates |
| Direct CUDA destinations, host staging, and ordinary Burn tensor materialization/upload | `j2k-ml` |
| Float conversion and normalization | Ordinary Burn tensor operations in the application |

One application batch can yield multiple codec groups when images have
different shapes, channel layouts, sample types, or decode requests. Training
code should either build uniform batches or handle every returned group and
its `source_indices`.

## Routes

| Feature | Codec execution | Burn destination |
| --- | --- | --- |
| `cpu` | Persistent CPU batch decoder | Any compatible Burn backend; placing the `TensorData` on a GPU backend performs an upload |
| `cuda` direct | Persistent CUDA decoder | Synchronous decode into Burn-owned CUDA storage |
| `cuda` staged | Persistent CUDA decoder, then decoded-pixel readback | Ordinary Burn CUDA upload |
| `metal` | Persistent Metal decoder, then decoded-pixel readback | Ordinary Burn wgpu/Metal upload |

The adapter returns ordinary rank-4 NCHW or NHWC `U8`, `U16`, or `I16` Burn
tensors. It does not own normalization, augmentation, dataset policy, or a
second decoding pipeline.

See the [Burn integration guide](../../docs/j2k-ml.md) for the full API,
support matrix, safety notes, test commands, and benchmark results. The API docs
are on [docs.rs](https://docs.rs/j2k-ml), the source is in the
[J2K repository](https://github.com/frames-sg/j2k), and codec support is listed
in [public support](../../docs/public-support.md).
