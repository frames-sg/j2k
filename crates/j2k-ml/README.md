# j2k-ml

`j2k-ml` is an independent integration for Burn 0.21 that adapts owned JPEG
2000 and HTJ2K batches. It is maintained by the `j2k` project and is not an official Tracel or Burn crate.

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
