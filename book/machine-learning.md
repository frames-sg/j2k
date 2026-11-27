# Machine learning

J2K can decode batches of images straight into tensors, so a training or
inference pipeline doesn't have to go through an image library first:

- [Burn](#burn): `j2k-ml` returns Burn integer tensors, decoded on the CPU,
  CUDA, or Metal.
- [MPSGraph](#mpsgraph): `j2k-mpsgraph` feeds Metal-resident batches to
  Apple's MPSGraph on Apple Silicon. It is experimental.
- [PyTorch and NumPy](#pytorch-and-numpy): the Python package decodes to
  tensors.
- [Wavelet coefficients](#wavelet-coefficients): train on the coded wavelet
  bands instead of pixels.

## Burn

`j2k-ml` turns batches of JPEG, JPEG 2000, and HTJ2K images into Burn 0.21
integer tensors. We maintain it; it isn't an official Tracel or Burn crate.

| Decoder | Feature | Decodes on | How pixels reach Burn |
| --- | --- | --- | --- |
| `CpuBurnDecoder<B>` | `cpu` | CPU | One `TensorData` per group, on any Burn backend. A GPU backend uploads it. |
| `CudaDirectBurnDecoder` | `cuda` | CUDA | Decoded into Burn-owned CUDA memory. No copy through the host. |
| `CudaUploadBurnDecoder` | `cuda` | CUDA | Copied to host memory, then uploaded with `Tensor::from_data`. |
| `MetalUploadBurnDecoder` | `metal` | Metal | Copied to host memory, then uploaded to Burn's wgpu device. |

No feature is on by default.

```rust
use std::sync::Arc;

use burn_flex::{Flex, FlexDevice};
use j2k::{BatchDecodeOptions, EncodedImage};
use j2k_ml::{BurnBatchTensor, CpuBurnDecoder};

let mut decoder = CpuBurnDecoder::<Flex>::new(FlexDevice, BatchDecodeOptions::default());
let inputs = files.iter().map(|bytes| EncodedImage::full(Arc::clone(bytes))).collect();
let batch = decoder.decode(inputs)?;
for group in batch.into_groups() {
    let rows = group.source_indices().to_vec(); // input index of each tensor row
    if let BurnBatchTensor::U8(tensor) = group.into_tensor() {
        let images = tensor.float().div_scalar(255.0); // [N, C, H, W]
    }
}
```

`files` is a `Vec<Arc<[u8]>>` of encoded images. The sample uses Burn's Flex
CPU backend; any Burn backend works. The repository has runnable examples for
each decoder:

```bash
cargo run -p j2k-ml --example training_batcher --features cpu
cargo run -p j2k-ml --example cuda_direct --features cuda
cargo run -p j2k-ml --example cuda_upload --features cuda
cargo run -p j2k-ml --example metal_upload --features metal
```

`training_batcher` is a full `Dataset`, `DataLoader`, and `Batcher` setup. It
keeps a `CpuBurnDecoder` behind a mutex, because Burn's `Batcher` receives
`&self`, and reorders labels to match each group's `source_indices`.

### What `j2k-ml` does

`j2k-ml` is an adapter, not a second codec. The codec crates parse, group,
and decode; `j2k-ml` puts the result in a Burn tensor. Dataset assembly,
labels, sampling, resizing, padding, augmentation, float conversion, and
normalization stay in your application. Cast and normalize the integer
tensors with ordinary Burn operations.

A codec group is not a training batch. Burn's `DataLoader` picks items and
calls your `Batcher`; the codec then splits those items into groups of the
same shape and sample type, so one batch can come back as several tensors.
Either build batches of uniform images, or handle every group and its
`source_indices`.

### Inputs and preparation

Inputs are owned `EncodedImage` values: an `Arc<[u8]>` holding JP2, JPH, or a
raw codestream, plus a full, region, reduced, or region-and-reduced request.
A decoder can:

- prepare and decode in one call, with `decode`;
- keep the `PreparedBatch` from `prepare` and call `decode_prepared` once per
  epoch;
- regroup `PreparedImage` values with `prepare_prepared_images` or
  `decode_prepared_images`, without parsing or copying the codestreams again.
  Source indices are then positions in the new list;
  `PreparedImage::source_index` still holds the original.

Regrouped images must agree on strict or lenient decoding; layout and worker
settings can change. Images the batch planner supports are prepared down to
their code blocks (`PreparationDepth::Htj2kOffsetPlan` or
`ClassicOffsetPlan`). Other valid images stay `MetadataOnly` and decode
through the general CPU decoder.

### Mixed JPEG and JPEG 2000 input

`CpuBurnDecoder::decode_batch(&[ImageInput<'_>])` takes both codec families in
one call. `ImageInput::Jpeg` carries a `j2k_jpeg::JpegBatchInput`, and
`ImageInput::Jpeg2000` borrows an `EncodedImage`.

- JPEG decodes on the CPU, full image only, to RGB `U8`, with grayscale
  expanded to three channels. Grayscale JPEG 2000 stays one channel.
- The session's strictness applies to both codecs. JPEG color hints are kept.
- Errors and warnings say which codec they came from (`ImageDecodeError`,
  `ImageDecodeWarning`). `j2k_info()` gives a group's JPEG 2000 metadata.

### Output

Each group is one rank-4 Burn integer tensor, NCHW by default or NHWC:

| Source samples | Tensor |
| --- | --- |
| Unsigned, up to 8 bits | `BurnBatchTensor::U8` |
| Unsigned, 9 to 16 bits | `BurnBatchTensor::U16` |
| Signed, up to 16 bits | `BurnBatchTensor::I16` |

- Images share a group only if they match in decoded size, channel count,
  sample type, layout, transform, execution shape, and alpha kind.
  Straight and premultiplied alpha never mix. Nothing is padded.
- Batches support Gray, RGB, and RGBA. Other component layouts need the
  codec's component-plane APIs; see [Decoding](decoding.md).
- `BurnBatchGroup` holds the codec metadata, the source indices in tensor
  order, the decoded rectangles, and warnings. Group order is unspecified, so
  use `source_indices`.
- `BurnBatchDecode` holds the groups, the per-input preparation errors, and
  one `BurnBatchGroupError` for each group that failed. A failed group doesn't
  affect the others. Read it with `groups()`, `errors()`, and
  `group_errors()`, or take the groups with `into_groups()`.
- Decoding is strict by default. In a lenient session, a JPEG that needed
  recovery comes back with `recovered()` set; a JPEG 2000 metadata recovery is
  reported as a warning on its group. Lenient JPEG 2000 decoding doesn't
  salvage damaged entropy-coded data.
- A CUDA or Metal decoder never falls back to the CPU. Unsupported inputs and
  transfer failures are errors.

### Synchronous and submitted decodes

`CudaDirectBurnDecoder::decode` and `decode_prepared` return once the tensors
are safe to use. There is no asynchronous form, because Burn 0.21 doesn't
expose the CUDA stream needed to order the work safely.

The upload decoders also have `submit` and `submit_prepared`, which return a
handle; its `wait` finishes the decode, the copy to the host, and the Burn
upload. Only the decode overlaps with your own work. On CUDA, RGB and RGBA
groups larger than 32 images decode in 32-image chunks, and `submit` waits for
every chunk except the last. A decoder can be reused after a submitted batch
is dropped.

### GPU adapter support

The GPU decoders support less than the CPU decoder. "All requests" means
full, region, reduced, and region-and-reduced, in NCHW and NHWC.

| Output | CPU | CUDA | Metal |
| --- | --- | --- | --- |
| Gray, RGB, RGBA as `U8`, `U16`, `I16` | Classic and HTJ2K, all requests | Classic and HTJ2K, all requests, single- and multi-tile | Classic and HTJ2K, all requests, single-tile. Multi-tile: see below |

- On Metal, multi-tile images are tested for HTJ2K Gray12 and RGB8 with all
  four requests, and for classic RGB8 full-image decodes.
- Reversible 5/3 and irreversible 9/7 output match the CPU bit for bit.
- Components deeper than 16 bits don't decode on a GPU.
- Both GPU backends decode ROI maxshift (RGN markers) when the coded bitplane
  count, including the shift, is at most 31. Classic code blocks must fit in
  64 x 64; HTJ2K limits depend on the backend.
- Each group also has a dense, codec-owned buffer in GPU memory
  (`MetalBatchGroup::resident_batch`, `CudaResidentBatchBuffer`).
- A decode error on one image names that image. A Metal command-buffer
  failure can't be traced to one image, so the whole group fails and its
  error lists every source index. A single image larger than the configured
  HTJ2K chunk limits is an error. Partly written tensors are never returned.

Speed results for the Burn decoders are in
[Performance](performance.md#cuda-to-burn).

## MPSGraph

`j2k-mpsgraph` is an experimental adapter for Apple Silicon Macs running
macOS 11 or later. It hands decoded JPEG 2000 and HTJ2K batches to Apple's
MPSGraph while they are still in GPU memory, without reading pixels back to
the CPU. (`j2k-ml`'s Metal decoder, by contrast, copies through host memory.)
On Intel Macs and other systems, its methods return
`Error::UnsupportedPlatform`.

Each successful group becomes one static rank-4 tensor:

| Codec group | NCHW | NHWC | MPS data type |
| --- | --- | --- | --- |
| Gray `U8` | `[N,1,H,W]` | `[N,H,W,1]` | `UInt8` |
| RGB `U8` | `[N,3,H,W]` | `[N,H,W,3]` | `UInt8` |
| RGBA `U8` | `[N,4,H,W]` | `[N,H,W,4]` | `UInt8` |
| Gray, RGB, RGBA `U16` | Same shapes | Same shapes | `UInt16` |
| Gray, RGB, RGBA `I16` | Same shapes | Same shapes | `Int16` |

The crate doesn't provide models or graph builders. You build an `MPSGraph`,
one static rank-4 image placeholder whose shape and data type match an
`MpsGraphTensorSpec`, and one or more target tensors, and pass them to
`MpsGraphProgram::new`. The image is the only runtime input: weights and other
model inputs must be graph constants.

There are three ways to run it:

- `MpsGraphBatchDecoder::decode` returns an `MpsGraphBatchDecode`. Its groups
  wrap the codec's `MTLBuffer` as an `MpsGraphInputGroup` without copying, and
  `MpsGraphProgram::submit_completed` runs the graph on one. The command queue
  must belong to the same Metal device; a queue from another device is
  rejected.
- `MpsGraphBatchDecoder::run_prepared_group` queues the decode and the graph
  on one command queue and waits for both. The CPU doesn't wait between
  decode and inference.
- `MpsGraphBatchDecoder::submit_prepared_group` does the same and returns at
  once with a `SubmittedMpsGraphRun`. `is_complete` doesn't block; `wait`
  checks the decode status and returns the graph outputs.

`SubmittedMpsGraphRun` is neither `Send` nor `Sync`. Dropping one that is
still running waits for it first, because MPSGraph doesn't promise to keep the
input buffer alive. We don't call the adapter zero-copy: MPSGraph may copy
internally.

```bash
cargo run -p j2k-mpsgraph --example resident_reference_graph
```

The example builds a small normalize-and-average graph and runs it all three
ways.

## PyTorch and NumPy

The Python package decodes single images and batches to NumPy arrays and
PyTorch tensors in the image's own sample type, including batches that mix
JPEG and JPEG 2000, regions, and reduced resolutions. Built with CUDA, it
decodes into PyTorch-owned GPU memory; built with Metal, it returns `mps`
tensors. See [Python, C, and the command line](bindings.md).

## Wavelet coefficients

JPEG 2000 stores an image as wavelet bands. `J2kDecoder::decode_wavelets`
returns the bands you select as exact integers (reversible 5/3) or
dequantized floats (irreversible 9/7), without running the inverse
transform, so a model can train on them directly. In Python, the same is
`j2k.decode_wavelets`. Selection, memory use, and limits are described in
[Wavelet coefficients](decoding.md#wavelet-coefficients).
