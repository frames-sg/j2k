# GPU decoding and encoding

J2K can use an NVIDIA GPU through CUDA (`j2k-cuda`) or an Apple GPU through
Metal (`j2k-metal`). Both crates sit on top of `j2k` and use the same
`J2kDecoder` concepts, batch types, and options described in
[Decoding](decoding.md) and [Encoding](encoding.md). JPEG and JPEG-to-HTJ2K
transcoding have their own GPU crates, covered in [JPEG](jpeg.md#gpu-jpeg) and
[Transcoding](transcoding.md#gpu-acceleration).

## How it works

The CPU still reads the file's structure: headers, packets, and where each
code block's data is. The GPU does the expensive parts: block decoding,
dequantization, the inverse wavelet transform, color conversion, and writing
out pixels. Decoded pixels can stay in GPU memory and go straight into a
GPU-side pipeline without a copy back to the CPU. The result's `residency()`
says where the pixels are, for example `SurfaceResidency::CudaResidentDecode`
or `MetalResidentDecode`.

Output values are the same as the CPU decoder's, including the [sample
conversions](decoding.md#output-sample-conversion) for signed and
high-bit-depth sources.

## Choosing a backend

You pick the backend with `BackendRequest`:

- `Auto`, the default, uses the GPU only for image types and sizes where it
  was measured to be faster than the CPU, with identical output. Everything
  else runs on the CPU. The measured cases are listed in
  `xtask/routing-promotion-evidence.json`; they cover 8-bit gray and RGB
  decodes of classic JPEG 2000 and HTJ2K above a minimum size, on both
  backends, and some 8-bit encodes. [Performance](performance.md) has the
  measurements and explains how the thresholds are chosen.
- `Cuda` or `Metal` forces the GPU. If the GPU can't handle the request, you
  get an error.
- `Cpu` never touches the GPU.

A decode never retries on the CPU after a GPU error, whether you asked for
the GPU or `Auto` chose it; you get the error. `Auto` choosing the CPU for an
image outside the measured cases is a decision made before any GPU work, not
a fallback.

Decode one image on Metal:

```rust
use j2k::{BackendRequest, PixelFormat};
use j2k_metal::{J2kDecoder, MetalDecodeRequest};

let mut decoder = J2kDecoder::new(&bytes)?;
let surface = decoder.decode_request_to_device(MetalDecodeRequest::full(
    PixelFormat::Rgb8,
    BackendRequest::Auto,
))?;
println!("{:?}", surface.residency());
```

`MetalDecodeRequest::region`, `scaled`, and `region_scaled` cover the other
requests. For many images, use the batch decoders, which take the same
inputs and operations as `CpuBatchDecoder` (see [Batch
decoding](decoding.md#batch-decoding)):

```rust
use j2k::EncodedImage;
use j2k_cuda::CudaBatchDecoder;

let mut decoder = CudaBatchDecoder::new();
let prepared = decoder.prepare(files.into_iter().map(EncodedImage::full).collect())?;
let result = decoder.decode_prepared(&prepared)?;
for group in result.groups() {
    let surfaces = group.surfaces(); // CUDA-resident images
}
```

## CUDA

`j2k-cuda` runs on Linux with an NVIDIA driver. Enable its `cuda-runtime`
feature:

```bash
cargo add j2k-cuda --features cuda-runtime
```

Without the feature, the crate still builds: the CPU paths and session types
work, and explicit CUDA requests return `CudaUnavailable` or an
unsupported-request error.

The CUDA kernels are written in Rust and compiled to PTX with the cuda-oxide
toolchain (`cargo oxide build`) when the crate is built. That only happens on
a Linux host with cuda-oxide installed; other builds embed placeholder PTX,
and calling a placeholder kernel returns an error saying the PTX was not
built. Set `J2K_REQUIRE_CUDA_OXIDE_BUILD=1` when building for a GPU machine
to turn missing PTX into a build error.

By default the kernels target compute capability 8.0 (`sm_80`), so they need
an Ampere or newer GPU; the driver compiles the PTX for newer architectures
when it loads it. To target a different architecture, set
`J2K_CUDA_OXIDE_ARCH` (for example `J2K_CUDA_OXIDE_ARCH=sm_90`) when building.
See [Environment variables](environment-variables.md).

What runs on CUDA:

- Decoding of classic JPEG 2000 and HTJ2K with the 5/3 or 9/7 wavelet, within
  the adapter's supported image and code-block profiles. Components deeper
  than 16 bits are not decoded on CUDA.
- Region decodes plan only the code blocks, packets, and wavelet levels the
  region needs, and run each inverse wavelet level over the region's window.
- Under `Auto`: `Gray8` or `Rgb8` output of classic codestreams, lossy HTJ2K
  codestreams, and lossless HTJ2K JPH files, as a full image, a region, a
  half-scale image, or a batch of 16 copies of one input, each above a
  minimum size.
- Encoding from CUDA buffers (`encode_lossless_from_cuda_buffer` and its
  batch and device-output forms). These need HTJ2K block coding and
  `J2kEncodeValidation::External`, and reject classic block coding.

### Lossless encode from host memory

`CudaLosslessEncoder` encodes host samples and follows the
`EncodeBackendPreference` in each job's options:

```rust
use j2k::{EncodeBackendPreference, J2kBlockCodingMode, J2kLosslessEncodeOptions, J2kLosslessSamples};
use j2k_cuda::CudaLosslessEncoder;

let samples = J2kLosslessSamples::new(&pixels, 2048, 2048, 3, 8, false)?;
let options = J2kLosslessEncodeOptions::default()
    .with_backend(EncodeBackendPreference::Auto)
    .with_block_coding_mode(J2kBlockCodingMode::HighThroughput);
let mut encoder = CudaLosslessEncoder::new();
let result = encoder.encode(samples, &options)?;
println!("{:?} {:?}", result.actual_backend(), result.fallback_reason());
let codestream = result.into_encoded().codestream;
```

- `CpuOnly` never touches CUDA.
- `Auto` uses CUDA only for a shape whose CUDA route was measured faster than
  the CPU. For any other shape it encodes on the CPU; that is the chosen
  route, so the result carries no fallback reason. A measured shape that
  CUDA can't run (no device, or a stage it doesn't cover) completes on the
  CPU and reports why in `fallback_reason()`.
- `RequireDevice` returns an error unless CUDA runs the whole encode.
  `encode_strict_cuda` and `encode_j2k_lossless_with_cuda` always behave this
  way.

A CUDA execution error is returned, never hidden by retrying on the CPU. The
result also reports the requested preference and the number of CUDA
dispatches per stage. The encoder can move between threads but not be
shared, since each call takes `&mut self`; after an error it drops its
cached GPU state, so the next job starts clean.

## Metal

`j2k-metal` runs on macOS. It builds on other platforms, but every Metal
request returns an error there. Functions that take Metal objects use
`objc2-metal` protocol objects, not `metal-rs` types: pass retained objects
for devices, queues, command buffers, and buffers that J2K should own, and
borrowed references otherwise.

- `J2kDecoder::decode_into_metal` decodes a full image into a caller-owned
  `MetalImageDestination`, so a stream of same-size frames can reuse one
  allocation. It writes `Gray8` or `Gray16` for one component, `Rgb8` for
  8-bit RGB, and `Rgb16` for 9- to 16-bit RGB, and returns once the GPU work
  is done. Codestreams Metal can't decode directly return
  `MetalDirectFallback`.
- The full-image and tile-batch decodes also handle RGB with subsampled
  components.
- `MetalBatchDecoder` decodes prepared batches (`decode_prepared`, or
  `submit_prepared` to overlap with other work). For groups of at least
  eight distinct classic JPEG 2000 images with unsigned 8-bit RGB and
  subsampled components, it moves block decoding to CPU workers on its own.
- `MetalBatchDecoder::decode_prepared_cooperative(&prepared, workers)` shares
  block decoding of a large classic RGB8 group between the GPU and up to
  `workers` CPU threads; reconstruction and output stay on the GPU. It needs
  at least eight distinct, full-resolution, reversible RGB8 images of at
  least 16,384 pixels each. Smaller batches, repeated images, grayscale,
  higher bit depths, and HTJ2K run entirely on the GPU. Whether it helps
  depends on the images, the CPU budget, and the device.

### Encoding on Metal

Pass a `MetalEncodeStageAccelerator` to the `j2k` encode functions:

```rust
use j2k::{encode_j2k_lossless_with_accelerator, BackendKind, J2kBlockCodingMode};
use j2k::{J2kLosslessEncodeOptions, J2kLosslessSamples};
use j2k_metal::MetalEncodeStageAccelerator;

let samples = J2kLosslessSamples::new(&pixels, 2048, 2048, 3, 8, false)?;
let options = J2kLosslessEncodeOptions::default()
    .with_block_coding_mode(J2kBlockCodingMode::HighThroughput);
let mut accelerator = MetalEncodeStageAccelerator::for_auto_host_output();
let encoded =
    encode_j2k_lossless_with_accelerator(samples, &options, BackendKind::Metal, &mut accelerator)?;
```

Under `Auto`, Metal is used only where it was measured faster:

- Lossless HTJ2K to host memory, in batches of more than one image: RGB8 at
  1024 x 1024, and Gray8 or RGB8 at 2048 x 2048. Metal prepares the
  coefficients and runs HT block coding; the CPU packetizes.
- Lossy 8-bit RGB of at least 2048 x 2048 pixels. Metal runs the transform
  and quantization.

Everything else encodes on the CPU under `Auto`, and for host output, classic
block coding, packetization, and codestream assembly always stay on the CPU.
An explicit Metal request has no size threshold: it runs on Metal when the
shape is supported and returns `UnsupportedMetalRequest` otherwise.

To keep the encoded codestream in a Metal buffer, use
`submit_lossless_batch_to_metal`. The encode is fully on the GPU only when
its `MetalLosslessEncodeResidency` reports coefficient preparation,
packetization, and codestream assembly all used. For input already in Metal
buffers without a copy, pass `MetalLosslessEncodeTile`s with
`MetalEncodeInputStaging::AlreadyPaddedContiguous`. `submit_lossless_batch`
does the same with host output.

Three examples show the routes on your machine:

```bash
cargo run -p j2k-metal --example decode_route_report      # which backend Auto picks
cargo run -p j2k-metal --example htj2k_encode_auto_report # encode backend and stage counts
cargo run -p j2k-metal --example resident_encode_buffer   # encode into a Metal buffer
```
