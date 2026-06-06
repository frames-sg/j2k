# WSI Decode API

This guide describes the public decode surfaces intended for whole-slide
imaging readers. It covers the stable caller contract shared by JPEG,
JPEG 2000 / HTJ2K, tile decompression, and the device-output adapters.

## Ownership Model

signinum does not own a viewer runtime. Callers own I/O, threading, tile
coordinates, pyramid selection, cache policy, and prefetch. Codec APIs only
parse compressed bytes and write decoded pixels into caller-provided storage.

Use caller-owned state for hot loops:

- `ScratchPool` reuses temporary allocations within one codec family.
- `DecoderContext` reuses codec tables and planning state across tile batches.
- `DeviceSubmission` lets adapter crates queue work and return a `DeviceSurface`
  after `wait()`.

The codec crates do not spawn worker threads, hold global decode queues, or
hide output allocation behind a runtime.

## CPU Decode Surfaces

Use `ImageDecode` when the caller has one compressed image or tile and wants
pixels in host memory.

Common shapes:

- `decode_into` decodes the full image.
- `decode_region_into` decodes a source-coordinate ROI.
- `decode_scaled_into` decodes the full image at a reduced resolution.
- `decode_region_scaled_into` decodes a source-coordinate ROI on a reduced
  resolution grid.

ROI coordinates are always expressed in source-image pixels. For
`decode_region_scaled_into`, the returned `DecodeOutcome::decoded` rectangle is
the floor-start / ceil-end projection of the source ROI into the scaled grid.
`Downscale::None` preserves the original ROI.

```rust
use signinum_core::{Downscale, ImageDecode, PixelFormat, Rect};
use signinum_j2k::{J2kDecoder, J2kScratchPool};

let bytes = std::fs::read("tile.jp2")?;
let mut decoder = J2kDecoder::new(&bytes)?;
let roi = Rect {
    x: 512,
    y: 512,
    w: 1024,
    h: 1024,
};
let scale = Downscale::Quarter;
let scaled = roi.scaled_covering(scale);
let stride = scaled.w as usize * PixelFormat::Gray8.bytes_per_pixel();
let mut out = vec![0_u8; stride * scaled.h as usize];

decoder.decode_region_scaled_into(
    &mut J2kScratchPool::new(),
    &mut out,
    stride,
    PixelFormat::Gray8,
    roi,
    scale,
)?;
```

## Row Streaming

Use `decode_rows` through `ImageDecodeRows` when a tile or image is too large
for one packed output buffer or when the caller wants to feed rows into a
streaming consumer. The caller implements `RowSink`, and signinum forwards sink
errors without converting them into silent decode success.

Row streaming is a host-memory API. Device adapters return surfaces instead of
row callbacks.

## Tile Batches

Use `TileBatchDecode` when a WSI reader is decoding many independent tile
payloads with the same codec. The caller keeps one `DecoderContext` and one
`ScratchPool`, then calls the stateless tile helper repeatedly.

```rust
use signinum_core::{DecoderContext, PixelFormat, TileBatchDecode};
use signinum_jpeg::{JpegCodec, ScratchPool};

let mut ctx = DecoderContext::<signinum_jpeg::DecoderContext>::new();
let mut pool = ScratchPool::new();

for tile in visible_tiles {
    JpegCodec::decode_tile(
        &mut ctx,
        &mut pool,
        tile,
        &mut output,
        stride,
        PixelFormat::Rgb8,
    )?;
}
```

Tile-batch helpers exist for full, ROI, scaled, and ROI+scaled decode. The
same source-coordinate ROI and reduced-grid coverage rules apply to tile-batch
ROI+scaled decode.

## Device Surfaces

Use `ImageDecodeDevice`, `ImageDecodeSubmit`, `TileBatchDecodeDevice`,
`TileBatchDecodeManyDevice`, or `TileBatchDecodeSubmit` when a downstream
pipeline wants a backend-tagged surface. Completed operations return a
`DeviceSurface`, which reports:

- backend kind
- dimensions
- pixel format
- byte length

Backend selection uses `BackendRequest`:

- `BackendRequest::Auto` / `BackendRequest::ACCELERATED` lets the adapter plan
  an adaptive accelerated route. Auto is conservative and may return CPU-backed
  surfaces when benchmark gates or shape support do not justify device
  execution.
- `BackendRequest::Cpu` / `BackendRequest::CPU_ONLY` requires host-backed CPU
  output.
- `BackendRequest::Metal` / `BackendRequest::STRICT_METAL` requires resident
  Metal execution on macOS.
  CPU-decoded bytes are not uploaded to satisfy this request. Call explicit
  CPU-staged upload APIs where the adapter exposes them when a Metal buffer is
  needed after CPU decode. Unsupported explicit Metal requests return an error.
- `BackendRequest::Cuda` / `BackendRequest::STRICT_CUDA` requires CUDA device memory
  output. When an adapter is built with `cuda-runtime` and a CUDA driver is
  available, explicit CUDA requests return CUDA-backed surfaces.
  `signinum-jpeg-cuda` uses Signinum-owned CUDA kernels for supported
  full-frame RGB8 4:2:0, 4:2:2, and 4:4:4 strict CUDA JPEG decode. Region,
  scaled, and non-RGB8 strict CUDA JPEG requests fail as unsupported rather
  than silently
  CPU-decoding and uploading pixels. `signinum-j2k-cuda` reserves this request for
  CUDA-resident HTJ2K codestream decode and lossless encode; it rejects
  unsupported classic JPEG 2000 or unsupported HTJ2K shapes instead of
  CPU-decoding and uploading pixels. Hosts without CUDA return unavailable.
  `Cpu` and ungated `Auto` remain CPU-backed host surfaces.

For Metal adapters, `BackendRequest::Auto` is a routing hint and may fall back
to host-backed CPU output when the request shape is not on the Metal-supported
path. `BackendRequest::Metal` is a strict request: supported shapes return
resident Metal-backed decode surfaces, unsupported shapes fail as unsupported,
and hosts without Metal fail as unavailable.
Adapters that expose `SurfaceResidency` mark true resident decode separately
from CPU-staged Metal upload so WSI pipelines do not count upload buffers as GPU
decode.

For JPEG routing, `JpegCapabilityReport` exposes parser-owned metadata and
backend eligibility without duplicating marker/table logic in higher layers.
Use `metal_fast` for broad Metal fast-packet shape support and
`metal_resident_rgb8_batch_output()` when routing to the current reusable
caller-owned RGB8 Metal buffer/texture batch APIs. The resident-output query is
narrower than `metal_fast`: it requires RGB8 output and a full, scaled, or
region-scaled batch shape supported by those reusable-output APIs.
`MetalBatchOutputBuffer::ensure_rgb8_tiles` and
`MetalBatchTextureOutput::ensure_rgba8_tiles` retain existing Metal allocations
when the requested tile shape already fits and replace them only when the
layout or capacity must change. Their scaled and region-scaled variants compute
the output shape from the full dimensions or source ROI. The
`Codec::decode_rgb8_*_into_resizable_metal_{buffer,textures}_with_session`
helpers combine request parsing, output resize, and resident batch decode for
viewport loops that reuse one caller-owned output across changing tile counts
or output shapes. For contiguous viewport workloads,
`decode_viewport_region_to_resizable_metal_{buffer,textures}_with_session`
exposes the same reusable-output behavior at the viewport layer and rejects
non-contiguous composition shapes instead of materializing a CPU viewport.
For sparse or non-contiguous RGB8 viewport composition,
`compose_viewport_to_resizable_metal_{buffer,textures}_with_session` decodes
component rows into reusable Metal plane buffers and packs the composed viewport
into a caller-owned Metal buffer or texture.

Callers should use explicit device requests only when they need that backend.
Use `Auto` for viewer paths where CPU fallback is acceptable.

## Error Contract

No decode path should fail silently. Unsupported formats, invalid regions,
too-small buffers, too-small strides, unavailable explicit backends, and row
sink aborts are returned as errors. Callers should handle `CodecError`
predicates for broad policy decisions and preserve detailed errors for logging.

## CUDA HTJ2K Lossless Encode

`signinum-j2k-cuda` exposes `encode_j2k_lossless_with_cuda` for on-device
HTJ2K lossless encode. The function targets a codestream byte-identical to
the public `signinum-j2k` lossless HTJ2K CPU encode path.

**Supported inputs:** reversible 5/3 DWT, HT cleanup-pass-only, single tile /
single quality layer / single precinct, 1-component (grayscale), 3-component
(RGB — MCT/RCT on all three planes), or 4-component (RGBA/CMYK — MCT/RCT on
the first three planes; 4th component passed through), bit depths 8–16 unsigned
or signed (signed = encode/codestream byte-parity only; native decode does not
reconstruct signed samples — see Non-goals), multi-level DWT, multi-codeblock.
Component subsampling must be (1,1).

**Parity contract:** byte-parity against the CPU reference is the contract
enforced by the `cuda-x86_64-compatibility` job in
`.github/workflows/gpu-validation.yml`. That job sets
`SIGNINUM_REQUIRE_CUDA_RUNTIME` and runs the `htj2k_encode_parity` tests on the
self-hosted CUDA runner with a fail-closed executed-count floor, so parity tests
cannot silently skip. This job is the authoritative gate before merging changes
to the CUDA encode path.

**No silent fallback:** out-of-scope or unavailable requests return a typed
error. The accelerator never silently falls back to the CPU path for an
in-scope input.

**Non-goals** (explicitly out of scope):

- Classic/tier-1 EBCOT coding — HTJ2K-only path.
- Lossy 9/7 DWT — never byte-exact.
- Multiple quality layers — native reference is single-layer.
- Multi-tile within one codestream — native reference is single-tile; tiling is
  done at the caller/per-codestream level.
- 2-component images — the native decoder rejects 2-component with
  `TooManyChannels`, so round-trip validation is not possible.
- Component subsampling != (1,1) — changes block geometry the strict kernel
  does not handle.
- HT SigProp/MagRef refinement passes — experimental; beyond the native
  cleanup-pass-only path and not round-trip-validated against the native
  reference.
- Native decode reconstruction of signed samples — the encoder produces
  spec-correct, byte-parity signed codestreams, but the shared native decoder
  ignores the SIZ `Ssiz` signed bit and does not reconstruct signed samples
  (output is offset by `+2^(depth-1)`). This affects the CPU and Metal decode
  paths identically; it is not a CUDA *encode* issue. The parity gate asserts
  byte-exact pixel round-trip for unsigned cells only; signed cells assert
  codestream byte-parity plus a successful decode.

## Current Validation Scope

Hosted CI validates CPU behavior, adapter fallback behavior, rustdoc, and
benchmark compilation. Runtime GPU validation is available through the manual
`.github/workflows/gpu-validation.yml` workflow on self-hosted runners:

- Apple Silicon runners labeled `self-hosted`, `macOS`, `ARM64`, `metal`
  validate Metal tests and optionally timed Metal benchmarks.
- x86_64 CUDA runners labeled `self-hosted`, `Linux`, `X64`, `cuda` validate
  CUDA device-memory output with `cuda-runtime`, the owned full-frame RGB8
  JPEG CUDA path, and the `htj2k_encode_parity` suite for the CUDA HTJ2K
  lossless encode path. Timed NVIDIA performance claims require the workflow's
  timed benchmark mode and recorded output.
