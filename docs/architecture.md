# Architecture

This document is the system map for `signinum`. It is the first thing a new
contributor or coding agent should read before changing code, and it is the
single source of truth for how the workspace is shaped, where each
responsibility lives, and which dependency directions are legal. Anything that
is not described here, in a referenced design note, or in a crate-level README
should be treated as undocumented and verified before it is relied on.

The structure of this file follows the harness-engineering convention of
keeping a repository-local map, an explicit layering, and cross-links to
design notes that an agent can reach without leaving the repo.

## Companion documents

- [`README.md`](../README.md) — public surface, supported APIs, MSRV, examples.
- [`CHANGELOG.md`](../CHANGELOG.md) — release history.
- [`docs/bench.md`](bench.md) — benchmark methodology and comparator policy.
- [`docs/parity.md`](parity.md) — parity expectations against reference decoders.
- [`docs/release.md`](release.md) — release staging notes.
- [`docs/support-matrix.md`](support-matrix.md) — stable surfaces, backend limits,
  MSRV, and benchmark-publication gates.
- [`docs/wsi-decode-api.md`](wsi-decode-api.md) — public WSI decode API guide.
- [`docs/jpeg-support-phases/`](jpeg-support-phases/README.md) — CPU-first JPEG
  compatibility expansion plan and Metal promotion gates.
- [`docs/wsi-dicom-passthrough.md`](wsi-dicom-passthrough.md) — passthrough-first
  policy for WSI/DICOM conversion layers built on these codec primitives.
- Crate-level `README.md` files where present — crate-scoped contracts and
  feature notes.

## System map

The workspace is a single Cargo workspace defined in [`Cargo.toml`](../Cargo.toml).
Most library and binary crates live under `crates/`; the workspace automation
crate lives under `xtask/`. Workspace crates share `edition = 2021` and
`rust-version = 1.88`. The current public line is `0.5.x`: the facade, core,
CPU codec, tilecodec, and CLI surfaces are treated as stable for the facade
release, while adapter and transcode APIs remain experimental until their
promotion gates are satisfied.

| Crate | Layer | Role |
|-------|-------|------|
| `signinum-core` | foundation | Shared traits, pixel/sample types, backend capability metadata, device-surface contracts, scratch/context contracts. No image-format logic. |
| `signinum-profile` | instrumentation helper | Shared profiling helpers used by implementation crates at runtime. Published only because public crates depend on it; not a user-facing API. |
| `signinum-cuda-runtime` | runtime helper | CUDA Driver API, CUDA memory, and bundled kernel launch helpers used by CUDA adapters. Published support crate, not the primary user-facing API. |
| `signinum-tilecodec` | codec | Tile decompression primitives: Deflate, Zstd, LZW, Uncompressed. Implements `TileDecompress` from `core`. |
| `signinum-jpeg` | codec | Native pure-Rust JPEG inspect/decode for supported WSI-oriented 8-bit Huffman classes. CPU-first. Owns SIMD backends and fused entropy/IDCT/upsample paths for current sequential fast paths; broader JPEG compatibility is tracked in [`docs/jpeg-support-phases`](jpeg-support-phases/README.md). Its baseline JPEG encoder is a compatibility/fallback utility, not the diagnostic WSI/DICOM encode path. |
| `signinum-j2k-native` | codec engine | Published implementation dependency for `signinum-j2k`; not the stable user-facing API. Lives under `#![forbid(unsafe_code)]` and uses `fearless_simd`. |
| `signinum-j2k` | codec | Public JPEG 2000 / HTJ2K crate. Wraps `j2k-native` with the signinum-core trait surface (inspect, decode, ROI, scaled, row-bounded, tile-batch, and lossless encode). |
| `signinum-j2k-compare` | dev-only | OpenJPEG FFI bindings used as a reference decoder for conformance and parity testing. Unpublished. |
| `signinum-jpeg-metal` | adapter | Apple Metal device-output adapter for `signinum-jpeg`. Hosts compute kernels for color conversion, interleave/pack, and `MTLBuffer` production. |
| `signinum-j2k-metal` | adapter | Apple Metal device-output adapter for `signinum-j2k`. Same shape as the JPEG adapter. |
| `signinum-jpeg-cuda` | adapter | CUDA-facing API adapter for JPEG. `Auto`/`Cpu` stay host-backed; explicit supported full-frame RGB8 4:2:0, 4:2:2, and 4:4:4 CUDA requests use Signinum-owned kernels, and unsupported strict CUDA shapes return clear errors. |
| `signinum-j2k-cuda` | adapter | CUDA-facing API adapter for J2K. Explicit CUDA requests are strict CUDA-resident HTJ2K codestream decode requests when `cuda-runtime` and a CUDA driver are available; CPU-decode-then-upload is exposed only through explicitly named CPU-staged APIs. |
| `signinum-transcode` | experimental | Coefficient-domain JPEG to HTJ2K transcode experiments. Owns the coupling between JPEG DCT extraction and native HTJ2K coefficient encode so codec crates stay independent. APIs in this crate are not stable until validation coverage and codestream integration land. |
| `signinum-transcode-cuda` | experimental adapter | CUDA accelerator for selected `signinum-transcode` stages. Uses `signinum-cuda-runtime` kernels for coefficient-domain DCT-grid to wavelet and fused 9/7 code-block paths. |
| `signinum-transcode-metal` | experimental adapter | Metal accelerator for selected `signinum-transcode` stages. Keeps JPEG parsing, entropy decode, dequantization, scheduling, and HTJ2K encode on CPU while optionally replacing direct DCT-grid to wavelet projection. |
| `signinum-test-support` | dev helper | Shared synthetic-image and benchmark input generators for tests, benches, and examples. Not a runtime dependency of public crates. |
| `signinum` | facade | Stable public import surface over `core`, the CPU codecs, tile decompression, and optional Metal/CUDA adapters behind facade features. |
| `signinum-cli` | binary | `signinum inspect <file>` entry point. Header parsing only, no decode. |
| `xtask` | workspace tool | Repository-local automation for tests, docs, benches, fuzz builds, coverage, and packaging. Lives under `xtask/` instead of `crates/` and is never published. |

Out-of-tree but in-repo:

- `corpus/` — committed and optional local test data: `conformance/`,
  `wsi-samples/`, `regressions/`, and `fuzz-seeds/`. Committed fixtures must
  carry source/license/tolerance metadata in the matching manifest or test
  documentation; optional real-slide corpora stay out of the default checkout.
- `paper/` — research paper materials.
- `target/` — build output (gitignored).

## Layered architecture and dependency rules

signinum is organized as foundation/helper crates, codec engines, codecs,
adapters, and facade/binary surfaces. Dependencies must flow inward only. An
agent adding a dependency edge that
points outward is changing the architecture and should stop and update this
document first.

```
foundation / helper crates  →  codec engines  →  codecs  →  adapters / transcode  →  facade / binary
```

| Layer | Members | May depend on |
|-------|---------|---------------|
| foundation | `signinum-core` | `thiserror` only. No other workspace crate. `no_std + alloc` posture. Contains only the x86 CPUID/XGETBV unsafe required for CPU feature detection. |
| helper crates | `signinum-profile`, `signinum-cuda-runtime` | No workspace crate. `signinum-profile` may be used by codec engines, codecs, and adapters for runtime instrumentation support. `signinum-cuda-runtime` may be used by CUDA adapters for driver/runtime integration. These crates must not become primary user-facing APIs and must not depend on codecs or adapters. |
| codec engines | `signinum-j2k-native` | helper crates. Internal only. Not re-exported. |
| codecs | `signinum-jpeg`, `signinum-j2k`, `signinum-tilecodec` | foundation, codec engines, helper crates. Must not depend on each other. Must not depend on adapters or `cli`. |
| adapters | `signinum-jpeg-metal`, `signinum-j2k-metal`, `signinum-jpeg-cuda`, `signinum-j2k-cuda` | foundation, helper crates, exactly one matching codec, optional engine for the matching codec. Adapters in different format families must not depend on each other. |
| experimental transcode | `signinum-transcode`, `signinum-transcode-cuda`, `signinum-transcode-metal` | `signinum-transcode` may depend on foundation and codec crates once integration begins. `signinum-transcode-cuda` may depend on `signinum-transcode`, `signinum-j2k-native`, and `signinum-cuda-runtime`; `signinum-transcode-metal` may depend on `signinum-transcode` and platform GPU APIs only. Neither crate may create codec-to-codec dependencies. |
| facade | `signinum` | foundation, codecs, tilecodec, optional adapters behind feature gates. |
| binary | `signinum-cli` | codecs. Must not depend on adapters (kept host-neutral). |
| dev-only | `signinum-j2k-compare` | Foundation, codecs under test, optional adapter crates, and comparator libraries needed for tests/benches. Never runtime dependencies of public crates. |

Hard rules enforced today (the goal is to mechanize these as the workspace
matures, mirroring harness-engineering structural tests):

1. `signinum-core` is a leaf in the import graph. It does not import any
   other workspace crate.
2. Codec crates do not import each other. Cross-format work goes through
   `core` types or through caller code.
3. Adapter crates are additive. Removing all adapter crates must leave the
   codec stack fully functional on CPU.
4. Metal sources are gated by `cfg(target_os = "macos")`. Non-macOS hosts
   compile the adapter crate to a thin fallback that exercises the same
   public API but reports unavailability.
5. CUDA sources expose the same device-output surface. Explicit CUDA requests
   produce CUDA device memory when `cuda-runtime` and a CUDA driver are
   available. JPEG full-frame RGB8 4:2:0, 4:2:2, and 4:4:4 requests use
   Signinum-owned CUDA kernels where supported; unsupported JPEG shapes use
   explicit CPU-staged upload APIs where exposed. J2K CUDA explicit
   requests are strict CUDA-resident HTJ2K codestream decode requests and must
   not silently CPU-decode and upload pixels.
6. `signinum-jpeg` keeps its NEON and x86 intrinsics scoped per-backend
   in `crates/signinum-jpeg/src/backend/`. `signinum-j2k-native` keeps
   its SIMD behind `fearless_simd` so the engine can stay
   `#![forbid(unsafe_code)]`.
7. Adapter crates consume codec planning hooks through public `adapter`
   modules. Imports from codec `__private` modules are not allowed.

## Crate dependency graph

Workspace edges (excluding external crates and `dev-dependencies`):

```
signinum-core         (leaf)
signinum-profile      (instrumentation helper)
signinum-cuda-runtime (CUDA runtime helper)
signinum-test-support (dev helper)
xtask                 (workspace automation)

signinum-tilecodec    -> signinum-core

signinum-jpeg         -> signinum-profile, signinum-core
signinum-jpeg-metal   -> signinum-jpeg, signinum-profile, signinum-core
signinum-jpeg-cuda    -> signinum-jpeg, signinum-cuda-runtime, signinum-profile, signinum-core

signinum-j2k-native   -> signinum-profile
signinum-j2k          -> signinum-j2k-native, signinum-core
signinum-j2k-metal    -> signinum-j2k, signinum-j2k-native, signinum-profile, signinum-core
signinum-j2k-cuda     -> signinum-j2k, signinum-j2k-native, signinum-cuda-runtime, signinum-profile, signinum-core

signinum-transcode    -> signinum-jpeg, signinum-j2k, signinum-j2k-native
signinum-transcode-cuda -> signinum-transcode, signinum-j2k-native, signinum-cuda-runtime
signinum-transcode-metal -> signinum-transcode

signinum              -> signinum-core, signinum-jpeg, signinum-j2k, signinum-tilecodec, signinum-jpeg-metal, signinum-j2k-metal, signinum-jpeg-cuda, signinum-j2k-cuda
signinum-cli          -> signinum-jpeg, signinum-j2k

signinum-j2k-compare  -> signinum-core, signinum-j2k (test/bench reference only)
```

## Core abstractions

These live in `signinum-core` and are the contract every codec and adapter
implements. New extension points belong here.

### Codec entry traits

- `ImageCodec` — base trait. Associated types: `Error`, `Warning`, `Pool`.
- `ImageDecode<'a>` — CPU decode surface. Methods include `inspect`, `parse`,
  `decode_into`, `decode_into_with_scratch`, `decode_region_into`,
  `decode_scaled_into`, and `decode_region_scaled_into`.
- `ImageDecodeRows<'a, S: RowSink<_>>` — row-bounded decode for large tiles.
- `ImageDecodeDevice<'a>` — synchronous device decode; returns a
  `DeviceSurface`.
- `ImageDecodeSubmit<'a>` — asynchronous device decode; returns a
  `DeviceSubmission` whose `wait()` produces a `DeviceSurface`.
- `TileBatchDecode` — stateless tile decode with a caller-owned
  `DecoderContext` reused across tiles.
- `TileDecompress` — generic tile decompression (used by `tilecodec`).

### Backend and surface model

- `BackendKind` — `Cpu | Metal | Cuda`.
- `BackendRequest` — `Auto | Cpu | Metal | Cuda`. Callers state intent.
  `Auto` may resolve to a CPU-backed surface; explicit unsupported device
  requests return an error before decode work.
- `DeviceSurface` — trait describing decoded data sitting on a backend
  (CPU buffer, `MTLBuffer`, etc.). Queryable for backend kind, dimensions,
  pixel format, byte length.
- `DeviceSubmission` — async handle with `wait() -> DeviceSurface`.
- `ReadySubmission<T, E>` — synchronous submission used by CPU fallbacks.
- `CpuFeatures` — runtime detection of AVX2 / SSE4.1 / NEON, cached.

### Pixel and layout types

- `PixelFormat` — `Rgb8 | Rgba8 | Gray8 | Rgb16 | Rgba16 | Gray16`.
- `PixelLayout` — `Rgb | Rgba | Gray`.
- `Sample`, `SampleType`.
- `Downscale` — reduced-resolution decode specifier.
- `RowSink<S>` — caller-implemented sink for row streaming.
- `Rect` — ROI.
- `Info` — image metadata: dimensions, colorspace, components, bit depth,
  tile layout, MCU/coded-unit layout, restart interval, resolution levels.
- `Colorspace` — `Grayscale | YCbCr | Rgb | Cmyk | Ycck | SRgb | SGray |
  IccTagged | Rct | Ict`.
- `DecodeOutcome<W>` — decoded `Rect` plus `Vec<W>` warnings.

### Caller-owned state

- `ScratchPool` — caller-owned reusable allocations, reset per operation.
- `CodecContext` — codec-specific state (e.g., JPEG quantization tables).
- `DecoderContext<C>` — wrapper holding a `CodecContext`, persistent across
  tiles in a batch.

These three types encode a deliberate harness contract: the codec never
hides allocation, threading, or runtime state. WSI readers compose those
themselves.

## Decode pipeline

The shape is the same for both image codecs, modulo the engine inside.

```
compressed bytes
    │
    ▼
inspect(bytes)              → Info (no decode)
    │
    ▼
view = View::parse(bytes)   → borrowed parse (no copy)
    │
    ▼
decoder = Decoder::from_view(view)
    │
    ├─ CPU path ───────────────────────────────────────────────┐
    │   decode_into / decode_into_with_scratch                 │
    │   decode_region_into / decode_scaled_into /              │
    │   decode_region_scaled_into / decode_rows                │
    │     entropy → IDCT or DWT → color conv → output buffer   │
    │     SIMD: AVX2 / SSE4.1 / NEON (jpeg)                    │
    │           fearless_simd (j2k-native)                     │
    │   returns DecodeOutcome<Warning>                         │
    │                                                           │
    └─ Device path (Metal today, CUDA API-only) ───────────────┘
        submit_to_device(session, fmt, BackendRequest::Metal)
            │
            ▼
        capability check
            │
            ├─ supported shape: prepare packet, upload to MTLBuffer,
            │   dispatch compute kernel (color conv, interleave/pack)
            │   → DeviceSubmission → wait() → Surface (with MTLBuffer)
            │
            └─ unsupported explicit backend: fail before decode;
                Auto/Cpu may wrap CPU output in a host-backed DeviceSurface
```

Current 8-bit sequential JPEG fast paths are fused on CPU: entropy decode, IDCT
scheduling, upsampling, ROI, and the fast 4:2:0 path live together in
`crates/signinum-jpeg/src/entropy/sequential.rs` because splitting them
regresses WSI tile-batch performance. Initial CMYK/YCCK `Rgb8`
full/ROI/scaled/region-scaled and `Rgba8` full/ROI CPU conversion now lives in
that fused sequential path, including RGB row streaming for the supported
four-component fixtures. Progressive 8-bit
full/ROI/scaled/region-scaled CPU decode uses full progressive coefficient
assembly followed by output projection. Initial 12-bit extended sequential
grayscale full-image/ROI/scaled/region-scaled decode writes native `Gray16`
samples or expanded `Rgb16` samples, including restart-coded grayscale streams.
Initial 12-bit progressive grayscale full-image/ROI/scaled/region-scaled decode
writes native `Gray16` samples or expanded `Rgb16` samples; initial 12-bit
APP14 RGB 4:4:4 and YCbCr 4:4:4/4:2:2/4:2:0 full-image/ROI/scaled/
region-scaled decode writes native `Rgb16` samples, including restart-coded
color streams. Initial SOF3 8-bit
grayscale/RGB row streaming has landed. Expanded four-component
subsampled/malformed coverage, other 12-bit subsampled color support, stronger
non-constant 12-bit oracle fixtures, and
broader lossless SOF3 YCbCr/16-bit color plus 16-bit row support remain separate
CPU parity work.
Initial SOF3 8-bit grayscale `Gray8` and 16-bit grayscale `Gray16`
full-image/ROI/scaled/region-scaled decode for predictors 1-7, including
restart-coded grayscale streams, plus 8-bit APP14 RGB `Rgb8` decode for
predictors 1-7, including restart-coded APP14 RGB streams, is implemented as a
non-DCT predictor path. Splitting the module is planned but gated on stable
benchmark and parity coverage.

J2K parses boxes (COD, QCD, QCC, etc.) and codestream structure on CPU, then
drives either the native CPU reconstruction path or a MetalDirect plan. ROI and
reduced-resolution requests share the same core contract: the ROI is expressed
in source coordinates, and the returned decoded rectangle covers the
floor-start/ceil-end projection onto the requested reduced-resolution grid.

The current MetalDirect path is first-class for grayscale and RGB J2K/HTJ2K
full-tile decode and ROI+scaled tile batches. Marker parsing and plan building
stay on CPU; supported classic Tier-1 or HT cleanup block jobs, grouped
sub-band decode, IDWT, optional MCT, and final store/pack run as one Metal
command sequence and return resident Metal surfaces. Distinct grayscale and RGB
WSI-style ROI+scaled batches are coalesced across separate codestreams. Cropped
ROI+scaled plans prune code-block jobs outside the requested store windows,
compact retained HTJ2K coded payloads, and crop every required IDWT output
level. Cropped IDWT outputs carry input-window origins and strides through the
resident band graph, so intermediate levels can feed later cropped levels
without returning to broad intermediate buffers.

Unsupported formats, unsupported codestream features, and non-macOS hosts fall
back through the CPU reconstruction and device-surface upload path according to
the requested backend. Explicit Metal requests fail for unsupported Metal
shapes; `Auto` is intentionally limited to measured grayscale/RGB batch cases.

## WSI/DICOM conversion policy

WSI/DICOM container readers and writers live outside this codec workspace. They
should inspect compressed tile payloads and pass them through unchanged whenever
the destination transfer syntax and frame metadata make that legal. The shared
contract is `signinum-core::PassthroughCandidate` plus
`PassthroughRequirements`; codec views build candidates from borrowed source
bytes, and the container layer remains responsible for DICOM-specific frame
ordering and fragment writing. If a new diagnostic codestream is required, use
the lossless J2K/HTJ2K encode surfaces. Baseline JPEG encode is reserved for
explicit fallback, generated fixtures, or non-diagnostic derived output.

Metal adapter routing is explicit after the facade release.
`BackendRequest::Cpu` returns host-backed CPU surfaces. `BackendRequest::Auto`
may select Metal only for validated adapter-supported shapes; otherwise it
falls back to a host-backed CPU surface. `BackendRequest::Metal` is strict: it
returns a resident Metal decode surface for supported shapes or a clear
unsupported/unavailable error. It does not silently return CPU output or
CPU-staged Metal upload. Adapters that expose staged upload do so as a separate
API, and resident-capable surfaces report residency so downstream WSI code can
reject CPU-staged buffers when end-to-end GPU residency is required.

## Backend story

There are three target backends. Selection is explicit in the public API.

- **CPU** — always available. Pure Rust, with SIMD where it is profitable.
  This is the baseline; every device path must have a working CPU fallback
  with equivalent results to within documented tolerance.
- **Metal** — Apple Silicon macOS. Compute kernels live next to the adapter
  crate. The adapter owns its `MTLDevice`/`MTLCommandQueue` session and
  produces `MTLBuffer`-backed `DeviceSurface`s so downstream GPU pipelines
  can consume the result without an extra download.
- **CUDA** — explicit device-memory output. `Auto` and `Cpu` return CPU-backed
  host surfaces. `BackendRequest::Cuda` returns CUDA device memory when the
  `cuda-runtime` feature and a CUDA driver are available, and otherwise reports
  CUDA as unavailable. JPEG full-frame RGB8 4:2:0, 4:2:2, and 4:4:4 can decode
  through Signinum-owned CUDA kernels; unsupported JPEG shapes use explicit
  CPU-staged upload APIs where exposed. J2K CUDA reserves explicit CUDA
  requests for CUDA-resident HTJ2K codestream decode and lossless encode (see
  [CUDA HTJ2K lossless encode](#cuda-htj2k-lossless-encode) below).

`BackendRequest::Auto` stays conservative: small or low-yield decodes are
served from CPU; larger batches with supported shapes can be routed to a
device backend. Public routing behavior is documented in the crate-level
READMEs, benchmark notes, and release notes.

## CUDA HTJ2K lossless encode

`signinum-j2k-cuda` exposes `encode_j2k_lossless_with_cuda`, a strict
on-device HTJ2K lossless encode path. Its design goal is a codestream
byte-identical to the public `signinum-j2k` CPU lossless path. Byte-parity is
the contract enforced by the
`cuda-x86_64-compatibility` job in `.github/workflows/gpu-validation.yml`: that
job sets `SIGNINUM_REQUIRE_CUDA_RUNTIME` and runs the `htj2k_encode_parity` test
suite with a fail-closed executed-count floor (at least 8 parity tests must
execute, not merely compile). The job runs on the self-hosted CUDA runner via
`workflow_dispatch` and is the authoritative gate before merging changes to the
CUDA encode path.

### Supported matrix

| Dimension | Supported values |
|-----------|-----------------|
| Wavelet | Reversible 5/3 DWT, multi-level |
| Coding passes | HT cleanup-pass-only |
| Tile / layer / precinct | Single tile, single quality layer, single precinct |
| Components | 1 (grayscale), 3 (RGB — MCT/RCT applied to all three planes), 4 (RGBA/CMYK — MCT/RCT on the first three planes; 4th component passed through) |
| Bit depth | 8–16; unsigned and signed (signed = encode/codestream byte-parity only — native decode does not reconstruct signed samples; see Non-goals) |
| Code-block sizes | Multi-codeblock layouts |
| Component subsampling | (1,1) only — equal subsampling for all components |

### Non-goals (explicitly out of scope)

The following are outside the strict CUDA encode path. Each is excluded for the
stated reason.

- **Classic/tier-1 EBCOT coding** — this path is HTJ2K-only; EBCOT codestreams
  cannot be byte-identical to the HTJ2K native reference.
- **Lossy 9/7 DWT (irreversible)** — lossy paths are never byte-exact.
- **Multiple quality layers** — the native CPU reference is single-layer; no
  multi-layer target exists to compare against.
- **Multi-tile within one codestream** — the native CPU reference is
  single-tile; tiling across codestreams is done at the caller/per-codestream
  level.
- **2-component images** — the native decoder rejects 2-component input with
  `TooManyChannels`, so a 2-component round-trip cannot be validated.
- **Component subsampling != (1,1)** — unequal subsampling changes block
  geometry in ways the strict encode kernel does not handle.
- **HT SigProp/MagRef refinement passes** — these are experimental and go
  beyond what the native cleanup-pass-only path produces; round-trip validation
  against the native reference is not established.
- **Native decode reconstruction of signed samples** — the encoder emits
  spec-correct signed codestreams (the SIZ `Ssiz` signed bit is set, and
  CUDA-vs-native codestream byte-parity holds for signed inputs across all
  component counts, bit depths, and decomposition levels). However, the *shared
  native decoder* reads the `Ssiz` signed bit and ignores it
  (`signinum-j2k-native/src/j2c/codestream.rs`) and unconditionally re-applies
  the unsigned inverse DC level-shift, then clamps negatives
  (`decode_native_with_context` in `signinum-j2k-native/src/lib.rs`), so a signed
  sample does not round-trip back through signinum's own decoder — it is offset
  by `+2^(depth-1)`. This is a native-*decoder* limitation that affects the CPU
  and Metal decode paths identically (decode reconstruction is shared), not a
  CUDA encode issue. The parity gate therefore asserts byte-exact pixel
  round-trip only for unsigned cells; signed cells assert codestream byte-parity
  plus a successful, correctly-sized decode. Signed sources are otherwise treated
  as unsupported in adjacent paths (e.g. the recode path rejects them).

### No silent fallback

Out-of-scope or unavailable requests from `encode_j2k_lossless_with_cuda`
return a typed error. The accelerator never silently falls back to the CPU
encoder for an in-scope input — if the CUDA path is requested, the caller
receives either a CUDA-produced codestream or an explicit error.

## Lifecycle and ownership

signinum deliberately externalizes anything that resembles a runtime.

- **Buffers** are caller-owned. Every `decode_*_into` writes into a slice the
  caller provides.
- **Scratch** is caller-owned via `ScratchPool` and reset per operation. WSI
  readers reuse pools across tiles to keep allocation off the hot path.
- **Decoder context** is caller-owned via `DecoderContext<C>` and reused
  across tiles in a batch.
- **Sessions** for device backends are owned by the adapter and held by the
  caller for the lifetime of a batch. They wrap `MTLDevice`/`MTLCommandQueue`
  state on Metal.
- **Threading and pyramid policy** are entirely the caller's responsibility.
  No crate spawns threads or owns I/O.

## Where to add things

| Change | Crate |
|--------|-------|
| New shared trait, pixel format, passthrough contract, or backend kind | `signinum-core` |
| New JPEG decode shape, marker, or CPU optimization | `signinum-jpeg` |
| New JPEG GPU shape | `signinum-jpeg-metal` (or `-cuda`) |
| New diagnostic encode/transcode path | Prefer passthrough in the caller/container layer; otherwise `signinum-j2k-native`, surfaced through `signinum-j2k` |
| New coefficient-domain JPEG→HTJ2K experiment | `signinum-transcode`; stage accelerators live in `signinum-transcode-metal` or a matching accelerator crate until validated and promoted |
| New J2K codestream feature, ROI/scaled support | `signinum-j2k-native`, surfaced through `signinum-j2k` |
| New tile decompression codec (e.g. LZ4) | `signinum-tilecodec` |
| New CLI subcommand | `signinum-cli` |
| New conformance fixture | `corpus/conformance/` plus its manifest |
| New regression repro | `corpus/regressions/issue-NNN.<ext>` |

When in doubt, prefer adding to the lowest layer that the new behavior
genuinely requires, and never bypass `signinum-core` to share types
between codec crates.

## Build and platform

- Rust edition `2021`, MSRV `1.88`, pinned by [`rust-toolchain.toml`](../rust-toolchain.toml).
- Supported decode hosts: `x86_64` and `aarch64` only. Other targets fail
  to build by design.
- Metal adapters compile and run on Apple Silicon macOS. On other hosts the
  adapter crate compiles to a fallback surface.
- CUDA adapter crates expose runtime device-memory output for explicit CUDA
  requests when built with `cuda-runtime` on hosts with a CUDA driver. Hosts
  without CUDA return the documented unavailable error. JPEG full-frame RGB8
  strict CUDA decode uses Signinum-owned kernels for supported 4:2:0, 4:2:2,
  and 4:4:4 inputs.
- Release profile: `lto = "fat"`, `codegen-units = 1`, `strip = "symbols"`,
  `opt-level = 3`. `release-bench` inherits `release` but keeps debug info.
- Notable feature flags:
  - `signinum-j2k-native`: `std` (default), `simd` (default, requires
    `std`), `logging`.
  - `signinum-jpeg`: `scalar-only` retained for fuzzing and reference.
  - `signinum-jpeg-cuda`, `signinum-j2k-cuda`: `cuda-runtime` enables CUDA Driver
    API device allocation and explicit CUDA requests.

## Active areas

These are the surfaces under active change. Treat anything here as
provisional and check the most recent commits before relying on it.

- Metal adapter hardening: aligning the adapter session model with the
  wgpu-hal style, exposing the underlying `MTLBuffer` from `DeviceSurface`,
  and pushing more of the J2K full-tile path onto the GPU.
- Adaptive backend routing: deciding when `BackendRequest::Auto` should
  upgrade a batch to Metal vs. stay on CPU.
- Keeping the public WSI decode API guide aligned with the core trait surface.
- Expanding JPEG compatibility through CPU parity first: initial CMYK/YCCK CPU
  `Rgb8` full/ROI/scaled/region-scaled plus `Rgba8` full/ROI conversion and
  progressive 8-bit ROI/scaled CPU output projection have landed. Initial
  12-bit extended sequential grayscale `Gray16`/`Rgb16`
  full-image/ROI/scaled/region-scaled decode, including restart-coded grayscale
  streams, initial 12-bit progressive grayscale `Gray16`/`Rgb16`
  full-image/ROI/scaled/region-scaled decode, and initial 12-bit APP14 RGB
  4:4:4 and YCbCr 4:4:4/4:2:2/4:2:0 `Rgb16` decode, including restart-coded
  color streams, have landed, while expanded four-component subsampled/malformed
  coverage, other 12-bit subsampled color support, stronger non-constant 12-bit
  oracle fixtures, and broader lossless SOF3 YCbCr/16-bit color plus 16-bit row
  support remain active parity work.
  Initial SOF3 8-bit
  grayscale `Gray8` and 16-bit grayscale `Gray16`
  full-image/ROI/scaled/region-scaled decode for predictors 1-7, including
  restart-coded grayscale streams, plus 8-bit APP14 RGB `Rgb8` decode for
  predictors 1-7, including restart-coded APP14 RGB streams, has landed.
  Metal acceleration for those classes is gated on
  parity and measured resident wins.
- Broadening release CI and adding self-hosted x86_64 GPU benchmark coverage.
- Coefficient-domain JPEG to HTJ2K experiments in `signinum-transcode` and
  stage accelerators in `signinum-transcode-metal`. These crates remain
  experimental until their promotion gate is satisfied: synthetic 1D/2D tests,
  real JPEG grayscale and YCbCr sampling tests, native and available external
  HTJ2K decoder acceptance, documented error histograms, loud unsupported-mode
  failures, benchmark evidence for optimization claims, and no direct
  dependency between `signinum-jpeg` and `signinum-j2k-native`.

## Stability posture

The facade release covers `signinum`, `signinum-core`, `signinum-jpeg`,
`signinum-j2k`, `signinum-tilecodec`, and `signinum-cli`.
`signinum-j2k-native`, `signinum-profile`, and `signinum-cuda-runtime` are
published support crates, not primary stable APIs.
Runtime backend selection defaults to `Auto`; supported compiled device paths
may run before falling back to CPU. The Metal and CUDA adapter APIs remain on
the hardening track while broader device decode, encode, and performance work
matures. Breaking changes to any of these surfaces should be reflected here and
in [`CHANGELOG.md`](../CHANGELOG.md).
