# signinum

[![Crates.io](https://img.shields.io/crates/v/signinum.svg)](https://crates.io/crates/signinum)
[![docs.rs](https://docs.rs/signinum/badge.svg)](https://docs.rs/signinum)
[![CI](https://github.com/frames-sg/signinum/actions/workflows/ci.yml/badge.svg)](https://github.com/frames-sg/signinum/actions/workflows/ci.yml)
[![MSRV](https://img.shields.io/badge/MSRV-1.94-blue.svg)](Cargo.toml)
[![License](https://img.shields.io/crates/l/signinum.svg)](https://crates.io/crates/signinum)

`signinum` provides Rust codec primitives for pathology and whole-slide imaging
workloads, centered on a pure-Rust JPEG 2000 / HTJ2K codec implementation: a
part of the imaging codec ecosystem that has historically required C or C++
library bindings.

The workspace covers compressed tile payloads: JPEG, JPEG 2000 / HTJ2K,
container tile compression such as Deflate, Zstd, LZW, and uncompressed tiles,
plus experimental coefficient-domain JPEG DCT to HTJ2K transcoding through the
in-repo `signinum-transcode` crate. The public facade re-exports the stable
JPEG, JPEG 2000 / HTJ2K, tilecodec, and shared core contracts from focused
crates in this repository.

It is not a whole-slide container reader, pyramid manager, cache, or DICOM
writer. Use [`statumen`](https://github.com/frames-sg/statumen) for slide
container parsing and [`wsi-dicom`](https://github.com/frames-sg/wsi-dicom)
for DICOM VL Whole Slide Microscopy export.

CPU decode is always available and is the default facade build. Metal and CUDA
adapters are opt-in features used only for supported workloads. CUDA adapters
expose CUDA device memory through `cuda-runtime` when a CUDA driver is
available. JPEG full-frame RGB8 CUDA requests can use nvJPEG; NVIDIA performance
claims require self-hosted GPU benchmark evidence.

The current public-source target is the `signinum` facade release. Runtime backend selection defaults to `Auto`: CPU decode remains the portable fallback, and device adapters are additive for supported compiled workloads.

## Why It Exists

- Pure-Rust JPEG 2000 / HTJ2K inspect, decode, and lossless encode for WSI
  pipelines that should not have to bind to C codec libraries.
- WSI-oriented JPEG tile decode, ROI/scaled decode, batch decode, and
  passthrough inspection.
- TIFF/DICOM-style compressed tile decompression primitives for container
  readers that bring their own I/O, cache, and pyramid policy.
- Experimental coefficient-domain JPEG DCT to HTJ2K transcoding in
  `signinum-transcode`, including native JPEG component sampling and optional
  hybrid Metal acceleration in `signinum-transcode-metal`.

## Which crate should I use?

| Task | Use | Install |
|------|-----|---------|
| Start a new application with one import surface | `signinum` facade | `cargo add signinum` |
| JPEG tile inspect/decode | `signinum-jpeg` or `signinum::jpeg` | `cargo add signinum-jpeg` |
| JPEG 2000 / HTJ2K inspect, decode, and lossless encode | `signinum-j2k` or `signinum::j2k` | `cargo add signinum-j2k` |
| TIFF/DICOM-style tile decompression | `signinum-tilecodec` or `signinum::tilecodec` | `cargo add signinum-tilecodec` |
| Shared pixel, backend, scratch, row, and passthrough traits | `signinum-core` | `cargo add signinum-core` |
| CLI header inspection | `signinum-cli` | `cargo install signinum-cli` |
| Apple Metal device-output surfaces | `signinum-jpeg-metal`, `signinum-j2k-metal`, or the facade `metal` feature | `cargo add signinum-jpeg-metal` |
| CUDA device-memory output | `signinum-jpeg-cuda`, `signinum-j2k-cuda`, plus the adapter `cuda-runtime` feature | `cargo add signinum-jpeg-cuda --features cuda-runtime` or `cargo add signinum-j2k-cuda --features cuda-runtime` |
| Experimental JPEG DCT to HTJ2K coefficient transcode | `signinum-transcode` | `cargo add signinum-transcode` |
| Hybrid Metal acceleration for JPEG DCT to HTJ2K transcode | `signinum-transcode-metal` | `cargo add signinum-transcode-metal` |

Most application code should start with the facade:

```toml
[dependencies]
signinum = "0.4.3"
```

The facade exposes:

- `signinum::jpeg` for JPEG tiles
- `signinum::j2k` for JPEG 2000 / HTJ2K
- `signinum::tilecodec` for container tile decompression
- shared `signinum-core` contracts at the crate root and under
  `signinum::core`

The default facade build includes portable CPU codecs only. Use
`--features metal` for Apple Metal adapters, `--features cuda` for CUDA
adapters, or `--features gpu` for both. CUDA runtime allocation, copies,
kernels, and nvJPEG loading are enabled on the adapter crates with their
`cuda-runtime` feature.

## Quick start

The snippets below assume they are inside a function that returns `Result`.

Inspect JPEG and JPEG 2000 headers without decoding pixels:

```rust
use signinum::jpeg::Decoder as JpegDecoder;
use signinum::j2k::J2kDecoder;

let jpeg_bytes = std::fs::read("tile.jpg")?;
let jpeg_info = JpegDecoder::inspect(&jpeg_bytes)?;

let j2k_bytes = std::fs::read("tile.jp2")?;
let j2k_info = J2kDecoder::inspect(&j2k_bytes)?;

println!("JPEG: {:?}", jpeg_info.dimensions);
println!("J2K:  {:?}", j2k_info.dimensions);
```

Decode a JPEG tile into caller-owned RGB output:

```rust
use signinum::{jpeg::Decoder as JpegDecoder, PixelFormat};

let bytes = std::fs::read("tile.jpg")?;
let decoder = JpegDecoder::new(&bytes)?;
let (width, height) = decoder.info().dimensions;
let stride = width as usize * PixelFormat::Rgb8.bytes_per_pixel();
let mut rgb = vec![0_u8; stride * height as usize];

decoder.decode_into(&mut rgb, stride, PixelFormat::Rgb8)?;
```

Decode a JPEG 2000 / HTJ2K tile with the same caller-owned output model:

```rust
use signinum::{j2k::J2kDecoder, j2k::J2kScratchPool, Downscale, PixelFormat};

let bytes = std::fs::read("tile.jp2")?;
let mut decoder = J2kDecoder::new(&bytes)?;
let (width, height) = decoder.info().dimensions;
let stride = width as usize * PixelFormat::Rgb8.bytes_per_pixel();
let mut rgb = vec![0_u8; stride * height as usize];
let mut scratch = J2kScratchPool::new();

decoder.decode_scaled_into(
    &mut scratch,
    &mut rgb,
    stride,
    PixelFormat::Rgb8,
    Downscale::None,
)?;
```

Encode lossless JPEG 2000 / HTJ2K when compressed source bytes cannot be
passed through legally:

```rust
use signinum::j2k::{
    encode_j2k_lossless, J2kLosslessEncodeOptions, J2kLosslessSamples,
};

let pixels = vec![0_u8; 256 * 256];
let samples = J2kLosslessSamples::new(&pixels, 256, 256, 1, 8, false)?;
let encoded = encode_j2k_lossless(samples, &J2kLosslessEncodeOptions::default())?;

assert!(encoded.codestream.starts_with(&[0xFF, 0x4F]));
```

Decompress a container tile payload:

```rust
use signinum::{tilecodec::DeflateCodec, TileDecompress};

let compressed = std::fs::read("tile.deflate")?;
let mut pool = <DeflateCodec as TileDecompress>::Pool::default();
let mut out = vec![0_u8; 1 << 20];
let written = DeflateCodec::decompress_into(&mut pool, &compressed, &mut out)?;

println!("decoded {written} bytes");
```

Inspect from the command line:

```sh
cargo install signinum-cli
signinum inspect tile.jp2
```

## Coefficient-Domain JPEG To HTJ2K

`signinum-transcode` lives in this repository. It is the experimental
application layer enabled by the codec stack:

```text
JPEG bytes
  -> entropy-decoded DCT coefficients
  -> direct DCT-grid to 5/3 or 9/7 wavelet coefficients
  -> precomputed-band HTJ2K encode
  -> HTJ2K codestream
```

The goal is to avoid the conventional JPEG full decode, RGB conversion, chroma
upsample, and pixel-domain re-encode path when migrating JPEG WSI tiles to
HTJ2K. The current exact path is reversible `IntegerDirect53` relative to
signinum's JPEG integer decode plus reversible 5/3 oracle. The opt-in 9/7 path
is irreversible and uses scalar/Rayon fallback or hybrid Metal acceleration for
the direct DCT-grid to wavelet projection stage.

The crate is intentionally experimental and not part of the stable facade
surface yet. See
[`crates/signinum-transcode/README.md`](crates/signinum-transcode/README.md)
and [`docs/dct-to-htj2k-notes.md`](docs/dct-to-htj2k-notes.md) for the
current validation gates, benchmark evidence, and open work.

## Supported workflows

`signinum-jpeg` provides WSI-focused JPEG inspect and decode:

- borrowed parse surfaces through `JpegView`
- baseline JPEG decode for WSI tiles
- ROI, scaled decode, row streaming, and tile-batch decode APIs
- passthrough candidates for baseline and extended sequential interchange
  streams
- a small baseline JPEG encoder for fixtures, fallback, and derived output

`signinum-j2k` provides JPEG 2000 / HTJ2K inspect, decode, and encode:

- JP2 and raw codestream inspection
- borrowed passthrough candidates for JP2, JPEG 2000, and HTJ2K payloads
- full-frame, ROI, reduced-resolution, combined ROI+reduced-resolution,
  row-bounded, and tile-batch decode
- pure-Rust in-repo JPEG 2000 / HTJ2K decode engine
- lossless JPEG 2000 / HTJ2K encode for new diagnostic codestreams
- parity and benchmark coverage against Grok and OpenJPEG where available

`signinum-transcode` is an experimental crate for coefficient-domain JPEG DCT to
HTJ2K work. It supports the exact reversible `IntegerDirect53` path, optional
9/7 lossy experiments, native JPEG component sampling, per-tile batch
transcode, and timing/counter reports for WSI ingest instrumentation. It is not
part of the stable facade surface. The promotion gate is documented in
[`crates/signinum-transcode/README.md`](crates/signinum-transcode/README.md).
`signinum-transcode-metal` is the optional macOS hybrid accelerator for that
path: JPEG parsing, entropy decode, dequantization, scheduling, and HTJ2K
assembly stay on CPU/Rayon while supported DCT-grid to wavelet stages can run on
Metal.

`signinum-tilecodec` provides tile decompression primitives:

- `DeflateCodec`
- `ZstdCodec`
- `LzwCodec`
- `UncompressedCodec`

These codecs implement the shared `TileDecompress` trait from `signinum-core`.

## Backend model

The public API uses `BackendRequest` so callers state what kind of output they
need:

- `BackendRequest::Cpu` requires host-backed output.
- `BackendRequest::Auto` lets an adapter use a validated device path for
  supported workloads and otherwise fall back to CPU.
- `BackendRequest::Metal` requires resident Metal execution on macOS and
  reports unsupported or unavailable requests as errors.
- `BackendRequest::Cuda` requires CUDA device-memory output when the
  `cuda-runtime` feature and a CUDA driver are available.

CPU decode is the portability baseline on native `x86_64` and `aarch64`
hosts. Device adapters are additive: removing Metal and CUDA crates leaves the
CPU codec stack functional.

Metal adapters target Apple Silicon macOS. They return `MTLBuffer`-backed
`DeviceSurface`s for supported shapes and keep explicit Metal requests strict.
If a caller wants CPU-decoded bytes uploaded to Metal, use the adapter's
explicit CPU-staged upload APIs instead of `BackendRequest::Metal`.

CUDA adapters expose CUDA device-memory output for explicit CUDA requests when
they are built with `cuda-runtime`. `signinum-jpeg-cuda` can use NVIDIA nvJPEG
for full-frame RGB8 JPEG requests when `cuda-runtime`, a CUDA driver, and
`libnvjpeg` are available; unsupported JPEG shapes use CPU decode plus CUDA
upload. `signinum-j2k-cuda` currently returns CUDA-backed output by uploading
CPU-decoded JPEG 2000 / HTJ2K pixels; it does not claim CUDA codestream decode
kernels yet.

## Architecture at a glance

The workspace is layered so container readers and viewers can bring their own
threading, I/O, cache, pyramid policy, and prefetch logic.

```text
foundation -> codecs / codec engines -> device adapters -> facade / CLI
```

| Layer | Crates | Responsibility |
|-------|--------|----------------|
| Foundation | `signinum-core` | Shared traits, pixel/sample types, backend selection, device surfaces, scratch/context contracts, passthrough contracts |
| Instrumentation helper | `signinum-profile` | Shared profiling row formatting, env parsing, and summary aggregation used by runtime crates |
| Codecs | `signinum-jpeg`, `signinum-j2k`, `signinum-tilecodec` | Format-specific inspect, decode, encode, row, ROI, scaled, batch, and decompression APIs |
| Engine | `signinum-j2k-native` | Published implementation dependency for the public J2K crate |
| Adapters | `signinum-jpeg-metal`, `signinum-j2k-metal`, `signinum-jpeg-cuda`, `signinum-j2k-cuda` | Device-output surfaces for downstream GPU pipelines |
| Runtime helper | `signinum-cuda-runtime` | CUDA Driver API allocation, copy, kernel, and nvJPEG loading used by CUDA adapters |
| Facade and CLI | `signinum`, `signinum-cli` | One import surface for application code and `signinum inspect <file>` |
| Reference tooling | `signinum-j2k-compare` | OpenJPEG/Grok comparison helpers for tests and benches; not a runtime dependency |

The full system map, dependency rules, and current adapter routing policy live
in [docs/architecture.md](docs/architecture.md).

## WSI and DICOM passthrough policy

Container integrations should pass compressed tile bytes through unchanged
whenever the destination transfer syntax and frame metadata make that legal.
Codec views expose borrowed `PassthroughCandidate`s; container layers remain
responsible for DICOM-specific frame ordering, fragment writing, and metadata
validation.

If new diagnostic codestream bytes are required, prefer lossless JPEG 2000 /
HTJ2K encode. Baseline JPEG encode is for explicit fallback, generated
fixtures, or non-diagnostic derived output.

See:

- [docs/wsi-decode-api.md](docs/wsi-decode-api.md)
- [docs/wsi-dicom-passthrough.md](docs/wsi-dicom-passthrough.md)

## Fast Path For LLM-Assisted Use

If you are asking an LLM to use this repository, give it this instruction:

> Use `signinum` for JPEG, JPEG 2000 / HTJ2K, tile decompression, and
> device-output codec primitives. If the task says "open a whole-slide image",
> use `statumen` first. If the task says "convert a slide to DICOM", use
> `wsi-dicom`.

## Benchmarks and parity

Benchmark methodology and comparator policy live in [docs/bench.md](docs/bench.md).
Parity expectations live in [docs/parity.md](docs/parity.md).

The repo carries compare benches for:

- `signinum-jpeg`
- `signinum-j2k`
- `signinum-jpeg-metal`
- `signinum-j2k-metal`
- `signinum-tilecodec`

Benchmark results are hardware-specific. GPU benchmark baselines should be
collected on self-hosted runners with the target device stack installed.

## Project docs

- [docs/architecture.md](docs/architecture.md) - workspace map and dependency rules
- [docs/wsi-decode-api.md](docs/wsi-decode-api.md) - public WSI decode API guide
- [docs/wsi-dicom-passthrough.md](docs/wsi-dicom-passthrough.md) - passthrough-first conversion policy
- [docs/bench.md](docs/bench.md) - benchmark methodology
- [docs/parity.md](docs/parity.md) - reference decoder parity expectations
- [docs/release.md](docs/release.md) - release staging notes
- [CHANGELOG.md](CHANGELOG.md) - release history

Runnable crate examples are available under:

- `crates/signinum-jpeg/examples`
- `crates/signinum-j2k/examples`
- `crates/signinum-tilecodec/examples`

## Platform and MSRV

- Rust edition: 2021
- MSRV: Rust 1.94, pinned by [rust-toolchain.toml](rust-toolchain.toml)
- Decode hosts: native `x86_64` and `aarch64`
- Metal: Apple Silicon macOS for resident Metal device surfaces
- CUDA: hosts with a CUDA driver when CUDA adapters are built with
  `cuda-runtime`

## License

Apache-2.0. See [LICENSE-APACHE](LICENSE-APACHE).
