# J2K — Pure-Rust JPEG 2000 and HTJ2K Codec

[![crates.io](https://img.shields.io/crates/v/j2k.svg)](https://crates.io/crates/j2k)
[![docs.rs](https://img.shields.io/docsrs/j2k)](https://docs.rs/j2k)
[![CI](https://github.com/frames-sg/j2k/actions/workflows/ci.yml/badge.svg)](https://github.com/frames-sg/j2k/actions/workflows/ci.yml)
[![downloads](https://img.shields.io/crates/d/j2k.svg)](https://crates.io/crates/j2k)
[![license](https://img.shields.io/crates/l/j2k.svg)](#license)

**Docs & guides:** [Pure-Rust JPEG 2000 codec documentation](https://frames-sg.github.io/j2k/rust-jpeg2000-codec/)

**Release status:** `0.11.3` is published and security-supported. See the
[release notes](CHANGELOG.md), [release policy](docs/release.md), and
[security policy](SECURITY.md).

**A JPEG 2000 and HTJ2K codec written in Rust, with a portable CPU
implementation and CUDA and Metal acceleration.**

J2K decodes, encodes, and recodes JPEG 2000 and HTJ2K, and transcodes baseline
JPEG to HTJ2K in the coefficient domain. It can decode whole images, regions,
reduced resolutions, single tiles, and batches, to host memory or to GPU
memory. Region and reduced-resolution decoding skip work outside the requested
area, which matters for large tiled images such as whole-slide scans. The
workspace is dual-licensed under MIT and Apache-2.0.

## GPU acceleration

On the GPU paths, parsing stays on the CPU and Tier-1 decoding,
dequantization, inverse wavelet transform, color transform, and output are done
on CUDA or Metal. No pipeline runs every stage on the GPU.

`BackendRequest::Auto` uses the GPU only for image shapes where it was measured
to be faster. On the external HTJ2K/JPH routing benchmark, `Auto` uses the GPU
for 8 of 10 CUDA cells and 3 of 10 Metal cells. In each of those cells the GPU
output matched the CPU output byte for byte, and the median time was at least
10% faster with non-overlapping 95% confidence intervals.

For lossless HTJ2K encode to host memory, Metal is used for RGB8 at
1024 x 1024 and for Gray8 and RGB8 at 2048 x 2048. Those paths run coefficient
preparation and HT Tier-1 on Metal and packetize on the CPU. The 512 x 512
cells and Gray8 at 1024 x 1024 stay on the CPU. These measurements are from an
Apple M4 Pro; full results are in
[docs/benchmark-evidence.md](docs/benchmark-evidence.md).

## Conformance

The CPU decoder in release `0.11.2` conforms to ISO/IEC 15444-4:2024 / ITU-T
T.803 v3 **Profile-1 Cclass-1**, **Profile-1 Cclass-1HF**, and the **Annex G
JP2 reader**. For HTJ2K (Part 15) it conforms to **DS1-HM Cclass-1h, MMAGB 15**
(including the DS1-HT, DS0-HM, and DS0-HT subsets), **Cclass-1HFh, MMAGB 20**,
and the **Annex G JPH reader at MMAGB 15**.

The CPU runs on macOS arm64, Linux x86-64, and Windows x86-64 each pass all 160
selected cases (90 Part 1, 70 Part 15) with no skips. The CUDA and Metal runs
pass the same 160 cases: 81 use the GPU for some stages and 79 run on the CPU
(Part 15: 33 and 37). The per-stage CPU/GPU labels come from dispatch counters
recorded during the run.

All five reports are for commit `75a3e0618e1963d8403e4edad0fa95ee1c217ec1` and
are attached to the [v0.11.2 release](https://github.com/frames-sg/j2k/releases/tag/v0.11.2).
The encoder passes 56/56 cases on the CPU and 35/35 on CUDA and on Metal; these
encoder results are informative (Annex D/F), not formal decoder conformance.
Scope and rules are in [docs/t803-conformance.md](docs/t803-conformance.md).
T.803 does not test robustness, security, or performance.

## Why J2K

JPEG 2000 is common in medical imaging, geospatial imagery, digital
preservation, and large tiled-image systems. The existing implementations each
have a drawback for some users:

| Option | Drawback |
| --- | --- |
| NVIDIA CUDA JPEG 2000 runtime | NVIDIA-only; no Metal or CPU-only deployment. |
| [OpenJPEG](https://github.com/uclouvain/openjpeg) | Mature, but written in C, so memory-safety bugs are the adopter's risk. |
| [Grok](https://github.com/GrokImageCompression/grok) | Capable C++ JPEG 2000 / HTJ2K implementation, but AGPL-licensed. |

J2K has a safe Rust public API, keeps `unsafe` code to FFI, GPU, and SIMD
boundaries, does not depend on NVIDIA's JPEG 2000 runtime, returns an error
instead of silently falling back when a requested GPU path is unsupported, and
is MIT/Apache-2.0 licensed.

## Memory safety

The public API is safe Rust and is meant to accept untrusted images. `unsafe`
code is limited to FFI, GPU integration, SIMD intrinsics, allocation, and
bounds-checked buffer access, and every such file is listed with its invariants
in [docs/unsafe-audit.md](docs/unsafe-audit.md). The decoders are fuzzed and
tested against malformed input. None of this proves the code is free of bugs.

## Quickstart

Use the public Rust API for application integration:

```bash
cargo add j2k
```

Run the command-line tool for quick inspection and JPEG-to-HTJ2K transcode
smoke tests:

```bash
cargo install j2k-cli
j2k inspect input.jp2
j2k transcode input.jpg output.j2k --htj2k --lossless-53
```

Runnable repository examples:

- `cargo run -p j2k --example decode_generated`
  ([crates/j2k/examples/decode_generated.rs](crates/j2k/examples/decode_generated.rs))
- `cargo run -p j2k-jpeg --example inspect`
  ([crates/j2k-jpeg/examples/inspect.rs](crates/j2k-jpeg/examples/inspect.rs))
- `cargo run -p j2k-transcode --example jpeg_to_htj2k`
  ([crates/j2k-transcode/examples/jpeg_to_htj2k.rs](crates/j2k-transcode/examples/jpeg_to_htj2k.rs))
- `cargo run -p j2k-transcode-metal --example jpeg_to_htj2k_route_report`
  ([crates/j2k-transcode-metal/examples/jpeg_to_htj2k_route_report.rs](crates/j2k-transcode-metal/examples/jpeg_to_htj2k_route_report.rs))
- `cargo run -p j2k-metal --example decode_route_report`
  ([crates/j2k-metal/examples/decode_route_report.rs](crates/j2k-metal/examples/decode_route_report.rs))
- `cargo run -p j2k-metal --example htj2k_encode_auto_report`
  ([crates/j2k-metal/examples/htj2k_encode_auto_report.rs](crates/j2k-metal/examples/htj2k_encode_auto_report.rs))
- `cargo run -p j2k-metal --example resident_encode_buffer`
  ([crates/j2k-metal/examples/resident_encode_buffer.rs](crates/j2k-metal/examples/resident_encode_buffer.rs))
- `cargo run -p j2k-ml --example training_batcher --features cpu`
  ([crates/j2k-ml/examples/training_batcher.rs](crates/j2k-ml/examples/training_batcher.rs))
- `cargo run -p j2k-ml --example cuda_upload --features cuda`
  ([crates/j2k-ml/examples/cuda_upload.rs](crates/j2k-ml/examples/cuda_upload.rs))
- `cargo run -p j2k-ml --example metal_upload --features metal`
  ([crates/j2k-ml/examples/metal_upload.rs](crates/j2k-ml/examples/metal_upload.rs))
- `cargo run -p j2k-mpsgraph --example resident_reference_graph`
  ([crates/j2k-mpsgraph/examples/resident_reference_graph.rs](crates/j2k-mpsgraph/examples/resident_reference_graph.rs))
- `cargo run -p j2k-tilecodec --example decompress`
  ([crates/j2k-tilecodec/examples/decompress.rs](crates/j2k-tilecodec/examples/decompress.rs))

To decode a region at a reduction finer than `Downscale::Eighth`, use
`J2kDecoder::decode_region_scaled_pow2_into`. The level is the number of
power-of-two halvings. A level beyond a component's resolution count returns
an unsupported error rather than decoding at a different scale.

## Backend selection

The default backend is `Auto`. The CPU path is always available; Metal or
CUDA is used only for shapes where it was benchmarked to be faster. Lossless
HTJ2K host-output encode uses Metal for the cells listed above and the CPU for
everything else. Encoding into Metal buffers is a separate batch API.

`BackendRequest::Cuda` and `BackendRequest::Metal` return an error for shapes
the device path does not support; they never switch to the CPU. If `Auto`
picks a GPU and the GPU then fails, that is also an error, not a CPU retry.
`Auto` does not promise to use a GPU just because one is present.

A new GPU threshold is added to `Auto` only when it produces identical output
on the external benchmark corpus and its median time is at least 10% faster
than the CPU and any other device route, with non-overlapping 95% confidence
intervals. Thresholds are fixed at build time; nothing is calibrated at
runtime.

CUDA support uses J2K's own CUDA Oxide kernels, enabled with the
`cuda-runtime` feature. NVIDIA performance numbers come from self-hosted
benchmark runs; hosted CI does not measure GPU performance.

## Batch decoding

The owned-batch API takes `EncodedImage` values (an `Arc<[u8]>` plus one of
`Full`, `Region`, `Reduced`, or `RegionReduced`). It prepares inputs in
parallel, groups outputs by shape without padding, and reports the source
index of every result and failure. Gray, RGB, and RGBA groups are returned as
`U8`, `U16`, or `I16` in NCHW or NHWC order. Conversion to float and
normalization are left to the caller.

Preparation can keep a `PreparedHtj2kPlan` or a `PreparedClassicPlan` with
per-tile packet, code-block, and destination geometry. Plans point into the
original `Arc<[u8]>` instead of copying the codestream, and `CpuBatchDecoder`
decodes single- and multi-tile plans without parsing again. Inputs that a plan
cannot describe fall back to the general CPU decoder.

`j2k-cuda` and `j2k-metal` provide persistent GPU sessions, GPU-resident
output, and decoding into caller-owned GPU buffers. Their final store writes
the requested data type and layout directly into the destination, so decoded
pixels never round-trip through host memory. The codec support matrix is in
[docs/public-support.md](docs/public-support.md), the Burn adapter's scope is
in [docs/j2k-ml.md](docs/j2k-ml.md), and hardware results are in
[docs/benchmark-evidence.md](docs/benchmark-evidence.md).

`j2k-ml` copies decoded output through host memory and builds ordinary Burn
tensors with Burn's public API. `CudaUploadBurnDecoder` and
`MetalUploadBurnDecoder` decode on the GPU but still copy through the host
before creating the tensor. Container readers such as `wsi-rs` are responsible
for locating the encoded bytes.

`j2k-mpsgraph` is the Apple Silicon path into MPSGraph. It either aliases
completed GPU-resident batches or queues decoding and the MPSGraph work on one
Metal command queue, so decoded pixels stay on the GPU. See
[docs/j2k-mpsgraph.md](docs/j2k-mpsgraph.md).

## Which crate should I use?

Use `cargo add j2k` for JPEG 2000 / HTJ2K application code. The lower-level
`j2k-*` crates are public but exist for specific integration points:

| Need | Crate |
| --- | --- |
| JPEG 2000 / HTJ2K inspect, decode, encode, and recode | `j2k` |
| Shared traits and backend types | `j2k-core` |
| Shared encode-stage types | `j2k-types` |
| Shared codec constants and pure helper algorithms | `j2k-codec-math` |
| JPEG inspect/decode and portable baseline encode | `j2k-jpeg` |
| Native JPEG 2000 and HTJ2K codec engine | `j2k-native` |
| JPEG-to-HTJ2K coefficient-domain transcode | `j2k-transcode` |
| CUDA adapters | `j2k-jpeg-cuda`, `j2k-cuda`, `j2k-transcode-cuda` |
| Metal adapters | `j2k-jpeg-metal`, `j2k-metal`, `j2k-transcode-metal` |
| Burn 0.21 native integer batch adapter | `j2k-ml` |
| Direct Apple Silicon MPSGraph batch adapter | `j2k-mpsgraph` |
| Tile compression codecs | `j2k-tilecodec` |
| Command-line inspection and JPEG-to-HTJ2K smoke transcode | `j2k-cli` |

`j2k-ml 0.7.5` shipped broken accelerator features; see the
[release policy](docs/release.md) for details. The current CUDA and Metal
adapters are tested as packaged crates before each release.

## Backend notes

The CPU path is the reference implementation. `BackendRequest::Auto` returns
CPU output when no GPU path exists for the shape or it was not measured to be
faster.

GPU routing is selective on purpose. A shape goes to Metal or CUDA only if it is
supported, matches the CPU output, is large or regular enough to pay for
dispatch and transfer, and measured faster. Small tiles, irregular packets,
entropy-heavy stages, and codestream assembly stay on the CPU unless a
GPU-resident path measures faster.

The Metal adapters are macOS-only and experimental. Explicit Metal requests
return GPU-resident surfaces or encode-stage dispatches for the supported
paths and return an error for everything else; not every encode route has a
Metal implementation.

The CUDA adapters require a CUDA driver. Supported paths return CUDA device
memory; unsupported explicit CUDA requests return an error.

Lossy HTJ2K encoding can use the OpenHTJ2K-compatible visual Qfactor profile
with `J2kLossyEncodeOptions::with_qfactor(Some(quality))`, where `quality` is
`1..=100`. Qfactor cannot be combined with byte, bits-per-pixel, PSNR,
quality-layer, or ROI targets.

## Public API and support policy

The stable APIs are `j2k`, the `j2k-core` traits and value types, `j2k-jpeg`,
and `j2k-tilecodec`. The Metal and CUDA adapters, the transcode crates, and
the backend encode-stage interface are experimental.

The main codec interfaces are `ImageDecode`, `decode_region_scaled_into`,
`decode_rows`, `TileBatchDecode`, `DeviceSurface`, `ScratchPool`, and the
`J2kContext` and `j2k_jpeg::DecoderContext` types. Row decoding of JPEG 2000 /
HTJ2K up to 24-bit component precision parses the tile once and reuses it for
every stripe, with stripe scratch limited by `J2kRowDecodeOptions`. Higher
precisions decode the full image and crop.

Container and storage code should copy compressed payloads through unchanged
when the payload type, dimensions, component count, bit depth, signedness, and
color interpretation already match the destination. Decode and re-encode only
when that is not possible.

Unsupported input returns an error, and error messages do not expose internal
details. Fuzzing and malformed-input tests run before each release. The MSRV
is set in the root `Cargo.toml`.

Reference documents:

- [docs/architecture.md](docs/architecture.md) - workspace layers and crate
  dependency graph
- [docs/benchmark-evidence.md](docs/benchmark-evidence.md) - benchmark
  commands and current CUDA/Metal results
- [docs/benchmark-corpora.md](docs/benchmark-corpora.md) - external benchmark
  corpora and manifest format
- [docs/env-vars.md](docs/env-vars.md) - `J2K_*` environment variables
- [docs/public-support.md](docs/public-support.md) - supported JPEG 2000 Part
  1, HTJ2K Part 15, and JP2/JPH features, and what is out of scope
- [docs/t803-conformance.md](docs/t803-conformance.md) - T.803 v3 decoder
  conformance results and encoder procedure
- [docs/j2k-ml.md](docs/j2k-ml.md) - Burn integer batch groups, plan reuse,
  and GPU decode/upload adapters
- [docs/j2k-mpsgraph.md](docs/j2k-mpsgraph.md) - Apple Silicon MPSGraph
  integration
- [docs/release.md](docs/release.md) - release and packaging process
- [docs/stable-api-1.0.md](docs/stable-api-1.0.md) - stable API snapshot
- [CHANGELOG.md](CHANGELOG.md) - release notes

## Benchmarks

Benchmark rules are in [docs/benchmark-corpora.md](docs/benchmark-corpora.md)
and current results in [docs/benchmark-evidence.md](docs/benchmark-evidence.md).
Run `cargo run -p xtask --features adoption -- adoption-benchmark` to produce
a benchmark bundle and
`cargo run -p xtask --features adoption -- adoption-report --run-dir <run-dir>`
to build the report. Comparisons with OpenJPEG, Grok, Kakadu, OpenJPH, CUDA,
or Metal require the comparator or hardware described in the benchmark docs;
skipped and emulated rows are for diagnosis only.

## Security

Report vulnerabilities as described in [SECURITY.md](SECURITY.md).

## License

Dual-licensed under either [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE), at your option.
