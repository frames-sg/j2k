# J2K

[![crates.io](https://img.shields.io/crates/v/j2k.svg)](https://crates.io/crates/j2k)
[![docs.rs](https://img.shields.io/docsrs/j2k)](https://docs.rs/j2k)
[![CI](https://github.com/frames-sg/j2k/actions/workflows/ci.yml/badge.svg)](https://github.com/frames-sg/j2k/actions/workflows/ci.yml)
[![downloads](https://img.shields.io/crates/d/j2k.svg)](https://crates.io/crates/j2k)
[![license](https://img.shields.io/crates/l/j2k.svg)](https://github.com/frames-sg/j2k/blob/main/LICENSE-APACHE)

**Docs & guides:** <https://frames-sg.github.io/j2k/>

J2K is a Rust image-codec workspace for JPEG 2000 / HTJ2K decode, encode,
GPU acceleration, and JPEG-to-J2K/HTJ2K transcoding. The public crate release
centers on `j2k`, with lower-level crates for native codec internals, device
adapters, JPEG input, and transcode pipelines.

The APIs are general codec APIs. Whole-slide imaging and DICOM tile workloads
are the main public examples and benchmark fixtures because they stress
large tiled images, strict color handling, and high-throughput GPU paths, but
the decoder, encoder, and transcode crates are not WSI-only.

Runtime backend selection defaults to `Auto`: CPU remains the portable baseline,
and Metal or CUDA paths are selected only for supported shapes with validation
and benchmark evidence. Explicit device requests are strict. Unsupported device
shapes return errors instead of silently changing the requested backend.
`Auto` is an optimization policy, not a promise to use a device whenever one is
available.

CUDA paths use J2K-owned CUDA kernels and `cuda-runtime` support for CUDA
device memory surfaces where implemented. NVIDIA performance claims require
self-hosted benchmark evidence; hosted CI is not treated as NVIDIA performance
evidence.

## Which crate should I use?

Use `cargo add j2k` for JPEG 2000 / HTJ2K application code. Lower-level
`j2k-*` crates remain public implementation and integration crates.

Use lower-level crates only when you need a specific integration point:

| Need | Crate |
| --- | --- |
| JPEG 2000 / HTJ2K inspect, decode, encode, and recode | `j2k` |
| Shared traits and backend types | `j2k-core` |
| JPEG inspect/decode and fixture/fallback encode | `j2k-jpeg` |
| Native JPEG 2000 and HTJ2K codec engine | `j2k-native` |
| JPEG to J2K/HTJ2K transcode | `j2k-transcode` |
| CUDA adapters | `j2k-jpeg-cuda`, `j2k-cuda`, `j2k-transcode-cuda` |
| Metal adapters | `j2k-jpeg-metal`, `j2k-metal`, `j2k-transcode-metal` |
| Tile compression codecs | `j2k-tilecodec` |
| Command-line inspection | `j2k-cli` |

The names `statumen` and `wsi-dicom` are not current package names.

## Fast Path For LLM-Assisted Use

For normal JPEG 2000 / HTJ2K work, start with the public codec crate:

```bash
cargo add j2k
```

The shared decode traits live in `j2k-core` and are implemented by codec
crates: `ImageDecode`, `ImageDecodeRows`, `TileBatchDecode`, and
device-surface traits.

Runnable examples:

- `crates/j2k/examples/decode_generated.rs`
- `crates/j2k-jpeg/examples/inspect.rs`
- `crates/j2k-tilecodec/examples/decompress.rs`
- `crates/j2k-transcode/examples/jpeg_to_htj2k.rs`

## Current backend posture

CPU is the correctness baseline. `BackendRequest::Auto` may return CPU-backed
outputs when a device path is unavailable, unsupported, or not benchmarked for
the requested shape.

GPU routing is intentionally selective. A Metal or CUDA path should be enabled
automatically only when the shape is supported, parity-covered, large or
regular enough to amortize dispatch and transfer costs, and backed by benchmark
evidence. Small tiles, irregular packet shapes, entropy-heavy stages, and
codestream assembly should remain CPU unless a measured resident path shows a
net win.

Metal adapters are macOS-only and experimental. Explicit Metal requests return
resident Metal surfaces or encode-stage dispatches only for supported adapter
paths. Metal encode support is not a blanket end-to-end guarantee for every
public encode route; unsupported explicit Metal shapes fail clearly.

CUDA adapters require a CUDA driver and adapter support. CUDA device memory
surfaces are available for supported paths; unsupported explicit CUDA requests
fail clearly. J2K-owned CUDA kernels are used for CUDA codec stages. NVIDIA
performance claims require recorded self-hosted benchmark output.

## Public API and support policy

Stable APIs are `j2k`, `j2k-core` traits and value types, `j2k-jpeg`,
and `j2k-tilecodec`. Experimental APIs are the Metal adapters, CUDA adapters,
transcode crates, and backend encode-stage adapter SPI.

Codec contracts include `ImageDecode`, `decode_region_scaled_into`,
`decode_rows`, `TileBatchDecode`, `DeviceSurface`, `ScratchPool`, and
`DecoderContext`. `BackendRequest::Auto` may return CPU output.
`BackendRequest::Metal` and `BackendRequest::Cuda` are strict and fail for
unsupported shapes.

Container and storage integrations should pass compatible compressed payloads
through when the payload kind, dimensions, component count, bit depth,
signedness, and color interpretation already match the destination
requirements. Decode and re-encode only when passthrough is invalid and the
source codec path is supported.

Unsupported input must fail explicitly. Error messages must avoid sensitive
internal details. Unsafe Rust inventory is tracked in
[docs/unsafe-audit.md](docs/unsafe-audit.md). Fuzzing and malformed-input tests
are part of release hardening. MSRV is declared in the root manifest.

Reference files:

- [docs/architecture.md](docs/architecture.md) - workspace layer rules and crate
  dependency graph
- [docs/env-vars.md](docs/env-vars.md) - supported `J2K_*`
  environment variables
- [docs/release.md](docs/release.md) - release and package validation policy
- [docs/stable-api-1.0.md](docs/stable-api-1.0.md) - stable API snapshot policy
- [CHANGELOG.md](CHANGELOG.md) - current release notes

## Benchmark and parity policy

A published benchmark must identify:

- host hardware and OS
- exact command
- input source
- whether input is j2k-generated or external
- comparator availability
- comparator version
- skipped paths and skip reason
- thread count, including `J2K_COMPARE_THREADS` when applicable

Public OpenJPEG and Grok comparison claims require explicit comparator gates and
cannot silently skip:

```bash
J2K_REQUIRE_OPENJPEG=1
J2K_REQUIRE_GROK=1
```

J2K-generated J2K/HTJ2K codestreams require native decoder round trips.
OpenJPEG and Grok comparisons are used where those tools support the feature.
Missing comparators cannot convert a parity signoff into a pass.

## Security

Report vulnerabilities according to [SECURITY.md](SECURITY.md). Codec errors
should be explicit, non-sensitive, and should not silently treat unsupported
input as successful decode.
