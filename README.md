# J2K

[![crates.io](https://img.shields.io/crates/v/j2k.svg)](https://crates.io/crates/j2k)
[![docs.rs](https://img.shields.io/docsrs/j2k)](https://docs.rs/j2k)
[![CI](https://github.com/frames-sg/j2k/actions/workflows/ci.yml/badge.svg)](https://github.com/frames-sg/j2k/actions/workflows/ci.yml)
[![downloads](https://img.shields.io/crates/d/j2k.svg)](https://crates.io/crates/j2k)
[![license](https://img.shields.io/crates/l/j2k.svg)](#license)

**Docs & guides:** <https://frames-sg.github.io/j2k/>

**Release status:** `0.7.1` is published and security-supported. See the
[release notes](CHANGELOG.md) and [release policy](docs/release.md).

**Safe public Rust APIs, audited unsafe boundaries, and vendor-independent JPEG 2000 / HTJ2K.**

J2K is a Rust image-codec workspace for JPEG 2000 / HTJ2K decode, encode,
recode, and JPEG-to-HTJ2K coefficient-domain transcoding. It is built for teams
that need safe Rust integration for untrusted still-image inputs, permissive
MIT/Apache-2.0 licensing, and optional acceleration across both CUDA and Apple
Metal without making a GPU vendor SDK the public API.

Speed matters, but it is not the reason this project exists. The strategic
gap is a memory-safety-oriented Rust codec with a portable CPU baseline,
multi-vendor GPU adapters, explicit support boundaries, and reproducible
benchmark gates. The public crate release centers on `j2k`, with lower-level
crates for native codec internals, device adapters, JPEG input, and transcode
pipelines.

The codec support claim is intentionally scoped and explicit: full JPEG 2000
Part 1 codestream support for still-image workflows, JP2 wrapping, HTJ2K
Part 15 codestream support, and JPH wrapping. JPX / JPEG 2000 Part 2
extensions are outside this claim unless a feature is required for standard
JP2/JPH still-image correctness. The living support boundary is
[docs/public-support.md](docs/public-support.md).

The APIs are general codec APIs. Whole-slide imaging and DICOM tile workloads
are the main public examples and benchmark fixtures because they stress
large tiled images, strict color handling, and high-throughput GPU paths, but
the decoder, encoder, and transcode crates are not WSI-only. The
[digital-pathology workflow audit](docs/digital-pathology-workflow-audit.md)
defines the container, indexing, color, memory, and validation responsibilities
that remain outside the codec layer.

## Why J2K exists

JPEG 2000 is still common in medical imaging, geospatial imagery, digital
preservation, and large tiled-image systems, but the implementation landscape
forces awkward tradeoffs:

| Option | Tradeoff J2K avoids |
| --- | --- |
| NVIDIA CUDA JPEG 2000 runtime | CUDA/NVIDIA GPU stacks are a good fit for NVIDIA-only deployments, but not for portable Rust applications that also need Metal or CPU-first operation. |
| [OpenJPEG](https://github.com/uclouvain/openjpeg) | Mature C implementation and useful comparator, but C codecs keep memory-safety risk on the adopter. |
| [Grok](https://github.com/GrokImageCompression/grok) | Capable C++ JPEG 2000 / HTJ2K implementation, but AGPL licensing is not usable for every commercial or embedded integration. |

J2K's intended position is different: a safe Rust public API, isolated
unsafe boundaries for FFI/GPU work, no active runtime dependency on NVIDIA's
JPEG 2000 runtime, strict errors for unsupported device routes, and dual
MIT/Apache-2.0 licensing.

## Memory Safety Posture

J2K is designed for safe Rust integration with untrusted image inputs. The
public codec API is safe Rust.
Unsafe code is isolated at audited FFI, GPU integration, architecture-specific
SIMD/intrinsic, allocation, and bounded pointer/buffer boundaries, where inputs
are validated and unsupported shapes fail with errors. The exhaustive inventory
is maintained in [docs/unsafe-audit.md](docs/unsafe-audit.md).

This is an engineering posture backed by an explicit unsafe inventory, tests,
fuzzing, and review—not a formal proof that all implementation defects are
impossible. It is also not a claim that every malformed codestream is accepted
or that every device path is faster than CPU. CPU remains the portable
correctness baseline; GPU acceleration is promoted only for measured paths.

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
- `cargo run -p j2k-tilecodec --example decompress`
  ([crates/j2k-tilecodec/examples/decompress.rs](crates/j2k-tilecodec/examples/decompress.rs))

Runtime backend selection defaults to `Auto`: CPU remains the portable baseline,
and Metal or CUDA paths are selected only for supported shapes with validation
and benchmark evidence. Single-frame HTJ2K host-output encode stays CPU by
default; resident Metal encode performance claims are batch claims. Explicit
device requests are strict. Unsupported device shapes return errors instead of
silently changing the requested backend. `Auto` is an optimization policy, not a
promise to use a device whenever one is available.

CUDA paths use J2K-owned CUDA Oxide device kernels through `cuda-runtime`.
NVIDIA performance claims require self-hosted benchmark evidence; hosted CI is
not treated as NVIDIA performance evidence.

## Which crate should I use?

Use `cargo add j2k` for JPEG 2000 / HTJ2K application code. Lower-level
`j2k-*` crates remain public implementation and integration crates.

Use lower-level crates only when you need a specific integration point:

| Need | Crate |
| --- | --- |
| JPEG 2000 / HTJ2K inspect, decode, encode, and recode | `j2k` |
| Shared traits and backend types | `j2k-core` |
| Shared encode-stage contracts | `j2k-types` |
| Shared codec constants and pure helper algorithms | `j2k-codec-math` |
| JPEG inspect/decode and fixture/fallback encode | `j2k-jpeg` |
| Native JPEG 2000 and HTJ2K codec engine | `j2k-native` |
| JPEG-to-HTJ2K coefficient-domain transcode | `j2k-transcode` |
| CUDA adapters | `j2k-jpeg-cuda`, `j2k-cuda`, `j2k-transcode-cuda` |
| Metal adapters | `j2k-jpeg-metal`, `j2k-metal`, `j2k-transcode-metal` |
| Experimental Burn 0.21 tensor decode integration | `j2k-ml` (unpublished) |
| Tile compression codecs | `j2k-tilecodec` |
| Command-line inspection and JPEG-to-HTJ2K smoke transcode | `j2k-cli` |

The names `statumen` and `wsi-dicom` are not current package names.

## Support Matrix

| Area | Current support | Notes |
| --- | --- | --- |
| JPEG 2000 Part 1 inspect | Raw J2K/J2C codestreams and JP2 still-image files | Unsupported or malformed input fails explicitly. |
| JPEG 2000 Part 1 decode | Full-frame, ROI, scaled, row, tile-batch, and component-plane API surfaces | CPU is the portable correctness baseline. |
| JPEG 2000 Part 1 encode | Native Rust encode APIs for codestream and JP2 output, including component-plane metadata | Stable public API is centered on `j2k`; adapter SPI remains experimental. |
| HTJ2K Part 15 inspect/decode/encode | Raw HT codestreams and JPH still-image files, including cleanup and refinement paths | HT requests beyond the Part 15 coded-bitplane limit reject explicitly. |
| JP2/JPH metadata | IHDR/COLR/BPCC/PCLR/CMAP/CDEF/ICC still-image metadata paths covered by repo-local tests | Broader external JP2/JPH metadata parity remains publication evidence. |
| Recode | J2K-to-HTJ2K coefficient recode where valid, pixel-preserving fallback otherwise | Palette/component-mapped fallbacks intentionally drop mapping metadata after resolving pixels. |
| JPEG input | JPEG inspect/decode through `j2k-jpeg` | Used by transcode and fixture workflows. |
| JPEG-to-HTJ2K coefficient-domain transcode | CPU transcode primitives plus CUDA/Metal stage adapters | The public workflow requires HT block coding. |
| CUDA acceleration | J2K-owned CUDA kernels with CUDA Oxide as the target device-kernel backend | Requires self-hosted CUDA validation before performance claims. |
| Metal acceleration | macOS Metal adapters for selected decode, encode-stage, and transcode stages | Auto routing stays conservative and benchmark-gated. |

## Fast Path For LLM-Assisted Use

For normal JPEG 2000 / HTJ2K work, start with the public codec crate:

```bash
cargo add j2k
```

The shared decode traits live in `j2k-core` and are implemented by codec
crates: `ImageDecode`, `ImageDecodeRows`, `TileBatchDecode`, and
device-surface traits.

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
- [docs/benchmark-evidence.md](docs/benchmark-evidence.md) - reproducible
  benchmark commands and current CUDA/Metal evidence
- [docs/benchmark-corpora.md](docs/benchmark-corpora.md) - external corpus and
  adoption-benchmark manifest policy
- [docs/env-vars.md](docs/env-vars.md) - supported `J2K_*`
  environment variables
- [docs/public-support.md](docs/public-support.md) - exact J2K Part 1,
  HTJ2K Part 15, JP2/JPH, and out-of-scope support boundary
- [docs/j2k-ml.md](docs/j2k-ml.md) - Burn tensor layouts, normalization,
  batching, and accelerator route guarantees
- [docs/release.md](docs/release.md) - release and package validation policy
- [docs/stable-api-1.0.md](docs/stable-api-1.0.md) - stable API snapshot policy
- [CHANGELOG.md](CHANGELOG.md) - current release notes

## Benchmark and parity policy

Benchmark publication requirements are maintained in
[docs/benchmark-corpora.md](docs/benchmark-corpora.md), with current run
evidence in [docs/benchmark-evidence.md](docs/benchmark-evidence.md).
Use `cargo run -p xtask --features adoption -- adoption-benchmark` for
publication bundles and
`cargo run -p xtask --features adoption -- adoption-report --run-dir <run-dir>`
for the guarded report.
OpenJPEG/Grok/CUDA/Metal/Kakadu/OpenJPH claims must use the required comparator
or hardware gates described in the benchmark docs; skipped rows and emulated
rows are diagnostic evidence only.

## Security

Report vulnerabilities according to [SECURITY.md](SECURITY.md). Codec errors
should be explicit, non-sensitive, and should not silently treat unsupported
input as successful decode.

## License

Dual-licensed under either [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE), at your option.
