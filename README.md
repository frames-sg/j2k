# Signinum

Signinum is a Rust image-codec workspace focused on JPEG, JPEG 2000, HTJ2K,
tile decompression, and GPU adapter experiments. The current public target is
the `signinum` facade release.

Runtime backend selection defaults to `Auto`: CPU remains the portable baseline,
and Metal or CUDA paths are selected only for supported shapes with validation
and benchmark evidence. Explicit device requests are strict. Unsupported device
shapes return errors instead of silently changing the requested backend.

CUDA paths use Signinum-owned CUDA kernels and `cuda-runtime` support for CUDA
device memory surfaces where implemented. NVIDIA performance claims require
self-hosted benchmark evidence; hosted CI is not treated as NVIDIA performance
evidence.

## Which crate should I use?

Use `cargo add signinum` for application code. It re-exports the stable facade
surface for JPEG, J2K/HTJ2K, and tile decompression.

Use lower-level crates only when you need a specific integration point:

| Need | Crate |
| --- | --- |
| Facade decode/decompress APIs | `signinum` |
| Shared traits and backend types | `signinum-core` |
| JPEG inspect/decode and fixture/fallback encode | `signinum-jpeg` |
| JPEG 2000 and HTJ2K inspect/decode/encode | `signinum-j2k` |
| Native J2K engine support | `signinum-j2k-native` |
| JPEG to HTJ2K transcode | `signinum-transcode` |
| CUDA adapters | `signinum-jpeg-cuda`, `signinum-j2k-cuda`, `signinum-transcode-cuda` |
| Metal adapters | `signinum-jpeg-metal`, `signinum-j2k-metal`, `signinum-transcode-metal` |
| Tile compression codecs | `signinum-tilecodec` |
| Command-line inspection | `signinum-cli` |

The names `statumen` and `wsi-dicom` are not current package names.

## Fast Path For LLM-Assisted Use

For normal decode work, start with the facade crate:

```bash
cargo add signinum
```

Then use the codec module matching the input format:

- `signinum::jpeg` for JPEG
- `signinum::j2k` for JPEG 2000 / HTJ2K
- `signinum::tilecodec` for raw tile decompression helpers

The public decode traits live in `signinum-core` and are implemented by codec
crates: `ImageDecode`, `ImageDecodeRows`, `TileBatchDecode`,
`TileBatchDecodeDevice`, and device-surface traits.

Runnable examples:

- `crates/signinum/examples/inspect_and_decode.rs`
- `crates/signinum/examples/tile_decompress.rs`
- `crates/signinum-transcode/examples/jpeg_to_htj2k.rs`

## Current backend posture

CPU is the correctness baseline. `BackendRequest::Auto` may return CPU-backed
outputs when a device path is unavailable, unsupported, or not benchmarked for
the requested shape.

Metal adapters are macOS-only and experimental. Explicit Metal requests return
resident Metal surfaces only for supported adapter paths.

CUDA adapters require a CUDA driver and adapter support. CUDA device memory
surfaces are available for supported paths; unsupported explicit CUDA requests
fail clearly. Signinum-owned CUDA kernels are used for CUDA codec stages. NVIDIA
performance claims require recorded self-hosted benchmark output.

## Public API and support policy

Stable APIs are the `signinum` facade, `signinum-core` traits and value types,
`signinum-jpeg`, `signinum-j2k`, and `signinum-tilecodec`. Experimental APIs are
the Metal adapters, CUDA adapters, transcode crates, and backend encode-stage
adapter SPI.

WSI decode contracts include `ImageDecode`, `decode_region_scaled_into`,
`decode_rows`, `TileBatchDecode`, `DeviceSurface`, `ScratchPool`, and
`DecoderContext`. `BackendRequest::Auto` may return CPU output.
`BackendRequest::Metal` and `BackendRequest::Cuda` are strict and fail for
unsupported shapes.

WSI/DICOM storage paths should pass compressed payloads through when transfer
syntax, payload kind, dimensions, component count, bit depth, signedness, and
color interpretation already match the destination requirements. Decode and
re-encode only when passthrough is invalid and the source codec path is
supported.

Unsupported input must fail explicitly. Error messages must avoid sensitive
internal details. Unsafe Rust inventory is tracked in
[docs/unsafe-audit.md](docs/unsafe-audit.md). Fuzzing and malformed-input tests
are part of release hardening. MSRV is declared in the root manifest.

Reference files:

- [docs/architecture.md](docs/architecture.md) - workspace layer rules and crate
  dependency graph
- [docs/env-vars.md](docs/env-vars.md) - supported `SIGNINUM_*`
  environment variables
- [docs/release.md](docs/release.md) - release and package validation policy
- [docs/stable-api-1.0.md](docs/stable-api-1.0.md) - stable API snapshot policy
- [CHANGELOG.md](CHANGELOG.md) - current release notes

## Benchmark and parity policy

A published benchmark must identify:

- host hardware and OS
- exact command
- input source
- whether input is signinum-generated or external
- comparator availability
- comparator version
- skipped paths and skip reason
- thread count, including `SIGNINUM_J2K_COMPARE_THREADS` when applicable

Public OpenJPEG and Grok comparison claims require explicit comparator gates and
cannot silently skip:

```bash
SIGNINUM_REQUIRE_OPENJPEG=1
SIGNINUM_REQUIRE_GROK=1
```

Signinum-generated J2K/HTJ2K codestreams require native decoder round trips.
OpenJPEG and Grok comparisons are used where those tools support the feature.
Missing comparators cannot convert a parity signoff into a pass.

## Security

Report vulnerabilities according to [SECURITY.md](SECURITY.md). Codec errors
should be explicit, non-sensitive, and should not silently treat unsupported
input as successful decode.
