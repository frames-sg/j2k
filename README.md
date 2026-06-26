# J2K

[![crates.io](https://img.shields.io/crates/v/j2k.svg)](https://crates.io/crates/j2k)
[![docs.rs](https://img.shields.io/docsrs/j2k)](https://docs.rs/j2k)
[![CI](https://github.com/frames-sg/j2k/actions/workflows/ci.yml/badge.svg)](https://github.com/frames-sg/j2k/actions/workflows/ci.yml)
[![downloads](https://img.shields.io/crates/d/j2k.svg)](https://crates.io/crates/j2k)
[![license](https://img.shields.io/crates/l/j2k.svg)](#license)

**Docs & guides:** <https://frames-sg.github.io/j2k/>

J2K is a Rust image-codec workspace for JPEG 2000 / HTJ2K decode, encode,
GPU acceleration, and JPEG-to-J2K/HTJ2K transcoding. The public crate release
centers on `j2k`, with lower-level crates for native codec internals, device
adapters, JPEG input, and transcode pipelines.

The APIs are general codec APIs. Whole-slide imaging and DICOM tile workloads
are the main public examples and benchmark fixtures because they stress
large tiled images, strict color handling, and high-throughput GPU paths, but
the decoder, encoder, and transcode crates are not WSI-only.

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
- `cargo run -p j2k-tilecodec --example decompress`
  ([crates/j2k-tilecodec/examples/decompress.rs](crates/j2k-tilecodec/examples/decompress.rs))

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
| Command-line inspection and JPEG-to-HTJ2K smoke transcode | `j2k-cli` |

The names `statumen` and `wsi-dicom` are not current package names.

## Support Matrix

| Area | Current support | Notes |
| --- | --- | --- |
| JPEG 2000 / HTJ2K inspect | J2K codestream and JP2 header inspection | Unsupported or malformed input fails explicitly. |
| JPEG 2000 / HTJ2K decode | Full-frame, ROI, scaled, row, and tile-batch API surfaces | CPU is the portable correctness baseline. |
| JPEG 2000 / HTJ2K encode | Native Rust encode APIs plus encode-stage accelerator hooks | Stable public API is centered on `j2k`; adapter SPI remains experimental. |
| JPEG input | JPEG inspect/decode through `j2k-jpeg` | Used by transcode and fixture workflows. |
| JPEG-to-J2K/HTJ2K transcode | CPU transcode primitives plus CUDA/Metal accelerator adapters | CLI exposes the conservative lossless JPEG-to-HTJ2K command first. |
| CUDA acceleration | J2K-owned CUDA kernels and optional cuda-oxide routes for selected stages | Requires self-hosted CUDA validation before performance claims. |
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
- benchmark mode, per-row decode method, and publication blockers
- comparator availability
- comparator version
- comparator gate status and skipped comparator list
- git revision, dirty state including untracked files, build profile, and host hardware
- skipped paths and skip reason
- thread count and internal decoder threading policy, including
  `J2K_COMPARE_THREADS` or `J2K_FIXTURE_COMPARE_THREADS` when applicable
- batch input policy and sample order policy
- separate decode fixture and encode source-image manifests when making both
  decoder and encoder claims

Use `cargo xtask adoption-materialize` for source-image corpora that need fixed
classic J2K/HTJ2K decode fixtures plus staged PGM/PPM encode inputs. It writes
raw and JP2-container decode variants, `staged-pnm/`, `fixtures.tsv`, and
`encode-fixtures.tsv` from the same source bytes. Materialized decode rows carry
`source_fnv1a64` so raw plus boxed variants do not inflate the unique-source
publication gate. Use `cargo xtask adoption-manifest` for existing native
J2K/JP2/JPH corpora such as conformance, OpenJPEG, OpenJPH, or parser test
data, or `cargo xtask adoption-curate` when a corpus mixes valid, invalid, and
non-comparable files. `adoption-curate` copies only supported 8-bit gray/RGB
files that pass full decode plus OpenJPEG/Grok full-image preflight, and records
rejected files in `skipped.tsv`. Pass the resulting manifests to
`cargo xtask adoption-benchmark --manifest ... --encode-manifest ...`.

Public OpenJPEG and Grok comparison claims require explicit comparator gates and
cannot silently skip:

```bash
J2K_REQUIRE_OPENJPEG=1
J2K_REQUIRE_GROK=1
```

Optional OpenJPH rows can be added for HTJ2K/JPH-compatible fixture context with
`J2K_INCLUDE_OPENJPH=1` or `cargo xtask adoption-benchmark --openjph`. Use
`J2K_REQUIRE_OPENJPH=1` or `--require-openjph` only when `ojph_expand`
availability is part of the claim. Set `J2K_OPENJPH_EXPAND_BIN=/path/to/ojph_expand`
for non-standard installs. These rows are labeled
`decode_method=openjph-cli-process-output-pnm` because they go through the
OpenJPH CLI and PGM/PPM file output, so report them separately from the default
in-process J2K/OpenJPEG/Grok matrix.

Optional Kakadu rows can be added as proprietary CLI/file-output context with
`J2K_INCLUDE_KAKADU=1` or `cargo xtask adoption-benchmark --kakadu`. Use
`J2K_REQUIRE_KAKADU=1` or `--require-kakadu` only when Kakadu availability is
part of the claim. Set `J2K_KDU_EXPAND_BIN=/path/to/kdu_expand` and
`J2K_KDU_COMPRESS_BIN=/path/to/kdu_compress` for non-standard installs. Kakadu
decode rows are labeled `decode_method=kakadu-cli-process-output-pnm`; encode
rows are CLI/process JP2 rows validated against the same classic lossless profile.
Report them separately from the default in-process/publication matrix.

J2K-generated J2K/HTJ2K codestreams require native decoder round trips.
OpenJPEG and Grok comparisons are used where those tools support the feature.
Missing comparators cannot convert a parity signoff into a pass.

The fixture comparator has three modes, controlled by
`J2K_FIXTURE_COMPARE_MODE`:

- `portable-native` is the default and the only mode intended for publishable
  head-to-head decoder speed tables. It excludes native operations that cannot
  be measured comparably across J2K, OpenJPEG, and Grok.
- `portable-emulated` keeps the same tasks but labels emulated comparator work,
  for example OpenJPEG HTJ2K JP2 ROI+scaled as
  `decode_method=emulated-full-scaled-crop`.
- `capability` keeps feature-coverage rows and emits explicit skips such as
  `skip_reason=openjpeg-htj2k-roi-scaled-noncomparable`.

Do not report skipped rows or emulated rows as native OpenJPEG speed numbers.

For local smoke/development decode comparisons, use the shared generated
fixture matrix so every supported decoder receives the same named input bytes.
Generated-only output is not adoption-facing benchmark evidence. By default the
fixture comparator measures every per-fixture detail row at batch `1` and mixed
external throughput rows at `1,16,256,1024`; setting
`J2K_FIXTURE_COMPARE_BATCH_SIZES` preserves the old behavior of applying one
batch list to both row types:

```bash
J2K_REQUIRE_OPENJPEG=1 J2K_REQUIRE_GROK=1 \
  cargo run -p j2k-compare --release --bin jp2k_fixture_compare
```

Set `J2K_FIXTURE_COMPARE_INPUT_DIRS=/path/to/iso:/path/to/openjpeg-data:/path/to/domain`
to add external J2K/JP2/JPH fixtures recursively. Adoption-facing reports
require external corpora, strict comparator gates, `correctness_preflight`,
`benchmark_mode=portable-native`, `publication_blockers=none`,
`benchmark_complete`, mixed external batch rows, and
`publication_eligible=true`, plus a corpus mix documented in
[docs/benchmark-corpora.md](docs/benchmark-corpora.md). The publication gate
counts distinct external input digests separately from derived operation cases,
so a file that contributes both full and ROI-scaled rows does not count twice
for corpus diversity. Repo-materialized natural-image codestreams are workload
diagnostic rows; publishable decode claims also require independently sourced
native compressed J2K and HTJ2K fixtures.

For external-only publication runs, set
`J2K_FIXTURE_COMPARE_INCLUDE_GENERATED=0` and provide
`J2K_FIXTURE_COMPARE_MANIFEST=/path/to/fixtures.tsv` so corpus category,
source/license status, encode command, expected hash, codec, and container are
explicit instead of inferred from paths. The harness builds rotating owned input
copies outside the timed loop and interleaves decoder sample order. External
runs also emit `external_mixed_*` rows that cycle through the same distinct
fixture sequence for every decoder in a compatible format/operation group; use
those mixed rows for huge-batch throughput claims and the per-fixture rows for
fixture-level diagnosis. Publication gates require mixed-batch coverage for
gray/RGB full-image decode groups and for ROI-scaled groups that remain in the
selected comparable mode; the report records `mixed_external_group_distinct_inputs`
so a single dominant group cannot hide a missing batch surface.

For CPU encoder comparisons against OpenJPEG and Grok, use
`jp2k_encode_compare` or `cargo xtask adoption-benchmark --encode-fixtures`.
That harness accepts common 8-bit gray/RGB source image formats, stages them as
canonical PGM/PPM outside the timed loop, feeds the same staged PNM bytes to all
encoders via CLI processes, rotates encoder measurement order, forces
OpenJPEG/Grok single-thread encode options where supported, records an explicit
classic lossless JP2 encode profile, validates produced codestreams against that
profile, reports input MiB/s for mixed batches, and has its own
manifest gate: `J2K_ENCODE_COMPARE_MANIFEST` records corpus, license, source
command, and decoded-pixel hash. Publication gates require separate gray/RGB
mixed encode rows when making large-batch source-matrix claims. Do not use
public API Criterion encode rows as OpenJPEG/Grok encoder comparison evidence.

For CUDA HTJ2K encode hardware claims, pass `--cuda --require-cuda` plus the
same `--encode-fixtures` and `--encode-manifest` used for CPU encode claims.
The adoption runner forwards those staged PGM/PPM sources to the CUDA encode
bench through `J2K_CUDA_ENCODE_INPUT_DIRS` and `J2K_CUDA_ENCODE_MANIFEST`.
Those rows compare J2K CPU HTJ2K encode with J2K CUDA HTJ2K encode on the same
manifest-pinned pixels; they are not OpenJPEG/Grok encoder comparison rows.
When `--require-cuda` is set, `cargo xtask adoption-report` requires CUDA decode
and encode steps to have run, requires manifest-backed external cases, requires
generated CUDA host inputs to be disabled, and requires Criterion estimates.
For large CUDA decode batch claims, pass an explicit batch list such as
`--cuda-decode-batch-sizes 1,16,256,1024`; the value is recorded in
`j2k_cuda_decode_batch_sizes` in the CUDA decode output and `summary.json`.
The self-hosted CUDA workflow exposes the same path via the
`run-adoption-benchmark` dispatch input. For the full pinned corpus, configure repository variables
`J2K_ADOPTION_FIXTURES`, `J2K_ADOPTION_MANIFEST`,
`J2K_ADOPTION_ENCODE_FIXTURES`, and `J2K_ADOPTION_ENCODE_MANIFEST` to the
pinned fixture locations on the runner; the workflow runs
`cargo xtask adoption-benchmark --cuda --require-cuda` and uploads the bundle.
If those variables are absent, the workflow builds a default public starter
corpus from Kodak plus curated OpenJPEG data under `target/j2k-public-corpora`.
Hybrid outputs also emit `j2k_cuda_decode_io_policy`,
`j2k_cuda_encode_io_policy`, or `j2k_metal_encode_io_policy` so reports state
that staged inputs are preloaded and filesystem I/O is outside timed loops.
For Metal auto-routing claims, use `--metal --require-metal` with the same
encode fixture arguments; the runner forwards them through
`J2K_METAL_ENCODE_INPUT_DIRS` and `J2K_METAL_ENCODE_MANIFEST` and the Metal
benchmark emits external rows as `mode=lossless_external`. When
`--require-metal` is set, the report requires the Metal step to have run,
requires manifest-backed external rows, requires generated Metal host inputs to
be disabled, and rejects skipped auto-routing rows or probe errors.
After a full run, render the guarded publication report with
`cargo xtask adoption-report --run-dir target/j2k-adoption-benchmark/full`.
The report command refuses nonpublishable bundles unless
`--allow-nonpublishable` is passed for diagnostics.

## Security

Report vulnerabilities according to [SECURITY.md](SECURITY.md). Codec errors
should be explicit, non-sensitive, and should not silently treat unsupported
input as successful decode.

## License

Dual-licensed under either [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE), at your option.
