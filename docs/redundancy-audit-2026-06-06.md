# Repo-Wide Redundancy Audit - 2026-06-06

## Executive Summary

This audit reviewed the current repository worktree for redundant code,
duplicated logic, repeated fixtures, and intentional backend symmetry. The
audit used four read-only sub-agents covering CUDA, Metal, CPU/native, and
core/tooling, plus a coordinator clone scan and manual spot checks.

No refactors were made. This report is the only intentional file change from the audit pass.

Top conclusions:

- One P0 correctness risk was found in in-flight JPEG lossless capability reporting:
  capability eligibility could be recomputed from `Info` after decoder planning
  rejected unsupported SOF3 details. Current SOF3 work addresses this by keeping
  `UnsupportedPredictor` and lossless `NotImplemented` planning failures as
  inspection errors and adding predictor/restart/scan-parameter rejection
  coverage.
- The largest maintainability issue is not exact function clones. It is families of near-duplicate backend flows, especially JPEG Metal 4:2:0/4:2:2/4:4:4 decode paths, Metal runtime/session infrastructure, CUDA ABI byte-view helpers, and shared J2K wavelet/packet planning math.
- Exact duplicate test and fixture helpers are widespread and low risk, but easy to clean up once the current JPEG work stabilizes.
- Several apparent duplicates should be left alone for now: SIMD-specific IDCT implementations, Rust/shader ABI mirrors, facade reexports, and crate-local fixture copies needed for independent package tests.

Current dirty worktree context during the audit included in-flight JPEG files
and docs. The original audit did not patch the P0 below; the current SOF3 work
adds the follow-up fix and regression coverage.

## Verification

Commands and checks used:

- `git status --short`
- `cargo metadata --no-deps --format-version=1`
- `rg --files` and targeted `rg -n` scans
- Source-like line inventory over 434 text files
- Normalized 18-line block hashing
- Function-body hashing across Rust, Metal, CUDA, C, and headers
- SHA-256 inventory for fixture and generated-like artifacts
- Manual spot checks of the highest-ranked production candidates

Reproducibility result:

- Function-body exact-clone scan found 144 cross-file exact clone groups.
- The exact-clone scan was rerun twice with matching digest `3340dccaddbeb1bcd82e7bd6b5c349658477f437927027c04519df3f79663e82`.
- The coordinator scanner observed 7,361 function-like bodies and 12 duplicate artifact groups.

No unit tests, lint, typecheck, or benchmarks were run because this audit did not change production code. Each refactor item below lists the validation needed before implementation is accepted.

Largest source buckets by scanned line count:

| Bucket | Lines |
| --- | ---: |
| `crates/signinum-j2k-metal` | 58,277 |
| `crates/signinum-j2k-native` | 38,940 |
| `crates/signinum-jpeg-metal` | 38,216 |
| `crates/signinum-jpeg` | 34,497 |
| `crates/signinum-cuda-runtime` | 25,764 |
| `crates/signinum-j2k-cuda` | 18,507 |
| `crates/signinum-j2k` | 13,784 |
| `crates/signinum-transcode` | 13,280 |

## Ranked Findings

### P0 - JPEG Lossless Capability Predicate Can Diverge From Decoder Support

Subsystem: CPU/native JPEG, current dirty worktree.

Status: addressed in the current SOF3 follow-up by removing lossless planning
failures from parsed-`Info` fallback reporting and adding capability tests for
predictors 1-7 plus explicit predictor-8, restart, and scan-parameter rejection.

Evidence:

- `JpegCapabilityReport::inspect` falls back to reporting from parsed `Info` for `UnsupportedPredictor` in `crates/signinum-jpeg/src/capabilities.rs:97` and `crates/signinum-jpeg/src/capabilities.rs:295`.
- `cpu_eligibility` marks full `Gray8` SOF3 as eligible from `Info` only in `crates/signinum-jpeg/src/capabilities.rs:147`.
- `build_lossless_plan` rejects unsupported predictor, restart interval, scan count, component count, and scan params in `crates/signinum-jpeg/src/decoder.rs:464`.
- Existing positive capability test only covers predictor 1 in `crates/signinum-jpeg/tests/device_plan.rs:215`.

Confidence: High.

Risk: Unsupported SOF3 lossless inputs can be reported as CPU-eligible even though decoder setup rejects them. That is a correctness and API-contract issue for capability reporting.

Proposed consolidation: Introduce one shared lossless-support predicate or carry full lossless scan support metadata into capability reporting. Do not recompute "eligible" from `Info` alone after `UnsupportedPredictor` or related lossless planning failures.

Validation if refactored:

- Add capability tests for SOF3 predictor other than 1.
- Add capability tests for SOF3 with restart interval.
- Add capability tests for invalid SOF3 scan params such as `Se`, `Ah`, and `Al`.
- Run focused JPEG capability/decode tests before merging the current JPEG work.

### P1 - JPEG Metal Fast-Path Families Are Mostly Copy-Variant Code

Subsystem: Metal JPEG.

Evidence:

- `crates/signinum-jpeg-metal/src/compute.rs` contains 80 `fast420`/`fast422`/`fast444` decode, grouped batch, texture, region, and scaled helper functions totaling about 9,639 lines.
- Token similarity after normalizing 420/422/444 shows exact or near-exact pairs, including:
  - `encode_fast420_region_batch_item` at `compute.rs:3000` and `encode_fast422_region_batch_item` at `compute.rs:3850`: 168 lines each, 1.00 token Jaccard.
  - `try_decode_fast420_region_scaled_rgb_batch_to_surfaces_with_output` at `compute.rs:9983` and `try_decode_fast422_region_scaled_rgb_batch_to_surfaces_with_output` at `compute.rs:10894`: 273 lines each, 1.00 token Jaccard.
  - `try_decode_fast420_region_scaled_rgba_batch_to_textures` at `compute.rs:10447` and `try_decode_fast422_region_scaled_rgba_batch_to_textures` at `compute.rs:11358`: 258 lines each, 1.00 token Jaccard.
- Shader variants in `crates/signinum-jpeg-metal/src/shaders.metal` repeat boundary and packing logic across kernels such as `jpeg_decode_fast420_*`, `jpeg_decode_fast422_*`, and `jpeg_decode_fast444_*`.

Confidence: High for near-duplicate family; medium for exact safe abstraction shape.

Risk: High maintenance drift. Region, restart, scaling, and texture bug fixes can land in one chroma variant but not its sibling.

Proposed consolidation: Introduce typed helper structs for common batch decode resources, restart slicing, entropy buffer binding, table binding, pack dispatch, and grouped output handling. Keep shader kernels specialized by sampling geometry unless parity and performance prove a shared shader path is safe.

Validation if refactored:

- `cargo test -p signinum-jpeg-metal`
- Metal shader integrity tests
- Fast 4:2:0, 4:2:2, and 4:4:4 full/region/scaled/region-scaled parity tests
- Texture batch tests
- `cargo bench --no-run -p signinum-jpeg-metal`
- Runtime benchmark comparison for the hot fast paths

### P1 - Metal Runtime, Buffer, Surface, And Batch Infrastructure Is Reimplemented Per Crate

Subsystem: Metal infrastructure.

Evidence:

- Runtime/pipeline setup repeats in `crates/signinum-jpeg-metal/src/compute.rs:735`, `crates/signinum-j2k-metal/src/compute.rs:2153`, and `crates/signinum-transcode-metal/src/metal.rs:31`.
- Unsafe buffer allocation/readback helpers repeat around `crates/signinum-jpeg-metal/src/compute.rs:106`, `crates/signinum-j2k-metal/src/compute.rs:2541`, `crates/signinum-j2k-metal/src/compute.rs:9556`, and `crates/signinum-transcode-metal/src/metal.rs:2048`.
- JPEG and J2K batch/session skeletons repeat in `crates/signinum-jpeg-metal/src/batch.rs:13`, `crates/signinum-jpeg-metal/src/lib.rs:694`, `crates/signinum-j2k-metal/src/batch.rs:32`, and `crates/signinum-j2k-metal/src/lib.rs:388`.
- Surface API shape repeats in `crates/signinum-jpeg-metal/src/lib.rs:117` and `crates/signinum-j2k-metal/src/lib.rs:177`.

Confidence: High.

Risk: Medium. Device acquisition, cache policy, command queue behavior, unsafe readback, and batch waiting semantics can drift across backends.

Proposed consolidation: Add a small shared Metal support layer for device/runtime caching, shader library compilation, named pipeline loading, nonzero buffer allocation, typed upload/download helpers, and generic batch submission scaffolding. Preserve codec-specific typed runtime structs and execution callbacks.

Validation if refactored:

- macOS and non-macOS builds for all Metal crates
- Runtime/session tests
- Surface download/readback tests
- Shader integrity tests
- One real dispatch path per crate

### P1 - CUDA Runtime Has Repeated Unsafe POD Byte-View Helpers

Subsystem: CUDA runtime.

Evidence:

- `crates/signinum-cuda-runtime/src/lib.rs:12515` through `crates/signinum-cuda-runtime/src/lib.rs:12912` repeats the same unsafe `from_raw_parts` and `from_raw_parts_mut` patterns for numeric slices and CUDA ABI structs.
- Examples include `f32_slice_as_bytes`, `i32_slice_as_bytes`, `cuda_jpeg_decode_statuses_as_bytes_mut`, `htj2k_encode_jobs_as_bytes`, `htj2k_jobs_as_bytes`, and `idwt_multi_jobs_as_bytes`.

Confidence: High.

Risk: Medium. Unsafe layout assumptions are scattered, and future ABI struct changes have many places to audit.

Proposed consolidation: Introduce internal `pod_slice_as_bytes`, `pod_slice_as_bytes_mut`, and `pod_ref_as_bytes` helpers behind an unsafe `CudaAbi` or `Pod` marker trait implemented only for `repr(C)` integer/POD structs and primitive numeric types used by the runtime.

Validation if refactored:

- Unit tests for byte lengths and alignment assumptions.
- Size and offset checks for CUDA ABI structs where practical.
- CUDA runtime tests and kernel smoke tests.
- Existing CUDA decode/encode/transcode parity tests.

### P1 - Shared J2K Wavelet And IDWT Planning Math Is Duplicated Across CPU, CUDA, Metal, And Transcode

Subsystem: J2K native, CUDA, Metal, transcode.

Evidence:

- Exact `add_idwt_input_required_regions` bodies are duplicated in `crates/signinum-j2k-cuda/src/direct_plan.rs:632` and `crates/signinum-j2k-metal/src/compute.rs:4730`.
- 9/7 forward lifting logic appears in native and transcode paths at `crates/signinum-j2k-native/src/j2c/fdwt.rs:251` and `crates/signinum-transcode/src/dct97_2d.rs:381`.
- Reversible 5/3 lifting appears in `crates/signinum-transcode/src/accelerator.rs:675` and `crates/signinum-transcode/src/jpeg_to_htj2k.rs:4726`.
- DWT output conversion helpers are byte-identical across `crates/signinum-j2k/src/encode.rs:1065`, `crates/signinum-j2k-cuda/src/encode.rs:4096`, and `crates/signinum-transcode/src/jpeg_to_htj2k.rs:2185`, with the 9/7 equivalent at `encode.rs:1090`, `cuda/src/encode.rs:4121`, and `jpeg_to_htj2k.rs:2210`.

Confidence: High for duplication; medium for a single abstraction because public/private crate boundaries differ.

Risk: Medium-high. Boundary extension, margin, level packing, and band sizing fixes can diverge across CPU/GPU planners.

Proposed consolidation: Move backend-neutral wavelet geometry, band-region propagation, and public/native DWT conversion helpers to one crate boundary that all relevant crates can depend on without cyclic dependencies. Keep actual kernels and SIMD/GPU implementations specialized.

Validation if refactored:

- J2K native DWT/IDWT tests
- J2K CUDA direct plan tests
- J2K Metal device tests
- Transcode DCT/DWT/oracle tests
- Corpus validation and parity tests

### P1 - J2K Metadata Parsing Has Two Production Authorities

Subsystem: J2K parser/native decode.

Evidence:

- Public J2K parsing logic exists in `crates/signinum-j2k/src/parse/codestream.rs:25`, `crates/signinum-j2k/src/parse/boxes.rs:14`, and call sites such as `crates/signinum-j2k/src/view.rs:41`.
- Native J2K parsing logic exists separately in `crates/signinum-j2k-native/src/j2c/codestream.rs:46` and `crates/signinum-j2k-native/src/jp2/box.rs:69`.

Confidence: High.

Risk: Medium-high. Inspect/pass-through metadata and decode acceptance can drift as COD/QCD/RGN/HTJ2K support expands.

Proposed consolidation: Expose native metadata inspection through a stable internal API or move the shared JP2/J2C parser behind one crate boundary and reuse it from public inspect and native decode.

Validation if refactored:

- J2K parse, inspect, recode, and batch tests
- Native conformance and parity tests
- Fuzz targets
- JP2/J2C/HTJ2K passthrough transfer-syntax checks

### P1 - JPEG Baseline Entropy Traversal Duplicates Generic And RGB Paths

Subsystem: CPU JPEG entropy decode.

Evidence:

- `decode_scan_baseline` and `decode_scan_baseline_rgb` in `crates/signinum-jpeg/src/entropy/sequential.rs:174` and `crates/signinum-jpeg/src/entropy/sequential.rs:330` duplicate MCU traversal, restart handling, ROI/scaled stripe windows, `BitReader` setup, and scan finalization.
- The primary difference is output emission policy.

Confidence: High.

Risk: Medium. Restart, ROI, or downscale fixes can land in one path only.

Proposed consolidation: Factor scan traversal into a core routine with an output emitter/policy for grayscale/RGB and special 4:2:0 context-window behavior.

Validation if refactored:

- JPEG ROI/scaled/restart tests
- Scratch reuse tests
- Libjpeg-turbo comparison tests
- Decode fuzz
- Fast-path benchmarks

### P1 - CUDA Resident Transcode Full And Compact Paths Duplicate Production Flow

Subsystem: CUDA transcode.

Evidence:

- Full and compact HTJ2K 9/7 resident transcode paths repeat validation, grouping, dispatch, empty handling, timing, and error flow in `crates/signinum-transcode-cuda/src/cuda.rs:585`, `cuda.rs:645`, `cuda.rs:711`, `cuda.rs:803`, and paired helpers around `cuda.rs:1074`, `cuda.rs:1113`, `cuda.rs:1153`, `cuda.rs:1230`, `cuda.rs:1313`, and `cuda.rs:1416`.

Confidence: High.

Risk: Medium. Validation and timing/accounting behavior can diverge between full and compact modes.

Proposed consolidation: Extract common resident batch/group dispatch and status handling. Keep final payload assembly and compact/full output materialization mode-specific.

Validation if refactored:

- `cargo test -p signinum-transcode-cuda`
- CUDA parity tests
- Grouped batch tests
- Vendor GPU baseline/performance checks where available

### P1 - Release Package And Benchmark Taxonomy Is Duplicated Across Tooling

Subsystem: core/tooling.

Evidence:

- Package lists repeat across `xtask/src/main.rs:10`, `scripts/publish-crate.sh:8`, and repo-integrity expectations around `crates/signinum-core/tests/repo_integrity.rs:787`.
- Benchmark lists and filters repeat across `xtask/src/main.rs:330`, `xtask/src/perf_guard.rs:62`, and `xtask/src/perf_guard.rs:134`.

Confidence: High.

Risk: Medium. Publish/package dry-run behavior and performance signoff gates can drift.

Proposed consolidation: Use a single structured manifest or one `xtask` source of truth for package taxonomy and benchmark targets. Scripts and tests should query that source instead of duplicating lists.

Validation if refactored:

- `cargo test -p xtask`
- `cargo xtask package`
- `cargo xtask bench-build`
- Perf guard dry-run path where practical

## P2 Findings

### Test And Fixture Builders Are Repeated Across JPEG, J2K, Transcode, CLI, And Benches

Evidence:

- JPEG restart/grayscale helpers repeat in `crates/signinum-jpeg/tests/wsi_parity.rs:169`, `tests/device_plan.rs:569`, `tests/view_and_rows.rs:196`, and `tests/inspect.rs:76`.
- J2K JP2 wrapper helper repeats in `crates/signinum-j2k/tests/decode.rs:67`, `tests/batch.rs:62`, `tests/recode.rs:36`, `tests/grok_parity.rs:387`, and `benches/public_api.rs:79`.
- CLI minimal J2K/JP2 helpers repeat in `crates/signinum-cli/src/main.rs:126` and `crates/signinum-cli/tests/inspect_cli.rs:116`.
- Transcode Metal structured block builders repeat in `crates/signinum-transcode-metal/tests/dct97.rs:471`, `tests/dct53.rs:358`, and `benches/dct97.rs:932`.
- J2K Metal gray fixture/stub helpers repeat in `crates/signinum-j2k-metal/src/idwt.rs:97`, `src/store.rs:80`, `src/classic.rs:318`, and `src/mct.rs:120`.

Confidence: High.

Risk: Low to medium. Test fixtures can drift or encode malformed cases inconsistently.

Proposed consolidation: Add narrowly scoped test-support helpers. For production package boundaries, prefer `signinum-test-support` or per-crate `tests/fixtures/mod.rs` modules rather than pulling production crates into test-only fixture dependencies.

Validation if refactored:

- Affected package tests
- Bench compile for benches that import helpers
- Preserve explicit malformed-stream tests as separate cases

### Benchmark, Report, And Profile Utilities Are Repeated

Evidence:

- JPEG Metal corpus helpers repeat in `crates/signinum-jpeg-metal/benches/compare.rs:158` and `crates/signinum-jpeg-metal/src/bin/viewport_report.rs:187`.
- Test-only CUDA comparison CLIs repeat config/report formatting across their
  decode and transcode report paths.
- CUDA/JPEG/Metal bench helpers repeat dimension and availability parsing, for example `crates/signinum-jpeg-metal/benches/device_upload.rs:109` and `crates/signinum-jpeg-cuda/benches/device_decode.rs:158`.
- CUDA/JPEG/J2K profile route helpers repeat in `crates/signinum-jpeg-cuda/src/profile.rs:3` and `crates/signinum-j2k-cuda/src/profile.rs:21`.

Confidence: High.

Risk: Low, mostly stale reporting and inconsistent benchmark environment behavior.

Proposed consolidation: Use dev-only helper modules for report escaping, corpus classification, env parsing, and profile summary formatting.

Validation if refactored:

- CLI unit tests
- JSON/CSV output golden checks where available
- Benchmark `--no-run` builds

### Documentation Command Blocks Repeat

Evidence:

- CUDA HTJ2K profile commands repeat between `docs/bench.md:150` and `docs/adaptive-j2k-gates.md:51`.
- CUDA benchmark command lists repeat around `docs/adaptive-j2k-gates.md:461`.

Confidence: High.

Risk: Low, but stale benchmark instructions are likely.

Proposed consolidation: Keep canonical commands in `docs/bench.md`; gate records should link to canonical commands and preserve only run-specific evidence.

Validation if refactored:

- Markdown link checks if available
- Repo-integrity text checks if added

### J2K Signpost Event Lists Repeat

Evidence:

- `crates/signinum-j2k-metal/src/signpost.c:7` repeats event IDs/names across enum and begin/end switch logic.

Confidence: Medium-high.

Risk: Low profiling drift.

Proposed consolidation: Use one static table or macro list.

Validation if refactored:

- macOS build
- Basic signpost smoke output

## P3 / Leave Alone For Now

The following patterns are redundant-looking but should not be treated as refactor targets without stronger evidence:

- SIMD-specific JPEG IDCT implementations in `crates/signinum-jpeg/src/idct/scalar.rs:34`, `idct/neon.rs:45`, and `idct/avx2.rs:47`. These are performance-sensitive and architecture-specific.
- Rust/shader ABI mirrors in Metal and CUDA crates. The duplication is intentional at language and device boundaries; prioritize size/offset tests before codegen or shared headers.
- CUDA/JPEG IDCT kernels in decode versus transcode flows. They share mathematical structure but have different output policies.
- Facade reexports and stable API snapshots in `crates/signinum/src/lib.rs` and `docs/stable-api-1.0.public-api.txt`.
- Crate-local fixture copies when independent crate packaging requires them.
- Small tilecodec trait adapters in `crates/signinum-tilecodec/src/*.rs`; bounded IO helpers are already shared.

## Duplicate Binary And Fixture Inventory

Coordinator SHA-256 scan found 12 duplicate artifact groups. These are mostly package-boundary fixtures, not immediate deletion targets.

| Hash prefix | Count | Size | Files |
| --- | ---: | ---: | --- |
| `f2ba3b799977` | 8 | 324 | `baseline_420_16x16.jpg` in `corpus/conformance`, JPEG fuzz corpus, JPEG fixtures, JPEG Metal fixtures, JPEG CUDA fixtures, transcode fixtures, transcode Metal fixtures, transcode CUDA fixtures |
| `eb9bad4e5df6` | 6 | 314 | `baseline_422_16x8.jpg` in `corpus/conformance`, JPEG, JPEG Metal, JPEG CUDA, transcode, transcode Metal |
| `502872ae9404` | 6 | 311 | `baseline_444_8x8.jpg` in `corpus/conformance`, JPEG, JPEG Metal, JPEG CUDA, transcode, transcode Metal |
| `53e5cebc7a61` | 6 | 165 | `grayscale_8x8.jpg` in `corpus/conformance`, JPEG fuzz corpus, JPEG, JPEG Metal, transcode, transcode Metal |
| `4ad864df2c3e` | 4 | 364 | `baseline_420_restart_32x16.jpg` in `corpus/conformance`, JPEG, JPEG Metal, transcode |
| `721b1f341574` | 3 | 1536 | `baseline_420_restart_32x16.rgb` in `corpus/conformance`, JPEG, transcode |
| `951f69e8c442` | 3 | 768 | `baseline_420_16x16.rgb` in `corpus/conformance`, JPEG, transcode |
| `4c8a9a97587b` | 3 | 384 | `baseline_422_16x8.rgb` in `corpus/conformance`, JPEG, transcode |
| `59dffe4f9798` | 3 | 192 | `baseline_444_8x8.rgb` in `corpus/conformance`, JPEG, transcode |
| `e19b9939d85b` | 3 | 64 | `grayscale_8x8.gray` in `corpus/conformance`, JPEG, transcode |
| `1837d4e37327` | 2 | 790 | OpenHTJ2K `.j2k` fixture in J2K native and J2K CUDA |
| `f4f367d009f1` | 2 | 629 | OpenHTJ2K `.gray` fixture in J2K native and J2K CUDA |

Recommended handling:

- Keep crate-local copies where they are needed for publish/package tests.
- Add checksum tests or a manifest linking crate-local fixtures to canonical corpus entries.
- Avoid symlinks unless packaging behavior is confirmed for crates.io and local CI.

## Refactor Roadmap

### Immediate Blocker

1. Keep JPEG SOF3/lossless capability rejection coverage green before merging
   the in-flight JPEG lossless work.

### Safe First-Pass Cleanups

1. Centralize JPEG and J2K test fixture builders.
2. Add checksum coverage for duplicate conformance fixtures.
3. Consolidate CUDA POD byte-view helpers behind one audited unsafe helper.
4. Consolidate benchmark/report utility helpers for corpus discovery, env parsing, and JSON/CSV escaping.
5. Consolidate package and benchmark taxonomies in `xtask`.

### Higher-Risk Architectural Consolidation

1. Extract shared Metal runtime, buffer, surface, and batch infrastructure.
2. Reduce JPEG Metal fast-path copy-variant code with common resource binding and dispatch helpers.
3. Consolidate J2K wavelet/IDWT geometry and DWT conversion helpers across native, CUDA, Metal, and transcode.
4. Reconcile public J2K inspect parsing with native parser authority.
5. Factor JPEG baseline entropy traversal behind an emitter/policy abstraction.
6. Unify CUDA transcode full/compact resident dispatch flow.

## Sub-Agent Appendices

### CUDA Slice

Raw observations:

- No P0 findings.
- P1 findings: repeated unsafe POD byte-view helpers; full/compact HTJ2K resident transcode flow duplication; CUDA codec submit forwarding duplicated across JPEG/J2K adapters.
- P2 findings: profile/report helpers, host-surface tests, bench scaffolding, and duplicate/generated fixture artifacts.
- P3 findings: runtime/surface wrapper symmetry and kernel/ABI mirrors are expected backend specialization.
- Ignored generated target outputs except where noted as generated duplicates.

### Metal Slice

Raw observations:

- No P0 findings.
- P1 findings: duplicated runtime/pipeline setup, unsafe Metal buffer helpers, batch/session skeletons, and surface APIs across Metal crates.
- P2 findings: transcode test builders, J2K module test fixtures, JPEG bench/report helpers, duplicate fixtures, and J2K signpost lists.
- P3 findings: Rust/Metal ABI mirrors and J2K decode/encode shader symmetry are intentional.
- Large files are real hotspots, but size alone is not proof of redundant behavior.

### CPU/Native Slice

Raw observations:

- One P0 was found and independently spot-checked: dirty JPEG lossless capability predicate can diverge from decoder support.
- P1 findings: two J2K parser authorities, duplicated JPEG baseline entropy traversal, and duplicated transcode DCT/DWT geometry/helper math.
- P2 findings: JPEG/transcode fixture duplication, repeated JPEG byte builders, and repeated J2K JP2 wrapper helpers.
- P3 findings: SIMD IDCT implementations, generic versus fast scaled JPEG paths, and small tilecodec adapters should remain specialized for now.

### Core/Tooling Slice

Raw observations:

- No P0 findings.
- P1 finding: release package taxonomy is duplicated across `xtask`, publish script, and repo-integrity checks.
- P2 findings: benchmark target matrices, minimal JP2/J2K builders, deterministic fixture generators, and docs command blocks.
- P3 finding: facade public contract mirroring and stable API snapshots are intentional.
- Fixture inventory confirmed duplicate conformance assets across package boundaries.
