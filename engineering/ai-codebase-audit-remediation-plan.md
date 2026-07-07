# AI-Codebase Audit — Combined Findings and Remediation Plan

Date: 2026-07-04 (supersedes the July revision-2 plan and its status dashboard)

This document merges two independent audit passes over the working tree plus
hand-verification of every disputed finding:

- Pass A: multi-agent read-only audit (god files, cross-crate duplication,
  self-enforcement tooling, docs/site hygiene, panic surface) grounded in
  current AI-codebase research (duplication pressure, comprehension debt,
  over-engineered edge cases, panic-prone error handling).
- Pass B: gate-execution audit (ran every documented quality gate with
  `--all-features`) plus targeted correctness review of native parse, color,
  progression, Metal routing, CUDA build paths, and JPEG GPU adapters.
- Adjudication: findings claimed by only one pass were verified by reading the
  code or by empirical repro before inclusion. Non-findings are recorded in
  Section 4 so future audits do not re-flag them.

The previous plan's dashboard claimed "58 of 59 items DONE, workspace compiles
clean, all 113 repo lints pass". That was not true of the working tree at audit
time; Section 2 lists the broken gates. The refactor work itself was real —
roughly 71,000 lines were removed across 196 files — but it is uncommitted,
partially gate-breaking, and its size ratchets were set too loose (Section 6).

## Current execution status (updated 2026-07-07)

This section is the live status record for the integrated remediation sweep.
Sections 1-8 remain the audit record for what was found at audit time.

- **Normal test gates must not run benchmark executables.** The routine test
  gate is `cargo xtask test`, which expands to workspace `cargo test`
  invocations with `--lib --bins --tests` plus separate doc tests. Do not use
  `cargo test --workspace --all-targets --all-features` as a correctness gate:
  Cargo includes `[[bench]]` targets there, and this repository has expensive
  Criterion/GPU benchmark binaries. All current `[[bench]]` targets now declare
  `test = false`, and `xtask/tests/repo_lint.rs` has a regression lint requiring
  that setting for new benchmark targets.
- **Benchmark and performance evidence is explicit signoff work.**
  `cargo xtask bench-build`, `cargo xtask j2k-perf-guard`, `cargo bench`, and
  hardware-dependent CUDA/Metal benchmark runs are not part of routine
  code-change verification. Run them only during the performance-evidence phase
  or when a change directly targets benchmark code or performance behavior.
- **Phase 0 gate restoration is implemented in the working tree and the latest
  non-benchmark gate slice is green.** The docs moved to
  `engineering/`, `.tmp-metadata.json` was removed/ignored, clippy blockers were
  fixed, CUDA JPEG decode calls moved to `DecodeRequest`, stable API and unsafe
  audit were refreshed, stale unsafe rows fail closed, and `panic-surface` is
  wired into both `ci()` and CI. As of the 2026-07-07 status refresh,
  `cargo fmt --all --check`, `cargo check --workspace --all-features --lib
  --bins --examples --tests`, `cargo clippy --workspace --all-features --lib
  --bins --examples --tests -- -D warnings`,
  `cargo test -p xtask --test repo_lint -- --nocapture`,
  `cargo run -p xtask -- unsafe-audit`, `cargo xtask stable-api`,
  `cargo xtask panic-surface`, `cargo xtask semver`, `cargo deny check`, and
  `cargo machete` pass. Benchmark and GPU signoff remain separate.
  After the CUDA transcode API cleanup, JPEG-Metal shader helper extraction,
  and fixture-compare type split follow-ups, `cargo fmt --all --check`,
  `cargo check --workspace --all-features --lib --bins --examples --tests`,
  `cargo clippy --workspace --all-features --lib --bins --examples --tests -- -D warnings`,
  full `cargo test -p xtask --test repo_lint -- --nocapture`, and
  `cargo xtask test` pass without running benchmark executables.
  After the `j2k` encode-stage root-facade cleanup, CUDA surface API shrink,
  and CUDA grayscale HTJ2K plan-builder API shrink, stable API regeneration and
  verification, formatter check, and `git diff --check` pass. The current broad
  non-benchmark gate slice also passes:
  `cargo check --workspace --all-features --lib --bins --examples --tests`,
  `cargo clippy --workspace --all-features --lib --bins --examples --tests -- -D warnings`,
  `cargo test --workspace --all-features --lib --bins --examples --tests --no-fail-fast`,
  and full `cargo test -p xtask --test repo_lint -- --nocapture`. The
  lightweight policy gates `cargo run -p xtask -- unsafe-audit`,
  `cargo xtask panic-surface` (`unwrap_used 17/17`, `expect_used 98/106`),
  `cargo xtask semver`, `cargo deny check`, and `cargo machete` pass. This
  sweep also fixed the `cargo deny check` RUSTSEC-2026-0204 failure by updating
  `crossbeam-epoch` from 0.9.18 to 0.9.20 with `cargo update -p
  crossbeam-epoch`, then reran `cargo deny check` successfully. The broad test
  run was the non-benchmark selector above; benchmark-named harness tests either
  exercised policy/CLI behavior or stayed ignored, and no `cargo bench`,
  `cargo xtask bench-build`, or `cargo xtask j2k-perf-guard` command was run.
  CUDA hardware validation has also passed on the synced current tree; GitHub
  `gpu-validation.yml` dispatch/policy evidence remains a separate merge-policy
  item. Performance signoff was refreshed on 2026-07-07: `cargo xtask
  bench-build` passed, and `cargo xtask j2k-perf-guard --baseline-ref HEAD
  --quick` passed against `HEAD` (`29143c8e`) with the +10% guard threshold.
  The final quick guard recorded the previously regressed HT cleanup
  distribution rows at -9.59% and -8.22% versus baseline, and
  `jpeg_cpu_encode_runtime/rgb8_512_420_restart_64` at +1.66%, all within
  threshold. The local macOS run skipped Linux-only cuda-oxide kernels, so
  strict CUDA runtime benchmark validation remains hardware-dependent.
- **Phase 1 correctness work has landed.** The working tree contains typed
  Metal fallback/error classification, structured backend errors with
  failure-mode `BackendErrorKind` classification, shared CUDA/Metal native-decode error
  mappers that preserve truncated/unsupported classifications, checked
  shrink-factor handling, JPEG inspect/decode agreement coverage, canonical
  Huffman unification through `j2k-codec-math` with CUDA/Metal repo-lint
  coverage, adapter/facade deduplication, and Metal readback hardening:
  public byte views, direct decode status, decode-dispatch status/plane, HT
  cleanup status, Tier-1 encode status/profile, resident Tier-1 profile/status,
  forward-transform/lossless-prep/result-harvest paths, JPEG-Metal
  encode/decode/surface readbacks, transcode-Metal DWT 9/7 coefficient staging,
  byte-validation status readbacks, and JPEG-Metal viewport-cache CPU-side row
  writes now use checked helpers. Remaining raw `contents()` calls are limited
  to central helper internals, and repo lint now fails if new raw access appears
  outside that allow-list. Transcode Auto
  threshold policy is now documented for CUDA/Metal and pinned by repo lint;
  transcode stage counters and combined HTJ2K offer/dispatch accounting are
  shared in `j2k-transcode`, and CUDA/Metal now use the shared
  `TranscodeStageDispatchMode` for Auto/Explicit unavailable and recoverable
  error policy; repo lint pins both.
- **Phase 2 structural split work is mostly complete; residual shader
  texture/region scaffolding and parameter/API debt remain.**
  `crates/j2k-jpeg/src/decoder.rs` now has scratch/memory-cap math in
  `decoder/scratch.rs`, routes full-image and region-scaled lossless
  output-format dispatch through one helper, has the `SinkWriter` row-sink
  adapter in `decoder/sink_writer.rs`, reuses that adapter from bench profiling
  through a black-box `RowSink`, routes `ComponentRowWriter` through a blanket
  `OutputWriter for &mut W` implementation instead of a forwarding adapter,
  shares 8/16-bit lossless RGB/YCbCr sampling dispatch by bit depth, routes
  lossless RGB/YCbCr region fallback selection and RGBA scratch-copy through
  `decoder/lossless_region.rs`, and validates lossless color component,
  sampled, and row-stream paths through `decoder/lossless_helpers.rs`, including
  shared restart marker cadence through `LosslessRestartTracker` and
  `Extended12RestartTracker`. It is ratcheted below 3,985 lines. The Metal
  JPEG viewport-cache row writers now share `PlaneRowTarget`, with repo-lint
  coverage preventing a duplicate full-row writer from returning.
  `crates/j2k-native/src/j2c/encode.rs` now has raw sample width/sign-extension
  helpers in `encode/samples.rs`, public API conversion/deinterleave helpers in
  `encode/api_helpers.rs`, high-bit exact single-tile i64 encode helpers in
  `encode/single_tile.rs`, and i64 packetization helpers in
  `encode/i64_packetize.rs`; the active ratchet is tightened below 3,900
  lines. The
  `crates/j2k-compare/src/fixture_compare.rs` compare driver now has TSV row
  construction in `fixture_compare/rows.rs`, domain model enums in
  `fixture_compare/types.rs`, batch input ownership in
  `fixture_compare/types.rs`, external comparator CLI plumbing in
  `fixture_compare/comparators.rs`, publication-gate logic in
  `fixture_compare/gates.rs`, and is ratcheted below 2,295 lines.
  `crates/j2k-jpeg-metal/src/lib.rs` now re-exports the public `Decoder`,
  `Error`, `MetalBackendSession`, `MetalSession`, surface, and reusable Metal
  output types plus `JpegTileBatch` from focused modules, routes `Codec` batch
  implementation and RGB8 batch request types through `codec_batch.rs`, routes
  single-decode request types through `decode_request.rs`, and is ratcheted
  below 930 lines. The final shell-ratchet cleanup moved the internal
  `JpegFastPackets` tuple helper to `fast_packets.rs`; `lib.rs` is now 916
  lines against the focused-shell ratchet.
  `crates/j2k-metal/src/compute/resident_codestream.rs` now has classic
  profiling labels in `resident_codestream/classic_labels.rs` and is ratcheted
  below 2,785 lines. `crates/j2k-metal/src/compute.rs` is ratcheted below 390
  lines, and `crates/j2k-transcode/src/jpeg_to_htj2k.rs` now has its inline
  test module split to `jpeg_to_htj2k/tests.rs` and is ratcheted below 1,770
  lines.
  `xtask/tests/repo_lint.rs` is now a 3-line shim over the
  `repo_lint_support` module tree. The six former god files are split and
  tight-ratcheted with single-digit headroom where practical:
  `decoder.rs` is 3,974 lines against `<3,985`, `encode.rs` is 3,893 lines
  against `<3,900`, `resident_codestream.rs` is 2,778 lines against `<2,785`,
  `compute.rs` is below 390, `jpeg_to_htj2k.rs` is 1,760 lines against
  `<1,770`, and
  `repo_lint.rs` is a module shell.
  The MQ-coder QE table now lives once in `crates/j2k-native/src/j2c/mq.rs`,
  with encoder/decoder reuse protected by repo lint.
  The DCT-to-DWT 9/7 transcode path now imports f64 DWT constants from
  `j2k-codec-math`, with repo lint coverage preventing copied constants from
  returning. Component-plane metadata accessor methods are generated from one
  hidden `j2k-native` macro and reused by the public `j2k` facade, with repo
  lint coverage preventing local macro copies from returning.
  JPEG cache FNV-1a digest helpers now live once in `j2k-core` and are reused by
  the CPU, CUDA, and Metal JPEG session/context caches, with repo lint coverage
  preventing local FNV constants from returning.
  JPEG color fast-packet accessors now live in `j2k-jpeg` and are consumed by
  the CUDA and Metal adapters, with repo lint coverage preventing local accessor
  trait/macro copies from returning.
  GPU decoder CPU-host `ImageDecode` facades now use the `j2k-core`
  `CpuBackedImageDecode` blanket implementation, with repo lint coverage
  preventing local host-output facade copies from returning.
  The HT code-block scalar fallback lives in the `HtCodeBlockDecoder` trait
  default, with repo lint coverage preventing stage-only Metal adapters from
  restating it.
  The precomputed-DWT encode wrappers now share one forwarding macro with the
  defaulted unrelated hooks documented and pinned by repo lint.
  `too_many_arguments` suppression attributes are ratcheted at <=4 after broad
  conversion of native helpers, JPEG parser/facade helpers, JPEG-Metal helpers,
  CUDA launch helpers, transcode parameter groups, and CUDA JPEG decode
  quant/Huffman pointer groups to focused request or ABI structs. The ratchet
  counts multiline and crate-level attributes; the remaining suppressions are
  the broad `j2k-native` crate allow plus the CUDA Oxide JPEG/J2K/HTJ2K encode
  crate-level kernel allowances. The sampled-color lossless
  output renderer now shares one generic 8/16-bit loop, full-output and
  row-stream lossless color paths share one per-pixel component decode helper,
  sampled-color MCU component/plane traversal lives in one helper,
  lossless and extended-12 restart cadence use focused trackers, and Metal
  decode shaders share the DC-only/full-IDCT branch through `idct_block` plus
  single-image and batch entropy setup through `JPEG_ENTROPY_THREAD_VARS`,
  `JPEG_CONFIGURE_ENTROPY_THREAD`, `JPEG_BATCH_ENTROPY_THREAD_VARS`,
  `JPEG_CONFIGURE_BATCH_ENTROPY_THREAD`, and the simple full-image
  decode/idct/deposit block path through `decode_idct_deposit_block`; fast444,
  fast422, and fast420 region/scaled decode/deposit-or-skip routing now share
  `jpeg_decode_idct_deposit_region_block_or_skip` and
  `jpeg_decode_deposit_scaled_region_block_or_skip` from
  `shaders_decode_helpers.metal`; fast444, fast422, and fast420 non-region
  scaled decode/deposit paths also share `jpeg_decode_deposit_scaled_block`;
  texture batch checkpoint setup now routes through `configure_batch_entropy_thread`
  in all three sampling-family texture kernels; repeated four-slot texture
  repair metadata clearing now uses `jpeg_decode_clear_meta_quad`; YCbCr
  texture-write scaffolding now uses `jpeg_write_ycbcr_rgba` instead of direct
  per-kernel `rgba_float_ycbcr(...); out.write(...)` calls; fast422 texture
  boundary interpolation now uses shared `h2v1_boundary_*_from_samples`
  helpers instead of restating the h2v1 weighted sample arithmetic in decode
  kernels, and fast420/fast422 texture-boundary clamped copy spans now use
  `jpeg_clamped_extent(...)`; fast420 h2v2 texture-boundary weighted chroma
  sums now route through `h2v2_weighted_sample_sum`, and paired fast420 h2v2
  horizontal boundary writes now route through `jpeg_write_h2v2_boundary_pair`;
  fast420 h2v2 horizontal boundary top/bottom repair-row skip logic now routes
  through `jpeg_skip_h2v2_boundary_repair_row`;
  remaining JPEG decoder duplication is now broader decoder-family routing,
  while remaining Metal texture sampling work is limited to sampling-specific
  local row/edge orchestration.
  Stale
  suppressions are being removed as clippy proves them unnecessary. The
  repo-lint module-seam split is complete; the remaining Section 7 tooling work
  is the broader data-driven lint-runner conversion.
- **Phase 3 remains partially complete.** GPU policy is wired as
  `workflow_dispatch`-only unless policy approves automatic triggers; the CI
  `gpu-path-policy` job now checks PR diffs and successful manual
  `gpu-validation.yml` runs by head SHA for GPU-touching changes, and
  `CONTRIBUTING.md` plus repo-lint pin the manual-dispatch requirement.
  Docs/site cleanup items in Section 8 are now done. Duplicate `weezl` versions are gone,
  `block v0.1.6` is patched through
  `third_party/block-0.1.6-patched`, and `cargo deny check` passes. The
  adoption stack is now behind the opt-in xtask `adoption` feature for default
  xtask builds. Semver-visible API shrinkage has removed internal backend error
  constructors including the accidental public `J2kError::adapter_backend`
  helper, the unused adaptive-route policy model, the CUDA
  `cuda_dwt53_output_to_j2k_for_test` export, and the duplicate public
  `j2k_jpeg::adapter::fast_packet` submodule path from the rendered public API.
  The JPEG Metal viewport module is now private, with root-level facade exports
  only for the intended planner/decode surface; helper and resident-output
  viewport entrypoints are test-only/internal and absent from the stable API
  snapshot. Repo-lint guards prevent those paths from returning. The deprecated
  pre-1.0 `j2k-jpeg` owned-output method cross-product was removed in favor of
  `DecodeRequest`, with a repo-lint guard preventing the wrappers from
  returning. The deprecated pre-1.0 `j2k-jpeg-metal` decoder and tile-batch
  request-wrapper methods were removed in favor of `MetalDecodeRequest`, with a
  repo-lint guard preventing those wrappers from returning. The deprecated
  pre-1.0 `j2k-metal` decoder and tile-batch request-wrapper methods were also
  removed in favor of `MetalDecodeRequest`, with a repo-lint guard preventing
  those wrappers from returning. The `j2k-jpeg` decoder implementation module
  is now internal, with the intended root facade exports preserved and a
  repo-lint guard preventing the duplicate public `j2k_jpeg::decoder` module
  path from returning. The `j2k` view implementation module is now internal as
  well, with the intended root facade exports preserved and a repo-lint guard
  preventing the duplicate public `j2k::view` module path from returning. The
  duplicate public `j2k::{context,error,scratch}` implementation module paths
  are now internal too, with the intended root facade exports preserved and a
  repo-lint guard preventing those duplicate module paths from returning. The
  duplicate public `j2k_jpeg::{info,context,batch_session,capabilities,
  output_buffer,segment,error,encoder}` module paths are now internal as well.
  The `j2k_jpeg::adapter` and `j2k_jpeg::transcode` module inventories are now
  hidden from the rendered stable API while remaining source-visible to
  first-party adapter/transcode crates; `DeviceBatchSummary` remains visible
  through the `j2k_jpeg` root because it is embedded in
  `JpegCapabilityReport`. The duplicate public
  `j2k_transcode::{dct53_2d,dct97_2d,htj2k97_codeblock_oracle}` module paths
  are now internal too, with intended transform/oracle items exposed through the
  crate root and repo-lint coverage preventing the duplicate module paths from
  returning. The shared transcode stage-counter mutation API is now one typed
  `record(event, count)` method instead of a 16-method public `record_*`
  fanout, with repo-lint coverage preventing the old mutators from returning.
  The prequantized HTJ2K 9/7 oracle builders are now in the unpublished
  `j2k-transcode-test-support` crate, keeping GPU parity tests covered without
  exporting those test helpers from the stable `j2k-transcode` API.
  The shared device-decode request normalizer now lives at the `j2k` root
  facade as `DeviceDecodePlan`/`DeviceDecodeRequest`; the duplicate public
  `j2k::adapter::device_plan` module path is internal and guarded from
  returning. CUDA transcode 9/7 batch request shapes now expose only the
  method-specific `*_WithPoolRequest` structs; the duplicate inner
  `CudaDwt97BatchRequest` and `CudaHtj2k97CodeblockBatchRequest` public types
  were flattened away and guarded from returning. CUDA HTJ2K cleanup multi
  decode now exposes only the timed/status-returning entrypoint; the duplicate
  non-timed `decode_htj2k_codeblocks_cleanup_multi_with_resources_and_pool`
  wrapper was removed and guarded from returning. CUDA HTJ2K single-codeblock
  encode wrappers were then retired from the stable API; production and
  external tests use the supported one-job batch encode path, and repo lint
  guards the old wrappers from returning. CUDA HTJ2K resident encode now keeps
  the table-upload wrappers and explicit `_and_pool` resource-reuse wrappers;
  the duplicate implicit-pool resident `*_with_resources` wrappers were removed
  and guarded from returning. CUDA HTJ2K simple decode now keeps the public
  table-upload path plus multi-cleanup/pool APIs; the duplicate simple
  resource-backed public wrappers were removed or made internal. CUDA HTJ2K
  multi-buffer dequantize now exposes the explicit caller-pool API; the
  duplicate non-pool public wrapper was removed and guarded from returning.
  CUDA HTJ2K cleanup packetization now exposes the explicit tag-state API; the
  duplicate stateless no-tag public wrapper was removed and guarded from
  returning. `j2k-core` now keeps passthrough types on the root facade while
  hiding the duplicate public `j2k_core::passthrough` module path, and keeps
  `RowSink`/`ScratchPool` on the root facade while hiding their duplicate
  implementation-module paths. The same facade cleanup now keeps
  `PixelFormat`/`PixelLayout`, `Sample`/`SampleType`, and `Downscale` on the
  root facade while hiding the duplicate public
  `j2k_core::{pixel,sample,scale}` implementation-module paths.
  Shared error/classification types now also stay on the root facade while the
  duplicate public `j2k_core::error` implementation-module path is hidden.
  Shared backend, batch, context, device, trait, and metadata contracts also
  now stay on the root facade while their duplicate public
  `j2k_core::{backend,batch,context,device,traits,types}` implementation-module
  paths are hidden; `j2k_core::accelerator` remains public for the intentional
  `GpuAbi` path. JPEG baseline GPU encode keeps the public shared adapter
  trait/types and tile/batch entrypoints, while plan-building and validation
  helpers that are only used inside `j2k-jpeg` are now crate-private. CUDA
  runtime HTJ2K decode now keeps only the one-shot public decode/encode
  entrypoints in the rendered stable API; reusable table/resource handles and
  `*_with_resources*` entrypoints remain source-visible to first-party CUDA
  adapters but are hidden from the 1.0 inventory. The encode-stage accelerator
  contracts now stay on the `j2k` root facade while the duplicate public
  `j2k::adapter::encode_stage` module path is hidden. The unused CUDA surface
  profiling convenience `Surface::download_into_profiled` is removed from the
  semver-visible API; `download_into` remains the supported download entrypoint.
  CUDA grayscale HTJ2K plan-builder profiling hooks are now crate-private, with
  plan-shape and kernel-parity regression coverage moved to crate-local tests.
  The raw CUDA HTJ2K direct-plan kernel metadata model is now crate-private as
  well; supported CUDA decode entrypoints stay public, but unconstructable
  `CudaHtj2k*Plan`/step/rect/block metadata no longer appears in the rendered
  stable API snapshot. The duplicate JPEG adapter packet-builder convenience
  wrappers `build_fast{420,422,444}_packet_for_decoder` and
  `build_gray_packet_for_decoder` were then removed from the public adapter
  surface; first-party Metal callers use `decoder_bytes(decoder)` with the
  byte-slice packet builders instead. The custom RGBA alpha convenience
  wrappers `Decoder::decode_rgba8_into_with_alpha` and
  `Decoder::decode_region_rgba8_into_with_alpha` were then removed from the
  semver-visible API; crate-local tests exercise the underlying full-image and
  region `OutputFormat::Rgba8 { alpha }` paths, while the public API keeps the
  supported `PixelFormat::Rgba8` default-alpha path. The redundant JPEG decoder
  native-region scratch wrapper
  `Decoder::decode_region_into_with_scratch` was also made crate-private; the
  remaining external CPU-upload fallback now calls
  `decode_region_scaled_into_with_scratch(..., Downscale::None)` explicitly.
  The full JPEG adapter device-plan/checkpoint planning API is now hidden from
  the rendered 1.0 inventory while remaining source-visible to crate
  integration tests and first-party diagnostics; the lightweight
  `DeviceBatchSummary` remains visible because it is part of
  `JpegCapabilityReport`. The JPEG fast-packet raw ABI surface is also hidden
  from the rendered 1.0 inventory while remaining source-visible to first-party
  CUDA/Metal adapters. Raw CUDA JPEG runtime decode/encode ABI surfaces are
  hidden from the rendered 1.0 inventory while remaining source-visible to
  `j2k-jpeg-cuda`. CUDA HTJ2K profiling report types, explicit
  `_and_profile` entrypoints, and their side-effect `emit(...)` helpers are now
  hidden from the rendered inventory while remaining source-visible for
  diagnostics and tests. CUDA runtime HTJ2K resource-reuse APIs are also hidden
  from the rendered 1.0 inventory while remaining available to first-party
  adapters. The `j2k-core::CpuBackedImageDecode` adapter hook and
  its blanket `ImageDecode` implementation are now hidden from the rendered
  1.0 inventory while remaining source-visible to first-party CUDA/Metal
  adapters. The lower-level `j2k-jpeg-metal` tile-submit helper is also hidden
  from the rendered 1.0 inventory while the public trait methods remain the
  supported submit surface. The JPEG Metal resident batch preflight report and
  report-consuming resize helpers are now hidden from the rendered inventory
  while the normal batch decode and explicit tile/texture allocation APIs
  remain visible. The `j2k-core` ready-submission/session helpers are
  hidden from the rendered 1.0 inventory while the public `DeviceSubmission`
  wait contract remains visible. Synchronous CUDA/Metal submit impl renderings
  that exposed the hidden `ReadySubmission` helper as an associated type are
  now hidden too; the public device-submit traits and inherent decode APIs
  remain visible. The Metal decode route-report diagnostic
  types and `decode_request_to_device_with_report` are now hidden from the
  rendered inventory while the normal `decode_request_to_device` API remains
  visible. CUDA lossless encode timing-report outcome types and `*_with_report`
  CUDA buffer encode entrypoints are also hidden from the rendered inventory
  while normal encode/submit APIs remain visible. Metal lossless encode
  timing/report outcome and stats types plus
  `encode_lossless_batch_with_report` are hidden from the rendered inventory
  while normal Metal submit/config/tile APIs remain visible. The JPEG Metal
  viewport helper `decode_viewport_to_surface` is hidden from the rendered
  inventory because it exposes the internal JPEG scratch-pool type and is used
  by first-party tests/reporting/benchmark code rather than the stable user
  surface. The root-level `j2k-jpeg` `_with_options` decode free-function
  wrappers are hidden from the rendered inventory while the explicit
  `JpegBatchSession` methods and request/view option paths remain visible.
  CUDA/Metal adapter `ImageCodec` impl renderings are now hidden so the
  rendered inventory no longer exposes the private
  `j2k_jpeg::internal::scratch::ScratchPool` and
  `j2k::scratch::J2kScratchPool` defining paths; the source trait impls remain
  available. The `j2k_jpeg::adapter` and `j2k_jpeg::transcode` rendered module
  inventories are hidden too; `DeviceBatchSummary` is re-exported through the
  `j2k_jpeg` root for the public capability-report field. The
  `j2k_transcode::accelerator` compatibility module is also hidden from the
  rendered inventory; the accelerator contract is root-defined and
  `j2k_transcode::accelerator::*` remains a hidden source-compatible re-export
  for first-party callers. The adjacent
  `idct_blocks_to_signed_samples_rayon` helper remains source-visible for the
  Metal transcode adapter but is hidden from the rendered inventory. The current
  decoder `ImageCodec`/`ImageDecode`/`ImageDecodeRows` trait-adapter impl
  renderings for `j2k::J2kDecoder` and `j2k_jpeg::Decoder` are hidden as
  duplicate documentation of already-public inherent decode methods. Concrete
  `CodecContext`/`ScratchPool` impl renderings for root JPEG 2000/JPEG
  contexts and caller-owned scratch pools are hidden too; the types and
  intended inherent constructors/accessors remain visible. The current stable
  API snapshot is still large but currently at 239,689 bytes with 665 `pub fn`
  entries; current rendered package counts include 114 `j2k` public functions,
  94 `j2k-jpeg` public functions, 76 `j2k-core` public functions, 74
  `j2k-native` public functions, 60 `j2k-jpeg-metal` public functions, 40
  `j2k-metal` public functions, 39 `j2k-cuda` public functions, 40
  `j2k-transcode` public functions, 38 `j2k-cuda-runtime` public functions, 30
  `j2k-metal-support` public functions, 21 `j2k-types` public functions, 16
  `j2k-transcode-metal` public functions, 12 `j2k-jpeg-cuda` public functions,
  7 `j2k-tilecodec` public functions, 2 `j2k-codec-math` public functions, and
  2 `j2k-transcode-cuda` public functions.
  The latest `j2k-core` backend memory helper slice hides
  `checked_surface_len`, `copy_tight_pixels_to_strided_output`,
  `ensure_allocation_within_cap`, `strided_output_len`,
  `strided_output_len_capped`, and `validate_strided_output_buffer` from the
  rendered 1.0 inventory while preserving source-visible first-party adapter
  access and keeping the user-facing allocation cap constant visible.
  A follow-up hides the `IndexedBatchResult` alias plus the
  `collect_indexed_batch_results`, `tile_batch_worker_count`, and
  `validate_cuda_surface_backend_request` first-party batch/backend helpers
  from the rendered inventory while preserving source-visible callers.
  The context-reuse `decode_tile_*_in_context` helpers in both `j2k` and
  `j2k-jpeg` are hidden from the rendered inventory as first-party adapter/tool
  hooks; ordinary one-shot and batch tile decode APIs remain visible.
  `JpegBatchSession::worker_count` and `retained_worker_slots` are also hidden
  from the rendered inventory as diagnostics; session construction, options,
  reset, and decode methods remain visible.
  The latest JPEG-CUDA public API slice hides owned-output cache/session helpers
  and direct RGB8 caller-owned CUDA buffer decode entrypoints from the rendered
  inventory while preserving source-visible first-party access; repo lint now
  guards those names from returning.
  The latest CUDA stats slice hides duplicate CUDA-specific `CudaSurfaceStats`/
  `CudaSurface::stats` renderings and `j2k_jpeg_cuda::CudaJpegDecodePath`;
  generic `DeviceSurface::execution_stats()` remains the public stats path.
  The latest tilecodec slice hides concrete `TileDecompress` impl renderings
  for `DeflateCodec`, `LzwCodec`, `UncompressedCodec`, and `ZstdCodec`; the
  codec types and the public `TileDecompress` trait remain source-visible.
  The latest CUDA/Metal adapter slice hides concrete device-trait impl
  renderings (`TileBatchDecode*`, `ImageDecode*`, `DeviceSurface`,
  `AcceleratorSession`, and submitted encode `DeviceSubmission` impls) while
  leaving the public `j2k_core` trait contracts and all source-visible impls
  intact.
  The latest root decoder slice hides the remaining concrete `ImageCodec`
  associated-type renderings for `j2k::J2kDecoder` and `j2k_jpeg::Decoder`;
  the public `j2k_core::ImageCodec` trait and source-visible impls remain
  intact.
  This public API slice was checked with `cargo xtask stable-api --write`,
  `cargo xtask stable-api`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`,
  `cargo check -p j2k -p j2k-jpeg --all-features --lib --tests`,
  `cargo clippy -p j2k -p j2k-jpeg --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k -p j2k-jpeg --all-features --lib --tests --no-fail-fast -- --nocapture`,
  `cargo check -p j2k-tilecodec --all-features --lib --tests`,
  `cargo clippy -p j2k-tilecodec --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-tilecodec --all-features --lib --tests --no-fail-fast -- --nocapture`,
  `cargo check -p j2k-cuda -p j2k-jpeg-cuda -p j2k-metal -p j2k-jpeg-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda -p j2k-jpeg-cuda -p j2k-metal -p j2k-jpeg-metal --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-cuda -p j2k-jpeg-cuda -p j2k-metal -p j2k-jpeg-metal --all-features --lib --tests --no-fail-fast -- --nocapture`,
  `cargo check -p j2k-transcode -p j2k-transcode-cuda -p j2k-transcode-metal -p j2k-transcode-test-support --all-features --lib --tests`,
  `cargo clippy -p j2k-transcode -p j2k-transcode-cuda -p j2k-transcode-metal -p j2k-transcode-test-support --all-features --lib --tests -- -D warnings`,
  `cargo check -p j2k-transcode -p j2k-transcode-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-transcode -p j2k-transcode-metal --all-features --lib --tests -- -D warnings`,
  `cargo check -p j2k-jpeg --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg --all-features --lib --tests -- -D warnings`,
  `cargo check -p j2k -p j2k-jpeg --all-features --lib --tests`,
  `cargo clippy -p j2k -p j2k-jpeg --all-features --lib --tests -- -D warnings`,
  `cargo check -p j2k-jpeg-metal -p j2k-jpeg-cuda -p j2k-metal -p j2k-cuda --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg-metal -p j2k-jpeg-cuda -p j2k-metal -p j2k-cuda --all-features --lib --tests -- -D warnings`,
  `cargo check -p j2k-cuda-runtime -p j2k-cuda -p j2k-transcode-cuda --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda-runtime -p j2k-cuda -p j2k-transcode-cuda --all-features --lib --tests -- -D warnings`,
  `cargo check -p j2k-core -p j2k-cuda-runtime -p j2k-metal-support --all-features --lib --tests`,
  `cargo clippy -p j2k-core -p j2k-cuda-runtime -p j2k-metal-support --all-features --lib --tests -- -D warnings`,
  `cargo check -p j2k -p j2k-core --all-features --lib --tests`,
  `cargo clippy -p j2k -p j2k-core --all-features --lib --tests -- -D warnings`,
  `cargo check -p j2k-jpeg -p j2k-core --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg -p j2k-core --all-features --lib --tests -- -D warnings`,
  `cargo check -p j2k-types -p j2k-transcode --all-features --lib --tests`,
  `cargo clippy -p j2k-types -p j2k-transcode --all-features --lib --tests -- -D warnings`,
  `cargo check -p j2k-cuda -p j2k-metal -p j2k-transcode-cuda -p j2k-transcode-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda -p j2k-metal -p j2k-transcode-cuda -p j2k-transcode-metal --all-features --lib --tests -- -D warnings`,
  `cargo check -p j2k-native --all-features --lib --tests`,
  `cargo clippy -p j2k-native --all-features --lib --tests -- -D warnings`,
  `cargo fmt --all --check`, and `git diff --check`; no benchmark executable
  was run.
  The latest CUDA runtime follow-up hid the generic
  `CudaKernel{,Batch,ContiguousBatch}Output`, `CudaPooledKernelOutput`, and
  contiguous-output range wrappers plus the typed `CudaDeviceBufferView`
  wrappers and `CudaDeviceBuffer::typed_view{,_mut}` methods from the rendered
  1.0 inventory, then hid raw HTJ2K code-block job and lookup-table structs
  from the rendered inventory while preserving source-visible first-party
  adapter/runtime access. It was checked with
  `cargo fmt --all --check`,
  `cargo check -p j2k-cuda-runtime --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda-runtime --all-features --lib --tests -- -D warnings`,
  `cargo check -p j2k-cuda-runtime -p j2k-cuda -p j2k-transcode-cuda --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda-runtime -p j2k-cuda -p j2k-transcode-cuda --all-features --lib --tests -- -D warnings`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`,
  and `git diff --check`.
  The latest trait-impl rendering cleanup hid concrete `CodecError` impl
  method inventories for `j2k`, `j2k-jpeg`, CUDA/Metal J2K adapters, and
  CUDA/Metal JPEG adapters. The trait contract remains visible through
  `j2k_core::CodecError`, and the concrete impls still satisfy downstream
  bounds. It was checked with `cargo fmt --all --check`,
  `cargo check -p j2k -p j2k-jpeg -p j2k-cuda -p j2k-metal -p j2k-jpeg-cuda -p j2k-jpeg-metal --all-features --lib --tests`,
  `cargo clippy -p j2k -p j2k-jpeg -p j2k-cuda -p j2k-metal -p j2k-jpeg-cuda -p j2k-jpeg-metal --all-features --lib --tests -- -D warnings`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`,
  and `git diff --check`.
  The latest `j2k` root-facade cleanup hid the duplicate encode-stage contract
  inventory from the rendered API while preserving source-visible access and
  the canonical `j2k-types` contract. It removed the root `j2k::` reexport
  renderings for encode-stage jobs, reports, code-block outputs,
  packetization structs, and precomputed/prequantized/preencoded HTJ2K images.
  It was checked with `cargo fmt --all --check`,
  `cargo check -p j2k --all-features --lib --tests`,
  `cargo clippy -p j2k --all-features --lib --tests -- -D warnings`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`,
  and `git diff --check`; no benchmark executable was run.
  Remaining public API review should focus on the largest surviving clusters:
  the `j2k`, `j2k-jpeg`, `j2k-transcode`, `j2k-native`, and `j2k-core`
  public-method inventories plus any remaining backend memory utility surfaces.
  The decoder clusters now render only their
  inherent public methods plus intentional root facade items, not duplicate
  trait-adapter method inventories. The
  performance evidence record is also still open: benchmark evidence has not
  been refreshed with a new dated run during this non-benchmark work slice.
  `repo_lint.rs` now fails closed when const-array-derived package lists parse
  empty, publishable package entries must resolve to real crate directories,
  and shared repo-lint helpers live in `repo_lint_support` with guarded
  file-must-contain and must-not-contain checks plus repeated required-pattern
  checks migrating to shared runners. `FilePatternCheck` now provides a
  data-driven file-pattern runner with file existence, empty-row, required,
  forbidden, and normalized-pattern guards; README codec API coverage, CI
  docs/benchmark compile-gate coverage, and backend surface
  metadata/residency checks use it, along with the decode-capability
  shrink-factor/progressive inspect guard and GPU coverage-exclusion/substitute
  evidence guard. Public-doc checks for public-crate release posture, current
  crate routing, support-matrix coverage, and the reset J2K Metal bench surface
  also use it. Source-policy checks for
  non-clobbering CUDA trace export documentation and reusable
  benchmark-generator ownership use it too. Release-policy checks for
  release-doc version policy, j2k-compare package exclusion, and staged
  dependency preflight text use it as well. Architecture-policy checks for
  public facade crate membership, architecture doc classification, unpublished
  tooling crates, and adaptive-route API exclusion use it too. Corpus-policy
  checks for OpenHTJ2K fixture notice and license coverage and stale
  empty-corpus README text also use it.
  `PatternCheck` now applies the same required/forbidden/normalized-pattern
  runner to already-extracted text sections; the scoped `xtask test` function,
  CI coverage job, xtask nextest/machete/strict-clippy dispatch/help/command
  guard, release
  cargo-metadata prerequisite check, CI semver job, stable-API prerequisite
  check, per-crate crates.io README/docs.rs metadata checks, and README public
  example links use it without broadening to whole-file matches where scoped
  text was already extracted. Public-doc benchmark publication, pinned
  starter-corpus fallback, publication-gate single-source, and Metal
  consistency guards now use it too, leaving `public_docs_policy.rs` without
  hand-written required/forbidden pattern loops. Env-var docs and README
  pointer checks now use `FilePatternCheck`. `RustSourceScanCheck` now fails
  closed for empty Rust-source scan sets, empty directory lists, empty
  forbidden patterns, and empty source directories; the adapter private-module
  import ban, production CUDA nvJPEG ban, and GPU hardware-gate silent-return
  ban use it. In
  `docs_and_workflows_policy.rs`, typed Metal retry classification, GPU adapter
  error classification, packet progression ordering, IDWT required-region
  propagation, Metal direct required-region retain, Metal direct sub-band group
  scan, and Metal hybrid region-scaled cache guards now use `PatternCheck`.
  Wavelet/IDCT constant ownership, JPEG GPU encode host orchestration, Metal
  session lifecycle, fast444 shared Metal paths, JPEG fast420 profiled scan
  sharing, and CUDA HTJ2K compact planner guards now use it too. Native
  classic/HT decoded-block copyback, CUDA Oxide SIMT prelude, copied fixture
  helper, Metal compute runtime split, and native encode option/tile-part
  split guards now use it as well. The remaining native encode helper
  ownership guards and JPEG decoder focused-module ownership guards also use
  `PatternCheck`. Transcode Auto threshold policy, transcode shared stage
  counters, and Metal direct plan type split guards now use `PatternCheck` too.
  Shared MQ table ownership, component-plane accessor ownership, FNV-1a JPEG
  cache helper ownership, and shared JPEG fast-packet accessor ownership now use
  `FilePatternCheck` where the assertions are literal file pattern policy; the
  component macro call-count assertions remain explicit structural checks. The
  CPU-backed GPU decoder facade ownership row and the j2k-metal
  `j2k-codec-math` dependency row now use `FilePatternCheck` too. CI fuzz
  budget, `deny.toml` paste-advisory metadata, GPU-validation workflow policy,
  CUDA Oxide strict-build documentation, retired GPU-comparator workflow
  checks, and the xtask adoption-stack feature-gate policy now use
  `FilePatternCheck` as literal file-pattern policy rows as well.
  Whitespace-normalized matching helpers now cover HT fallback and CI
  permissions checks. The `repo_lint.rs` entry point is now
  a thin module shell, with
  `architecture_policy`, `corpus_policy`, `dependency_policy`,
  `docs_and_workflows_policy`, `public_docs_policy`, `release_policy`,
  `shader_policy`, `source_policy`, and `workflow_policy` in focused files
  under `repo_lint_support`; the broader data-driven lint-runner migration
  remains open.
- **Latest non-benchmark evidence:** As of the 2026-07-07 gate sweep,
  `cargo fmt --all --check`,
  `cargo check --workspace --all-features --lib --bins --examples --tests`,
  `cargo clippy --workspace --all-features --lib --bins --examples --tests -- -D warnings`,
  `cargo test -p xtask --test repo_lint -- --nocapture`,
  `cargo run -p xtask -- unsafe-audit`, `cargo xtask stable-api`,
  `cargo xtask panic-surface` (`unwrap_used 17/17`, `expect_used 98/106`),
  `cargo xtask semver`, `cargo deny check`, `cargo machete`, and
  `cargo xtask test` pass. The post-run `cargo fmt --all --check` and
  `git diff --check` sanity checks also pass. The `xtask` test gate ran the workspace
  non-benchmark test/doc-test path plus the serialized `j2k-metal` leg;
  benchmark-named harness tests that are intentionally ignored remained
  ignored, and no `cargo bench`, Criterion, `bench-build`, or perf-guard command
  was run. This sweep is current after the `j2k-jpeg-metal` focused-shell
  ratchet fix that moved `JpegFastPackets` into `fast_packets.rs`. The earlier
  reported `J2kError::backend` visibility build break is not present in the
  current tree. Hardware-dependent CUDA validation has since
  been rerun manually on the provided self-hosted CUDA host; GitHub
  `gpu-validation.yml` dispatch evidence/policy signoff remains a separate
  merge-policy item. Dated benchmark/performance evidence is now recorded in
  `docs/benchmark-evidence.md` from the 2026-07-07 quick perf-guard pass.
  After the latest CUDA runtime resource-reuse API hiding slice,
  `cargo check --workspace --all-features --lib --bins --examples --tests`
  also passes with the non-benchmark selector; no `--all-targets`, benchmark
  executable, Criterion command, `bench-build`, or perf-guard command was run.
  CUDA host validation used `jcwal@100.75.125.59` (WSL2 Linux, RTX 4070 SUPER,
  NVIDIA driver 596.49) with the 2026-07-07 current working tree synced to
  `/home/jcwal/codex-runs/j2k-remediation-20260707`, excluding build outputs
  and large WSI sample artifacts. The final `rsync -ani --delete` dry-run
  reported no pending source delta under those validation excludes. A strict
  cuda-oxide staging bug was found earlier and
  fixed by copying `src/cuda_oxide_simt_prelude.rs` into `OUT_DIR` before
  building generated SIMT crates; repo lint now pins that staging. With
  `LIBCLANG_PATH=/home/jcwal/.local/llvm18/usr/lib/llvm-18/lib` and
  `BINDGEN_EXTRA_CLANG_ARGS=-I/home/jcwal/.local/llvm18/usr/lib/llvm-18/lib/clang/18/include`,
  the remote `J2K_REQUIRE_CUDA_OXIDE_BUILD=1 cargo check -p j2k-cuda-runtime
  --all-features --lib` passed and built all enabled cuda-oxide projects for
  `sm_80`. Runtime-required CUDA tests also passed:
  `J2K_REQUIRE_CUDA_RUNTIME=1 J2K_REQUIRE_CUDA_OXIDE_BUILD=1 cargo test -p
  j2k-cuda-runtime -p j2k-cuda -p j2k-jpeg-cuda -p j2k-transcode-cuda
  --all-features --lib --tests -- --nocapture`, followed by
  `J2K_REQUIRE_CUDA_RUNTIME=1 J2K_REQUIRE_CUDA_OXIDE_BUILD=1
  J2K_REQUIRE_CUDA_JPEG_HARDWARE_DECODE=1 cargo test -p j2k-jpeg-cuda
  --all-features --lib --tests -- --nocapture`. After the latest JPEG-CUDA
  rendered-API hiding slice, the current tree was synced again with an empty
  final `rsync -ani` delta and the same JPEG-CUDA hardware command passed
  again. After the latest CUDA stats hiding slice, the current tree was synced
  again with an empty final `rsync -ani` delta, and
  `J2K_REQUIRE_CUDA_RUNTIME=1 J2K_REQUIRE_CUDA_OXIDE_BUILD=1
  J2K_REQUIRE_CUDA_JPEG_HARDWARE_DECODE=1 cargo test -p j2k-cuda -p
  j2k-jpeg-cuda --all-features --lib --tests -- --nocapture` passed on the
  CUDA host. No benchmark executable or perf-guard command was run for this
  CUDA validation.
  After updating the transcode-Metal harness guard from the removed
  `debug_report` helper to structured `TranscodePipelineMap` fields, the
  focused `cargo test -p j2k-transcode-metal --test bench_harness -- --nocapture`
  check and the full `cargo xtask test` wrapper both pass.
  `cargo test --workspace --all-features --lib --bins --tests` and
  `cargo test --workspace --all-features --doc` also pass with explicit
  non-benchmark selectors through the `cargo xtask test` wrapper. After removing
  the stale
  `crates/j2k-jpeg-metal/tests/viewport.rs` unsafe-audit row, current
  non-benchmark evidence also includes `cargo run -p xtask -- unsafe-audit`,
  `cargo xtask stable-api`, `cargo xtask panic-surface`
  (`unwrap_used 17/17`, `expect_used 98/106`), `cargo xtask semver`,
  `cargo deny check`, `cargo machete`, and full
  `cargo test -p xtask --test repo_lint -- --nocapture` (131 passed). After the
  core tile region-scaled request-object cleanup, current evidence includes
  `cargo check --workspace --all-features --lib --bins --examples --tests`,
  `cargo clippy -p j2k-core -p j2k -p j2k-jpeg -p j2k-cuda -p j2k-jpeg-cuda -p j2k-metal -p j2k-jpeg-metal --all-features --lib --tests -- -D warnings`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`, and full
  `cargo test -p xtask --test repo_lint -- --nocapture` (131 passed). After the
  unused CUDA JPEG 4:2:0-specific owned-decode facade removal, current evidence
  includes
  `cargo check -p j2k-cuda-runtime -p j2k-jpeg-cuda --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda-runtime -p j2k-jpeg-cuda --all-features --lib --tests -- -D warnings`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`, and
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`;
  the generic `decode_jpeg_rgb8_owned*` entrypoints remain and no benchmark
  executable was run. After removing the unused public CUDA runtime IDWT
  untimed wrappers `j2k_inverse_dwt_single_device_untimed` and
  `j2k_inverse_dwt_batch_device_untimed_with_pool` while preserving the timed
  single-device/batch paths, the pooled steady-state path used by `j2k-cuda`,
  and the async enqueue path, current evidence includes
  `cargo check -p j2k-cuda-runtime -p j2k-cuda --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda-runtime -p j2k-cuda --all-features --lib --tests -- -D warnings`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`,
  `cargo test -p j2k-cuda --test bench_harness --all-features -- --nocapture`,
  and
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`;
  no benchmark executable was run. After making CUDA/Metal encode-stage
  per-stage attempt and dispatch getters test-only internals while preserving
  public `dispatch_report()`, current evidence includes
  `cargo check -p j2k-cuda -p j2k-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda -p j2k-metal --all-features --lib --tests -- -D warnings`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`,
  `cargo test -p j2k-cuda --test encode_stage_api cuda_encode_stage_can_prefer_cpu_packetization --all-features -- --nocapture`,
  `cargo test -p j2k-cuda --lib --all-features prefer_cpu_ht_subband_declines_fused_subband_but_counts_attempts -- --nocapture`,
  `cargo test -p j2k-cuda --lib --all-features cuda_auto_host_output_declines_packetization_before_flattening -- --nocapture`,
  `cargo test -p j2k-metal --lib --all-features metal_encode_stage_accelerator_can_leave_forward_rct_on_cpu -- --nocapture`,
  and
  `cargo test -p j2k-metal --lib --all-features auto_lossy_packet_marker_shape_stays_cpu_without_packetization_dispatch -- --nocapture`;
  no benchmark executable was run. After the
  `repo_lint_support` split,
  `cargo check -p xtask --all-features --test repo_lint`,
  `cargo clippy -p xtask --all-features --test repo_lint -- -D warnings`, and
  `cargo test -p xtask --test repo_lint -- --nocapture` pass; repo lint now
  runs 131 lints. After the latest docs/workflows `FilePatternCheck` migration,
  `cargo fmt --all`,
  `cargo test -p xtask --test repo_lint ci_fuzz_run_budgets_are_nontrivial -- --nocapture`,
  `cargo test -p xtask --test repo_lint deny_paste_advisory_ignore_has_review_metadata -- --nocapture`,
  `cargo test -p xtask --test repo_lint gpu_validation_workflow_is_self_hosted_and_explicit -- --nocapture`,
  `cargo test -p xtask --test repo_lint cuda_oxide_shared_strict_build_gate_is_wired_and_documented -- --nocapture`,
  and
  `cargo test -p xtask --test repo_lint nvidia_baseline_workflow_is_retired -- --nocapture`
  pass. After moving the xtask adoption-stack feature-gate policy to
  `FilePatternCheck`,
  `cargo test -p xtask --test repo_lint xtask_adoption_stack_is_feature_gated -- --nocapture`
  and `cargo clippy -p xtask --all-features --test repo_lint -- -D warnings`
  pass. After moving the xtask nextest/machete/strict-clippy dispatch and help
  rows to `PatternCheck`,
  `cargo test -p xtask --test repo_lint xtask_exposes_nextest_machete_and_strict_clippy_gates -- --nocapture`
  and `cargo clippy -p xtask --all-features --test repo_lint -- -D warnings`
  pass. After moving DWT diff assertions to shared integration-test support and
  splitting the `jpeg_to_htj2k.rs` inline test module,
  `cargo test -p j2k-transcode --test dct53_2d --test dct97_2d --all-features -- --nocapture`,
  `cargo test -p xtask --test repo_lint copied_test_fixture_helpers_live_in_shared_support -- --nocapture`,
  `cargo test -p xtask --test repo_lint jpeg_to_htj2k_options_live_in_focused_module -- --nocapture`,
  `cargo test -p xtask --test repo_lint nvidia_codec_comparator_is_historical_only -- --nocapture`,
  `cargo test -p xtask --test repo_lint docs_and_workflows_policy -- --nocapture`,
  `cargo clippy -p j2k-transcode --all-features --lib --tests -- -D warnings`,
  `cargo clippy -p xtask --all-features --test repo_lint -- -D warnings`,
  and full `cargo test -p xtask --test repo_lint -- --nocapture` (131 passed)
  pass. After removing the public `j2k_jpeg::Decoder::decode_tile` row-sink
  helper,
  `cargo check -p j2k-jpeg --all-features --lib --tests --benches`,
  `cargo test -p j2k-jpeg --test batch --all-features -- --nocapture`,
  `cargo clippy -p j2k-jpeg --all-features --lib --tests --benches -- -D warnings`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`, and
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`
  plus full `cargo test -p xtask --test repo_lint -- --nocapture` (131 passed)
  pass. After removing the duplicate public
  `j2k_jpeg::Decoder::inspect_with_options` helper in favor of
  `JpegView::parse_with_options(...).info()`,
  `cargo test -p j2k-jpeg --test inspect --all-features -- --nocapture`,
  `cargo clippy -p j2k-jpeg --all-features --lib --tests --benches -- -D warnings`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`, and
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`
  plus full `cargo test -p xtask --test repo_lint -- --nocapture` (131 passed)
  pass. After removing duplicate public `j2k_jpeg::Decoder::new_with_options` in
  favor of `JpegView::parse_with_options(...)` plus `Decoder::from_view(...)`,
  `cargo check -p j2k-jpeg -p j2k-jpeg-metal --all-features --lib --tests`,
  `cargo test -p j2k-jpeg --test batch --test encode_baseline --all-features -- --nocapture`,
  `cargo clippy -p j2k-jpeg -p j2k-jpeg-metal --all-features --lib --tests -- -D warnings`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`, and
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`
  pass; no Metal GPU tests or benchmark executables were run for this public
  API slice.
  After making duplicate public
  `j2k_jpeg::Decoder::decode_request_with_scratch` private behind
  `Decoder::decode_request`,
  `cargo check -p j2k-jpeg --all-features --lib --tests`,
  `cargo test -p j2k-jpeg --test decode_into --all-features -- decode_owned --nocapture`,
  `cargo clippy -p j2k-jpeg --all-features --lib --tests -- -D warnings`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`,
  `cargo test -p xtask --test repo_lint jpeg_decoder_owned_outputs_use_decode_request -- --nocapture`,
  `cargo xtask stable-api --write`, and `cargo xtask stable-api` pass; no
  benchmark executables were run for this public API slice.
  After keeping `j2k_jpeg::Decoder` custom-alpha RGBA behavior public while
  making the duplicate caller-owned scratch helpers private,
  `cargo check -p j2k-jpeg --all-features --lib --tests`,
  `cargo test -p j2k-jpeg --test decode_into --all-features -- decode_into_rgba8_writes_alpha_byte --nocapture`,
  `cargo clippy -p j2k-jpeg --all-features --lib --tests -- -D warnings`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`,
  `cargo xtask stable-api --write`, and `cargo xtask stable-api` pass; no
  benchmark executables were run for this public API slice.
  After removing duplicate public `j2k::J2kDecoder::bytes` in favor of
  `J2kView::bytes()` and storing source bytes inside first-party CUDA/Metal
  adapter decoder wrappers,
  `cargo check -p j2k -p j2k-cuda -p j2k-metal --all-features --lib --tests`,
  `cargo clippy -p j2k -p j2k-cuda -p j2k-metal --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k --test inspect --all-features -- --nocapture`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`,
  `cargo xtask stable-api --write`, and `cargo xtask stable-api` pass; no
  benchmark executables were run for this public API slice.
  After removing duplicate public `j2k::J2kDecoder::support_info` in favor of
  `J2kView::support_info()` and `J2kDecoder::inspect_support(...)`,
  `cargo check -p j2k --all-features --lib --tests`,
  `cargo test -p j2k --test inspect --all-features -- --nocapture`,
  `cargo clippy -p j2k --all-features --lib --tests -- -D warnings`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`,
  `cargo xtask stable-api --write`, and `cargo xtask stable-api` pass; no
  benchmark executables were run for this public API slice.
  After narrowing CUDA runtime helpers that are only used inside
  `j2k-cuda-runtime`, `CudaContext::pinned_host_buffer`,
  `CudaContext::upload_i32_pinned`, `CudaContext::time_default_stream_us`, and
  `CudaPinnedHostBuffer` are crate-local/test-only and guarded out of the stable
  API snapshot. `cargo check -p j2k-cuda-runtime -p j2k-cuda -p j2k-jpeg-cuda -p j2k-transcode-cuda --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda-runtime -p j2k-cuda -p j2k-jpeg-cuda -p j2k-transcode-cuda --all-features --lib --tests -- -D warnings`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`,
  `cargo xtask stable-api --write`, and `cargo xtask stable-api` pass; no
  benchmark executables were run for this public API slice.
  After making `CudaHtj2kProfileReport::emit` and
  `CudaHtj2kEncodeProfileReport::emit` crate-private diagnostic helpers,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`,
  `cargo fmt --all --check`, `cargo check -p j2k-cuda --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda --all-features --lib --tests -- -D warnings`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`,
  `cargo test -p j2k-cuda --all-features --lib --tests -- --nocapture`, and
  `git diff --check` pass; no benchmark executable, Criterion command,
  perf-guard command, or bench-build command was run for this public API slice.
  After making the raw CUDA HTJ2K direct-plan structs, fields, and accessors
  crate-private and removing their public facade reexport,
  `cargo fmt --all --check`, `cargo check -p j2k-cuda --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda --all-features --lib --tests -- -D warnings`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`,
  `cargo test -p j2k-cuda --all-features --lib -- --nocapture`, and
  `git diff --check` pass; no benchmark executable, Criterion command,
  perf-guard command, or bench-build command was run for this public API slice.
  After hiding CUDA runtime HTJ2K decode/encode reusable table/resource handles
  and `*_with_resources*` entrypoints from the rendered stable API while
  preserving first-party adapter access, `cargo fmt --all --check`,
  `cargo check -p j2k-cuda-runtime -p j2k-cuda -p j2k-transcode-cuda --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda-runtime -p j2k-cuda -p j2k-transcode-cuda --all-features --lib --tests -- -D warnings`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`,
  `cargo test -p j2k-cuda-runtime -p j2k-cuda -p j2k-transcode-cuda --all-features --lib -- --nocapture`,
  and `git diff --check` pass; no benchmark executable, Criterion command,
  perf-guard command, or bench-build command was run for this public API slice.
  After hiding the `CpuBackedImageDecode` first-party adapter hook and its
  blanket host-output impl from the rendered 1.0 inventory while preserving
  source access for CUDA/Metal adapters, `cargo fmt --all --check`,
  `cargo check -p j2k-core -p j2k-cuda -p j2k-metal -p j2k-jpeg-cuda -p j2k-jpeg-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-core -p j2k-cuda -p j2k-metal -p j2k-jpeg-cuda -p j2k-jpeg-metal --all-features --lib --tests -- -D warnings`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`, and
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`
  pass; no benchmark executable, Criterion command, perf-guard command, or
  bench-build command was run for this public API slice.
  After hiding the lower-level
  `j2k_jpeg_metal::Codec::submit_tile_request_to_device` helper from the
  rendered inventory while preserving the public trait submit methods,
  `cargo fmt --all --check`,
  `cargo check -p j2k-jpeg-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg-metal --all-features --lib --tests -- -D warnings`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`, and
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`
  pass; no benchmark executable, Criterion command, perf-guard command, or
  bench-build command was run for this public API slice.
  After hiding `j2k-core`'s ready-submission/session helper APIs from the
  rendered inventory while preserving the visible `DeviceSubmission` wait
  contract and source access for first-party adapters,
  `cargo fmt --all --check`,
  `cargo check -p j2k-core -p j2k-cuda -p j2k-metal -p j2k-jpeg-cuda -p j2k-jpeg-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-core -p j2k-cuda -p j2k-metal -p j2k-jpeg-cuda -p j2k-jpeg-metal --all-features --lib --tests -- -D warnings`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`, and
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`
  pass; no benchmark executable, Criterion command, perf-guard command, or
  bench-build command was run for this public API slice.
  After hiding CUDA HTJ2K profiling report types, explicit `_and_profile`
  entrypoints, and the profile-collection accelerator constructor from the
  rendered inventory while preserving diagnostics/tests source access,
  `cargo fmt --all --check`,
  `cargo check -p j2k-cuda --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda --all-features --lib --tests -- -D warnings`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`, and
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`
  pass; no benchmark executable, Criterion command, perf-guard command, or
  bench-build command was run for this public API slice.
  After hiding synchronous CUDA/Metal submit impl renderings that leaked the
  hidden `ReadySubmission` helper through associated types,
  `cargo fmt --all --check`,
  `cargo check -p j2k-cuda -p j2k-metal -p j2k-jpeg-cuda --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda -p j2k-metal -p j2k-jpeg-cuda --all-features --lib --tests -- -D warnings`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`, and
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`
  pass; no benchmark executable, Criterion command, perf-guard command, or
  bench-build command was run for this public API slice.
  After hiding Metal decode route-report diagnostic types and
  `J2kDecoder::decode_request_to_device_with_report` from the rendered
  inventory while preserving the normal `decode_request_to_device` API and
  diagnostics source access, `cargo fmt --all --check`,
  `cargo check -p j2k-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-metal --all-features --lib --tests -- -D warnings`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`, and
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`
  pass; no benchmark executable, Criterion command, perf-guard command, or
  bench-build command was run for this public API slice.
  After hiding CUDA lossless encode timing-report outcome types and
  `*_with_report` CUDA buffer encode entrypoints from the rendered inventory
  while preserving normal encode/submit APIs and diagnostics source access,
  `cargo fmt --all --check`,
  `cargo check -p j2k-cuda --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda --all-features --lib --tests -- -D warnings`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`, and
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`
  pass; no benchmark executable, Criterion command, perf-guard command, or
  bench-build command was run for this public API slice.
  After hiding Metal lossless encode timing/report outcome and stats types plus
  `encode_lossless_batch_with_report` from the rendered inventory while
  preserving normal Metal submit/config/tile APIs and diagnostics source access,
  `cargo fmt --all --check`,
  `cargo check -p j2k-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-metal --all-features --lib --tests -- -D warnings`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`, and
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`
  pass; no benchmark executable, Criterion command, perf-guard command, or
  bench-build command was run for this public API slice.
  After hiding the JPEG Metal resident batch preflight report,
  `Codec::inspect_rgb8_decoder_batch_metal_output`, and the report-consuming
  `ensure_*_batch_report` allocation helpers from the rendered inventory while
  preserving normal batch decode and explicit allocation APIs,
  `cargo fmt --all --check`,
  `cargo check -p j2k-jpeg-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg-metal --all-features --lib --tests -- -D warnings`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`, and
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`
  pass; no benchmark executable, Criterion command, perf-guard command, or
  bench-build command was run for this public API slice.
  After hiding `j2k-core`'s first-party buffer/allocation helpers from the
  rendered inventory while preserving source-visible cross-crate adapter use
  and the public allocation cap constant, the stable API snapshot regenerated
  to 242,480 bytes / 677 `pub fn` entries. `cargo fmt --all --check`,
  `cargo check -p j2k-core -p j2k-jpeg -p j2k-cuda -p j2k-metal -p j2k-jpeg-cuda -p j2k-jpeg-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-core -p j2k-jpeg -p j2k-cuda -p j2k-metal -p j2k-jpeg-cuda -p j2k-jpeg-metal --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-core -p j2k-jpeg --all-features --lib --tests --no-fail-fast -- --nocapture`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`,
  and `git diff --check` pass; no benchmark executable, Criterion command,
  perf-guard command, or bench-build command was run for this public API slice.
  After hiding the `j2k-core` first-party batch/backend helpers
  `IndexedBatchResult`, `collect_indexed_batch_results`,
  `tile_batch_worker_count`, and `validate_cuda_surface_backend_request` from
  the rendered inventory while preserving source-visible first-party use, the
  stable API snapshot regenerated to 241,944 bytes / 675 `pub fn` entries.
  `cargo fmt --all --check`,
  `cargo check -p j2k-core -p j2k -p j2k-jpeg -p j2k-cuda -p j2k-jpeg-cuda -p j2k-compare --all-features --lib --bins --tests`,
  `cargo clippy -p j2k-core -p j2k -p j2k-jpeg -p j2k-cuda -p j2k-jpeg-cuda -p j2k-compare --all-features --lib --bins --tests -- -D warnings`,
  `cargo test -p j2k-core --all-features --lib --tests --no-fail-fast -- --nocapture`,
  `cargo test -p j2k-jpeg --all-features --test batch -- --nocapture`,
  `cargo test -p j2k --all-features --lib batch -- --nocapture`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`,
  and `git diff --check` pass; no benchmark executable, Criterion command,
  perf-guard command, or bench-build command was run for this public API slice.
  After hiding the source-visible `j2k` and `j2k-jpeg`
  `decode_tile_*_in_context` helpers from the rendered inventory while
  preserving ordinary one-shot and batch tile decode APIs, the stable API
  snapshot regenerated to 239,826 bytes / 667 `pub fn` entries.
  `cargo fmt --all --check`,
  `cargo check -p j2k -p j2k-jpeg -p j2k-jpeg-cuda -p j2k-compare --all-features --lib --bins --tests`,
  `cargo clippy -p j2k -p j2k-jpeg -p j2k-jpeg-cuda -p j2k-compare --all-features --lib --bins --tests -- -D warnings`,
  `cargo test -p j2k-jpeg --all-features --test batch -- --nocapture`,
  `cargo test -p j2k --all-features --test batch -- --nocapture`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`,
  and `git diff --check` pass; no benchmark executable, Criterion command,
  perf-guard command, or bench-build command was run for this public API slice.
  After hiding the `JpegBatchSession::worker_count` and
  `JpegBatchSession::retained_worker_slots` diagnostic methods from the
  rendered inventory while preserving source-visible test access, the stable
  API snapshot regenerated to 239,689 bytes / 665 `pub fn` entries.
  `cargo fmt --all --check`,
  `cargo check -p j2k-jpeg --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-jpeg --all-features --test batch -- --nocapture`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`,
  and `git diff --check` pass; no benchmark executable, Criterion command,
  perf-guard command, or bench-build command was run for this public API slice.
  After these public API slices, the broad non-benchmark workspace compile gate
  `cargo check --workspace --all-features --lib --bins --tests --examples`
  passes; this intentionally excludes bench targets.
  After the latest repo-lint helper migration,
  `cargo check -p xtask --all-features --test repo_lint`,
  `cargo clippy -p xtask --all-features --test repo_lint -- -D warnings`,
  `cargo test -p xtask --test repo_lint normalized_match_helpers -- --nocapture`,
  `cargo test -p xtask --test repo_lint docs_and_workflows_policy -- --nocapture`,
  and full `cargo test -p xtask --test repo_lint -- --nocapture` pass.
  After adding `PatternCheck` for extracted text sections,
  `cargo fmt --all --check`,
  `cargo check -p xtask --all-features --test repo_lint`,
  `cargo clippy -p xtask --all-features --test repo_lint -- -D warnings`,
  `cargo test -p xtask --test repo_lint file_pattern_runner_checks_files_and_rejects_empty_pattern_rows -- --nocapture`,
  `cargo test -p xtask --test repo_lint xtask_test_does_not_run_benchmarks_as_tests -- --nocapture`,
  `cargo test -p xtask --test repo_lint ci_coverage_job_is_a_required_gate -- --nocapture`,
  `cargo test -p xtask --test repo_lint xtask_exposes_nextest_machete_and_strict_clippy_gates -- --nocapture`,
  and full `cargo test -p xtask --test repo_lint -- --nocapture` pass.
  After adding `RustSourceScanCheck` for forbidden Rust-source scans,
  `cargo fmt --all --check`,
  `cargo check -p xtask --all-features --test repo_lint`,
  `cargo clippy -p xtask --all-features --test repo_lint -- -D warnings`,
  `cargo test -p xtask --test repo_lint adapter_crates_do_not_import_codec_private_modules -- --nocapture`,
  `cargo test -p xtask --test repo_lint production_j2k_cuda_code_does_not_reference_nvjpeg -- --nocapture`,
  and full `cargo test -p xtask --test repo_lint -- --nocapture` pass.
  The GPU hardware-gate silent-return scan was then moved from a bespoke loop
  onto `RustSourceScanCheck`; `cargo fmt --all --check`,
  `cargo test -p xtask --test repo_lint source_policy -- --nocapture`, and
  `cargo clippy -p xtask --all-features --test repo_lint -- -D warnings`
  pass.
  After expanding `PatternCheck`/`FilePatternCheck` across release, semver, and
  public-doc guards, `cargo fmt --all --check`,
  `cargo check -p xtask --all-features --test repo_lint`,
  `cargo clippy -p xtask --all-features --test repo_lint -- -D warnings`,
  `cargo test -p xtask --test repo_lint release_docs_use_manifest_versions_for_publish_order -- --nocapture`,
  `cargo test -p xtask --test repo_lint ci_workflow_runs_semver_checks_for_stable_library_crates -- --nocapture`,
  `cargo test -p xtask --test repo_lint supported_j2k_env_vars_are_documented -- --nocapture`,
  `cargo test -p xtask --test repo_lint published_crates_have_crates_io_landing_readmes -- --nocapture`,
  `cargo test -p xtask --test repo_lint publishable_crates_configure_docs_rs_metadata -- --nocapture`,
  `cargo test -p xtask --test repo_lint public_codec_and_transcode_examples_are_publicly_linked -- --nocapture`,
  and full `cargo test -p xtask --test repo_lint -- --nocapture` pass.
  After finishing the public-doc pattern-loop migration,
  `cargo fmt --all --check`,
  `cargo check -p xtask --all-features --test repo_lint`,
  `cargo clippy -p xtask --all-features --test repo_lint -- -D warnings`,
  `cargo test -p xtask --test repo_lint benchmark_docs_define_publication_gate_for_openjpeg_and_grok -- --nocapture`,
  `cargo test -p xtask --test repo_lint adoption_starter_corpus_fallback_is_pinned -- --nocapture`,
  `cargo test -p xtask --test repo_lint benchmark_publication_gate_rules_are_single_sourced -- --nocapture`,
  `cargo test -p xtask --test repo_lint metal_consistency_cleanup_keeps_names_status_buffers_and_marker_sizes_single_sourced -- --nocapture`,
  and full `cargo test -p xtask --test repo_lint -- --nocapture` pass.
  After the first `docs_and_workflows_policy.rs` `PatternCheck` cluster
  migration, `cargo fmt --all --check`,
  `cargo check -p xtask --all-features --test repo_lint`,
  `cargo clippy -p xtask --all-features --test repo_lint -- -D warnings`,
  `cargo test -p xtask --test repo_lint metal_resident_retry_uses_typed_error_classification -- --nocapture`,
  `cargo test -p xtask --test repo_lint gpu_adapter_error_classification_uses_shared_core_impl -- --nocapture`,
  `cargo test -p xtask --test repo_lint packet_progression_ordering_uses_shared_packetization_contract -- --nocapture`,
  `cargo test -p xtask --test repo_lint idwt_required_region_propagation_uses_shared_native_helper -- --nocapture`,
  `cargo test -p xtask --test repo_lint metal_direct_required_region_retain_uses_shared_job_helper -- --nocapture`,
  `cargo test -p xtask --test repo_lint metal_direct_sub_band_group_scan_uses_shared_helper -- --nocapture`,
  `cargo test -p xtask --test repo_lint metal_hybrid_region_scaled_cache_uses_shared_scope -- --nocapture`,
  and full `cargo test -p xtask --test repo_lint -- --nocapture` pass.
  After the second `docs_and_workflows_policy.rs` `PatternCheck` cluster
  migration, `cargo fmt --all --check`,
  `cargo check -p xtask --all-features --test repo_lint`,
  `cargo clippy -p xtask --all-features --test repo_lint -- -D warnings`,
  `cargo test -p xtask --test repo_lint wavelet_and_idct_constants_use_codec_math_sources -- --nocapture`,
  `cargo test -p xtask --test repo_lint jpeg_gpu_encode_host_orchestration_uses_shared_adapter_helper -- --nocapture`,
  `cargo test -p xtask --test repo_lint metal_backend_session_lifecycle_lives_in_support_crate -- --nocapture`,
  `cargo test -p xtask --test repo_lint fast444_region_scaled_batches_use_shared_region_scaled_metal_path -- --nocapture`,
  `cargo test -p xtask --test repo_lint fast444_full_batches_use_shared_fastsubsampled_metal_path -- --nocapture`,
  `cargo test -p xtask --test repo_lint jpeg_fast420_profiled_decode_uses_shared_scan_loop -- --nocapture`,
  `cargo test -p xtask --test repo_lint cuda_htj2k_compact_jobs_use_shared_planner -- --nocapture`,
  and full `cargo test -p xtask --test repo_lint -- --nocapture` pass.
  After the third `docs_and_workflows_policy.rs` `PatternCheck` cluster
  migration, `cargo fmt --all --check`,
  `cargo check -p xtask --all-features --test repo_lint`,
  `cargo clippy -p xtask --all-features --test repo_lint -- -D warnings`,
  `cargo test -p xtask --test repo_lint native_classic_and_ht_parallel_copyback_share_one_helper -- --nocapture`,
  `cargo test -p xtask --test repo_lint cuda_oxide_simt_helpers_use_shared_prelude -- --nocapture`,
  `cargo test -p xtask --test repo_lint copied_test_fixture_helpers_live_in_shared_support -- --nocapture`,
  `cargo test -p xtask --test repo_lint metal_compute_runtime_registry_is_split_from_compute_god_file -- --nocapture`,
  `cargo test -p xtask --test repo_lint native_encode_options_and_tile_parts_live_in_focused_modules -- --nocapture`,
  and full `cargo test -p xtask --test repo_lint -- --nocapture` pass.
  After the fourth `docs_and_workflows_policy.rs` `PatternCheck` cluster
  migration, `cargo fmt --all --check`,
  `cargo check -p xtask --all-features --test repo_lint`,
  `cargo clippy -p xtask --all-features --test repo_lint -- -D warnings`,
  `cargo test -p xtask --test repo_lint native_encode_options_and_tile_parts_live_in_focused_modules -- --nocapture`,
  `cargo test -p xtask --test repo_lint jpeg_decoder_view_and_output_format_live_in_focused_modules -- --nocapture`,
  and full `cargo test -p xtask --test repo_lint -- --nocapture` pass.
  After the fifth `docs_and_workflows_policy.rs` `PatternCheck` cluster
  migration, `cargo fmt --all --check`,
  `cargo check -p xtask --all-features --test repo_lint`,
  `cargo clippy -p xtask --all-features --test repo_lint -- -D warnings`,
  `cargo test -p xtask --test repo_lint transcode_gpu_auto_threshold_policy_is_documented -- --nocapture`,
  `cargo test -p xtask --test repo_lint transcode_stage_counters_are_shared_between_gpu_adapters -- --nocapture`,
  `cargo test -p xtask --test repo_lint metal_direct_plan_types_live_in_focused_module -- --nocapture`,
  and full `cargo test -p xtask --test repo_lint -- --nocapture` pass.
  After the first `FilePatternCheck` migration, `cargo fmt --all --check`,
  `cargo check -p xtask --all-features --test repo_lint`,
  `cargo clippy -p xtask --all-features --test repo_lint -- -D warnings`,
  `cargo test -p xtask --test repo_lint file_pattern_runner -- --nocapture`,
  and `cargo test -p xtask --test repo_lint docs_and_workflows_policy -- --nocapture`
  pass; full `cargo test -p xtask --test repo_lint -- --nocapture` passes 130
  tests.
  After expanding `FilePatternCheck` into public-doc guards,
  `cargo fmt --all --check`,
  `cargo check -p xtask --all-features --test repo_lint`,
  `cargo clippy -p xtask --all-features --test repo_lint -- -D warnings`,
  `cargo test -p xtask --test repo_lint public_docs_policy -- --nocapture`,
  and full `cargo test -p xtask --test repo_lint -- --nocapture` pass.
  After expanding `FilePatternCheck` into source-policy guards,
  `cargo fmt --all --check`,
  `cargo check -p xtask --all-features --test repo_lint`,
  `cargo clippy -p xtask --all-features --test repo_lint -- -D warnings`,
  `cargo test -p xtask --test repo_lint source_policy -- --nocapture`,
  and full `cargo test -p xtask --test repo_lint -- --nocapture` pass.
  After expanding `FilePatternCheck` into release-policy guards,
  `cargo fmt --all --check`,
  `cargo check -p xtask --all-features --test repo_lint`,
  `cargo clippy -p xtask --all-features --test repo_lint -- -D warnings`,
  `cargo test -p xtask --test repo_lint release_policy -- --nocapture`,
  and full `cargo test -p xtask --test repo_lint -- --nocapture` pass.
  After moving release publishable-package, publish-workflow, and publish-script
  coverage assertions onto shared pattern helpers, `cargo fmt --all --check`,
  `cargo clippy -p xtask --all-features --test repo_lint -- -D warnings`, and
  `cargo test -p xtask --test repo_lint release_policy -- --nocapture` pass.
  After expanding `FilePatternCheck` into architecture-policy guards,
  `cargo fmt --all --check`,
  `cargo check -p xtask --all-features --test repo_lint`,
  `cargo clippy -p xtask --all-features --test repo_lint -- -D warnings`,
  `cargo test -p xtask --test repo_lint architecture_policy -- --nocapture`,
  and full `cargo test -p xtask --test repo_lint -- --nocapture` pass.
  After expanding `FilePatternCheck` into corpus-policy guards,
  `cargo fmt --all --check`,
  `cargo check -p xtask --all-features --test repo_lint`,
  `cargo clippy -p xtask --all-features --test repo_lint -- -D warnings`,
  `cargo test -p xtask --test repo_lint corpus_policy -- --nocapture`,
  and full `cargo test -p xtask --test repo_lint -- --nocapture` pass.
  After expanding `FilePatternCheck` into the decode-capability correctness
  guard, `cargo fmt --all --check`,
  `cargo check -p xtask --all-features --test repo_lint`,
  `cargo clippy -p xtask --all-features --test repo_lint -- -D warnings`,
  `cargo test -p xtask --test repo_lint decode_capability_correctness_regressions_are_guarded -- --nocapture`,
  and full `cargo test -p xtask --test repo_lint -- --nocapture` pass.
  After expanding `FilePatternCheck` into the GPU coverage-exclusion and
  substitute-evidence guard, `cargo fmt --all --check`,
  `cargo check -p xtask --all-features --test repo_lint`,
  `cargo clippy -p xtask --all-features --test repo_lint -- -D warnings`,
  `cargo test -p xtask --test repo_lint coverage_excludes_hardware_only_gpu_adapter_crates -- --nocapture`,
  and full `cargo test -p xtask --test repo_lint -- --nocapture` pass.
  After expanding `FilePatternCheck` into the MQ table/FNV/JPEG fast-packet
  ownership rows and the literal component-plane accessor checks,
  fmt, xtask repo-lint check, xtask repo-lint clippy with `-D warnings`, and
  full `cargo test -p xtask --test repo_lint -- --nocapture` pass. No benchmark
  or performance commands were run for this tooling slice.
  After expanding `FilePatternCheck` into the CPU-backed GPU decoder facade
  ownership row and j2k-metal codec-math dependency row, fmt, xtask repo-lint
  clippy with `-D warnings`, and the docs/workflows repo-lint subset pass.
  After extracting `architecture_policy`, `corpus_policy`, `dependency_policy`,
  `docs_and_workflows_policy`, `public_docs_policy`, `release_policy`, and
  `source_policy` into `repo_lint_support`,
  `cargo fmt --all --check`,
  `cargo check -p xtask --all-features --test repo_lint`,
  `cargo clippy -p xtask --all-features --test repo_lint -- -D warnings`,
  `cargo test -p xtask --test repo_lint docs_and_workflows_policy -- --nocapture`,
  `cargo test -p xtask --test repo_lint source_policy -- --nocapture`,
  `cargo test -p xtask --test repo_lint architecture_policy -- --nocapture`,
  `cargo test -p xtask --test repo_lint dependency_policy -- --nocapture`,
  `cargo test -p xtask --test repo_lint release_policy -- --nocapture`, and
  full `cargo test -p xtask --test repo_lint -- --nocapture` pass. The shared
  `is_repo_lint_test_source` exclusion was added for moved test-source modules
  and checked with `cargo test -p xtask --test repo_lint public_docs_policy -- --nocapture`.
  `cargo run -p xtask -- unsafe-audit`, `cargo xtask
  stable-api`, `cargo xtask panic-surface`, `cargo xtask semver`, `cargo deny
  check`, and `cargo machete` passed in the current remediation sweep.
  Stable API evidence was regenerated after the `BackendErrorKind` constructor
  shrink, the adaptive-route policy model removal, the shared JPEG GPU encode
  adapter-driver addition, and the shared transcode dispatch-mode contract.
  `cargo deny check` also passed in the current remediation sweep; duplicate
  `weezl` is absent and the `metal` -> `block` path resolves through the local
  patched `third_party/block-0.1.6-patched` crate.
  Focused package tests pass for `cargo test -p j2k --all-features --lib
  --tests`, `cargo test -p j2k-jpeg --all-features --lib --tests`,
  `cargo test -p j2k-native --all-features --lib --tests`, and
  `cargo test -p j2k-compare --all-features --lib --tests`. The latest
  focused `j2k` checks also passed
  `cargo check -p j2k --all-features --lib --bins --tests` and
  `cargo clippy -p j2k --all-features --lib --bins --tests -- -D warnings`.
  The latest focused `j2k-compare` checks also passed
  `cargo check -p j2k-compare --all-features --lib --bins --tests`,
  `cargo clippy -p j2k-compare --all-features --lib --bins --tests -- -D warnings`,
  and `cargo test -p xtask --test repo_lint compare_bins_use_library_common_helpers -- --nocapture`.
  The latest focused `j2k-jpeg` checks after the lossless routing, sampled
  output-renderer cleanup, shared lossless color per-pixel decoder, and sampled
  MCU decode helper
  passed `cargo check -p j2k-jpeg --all-features --lib --bins --tests`,
  `cargo clippy -p j2k-jpeg --all-features --lib --bins --tests -- -D warnings`,
  `cargo test -p j2k-jpeg --all-features --lib --tests`,
  `cargo test -p j2k-jpeg --all-features --test decode_into lossless -- --nocapture`, and
  `cargo test -p xtask --test repo_lint jpeg_decoder_view_and_output_format_live_in_focused_modules -- --nocapture`.
  After the `LosslessRestartTracker` split, focused checks passed
  `cargo fmt --all --check`,
  `cargo check -p j2k-metal --all-features --lib --tests`,
  `cargo check -p j2k-jpeg --all-features --lib --bins --tests`,
  `cargo clippy -p j2k-jpeg --all-features --lib --bins --tests -- -D warnings`,
  `cargo test -p j2k-jpeg --all-features --test decode_into lossless -- --nocapture`,
  `cargo check -p xtask --all-features --test repo_lint`,
  `cargo clippy -p xtask --all-features --test repo_lint -- -D warnings`, and
  `cargo test -p xtask --test repo_lint jpeg_decoder_view_and_output_format_live_in_focused_modules -- --nocapture`;
  after lowering the decoder ratchet to `<4,005`, full
  `cargo test -p xtask --test repo_lint -- --nocapture` passed 128 lints.
  After the `Extended12RestartTracker` split, focused checks passed
  `cargo check -p j2k-jpeg --all-features --lib --bins --tests`,
  `cargo clippy -p j2k-jpeg --all-features --lib --bins --tests -- -D warnings`,
  `cargo test -p j2k-jpeg --all-features --test decode_into extended12 -- --nocapture`,
  `cargo check -p xtask --all-features --test repo_lint`,
  `cargo clippy -p xtask --all-features --test repo_lint -- -D warnings`, and
  `cargo test -p xtask --test repo_lint jpeg_decoder_view_and_output_format_live_in_focused_modules -- --nocapture`;
  after lowering the decoder ratchet to `<3,985`, full
  `cargo test -p xtask --test repo_lint -- --nocapture` passed 128 lints.
  After the row-sink adapter split and bench-profile writer unification,
  focused checks passed
  `cargo check -p j2k-jpeg --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-jpeg --all-features --lib --tests`, and
  `cargo test -p xtask --test repo_lint jpeg_decoder_view_and_output_format_live_in_focused_modules -- --nocapture`.
  After removing the component-writer forwarding adapter, focused checks passed
  `cargo fmt --all --check`,
  `cargo clippy -p j2k-jpeg --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-jpeg --all-features --lib --tests`,
  `cargo test -p xtask --test repo_lint jpeg_decoder_view_and_output_format_live_in_focused_modules -- --nocapture`,
  full `cargo test -p xtask --test repo_lint -- --nocapture`, and
  `cargo clippy -p xtask --all-features --bins --tests -- -D warnings`.
  The progressive-scan parser request-object cleanup was checked with
  `cargo check -p j2k-jpeg --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-jpeg --all-features --lib parse::header -- --nocapture`,
  and `cargo test -p xtask --test repo_lint too_many_arguments_suppressions_stay_below_current_ratchet -- --nocapture`.
  The public facade sample-geometry request-object cleanup was checked with
  `cargo check -p j2k --all-features --lib --tests`,
  `cargo clippy -p j2k --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k --all-features --lib encode -- --nocapture`, and
  `cargo test -p xtask --test repo_lint too_many_arguments_suppressions_stay_below_current_ratchet -- --nocapture`.
  The native prequantized subband test request-object cleanup was checked with
  `cargo fmt --all --check`, `cargo check -p j2k-native --all-features --lib --tests`,
  `cargo clippy -p j2k-native --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-native --all-features --lib prequantized_htj2k97 -- --nocapture`,
  and `cargo test -p xtask --test repo_lint too_many_arguments_suppressions_stay_below_current_ratchet -- --nocapture`.
  The private CUDA/JPEG-CUDA region-scaled surface request-object cleanup was
  checked with `cargo fmt --all --check`,
  `cargo check -p j2k-cuda -p j2k-jpeg-cuda --all-features --lib --tests`,
  and `cargo clippy -p j2k-cuda -p j2k-jpeg-cuda --all-features --lib --tests -- -D warnings`.
  The CUDA encode test-helper request-object cleanup was checked with
  `cargo fmt --all --check`, `cargo check -p j2k-cuda --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda --all-features --lib --tests -- -D warnings`,
  and `cargo test -p j2k-cuda --all-features --lib cuda_encode_stage_accelerator_preserves_cpu_codestream_validity -- --nocapture`.
  The CUDA resident color decode completion request-object cleanup was checked
  with `cargo check -p j2k-cuda --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda --all-features --lib --tests -- -D warnings`,
  and `cargo test -p j2k-cuda --all-features --lib cuda_batch_decode_two_color_images_matches_single_when_runtime_required -- --nocapture`
  (hardware gate skipped on this host).
  The stale transcode CUDA subband-plan suppression removal was checked with
  `cargo check -p j2k-transcode-cuda --all-features --lib --tests` and
  `cargo clippy -p j2k-transcode-cuda --all-features --lib --tests -- -D warnings`.
  The stale JPEG four-component color-row suppression removal was checked with
  `cargo check -p j2k-jpeg --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg --all-features --lib --tests -- -D warnings`,
  and `cargo test -p j2k-jpeg --all-features --test batch cmyk -- --nocapture`.
  The stale CUDA oxide JPEG RGB444 MCU suppression removal was checked with
  `cargo fmt --all --check`,
  `cargo check -p j2k-cuda-runtime --all-features --lib --tests`, and
  `cargo clippy -p j2k-cuda-runtime --all-features --lib --tests -- -D warnings`;
  clippy confirmed nearby CUDA runtime method suppressions still count `self`
  and remain required until those methods gain request objects.
  The lossless RGB/YCbCr sampling-dispatch collapse was checked with
  `cargo clippy -p j2k-jpeg --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-jpeg --all-features --lib --tests`,
  `cargo test -p xtask --test repo_lint jpeg_decoder_view_and_output_format_live_in_focused_modules -- --nocapture`,
  full `cargo test -p xtask --test repo_lint -- --nocapture`, and
  `cargo clippy -p xtask --all-features --bins --tests -- -D warnings`.
  The lossless RGBA region fallback selector cleanup was checked with
  `cargo clippy -p j2k-jpeg --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-jpeg --all-features --test decode_into lossless -- --nocapture`,
  `cargo test -p xtask --test repo_lint jpeg_decoder_view_and_output_format_live_in_focused_modules -- --nocapture`,
  full `cargo test -p xtask --test repo_lint -- --nocapture`, and
  `cargo clippy -p xtask --all-features --bins --tests -- -D warnings`.
  The lossless RGBA region scratch-copy extraction was checked with
  `cargo fmt --all --check`,
  `cargo check -p j2k-jpeg --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg --all-features --lib --tests -- -D warnings`, and
  `cargo test -p j2k-jpeg --all-features --test decode_into decode_region_scaled_into_rgba -- --nocapture`.
  The shared lossless color validation helper was checked with
  `cargo check -p j2k-jpeg --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-jpeg --all-features --test decode_into lossless -- --nocapture`,
  `cargo test -p xtask --test repo_lint jpeg_decoder_view_and_output_format_live_in_focused_modules -- --nocapture`,
  full `cargo test -p xtask --test repo_lint -- --nocapture`, and
  `cargo clippy -p xtask --all-features --bins --tests -- -D warnings`.
  The Metal JPEG viewport-cache row-target cleanup was checked with
  `cargo clippy -p j2k-jpeg-metal --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-jpeg-metal --all-features --lib --tests`,
  `cargo test -p xtask --test repo_lint jpeg_metal_viewport_plane_rows_use_shared_target -- --nocapture`,
  full `cargo test -p xtask --test repo_lint -- --nocapture`, and
  `cargo clippy -p xtask --all-features --bins --tests -- -D warnings`.
  The follow-up viewport-cache raw-contents removal moved row writes and plane
  fills to checked byte-write helpers and removed
  `crates/j2k-jpeg-metal/src/compute/viewport_cache.rs` from the raw-contents
  allow-list; verification passed `cargo fmt --all --check`,
  `cargo clippy -p j2k-jpeg-metal --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-jpeg-metal --all-features --lib --tests`,
  `cargo test -p xtask --test repo_lint -- --nocapture`,
  `cargo check -p xtask --all-features --test repo_lint`,
  `cargo clippy -p xtask --all-features --test repo_lint -- -D warnings`, and
  `cargo check --workspace --all-features --lib --bins --examples --tests`.
  The shared JPEG-Metal decode `idct_block` helper was checked with
  `cargo check -p j2k-jpeg-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg-metal --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-jpeg-metal --test shader_integrity -- --nocapture`,
  `cargo check -p xtask --all-features --test repo_lint`,
  `cargo clippy -p xtask --all-features --test repo_lint -- -D warnings`, and
  `cargo test -p xtask --test repo_lint jpeg_metal_shader_is_split_by_subsystem -- --nocapture`.
  The shared JPEG-Metal global batch entropy setup was checked with
  `cargo check -p j2k-jpeg-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg-metal --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-jpeg-metal --test shader_integrity -- --nocapture`,
  `cargo check -p xtask --all-features --test repo_lint`,
  `cargo clippy -p xtask --all-features --test repo_lint -- -D warnings`, and
  `cargo test -p xtask --test repo_lint jpeg_metal_shader_is_split_by_subsystem -- --nocapture`.
  The shared JPEG-Metal full-image decode/idct/deposit helper was checked with
  `cargo fmt --all --check`,
  `cargo test -p xtask --test repo_lint jpeg_metal_shader_is_split_by_subsystem -- --nocapture`,
  `cargo check -p j2k-jpeg-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg-metal --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-jpeg-metal --test shader_integrity -- --nocapture`,
  `cargo test -p j2k-jpeg-metal --all-features --lib fast420 -- --nocapture`,
  `cargo test -p j2k-jpeg-metal --all-features --lib fast422 -- --nocapture`,
  and `cargo test -p j2k-jpeg-metal --all-features --lib fast444 -- --nocapture`,
  followed by `cargo fmt --all --check`,
  `cargo test -p xtask --test repo_lint -- --nocapture`,
  `cargo check -p j2k-jpeg-metal -p xtask --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg-metal -p xtask --all-features --lib --tests -- -D warnings`,
  `git diff --check`, and an explicit trailing-whitespace scan over the touched
  shader/doc/lint files.
  The fast444 region/scaled decode/deposit-or-skip helper extraction was
  checked with
  `cargo test -p xtask --test repo_lint jpeg_metal_shader_is_split_by_subsystem -- --nocapture`,
  `cargo test -p j2k-jpeg-metal --test shader_integrity -- --nocapture`, and
  `cargo test -p j2k-jpeg-metal --all-features --lib fast444 -- --nocapture`,
  followed by `cargo fmt --all --check`,
  `cargo test -p xtask --test repo_lint -- --nocapture`,
  `cargo check -p j2k-jpeg-metal --all-features --lib --tests`,
  `git diff --check`, and an explicit trailing-whitespace scan over the touched
  untracked shader/doc files.
  The fast422 region/scaled decode/deposit-or-skip helper extraction was
  checked with
  `cargo test -p xtask --test repo_lint jpeg_metal_shader_is_split_by_subsystem -- --nocapture`,
  `cargo test -p j2k-jpeg-metal --test shader_integrity -- --nocapture`, and
  `cargo test -p j2k-jpeg-metal --all-features --lib fast422 -- --nocapture`,
  followed by `cargo fmt --all --check`,
  `cargo test -p xtask --test repo_lint -- --nocapture`,
  `cargo check -p j2k-jpeg-metal -p xtask --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg-metal -p xtask --all-features --lib --tests -- -D warnings`,
  `git diff --check`, and an explicit trailing-whitespace scan over the touched
  shader/doc/lint files.
  The texture batch checkpoint setup consolidation was checked with
  `cargo test -p xtask --test repo_lint jpeg_metal_shader_is_split_by_subsystem -- --nocapture`,
  `cargo test -p j2k-jpeg-metal --test shader_integrity -- --nocapture`,
  `cargo test -p j2k-jpeg-metal --all-features --lib fast420 -- --nocapture`,
  `cargo test -p j2k-jpeg-metal --all-features --lib fast422 -- --nocapture`,
  and `cargo test -p j2k-jpeg-metal --all-features --lib fast444 -- --nocapture`.
  The fast422/fast420 non-region scaled decode/deposit helper extraction was
  checked with
  `cargo test -p xtask --test repo_lint jpeg_metal_shader_is_split_by_subsystem -- --nocapture`,
  `cargo test -p j2k-jpeg-metal --test shader_integrity -- --nocapture`,
  `cargo test -p j2k-jpeg-metal --all-features --lib fast422 -- --nocapture`,
  and `cargo test -p j2k-jpeg-metal --all-features --lib fast420 -- --nocapture`,
  followed by `cargo fmt --all --check`,
  `cargo test -p xtask --test repo_lint -- --nocapture`,
  `cargo check -p j2k-jpeg-metal -p xtask --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg-metal -p xtask --all-features --lib --tests -- -D warnings`,
  `git diff --check`, and an explicit trailing-whitespace scan over the touched
  shader/doc/lint files.
  The fast420 region/scaled decode/deposit-or-skip helper extraction was
  checked with
  `cargo test -p xtask --test repo_lint jpeg_metal_shader_is_split_by_subsystem -- --nocapture`,
  `cargo test -p j2k-jpeg-metal --test shader_integrity -- --nocapture`, and
  `cargo test -p j2k-jpeg-metal --all-features --lib fast420 -- --nocapture`,
  followed by `cargo fmt --all --check`,
  `cargo test -p xtask --test repo_lint -- --nocapture`,
  `cargo check -p j2k-jpeg-metal -p xtask --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg-metal -p xtask --all-features --lib --tests -- -D warnings`,
  `git diff --check`, and an explicit trailing-whitespace scan over the touched
  shader/doc/lint files.
  The shared JPEG-Metal region/scaled decode/deposit helper chunk was checked
  with `cargo check -p j2k-jpeg-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg-metal --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-jpeg-metal --test shader_integrity --all-features -- --nocapture`,
  and
  `cargo test -p xtask --test repo_lint jpeg_metal_shader_is_split_by_subsystem -- --nocapture`.
  It moved the shared decode/deposit helpers into
  `shaders_decode_helpers.metal`, reduced the JPEG Metal shader total to 6,051
  lines, and tightened `shaders_decode_fast422_regions.metal` to <955 lines and
  `shaders_decode_fast444.metal` to <440 lines.
  The texture repair metadata clear was then shared through
  `jpeg_decode_clear_meta_quad`, lowering the JPEG Metal shader total to 6,034
  lines and tightening `shaders_decode_fast420.metal` to <1,005 lines and
  `shaders_decode_fast422_regions.metal` to <950 lines. This source-level
  shader slice was checked with
  `cargo test -p j2k-jpeg-metal --test shader_integrity -- --nocapture`;
  the Metal compiler pass remains hardware/toolchain-dependent and was not
  rerun in this slice.
  The YCbCr texture-write scaffolding was then shared through
  `jpeg_write_ycbcr_rgba`, leaving direct `rgba_float_ycbcr(` calls only in the
  helper for the split decode shader family. The decode shader total was then
  2,494 lines across `shaders_decode_fast420.metal` (986),
  `shaders_decode_fast422_regions.metal` (933),
  `shaders_decode_fast444.metal` (436), and
  `shaders_decode_helpers.metal` (139), with repo-lint ratchets tightened to
  <990, <940, <440, and <145 respectively. This shader slice was checked with
  `cargo fmt --all --check`,
  `cargo test -p xtask --test repo_lint jpeg_metal_shader_is_split_by_subsystem -- --nocapture`,
  `cargo test -p j2k-jpeg-metal --test shader_integrity -- --nocapture`, and
  `git diff --check`.
  The fast422 texture boundary interpolation arithmetic was then shared through
  `h2v1_boundary_left_from_samples` and
  `h2v1_boundary_right_from_samples` in `shaders_decode_helpers.metal`,
  replacing direct h2v1 weighted sample formulas in
  `shaders_decode_fast422_regions.metal` and adding a repo-lint guard against
  those formulas returning. The decode shader total is now 2,498 lines across
  `shaders_decode_fast420.metal` (986),
  `shaders_decode_fast422_regions.metal` (933),
  `shaders_decode_fast444.metal` (436), and
  `shaders_decode_helpers.metal` (143), still under the active <990, <940,
  <440, and <145 ratchets. `shaders_encode.metal` remains under its <1,815
  ratchet at 1,810 lines. This source-level shader slice was checked with
  `cargo fmt --all --check`,
  `cargo check -p j2k-jpeg-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg-metal --all-features --lib --tests -- -D warnings`,
  `cargo test -p xtask --test repo_lint jpeg_metal_shader_is_split_by_subsystem -- --nocapture`,
  `cargo test -p j2k-jpeg-metal --all-features --test shader_integrity -- --nocapture`, and
  `cargo test -p j2k-jpeg-metal --all-features --lib shader_source_keeps_entropy_fast_paths -- --nocapture`.
  Repeated clamped copy-span math in the fast420/fast422 texture-boundary
  paths was then routed through `jpeg_clamped_extent(...)` in
  `shaders_decode_helpers.metal`, with repo-lint guards preventing the old
  inline `min(span, limit - min(origin, limit))` expressions from returning in
  those split shader chunks. The decode shader total is now 2,502 lines across
  `shaders_decode_fast420.metal` (986),
  `shaders_decode_fast422_regions.metal` (933),
  `shaders_decode_fast444.metal` (436), and
  `shaders_decode_helpers.metal` (147), still under the active <990, <940,
  <440, and tightened <150 helper ratchets. This source-level shader slice was
  checked with
  `cargo test -p xtask --test repo_lint jpeg_metal_shader_is_split_by_subsystem -- --nocapture`,
  `cargo test -p j2k-jpeg-metal --test shader_integrity --all-features -- --nocapture`, and
  `cargo test -p j2k-jpeg-metal --all-features shader_source -- --nocapture`.
  The current non-benchmark ratchet-only follow-up tightened the same shader
  line guards to `shaders_encode.metal` <1,812,
  `shaders_decode_helpers.metal` <149, `shaders_decode_fast420.metal` <988,
  `shaders_decode_fast422_regions.metal` <935, and
  `shaders_decode_fast444.metal` <438, matching the current source sizes
  1,810 / 147 / 986 / 933 / 436. It was checked with
  `cargo test -p xtask --test repo_lint jpeg_metal_shader_is_split_by_subsystem -- --nocapture`.
  The fast444 non-region scaled decode/deposit path then joined the shared
  `jpeg_decode_deposit_scaled_block` helper already used by the fast422/fast420
  scaled kernels, reducing `shaders_decode_fast444.metal` from 436 to 399
  lines and tightening its ratchet to <405. This source-level shader slice was
  checked with `cargo fmt --all --check`,
  `cargo test -p xtask --test repo_lint shader_policy -- --nocapture`
  (2 passed),
  `cargo clippy -p xtask --all-features --test repo_lint -- -D warnings`,
  and
  `cargo test -p j2k-jpeg-metal --test shader_integrity --all-features -- --nocapture`
  (1 passed).
  The fast420 h2v2 texture-boundary weighted chroma sums then joined the shared
  `h2v2_weighted_sample_sum` helper used by h2v2 row sampling and corner
  interpolation, reducing `shaders_decode_fast420.metal` from 986 to 978 lines
  and tightening its ratchet to <982. This source-level shader slice was
  checked with `cargo fmt --all --check`,
  `cargo test -p xtask --test repo_lint shader_policy -- --nocapture`
  (2 passed),
  `cargo clippy -p xtask --all-features --test repo_lint -- -D warnings`,
  `cargo test -p j2k-jpeg-metal --test shader_integrity --all-features -- --nocapture`
  (1 passed), and `git diff --check`.
  The paired fast420 h2v2 horizontal boundary texture writes then joined the
  shared `jpeg_write_h2v2_boundary_pair` helper, leaving direct
  `h2v2_boundary_{left,right}_from_sums` calls out of
  `shaders_decode_fast420.metal`. This reduced `shaders_decode_fast420.metal`
  from 978 to 970 lines, moved the pair-write body into the 167-line helper
  chunk, and tightened the then-active ratchets to
  `shaders_decode_helpers.metal` <170 and `shaders_decode_fast420.metal` <974.
  This source-level shader slice was checked with `cargo fmt --all --check`,
  `cargo test -p xtask --test repo_lint shader_policy -- --nocapture`
  (2 passed),
  `cargo clippy -p xtask --all-features --test repo_lint -- -D warnings`,
  `cargo test -p j2k-jpeg-metal --test shader_integrity --all-features -- --nocapture`
  (1 passed), and `git diff --check`.
  The repeated fast420 h2v2 horizontal boundary top/bottom repair-row skip
  rule then moved into shared `jpeg_skip_h2v2_boundary_repair_row`. This
  reduced `shaders_decode_fast420.metal` from 970 to 964 lines, grew the helper
  chunk from 167 to 171 lines, and tightened the active ratchets to
  `shaders_decode_helpers.metal` <174 and `shaders_decode_fast420.metal` <966.
  This source-level shader slice was checked with `cargo fmt --all --check`,
  `cargo test -p xtask --test repo_lint shader_policy -- --nocapture`
  (2 passed),
  `cargo clippy -p xtask --all-features --test repo_lint -- -D warnings`,
  and
  `cargo test -p j2k-jpeg-metal --test shader_integrity --all-features -- --nocapture`
  (1 passed).
  A direct `xcrun -sdk macosx metal -c` syntax pass was attempted against the
  concatenated shader source, but this machine reports the Xcode Metal
  Toolchain component is missing and suggests
  `xcodebuild -downloadComponent MetalToolchain`; do not record shader compiler
  evidence until that component is installed or CI supplies it.
  The shared JPEG GPU encode host driver was checked with
  `cargo check -p j2k-jpeg -p j2k-jpeg-cuda -p j2k-jpeg-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg -p j2k-jpeg-cuda -p j2k-jpeg-metal --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-jpeg -p j2k-jpeg-cuda -p j2k-jpeg-metal --all-features --lib --tests`,
  `cargo test -p xtask --test repo_lint jpeg_gpu_encode_host_orchestration_uses_shared_adapter_helper -- --nocapture`,
  full `cargo test -p xtask --test repo_lint -- --nocapture`,
  `cargo clippy -p xtask --all-features --bins --tests -- -D warnings`, and
  `cargo xtask stable-api`.
  The shared transcode dispatch/recover policy was checked with
  `cargo check -p j2k-transcode -p j2k-transcode-cuda -p j2k-transcode-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-transcode -p j2k-transcode-cuda -p j2k-transcode-metal --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-transcode -p j2k-transcode-cuda -p j2k-transcode-metal --all-features --lib --tests`,
  `cargo test -p xtask --test repo_lint transcode_stage_counters_are_shared_between_gpu_adapters -- --nocapture`,
  `cargo clippy -p xtask --all-features --bins --tests -- -D warnings`, and
  `cargo xtask stable-api`.
  The focused `j2k-jpeg-metal` package tests passed after the decoder and
  codec-batch splits. After the follow-on request-type split, focused checks
  passed
  `cargo check -p j2k-jpeg-metal --all-features --lib --bins --tests`,
  `cargo clippy -p j2k-jpeg-metal --all-features --lib --bins --tests -- -D warnings`,
  `cargo test -p xtask --test repo_lint jpeg_metal_single_decode_uses_request_api -- --nocapture`,
  and
  `cargo test -p xtask --test repo_lint gpu_adapter_error_classification_uses_shared_core_impl -- --nocapture`.
  The JPEG-Metal route and region-scaled fast-packet bundle cleanup was checked
  with `cargo check -p j2k-jpeg-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg-metal --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-jpeg-metal --all-features --lib region_scaled -- --nocapture`,
  `cargo test -p j2k-jpeg-metal --all-features --test viewport hybrid -- --nocapture`,
  `cargo test -p xtask --test repo_lint`, and
  `cargo clippy -p xtask --all-features --bins --tests -- -D warnings`.
  The transcode DCT 9/7 row and strided-split context cleanup was checked with
  `cargo check -p j2k-transcode --all-features --lib --tests`,
  `cargo clippy -p j2k-transcode --all-features --lib --tests -- -D warnings`,
  and
  `cargo test -p xtask --test repo_lint too_many_arguments_suppressions_stay_below_current_ratchet`.
  The component transcode plan cleanup was checked with
  `cargo check -p j2k-transcode --all-features --lib --tests`,
  `cargo clippy -p j2k-transcode --all-features --lib --tests -- -D warnings`,
  and
  `cargo test -p xtask --test repo_lint too_many_arguments_suppressions_stay_below_current_ratchet`.
  The Metal lossless DWT 5/3 component-dispatch request-object cleanup was
  checked with `cargo check -p j2k-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-metal --all-features --lib --tests -- -D warnings`,
  and
  `cargo test -p xtask --test repo_lint too_many_arguments_suppressions_stay_below_current_ratchet`.
  The Metal resident Tier-1 status-readback request-object cleanup was checked
  with `cargo check -p j2k-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-metal --all-features --lib --tests -- -D warnings`,
  and
  `cargo test -p xtask --test repo_lint too_many_arguments_suppressions_stay_below_current_ratchet`.
  The Metal checked-readback helper plus direct-status, decode-dispatch, HT
  cleanup, Tier-1 encode, resident Tier-1, forward-transform/lossless-prep,
  result-harvest, JPEG-Metal encode/decode/surface readbacks,
  transcode-Metal DWT 9/7 coefficient staging, and validation-status migration
  was checked with
  `cargo check -p j2k-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-metal --all-features --lib --tests -- -D warnings`,
  `cargo check -p j2k-jpeg-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg-metal --all-features --lib --tests -- -D warnings`,
  `cargo check -p j2k-transcode-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-transcode-metal --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-metal --all-features --lib direct_buffers::tests::checked_buffer_required_bytes_rejects_overflow_and_zero_sized_elements -- --nocapture`,
  and
  `cargo test -p j2k-jpeg-metal --all-features --lib buffers::tests::checked_buffer_required_range_rejects_overflow_and_alignment_errors -- --nocapture`.
  The stale unsafe-audit rows removed by the readback migration were checked
  with `cargo run -p xtask -- unsafe-audit`.
  The Metal classic cleanup batch dispatch descriptor cleanup was checked with
  `cargo check -p j2k-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-metal --all-features --lib --tests -- -D warnings`,
  and
  `cargo test -p xtask --test repo_lint too_many_arguments_suppressions_stay_below_current_ratchet`.
  The Metal repeated classic cleanup/store dispatch descriptor cleanup was
  checked with `cargo check -p j2k-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-metal --all-features --lib --tests -- -D warnings`,
  and
  `cargo test -p xtask --test repo_lint too_many_arguments_suppressions_stay_below_current_ratchet`.
  The Metal IDWT sub-band dispatch request-object cleanup was checked with
  `cargo check -p j2k-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-metal --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-metal --all-features --lib idwt -- --nocapture`,
  and
  `cargo test -p xtask --test repo_lint too_many_arguments_suppressions_stay_below_current_ratchet`.
  The HT repeated cleanup dispatch descriptor cleanup was checked with
  `cargo check -p j2k-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-metal --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-metal --all-features --lib ht -- --nocapture`,
  and
  `cargo test -p xtask --test repo_lint too_many_arguments_suppressions_stay_below_current_ratchet`.
  The JPEG-Metal surface-pack request-object cleanup was checked with
  `cargo check -p j2k-jpeg-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg-metal --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-jpeg-metal --all-features --lib fast444 -- --nocapture`,
  and
  `cargo test -p xtask --test repo_lint too_many_arguments_suppressions_stay_below_current_ratchet`.
  The JPEG-Metal batch-item request-object cleanup was checked with
  `cargo check -p j2k-jpeg-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg-metal --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-jpeg-metal --all-features --lib fast420 -- --nocapture`,
  `cargo test -p j2k-jpeg-metal --all-features --lib fast444 -- --nocapture`,
  and
  `cargo test -p xtask --test repo_lint too_many_arguments_suppressions_stay_below_current_ratchet`.
  The JPEG-Metal split coeff/IDCT pass request-object cleanup was checked with
  `cargo check -p j2k-jpeg-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg-metal --all-features --lib --tests -- -D warnings`,
  and
  `cargo test -p xtask --test repo_lint too_many_arguments_suppressions_stay_below_current_ratchet`.
  The Metal repeated direct grayscale plan request-object cleanup was checked
  with `cargo check -p j2k-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-metal --all-features --lib --tests -- -D warnings`,
  and
  `cargo test -p xtask --test repo_lint too_many_arguments_suppressions_stay_below_current_ratchet`.
  The Metal direct color plan request-object cleanup was checked with
  `cargo check -p j2k-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-metal --all-features --lib --tests -- -D warnings`,
  and
  `cargo test -p xtask --test repo_lint too_many_arguments_suppressions_stay_below_current_ratchet`.
  The Metal stacked direct color/component batch request-object cleanup was
  checked with `cargo check -p j2k-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-metal --all-features --lib --tests -- -D warnings`,
  and
  `cargo test -p xtask --test repo_lint too_many_arguments_suppressions_stay_below_current_ratchet`.
  The Metal direct component-plane request-object cleanup was checked with
  `cargo check -p j2k-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-metal --all-features --lib --tests -- -D warnings`,
  and
  `cargo test -p xtask --test repo_lint too_many_arguments_suppressions_stay_below_current_ratchet`.
  The stale Metal lossless validation suppression removal was checked with
  `cargo clippy -p j2k-metal --all-features --lib --tests -- -D warnings` and
  `cargo test -p xtask --test repo_lint too_many_arguments_suppressions_stay_below_current_ratchet`.
  The stale native 5/3 multitile encode suppression removal was checked with
  `cargo clippy -p j2k-native --all-features --lib --tests -- -D warnings` and
  `cargo test -p xtask --test repo_lint too_many_arguments_suppressions_stay_below_current_ratchet`.
  The native i64 packetization request-object cleanup was checked with
  `cargo check -p j2k-native --all-features --lib --tests`,
  `cargo clippy -p j2k-native --all-features --lib --tests -- -D warnings`,
  and
  `cargo test -p xtask --test repo_lint native_encode_options_and_tile_parts_live_in_focused_modules`.
  The native i64 subband settings cleanup was checked with
  `cargo check -p j2k-native --all-features --lib --tests`,
  `cargo clippy -p j2k-native --all-features --lib --tests -- -D warnings`,
  and
  `cargo test -p xtask --test repo_lint too_many_arguments_suppressions_stay_below_current_ratchet`.
  The native i64 component-plane packet and single-tile request-object cleanup
  was checked with `cargo check -p j2k-native --all-features --lib --tests`,
  `cargo clippy -p j2k-native --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-native --all-features --test component_planes i64 -- --nocapture`,
  `cargo test -p xtask --test repo_lint native_encode_options_and_tile_parts_live_in_focused_modules -- --nocapture`,
  and
  `cargo test -p xtask --test repo_lint too_many_arguments_suppressions_stay_below_current_ratchet`.
  The latest focused `j2k-metal` checks after the resident codestream label
  split and Metal padded-copy parameter-object cleanup passed
  `cargo check -p j2k-metal --all-features --lib --bins --tests`,
  `cargo clippy -p j2k-metal --all-features --lib --bins --tests -- -D warnings`,
  `cargo test -p j2k-metal --all-features --lib --tests`, and
  `cargo test -p xtask --test repo_lint metal_compute_runtime_registry_is_split_from_compute_god_file -- --nocapture`.
  The latest focused `j2k-native` checks after the direct-CPU parameter-object
  cleanup passed `cargo check -p j2k-native --all-features --lib --bins --tests`,
  `cargo clippy -p j2k-native --all-features --lib --bins --tests -- -D warnings`,
  `cargo test -p j2k-native --all-features --lib --tests`, and the native
  encode split/precomputed-DWT forwarding guard was checked with
  `cargo test -p xtask --test repo_lint native_encode_options_and_tile_parts_live_in_focused_modules -- --nocapture`.
  The native HT adapter table/SigProp helper extraction was checked with
  `cargo fmt --all --check`, `cargo check -p j2k-native --all-features --lib --tests`,
  `cargo clippy -p j2k-native --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-native --all-features --lib --tests`, and
  `cargo test -p xtask --test repo_lint native_ -- --nocapture`.
  The current too-many-arguments ratchet was checked with
  `cargo check -p xtask --all-features --bins --tests`,
  `cargo clippy -p xtask --all-features --bins --tests -- -D warnings`,
  `cargo test -p xtask --all-features --bins --tests`,
  `cargo test -p xtask --test repo_lint too_many_arguments_suppressions_stay_below_current_ratchet -- --nocapture`,
  and full `cargo test -p xtask --test repo_lint -- --nocapture`.
  The latest line-count ratchet tightening for god-file guards was checked
  with `cargo fmt --all --check`,
  `cargo clippy -p xtask --all-features --bins --tests -- -D warnings`, and
  full `cargo test -p xtask --test repo_lint -- --nocapture`.
  The JPEG-Metal fast-decode binding parameter-object cleanup was checked with
  `cargo check -p j2k-jpeg-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg-metal --all-features --lib --tests -- -D warnings`,
  and `cargo test -p j2k-jpeg-metal --all-features --lib --tests`.
  The shared MQ QE table guard was checked with
  `cargo test -p xtask --test repo_lint mq_qe_table_is_shared_by_encoder_and_decoder -- --nocapture`.
  The session split was also checked with
  `cargo test -p xtask --test repo_lint metal_backend_session_lifecycle_lives_in_support_crate -- --nocapture`.
  The surface split was checked with
  `cargo test -p xtask --test repo_lint backend_surfaces_use_core_metadata_and_residency -- --nocapture`.
  The tile-batch split was checked with
  `cargo test -p xtask --test repo_lint jpeg_metal_single_decode_uses_request_api -- --nocapture`.
  The deprecated pre-1.0 JPEG-Metal `Codec::submit_tile_region_scaled_to_device`
  inherent wrapper was removed in favor of `submit_tile_request_to_device` plus
  `MetalDecodeRequest::region_scaled`, lowering the current
  `too_many_arguments` ratchet from 103 to 102 and regenerating the stable API
  snapshot. This API-shrink slice was checked with `cargo fmt --all --check`,
  `cargo clippy -p j2k-jpeg-metal --all-features --lib --tests -- -D warnings`,
  `cargo test -p xtask --test repo_lint too_many_arguments_suppressions_stay_below_current_ratchet -- --nocapture`,
  `cargo xtask stable-api --write`, and `cargo xtask stable-api`.
  The root `j2k` context-reused tile helpers now accept `TileDecodeOutput`
  for output buffer, stride, and pixel format, lowering the current
  `too_many_arguments` ratchet from 43 to 42 and regenerating stable API to
  621,845 bytes / 1,973 `pub fn` entries. This semver-visible pre-1.0 cleanup
  was checked with `cargo fmt --all --check`,
  `cargo check -p j2k -p j2k-compare --all-features --lib --bins --tests`,
  `cargo clippy -p j2k -p j2k-compare --all-features --lib --bins --tests -- -D warnings`,
  `cargo test -p j2k --all-features --lib --tests`,
  `cargo test -p xtask --test repo_lint too_many_arguments_suppressions_stay_below_current_ratchet -- --nocapture`,
  `cargo xtask stable-api --write`, and `cargo xtask stable-api`.
  The private JPEG baseline encoder entropy helpers now share one borrowed
  `EntropyEncodeContext` and one MCU block-position object, lowering the
  current `too_many_arguments` ratchet from 102 to 97 without public API churn.
  This encoder slice was checked with `cargo fmt --all --check`,
  `cargo check -p j2k-jpeg --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg --all-features --lib --tests -- -D warnings`,
  and `cargo test -p j2k-jpeg --all-features --lib encoder -- --nocapture`.
  The progressive JPEG entropy block decoder now passes a single
  `ProgressiveBlockTarget` for component/scan/block coordinates, lowering the
  current `too_many_arguments` ratchet from 97 to 96 without public API churn.
  This progressive slice was checked with `cargo fmt --all --check`,
  `cargo check -p j2k-jpeg --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-jpeg --all-features --test inspect progressive -- --nocapture`,
  and
  `cargo test -p j2k-jpeg --all-features --test decode_into progressive -- --nocapture`.
  The CUDA HTJ2K packetize cleanup host-launch helper now accepts one
  `Htj2kPacketizeCleanupLaunch` request object while preserving the kernel
  parameter order, lowering the current `too_many_arguments` ratchet from 96 to
  95. This CUDA runtime slice was checked with `cargo fmt --all --check`,
  `cargo check -p j2k-cuda-runtime --all-features --lib --tests`, and
  `cargo clippy -p j2k-cuda-runtime --all-features --lib --tests -- -D warnings`
  on macOS with the expected Linux-only cuda-oxide build-script skips.
  The sequential JPEG scan skipper now shares one `McuSkipState` plus
  `McuSkipTarget` for restart/position state, lowering the current
  `too_many_arguments` ratchet from 93 to 92. This sequential entropy slice was
  checked with `cargo fmt --all --check`,
  `cargo check -p j2k-jpeg --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg --all-features --lib --tests -- -D warnings`, and
  `cargo test -p j2k-jpeg --all-features --test decode_into -- --nocapture`.
  The sequential JPEG fast-tile scaled/full block helpers now share
  `EntropyBlockState`, `ReducedIdctScratch`, and `PlaneBlockTarget` for
  entropy state, reduced-IDCT scratch, and output placement, lowering the
  current `too_many_arguments` ratchet from 92 to 88. This sequential entropy
  slice was checked with `cargo fmt --all --check`,
  `cargo check -p j2k-jpeg --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg --all-features --lib --tests -- -D warnings`, and
  `cargo test -p j2k-jpeg --all-features --test decode_into -- --nocapture`.
  The sequential JPEG fast-tile 4:2:0 row helpers now share
  `FastTile420Components`, `FastTile420DcState`, `FastTile420EntropyState`,
  and `FastTile420Window` for component slices, DC predictors, entropy
  routing, and MCU windows, lowering the current `too_many_arguments` ratchet
  from 88 to 84. This sequential entropy slice was checked with
  `cargo fmt --all --check`,
  `cargo check -p j2k-jpeg --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg --all-features --lib --tests -- -D warnings`, and
  `cargo test -p j2k-jpeg --all-features --test decode_into -- --nocapture`.
  The JPEG AVX2/NEON IDCT private helpers now pass their eight lane vectors as
  fixed arrays, lowering the current `too_many_arguments` ratchet from 84 to
  82 without changing backend public behavior. This SIMD IDCT slice was
  checked with `cargo fmt --all --check`,
  `cargo check -p j2k-jpeg --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg --all-features --lib --tests -- -D warnings`, and
  `cargo test -p j2k-jpeg --all-features --test idct_parity -- --nocapture`.
  The CUDA JPEG baseline entropy launch helpers now share
  `CudaJpegBaselineEntropyLaunch`, `CudaJpegBaselineEntropyBatchLaunch`,
  `CudaJpegBaselineQuantLaunch`, and `CudaJpegBaselineHuffmanLaunch` while
  preserving kernel parameter order, lowering the current
  `too_many_arguments` ratchet from 82 to 80. This CUDA runtime slice was
  checked with `cargo fmt --all --check`,
  `cargo check -p j2k-cuda-runtime --all-features --lib --tests`, and
  `cargo clippy -p j2k-cuda-runtime --all-features --lib --tests -- -D warnings`
  on macOS with the expected Linux-only cuda-oxide build-script skips.
  The CUDA HTJ2K encode launch helpers now share
  `CudaHtj2kEncodeCodeblockLaunch`, `CudaHtj2kEncodeCodeblocksLaunch`,
  `CudaHtj2kEncodeMultiInputLaunch`, and `CudaHtj2kEncodeLaunchTables`,
  lowering the current `too_many_arguments` ratchet from 80 to 75 while
  preserving kernel parameter order. This CUDA runtime slice was checked with
  `cargo fmt --all --check`,
  `cargo check -p j2k-cuda-runtime --all-features --lib --tests`, and
  `cargo clippy -p j2k-cuda-runtime --all-features --lib --tests -- -D warnings`
  on macOS with the expected Linux-only cuda-oxide build-script skips.
  The CUDA transcode private DWT 9/7 pipeline and launch helpers now share
  `Dwt97BatchDeviceRequest`, `Dwt97BatchGeometry`,
  `Dwt97ColumnLiftBatchLaunch`,
  `Dwt97ColumnLiftQuantizeCodeblocksBatchLaunch`, and
  `Dwt97QuantizeCodeblocksLaunch`, lowering the current
  `too_many_arguments` ratchet from 75 to 69 while leaving the remaining
  transcode suppressions on semver-visible public entrypoints. This CUDA
  runtime slice was checked with `cargo fmt --all --check`,
  `cargo check -p j2k-cuda-runtime --all-features --lib --tests`, and
  `cargo clippy -p j2k-cuda-runtime --all-features --lib --tests -- -D warnings`
  on macOS with the expected Linux-only cuda-oxide build-script skips.
  The sequential JPEG stripe upsample helpers now share `StripeNeighbors`,
  `StripeComponentUpsampleSpec`, `StripeComponentUpsample`,
  `Stripe420PairSpec`, and `Stripe420PairUpsample`, lowering the current
  `too_many_arguments` ratchet from 69 to 67. This sequential entropy slice
  was checked with `cargo fmt --all --check`,
  `cargo check -p j2k-jpeg --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg --all-features --lib --tests -- -D warnings`, and
  `cargo test -p j2k-jpeg --all-features --test decode_into -- --nocapture`.
  The JPEG backend 4:2:0 row-pair dispatch, scalar, AVX2, and NEON wrappers
  now share `Rgb420ChromaRows`, `Rgb420RowPair`, `Rgb420Crop`, and
  `Rgb420CroppedRowPair`, lowering the current `too_many_arguments` ratchet
  from 67 to 59. This backend slice was checked with `cargo fmt --all --check`,
  `cargo check -p j2k-jpeg --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-jpeg --all-features --lib backend::tests -- --nocapture`,
  and `cargo test -p j2k-jpeg --all-features --test decode_into -- --nocapture`.
  The JPEG sequential MCU row decoders now use `McuRowContext`/`McuRowState`
  and `FastRgb444McuRowContext`/`FastRgb444McuRowState` for static scan
  geometry and mutable entropy/DC state, lowering the current
  `too_many_arguments` ratchet from 51 to 49 without public API churn. This
  production-internal decode slice was checked with `cargo fmt --all --check`,
  `cargo check -p j2k-jpeg --all-features --lib --bins --tests`, and
  `cargo clippy -p j2k-jpeg --all-features --lib --bins --tests -- -D warnings`,
  `cargo test -p j2k-jpeg --all-features --test decode_into -- --nocapture`,
  and
  `cargo test -p xtask --test repo_lint too_many_arguments_suppressions_stay_below_current_ratchet -- --nocapture`.
  The same sequential decode slice now routes generic and RGB stripe emission
  through `StripeEmit`, lowering the current `too_many_arguments` ratchet from
  49 to 47 without public API churn. This production-internal decode slice was
  checked with `cargo fmt --all --check`,
  `cargo check -p j2k-jpeg --all-features --lib --bins --tests`, and
  `cargo clippy -p j2k-jpeg --all-features --lib --bins --tests -- -D warnings`,
  `cargo test -p j2k-jpeg --all-features --test decode_into -- --nocapture`,
  and
  `cargo test -p xtask --test repo_lint too_many_arguments_suppressions_stay_below_current_ratchet -- --nocapture`.
  The public context-reused JPEG tile helpers now accept `TileDecodeOutput`
  for output buffer, stride, and pixel format, lowering the current
  `too_many_arguments` ratchet from 47 to 43 and regenerating stable API to
  621,745 bytes / 1,973 `pub fn` entries. This semver-visible pre-1.0 cleanup
  was checked with `cargo fmt --all --check`,
  `cargo check -p j2k-jpeg -p j2k-jpeg-cuda -p j2k-compare --all-features --lib --bins --tests`,
  `cargo clippy -p j2k-jpeg -p j2k-jpeg-cuda -p j2k-compare --all-features --lib --bins --tests -- -D warnings`,
  `cargo test -p j2k-jpeg --all-features --test decode_into -- --nocapture`,
  `cargo test -p j2k-jpeg --all-features --test batch -- --nocapture`,
  `cargo test -p xtask --test repo_lint too_many_arguments_suppressions_stay_below_current_ratchet -- --nocapture`,
  `cargo xtask stable-api --write`, and `cargo xtask stable-api`.
  The x86 AVX2 4:2:0 row-pair helper now also accepts the shared
  `Rgb420RowPair` request plus scratch, lowering the current
  `too_many_arguments` ratchet from 59 to 58. This x86 backend slice was
  checked with `cargo fmt --all --check`,
  `cargo check -p j2k-jpeg --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-jpeg --all-features --lib backend::tests -- --nocapture`,
  and `cargo check -p j2k-jpeg --all-features --lib --target x86_64-apple-darwin`.
  The NEON 4:2:0 row-pair dual/top-only dispatch helpers now accept the same
  shared request objects, lowering the current `too_many_arguments` ratchet
  from 58 to 56. This NEON backend slice was checked with
  `cargo fmt --all --check`,
  `cargo check -p j2k-jpeg --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-jpeg --all-features --lib backend::tests -- --nocapture`,
  and `cargo test -p j2k-jpeg --all-features --test decode_into -- --nocapture`.
  The doc-hidden JPEG 4:2:0 bench-internals helpers now use
  `BenchRgb420ChromaRows` and `BenchRgb420RowPair` request objects instead of
  flat row arguments, lowering the current `too_many_arguments` ratchet from
  56 to 53 without touching production decode paths. This hidden bench-support
  slice was checked with `cargo fmt --all --check`,
  `cargo check -p j2k-jpeg --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg --all-features --lib --tests -- -D warnings`, and
  `cargo test -p j2k-jpeg --all-features --test neon_hot_paths -- --nocapture`.
  The JPEG fast-tile RGB region-scaled decode entry now uses
  `FastTileRegionScaledRequest` for ROI, downscale, and checkpoint state,
  lowering the current `too_many_arguments` ratchet from 53 to 52. This
  production-internal decode slice was checked with `cargo fmt --all --check`,
  `cargo check -p j2k-jpeg --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg --all-features --lib --tests -- -D warnings`, and
  `cargo test -p j2k-jpeg --all-features --test decode_into -- --nocapture`.
  The JPEG 4:2:0 region stripe emitter now uses `Fast420RegionStripe` for
  stripe neighbors, ROI/layout, scratch rows, and downscale, lowering the
  current `too_many_arguments` ratchet from 52 to 51. This production-internal
  decode slice was checked with `cargo fmt --all`, `cargo check -p j2k-jpeg
  --all-features --lib --tests`, `cargo clippy -p j2k-jpeg --all-features
  --lib --tests -- -D warnings`, and `cargo test -p j2k-jpeg --all-features
  --test decode_into -- --nocapture`.
  The JPEG NEON 4:2:0 cropped, edge, tail, and interior helpers now carry
  shared chroma rows plus partial/tail chunk metadata instead of flat row and
  geometry argument lists, lowering the current `too_many_arguments` ratchet
  from 42 to 31 without public API churn. This aarch64 backend slice was
  checked with `cargo fmt --all --check`,
  `cargo check -p j2k-jpeg --all-features --lib --bins --tests`,
  `cargo clippy -p j2k-jpeg --all-features --lib --bins --tests -- -D warnings`,
  and `cargo test -p j2k-jpeg --all-features --test neon_hot_paths -- --nocapture`.
  The native HT cleanup encoder now routes quad-pair walk state and initial /
  non-initial row state through private request structs instead of repeated flat
  hot-path argument lists; the HT encode file now has no local
  `too_many_arguments` suppressions. This HT encode slice was checked with
  `cargo fmt --all`, `cargo check -p j2k-native --all-features --lib --tests`,
  `cargo clippy -p j2k-native --all-features --lib --tests -- -D warnings`,
  and `cargo test -p j2k-native --all-features --lib ht_block_encode -- --nocapture`.
  The repo-lint ratchet now scans full `allow` attributes instead of only the
  literal `#[allow(clippy::too_many_arguments` prefix, so multiline and
  crate-level suppressions are included; the initial corrected ratchet was <=33.
  The private CUDA HTJ2K decode launch helpers now use single/multi launch
  request structs plus a shared table-buffer bundle instead of flat kernel
  argument lists, lowering the corrected ratchet from 33 to 31 without public API
  churn. This CUDA runtime slice was checked with `cargo fmt --all --check`,
  `cargo check -p j2k-cuda-runtime --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda-runtime --all-features --lib --tests -- -D warnings`,
  and `cargo test -p j2k-cuda-runtime --all-features --lib htj2k -- --nocapture`
  on macOS with expected Linux-only cuda-oxide build-script skips.
  The private CUDA JPEG RGB8 decode and entropy self-sync launch helpers now use
  quant/Huffman/request structs instead of repeated flat kernel argument lists,
  lowering the corrected ratchet from 31 to 28 without public API churn. This
  CUDA JPEG runtime slice was checked with `cargo fmt --all --check`,
  `cargo check -p j2k-cuda-runtime --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda-runtime --all-features --lib --tests -- -D warnings`,
  and `cargo test -p j2k-cuda-runtime --all-features --lib jpeg -- --nocapture`
  on macOS with expected Linux-only cuda-oxide build-script skips.
  The private fused i16 HTJ2K 9/7 resident transcode helper now takes
  `Htj2k97I16ResidentFusedRequest` around existing batch geometry, quantization,
  input, and pool state, lowering the corrected ratchet from 28 to 27 without
  public API churn. This CUDA transcode runtime slice was checked with
  `cargo fmt --all`, `cargo clippy -p j2k-cuda-runtime --all-features --lib --tests -- -D warnings`,
  and `cargo test -p j2k-cuda-runtime --all-features --lib transcode -- --nocapture`
  on macOS with expected Linux-only cuda-oxide build-script skips.
  The public CUDA 9/7 transcode batch APIs now take named request structs for
  shared geometry, quantization parameters, and caller-owned buffer pools instead
  of long positional argument lists, lowering the corrected
  `too_many_arguments` ratchet from 27 to 21. This semver-visible pre-1.0 cleanup
  was checked with `cargo fmt --all`,
  `cargo check -p j2k-cuda-runtime -p j2k-transcode-cuda --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda-runtime -p j2k-transcode-cuda --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-cuda-runtime --all-features --lib transcode -- --nocapture`,
  `cargo test -p j2k-transcode-cuda --all-features --lib -- --nocapture`,
  `cargo xtask stable-api --write`,
  `cargo xtask stable-api`,
  and `cargo test -p xtask --test repo_lint too_many_arguments_suppressions_stay_below_current_ratchet -- --nocapture`
  on macOS with expected Linux-only cuda-oxide build-script skips.
  Redundant function-level `too_many_arguments` suppressions in generated
  CUDA-Oxide JPEG encode and J2K packetization sources were removed because the
  files already carry crate-level generated-kernel allows, lowering the ratchet
  from 21 to 16 without ABI or behavior changes. This cleanup was checked with
  `cargo fmt --all --check`,
  `cargo check -p j2k-cuda-runtime --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda-runtime --all-features --lib --tests -- -D warnings`,
  and `cargo test -p xtask --test repo_lint too_many_arguments_suppressions_stay_below_current_ratchet -- --nocapture`
  on macOS with expected Linux-only cuda-oxide build-script skips.
  A suppression recount confirmed 16 remaining `too_many_arguments` allows,
  including multiline crate-level generated-kernel allows that simple line-based
  scans miss, so the repo-lint ratchet was held at the tested count. The CUDA
  JPEG decode SIMT `decode_block` helper now takes `JpegDecodeBlockContext`
  for entropy, table, and quantization inputs, lowering the current ratchet from
  16 to 15 without changing kernel ABI. This slice was checked with
  `cargo fmt --all`,
  `cargo check -p j2k-cuda-runtime --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda-runtime --all-features --lib --tests -- -D warnings`,
  and `cargo test -p xtask --test repo_lint too_many_arguments_suppressions_stay_below_current_ratchet -- --nocapture`
  on macOS with expected Linux-only cuda-oxide build-script skips.
  The core tile region-scaled trait APIs now take
  `TileRegionScaledDecodeJob` / `TileRegionScaledDeviceDecodeRequest`, removing
  the two non-generated core trait suppressions and tightening the ratchet from
  15 to 13. This semver-visible pre-1.0 API cleanup was checked with
  `cargo check -p j2k-core -p j2k -p j2k-jpeg -p j2k-cuda -p j2k-jpeg-cuda -p j2k-metal -p j2k-jpeg-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-core -p j2k -p j2k-jpeg -p j2k-cuda -p j2k-jpeg-cuda -p j2k-metal -p j2k-jpeg-metal --all-features --lib --tests -- -D warnings`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`, and
  `cargo test -p xtask --test repo_lint too_many_arguments_suppressions_stay_below_current_ratchet -- --nocapture`;
  no benchmark executable was run.
  The CUDA-Oxide JPEG decode SIMT RGB 4:2:0/4:2:2 store helpers now take
  private MCU block bundles, and the entropy symbol scanner now takes a private
  table bundle, reducing private helper suppressions without changing public
  kernel ABI. This lowered the current `too_many_arguments` ratchet from 13 to
  10 and was checked with
  `cargo fmt --all --check`,
  `cargo check -p j2k-cuda-runtime --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda-runtime --all-features --lib --tests -- -D warnings`;
  no benchmark executable was run.
  The CUDA-Oxide transcode fused DWT 9/7 column-lift/quantize kernel now takes
  a private `Dwt97ColumnLiftQuantizeCodeblocksParams` scalar ABI block for
  code-block geometry and low/high quantization deltas, lowering the current
  `too_many_arguments` ratchet from 10 to 9 while keeping the entrypoint name
  stable. This was checked with local
  `cargo fmt --all --check`,
  `cargo check -p j2k-cuda-runtime --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda-runtime --all-features --lib --tests -- -D warnings`,
  `cargo test -p xtask --test repo_lint too_many_arguments_suppressions_stay_below_current_ratchet -- --nocapture`,
  and remote CUDA validation on `jcwal@100.75.125.59`:
  `J2K_REQUIRE_CUDA_OXIDE_BUILD=1 cargo check -p j2k-cuda-runtime --all-features --lib`
  plus
  `J2K_REQUIRE_CUDA_RUNTIME=1 J2K_REQUIRE_CUDA_OXIDE_BUILD=1 cargo test -p j2k-transcode-cuda --all-features --test htj2k97_codeblock_parity -- --nocapture`
  (3 passed). No benchmark executable was run.
  CUDA JPEG decode launch/device ABIs then gained private quant-table and
  Huffman-table pointer groups, removing the five function-level
  `too_many_arguments` suppressions from the 4:2:0 entropy diagnostic kernels
  and 4:2:0/4:2:2/4:4:4 RGB decode kernels. The current ratchet is now 4; the
  remaining counted suppressions are the broad `j2k-native` crate allow and the
  three CUDA Oxide encode crate-level generated-kernel allowances. This
  non-benchmark slice is checked locally with
  `cargo fmt --all --check`,
  `cargo check -p j2k-cuda-runtime --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda-runtime --all-features --lib --tests -- -D warnings`,
  and
  `cargo test -p xtask --test repo_lint too_many_arguments_suppressions_stay_below_current_ratchet -- --nocapture`;
  remote CUDA validation on `jcwal@100.75.125.59` passed
  `J2K_REQUIRE_CUDA_OXIDE_BUILD=1 cargo check -p j2k-cuda-runtime --all-features --lib`,
  `J2K_REQUIRE_CUDA_RUNTIME=1 J2K_REQUIRE_CUDA_OXIDE_BUILD=1 cargo test -p j2k-cuda-runtime --all-features --lib cuda_oxide_jpeg_entropy_self_sync_decodes_zero_stream_when_required -- --nocapture`,
  and
  `J2K_REQUIRE_CUDA_RUNTIME=1 J2K_REQUIRE_CUDA_OXIDE_BUILD=1 J2K_REQUIRE_CUDA_JPEG_HARDWARE_DECODE=1 cargo test -p j2k-jpeg-cuda --all-features --test host_surface explicit_cuda_full_frame -- --nocapture`
  (3 passed). No benchmark executable was run.
  The unused public CUDA transcode no-pool batch wrappers
  (`j2k_transcode_dwt97_batch`, `j2k_transcode_htj2k97_codeblock_batch`, and
  `j2k_transcode_htj2k97_codeblock_batch_resident`) were then removed after an
  exact-call scan found no in-repo callers; the `*_with_pool` request-struct
  methods remain the used API. This public API shrink regenerated
  `docs/stable-api-1.0.public-api.txt` to 622,733 bytes / 1,970 `pub fn`
  entries and was checked with `cargo fmt --all`,
  `cargo check -p j2k-cuda-runtime -p j2k-transcode-cuda --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda-runtime -p j2k-transcode-cuda --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-cuda-runtime --all-features --lib transcode -- --nocapture`,
  `cargo test -p j2k-transcode-cuda --all-features --lib -- --nocapture`,
  `cargo xtask stable-api --write`, and `cargo xtask stable-api`.
  The duplicate public `j2k_jpeg::decoder` module path was then removed by
  making the implementation module internal while preserving the root
  `j2k_jpeg::{Decoder, JpegView, DecodeRequest, ...}` facade. This regenerated
  `docs/stable-api-1.0.public-api.txt` to 604,439 bytes / 1,907 `pub fn`
  entries; the `j2k-jpeg` package now renders 374 public functions and
  `j2k_jpeg::Decoder` renders 37 public functions through the root facade.
  The guard in `architecture_policy.rs` prevents `pub mod j2k_jpeg::decoder`
  from returning. This follow-up was checked with
  `cargo check -p j2k-jpeg --all-features --lib --tests`,
  `cargo check -p j2k-jpeg-metal --all-features --lib --tests`,
  `cargo check -p j2k-jpeg-cuda --all-features --lib --tests`,
  `cargo fmt --all --check`, `cargo xtask stable-api`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`,
  `cargo clippy -p j2k-jpeg -p j2k-jpeg-metal -p j2k-jpeg-cuda --all-features --lib --tests -- -D warnings`,
  and `cargo test -p xtask --test repo_lint -- --nocapture`.
  The duplicate public `j2k::view` module path was then removed by making the
  implementation module internal while preserving the root
  `j2k::{J2kDecoder, J2kView, J2kCodec, J2kRowDecodeOptions}` facade. This
  regenerated `docs/stable-api-1.0.public-api.txt` to 595,099 bytes / 1,866
  `pub fn` entries; the `j2k` package now renders 162 public functions and the
  duplicate `pub mod j2k::view` entry is absent from the snapshot. The guard in
  `architecture_policy.rs` prevents `pub mod j2k::view` from returning. This
  follow-up was checked with
  `cargo check -p j2k --all-features --lib --tests`,
  `cargo check -p j2k -p j2k-metal -p j2k-cuda --all-features --lib --tests`,
  `cargo fmt --all --check`, `cargo xtask stable-api`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`,
  and `cargo clippy -p j2k -p j2k-metal -p j2k-cuda --all-features --lib --tests -- -D warnings`.
  The duplicate public `j2k::{context,error,scratch}` module paths were then
  removed by making those implementation modules internal while preserving the
  root `j2k::{J2kContext, J2kError, BackendError, BackendErrorKind,
  J2kScratchPool}` facade. This regenerated
  `docs/stable-api-1.0.public-api.txt` to 589,817 bytes / 1,847 `pub fn`
  entries; the `j2k` package now renders 143 public functions and the duplicate
  `pub mod j2k::{context,error,scratch,view}` entries are absent from the
  snapshot. The guard in `architecture_policy.rs` prevents those duplicate
  module paths from returning. This follow-up was checked with
  `cargo check -p j2k --all-features --lib --tests` and
  `cargo xtask stable-api --write`, then with `cargo fmt --all --check`,
  `cargo check -p j2k -p j2k-metal -p j2k-cuda --all-features --lib --tests`,
  `cargo xtask stable-api`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`,
  and
  `cargo clippy -p j2k -p j2k-metal -p j2k-cuda --all-features --lib --tests -- -D warnings`.
  The duplicate public `j2k_jpeg::{info,context,batch_session,capabilities,
  output_buffer,segment,error,encoder}` module paths were then removed by
  making those implementation modules internal while preserving the root
  `j2k_jpeg::{Info, Rect, DecoderContext, JpegBatchSession,
  JpegCapabilityReport, JpegOutputBuffer, JpegError, JpegEncodeOptions, ...}`
  facade. This regenerated `docs/stable-api-1.0.public-api.txt` to
  555,449 bytes / 1,770 `pub fn` entries; the `j2k-jpeg` package now renders
  295 public functions, `j2k_jpeg::Decoder` still renders 37 public functions
  through the root facade, and only `j2k_jpeg::adapter` plus
  `j2k_jpeg::transcode` remain public JPEG submodules. The guard in
  `architecture_policy.rs` prevents the duplicate module paths from returning.
  A later public API slice hides those two rendered module inventories as well
  while keeping them source-visible for first-party crates and re-exporting
  `DeviceBatchSummary` at the `j2k_jpeg` root.
  This follow-up has so far been checked with
  `cargo check -p j2k-jpeg --all-features --lib --tests`,
  `cargo check -p j2k-jpeg-metal -p j2k-jpeg-cuda --all-features --lib --tests`,
  `cargo xtask stable-api --write`, `cargo fmt --all --check`,
  `cargo xtask stable-api`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`,
  and
  `cargo clippy -p j2k-jpeg -p j2k-jpeg-metal -p j2k-jpeg-cuda --all-features --lib --tests -- -D warnings`.
  The duplicate public `j2k_native::error` module path was then removed by
  making the implementation module internal while preserving the root
  `j2k_native::{DecodeError, DecodingError, FormatError, MarkerError,
  TileError, ValidationError, ColorError, DirectPlanUnsupportedReason, Result}`
  facade. This regenerated `docs/stable-api-1.0.public-api.txt` to
  546,146 bytes / 1,750 `pub fn` entries; the `j2k-native` package now renders
  103 public functions and no `j2k_native::error` path remains in the snapshot.
  The guard in `architecture_policy.rs` prevents the duplicate module path from
  returning. This follow-up has so far been checked with
  `cargo check -p j2k-native -p j2k --all-features --lib --tests` and
  `cargo xtask stable-api --write`, then with `cargo fmt --all --check`,
  `cargo xtask stable-api`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`,
  and `cargo clippy -p j2k-native -p j2k --all-features --lib --tests -- -D warnings`.
  The CUDA J2K strided deinterleave and single-IDWT-with-pool host helpers now
  use `J2kStridedDeinterleaveLaunch` and `J2kInverseDwtSinglePoolRequest`
  request objects while preserving kernel parameter order and public method
  signatures, lowering the current `too_many_arguments` ratchet from 95 to 93.
  This CUDA runtime slice was checked with `cargo fmt --all --check`,
  `cargo check -p j2k-cuda-runtime --all-features --lib --tests`, and
  `cargo clippy -p j2k-cuda-runtime --all-features --lib --tests -- -D warnings`
  on macOS with the expected Linux-only cuda-oxide build-script skips.
  The duplicate public
  `j2k_transcode::{dct53_2d,dct97_2d,htj2k97_codeblock_oracle}` module paths
  were then removed by making those implementation modules internal, moving the
  public wavelet result structs to the crate root, and preserving the intended
  root transform/oracle exports. This regenerated
  `docs/stable-api-1.0.public-api.txt` to 545,288 bytes / 1,750 `pub fn`
  entries; no `j2k_transcode::dct53_2d`,
  `j2k_transcode::dct97_2d`, or
  `j2k_transcode::htj2k97_codeblock_oracle` path remains in the snapshot. The
  guard in `architecture_policy.rs` prevents those duplicate module paths from
  returning. This follow-up was checked with
  `cargo check -p j2k-transcode -p j2k-transcode-metal -p j2k-transcode-cuda --all-features --lib --tests`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`, and
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`.
  The shared transcode stage-counter mutation API was then collapsed from a
  16-method public `record_*` fanout to one typed
  `DctToWaveletStageCounters::record(event, count)` entry point plus
  `DctToWaveletStageCounterEvent`. This regenerated
  `docs/stable-api-1.0.public-api.txt` to 544,791 bytes / 1,735 `pub fn`
  entries. The guards in `architecture_policy.rs` and
  `docs_and_workflows_policy.rs` prevent the old counter mutators and adapter
  bookkeeping forks from returning. This follow-up was checked with
  `cargo check -p j2k-transcode -p j2k-transcode-metal -p j2k-transcode-cuda --all-features --lib --tests`,
  `cargo clippy -p j2k-transcode -p j2k-transcode-metal -p j2k-transcode-cuda --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-transcode --all-features --lib --tests`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`,
  `cargo test -p xtask --test repo_lint transcode_stage_counters_are_shared_between_gpu_adapters -- --nocapture`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`,
  `cargo check -p xtask --all-features --test repo_lint`,
  `cargo clippy -p xtask --all-features --test repo_lint -- -D warnings`,
  and full `cargo test -p xtask --test repo_lint -- --nocapture`.
  The prequantized HTJ2K 9/7 oracle builders
  (`prequantized_component_from_dwt97` and `quantize_codeblock_subband`) were
  then moved out of semver-visible `j2k-transcode` and into the unpublished
  `j2k-transcode-test-support` crate used by CUDA/Metal parity tests. The
  native codestream pin test moved with the oracle, preserving the regression
  check while regenerating `docs/stable-api-1.0.public-api.txt` to
  544,397 bytes / 1,733 `pub fn` entries. This follow-up was checked with
  `cargo check -p j2k-transcode -p j2k-transcode-cuda -p j2k-transcode-metal -p j2k-transcode-test-support --all-features --lib --tests`,
  `cargo clippy -p j2k-transcode -p j2k-transcode-cuda -p j2k-transcode-metal -p j2k-transcode-test-support --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-transcode-test-support --all-features --lib --tests`,
  `cargo test -p j2k-transcode --all-features --lib --tests`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`,
  `cargo test -p xtask --test repo_lint architecture_docs_classify_workspace_and_in_repo_tool_crates -- --nocapture`,
  `cargo test -p xtask --test repo_lint tooling_and_validation_crates_stay_unpublished -- --nocapture`,
  `cargo test -p xtask --test repo_lint architecture_dependency_graph_matches_cargo_metadata -- --nocapture`,
  and `cargo fmt --all --check`.
  The reversible 5/3 `reversible_dwt53_first_level_from_block_samples` helper
  was then made crate-private after confirming first-party adapters only use
  the adjacent source-visible `idct_blocks_to_signed_samples_rayon` helper
  (now hidden from the rendered public inventory). The stable
  API snapshot was regenerated to 544,179 bytes / 1,732 `pub fn` entries, and
  `architecture_policy.rs` now prevents that helper from returning to the
  rendered public API. This follow-up was checked with
  `cargo check -p j2k-transcode -p j2k-transcode-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-transcode -p j2k-transcode-metal --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-transcode --all-features --lib --tests`,
  `cargo xtask stable-api --write`, and `cargo xtask stable-api`.
  The DCT-grid scratch capacity inspectors
  (`Dct53GridScratch::weight_row_capacity` and
  `Dct97GridScratch::spatial_sample_capacity`) were then moved out of the
  stable API by keeping the reuse/sparsity assertions as crate-local unit
  tests and making the inspectors test-only internals. The stable API snapshot
  was regenerated to 543,949 bytes / 1,730 `pub fn` entries, with
  `architecture_policy.rs` guards preventing the source methods or rendered
  stable API entries from returning. This follow-up was checked with
  `cargo check -p j2k-transcode -p j2k-transcode-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-transcode -p j2k-transcode-metal --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-transcode --all-features --lib --tests`,
  `cargo test -p j2k-transcode --all-features --lib dct8x8_grid_scratch -- --nocapture`,
  `cargo test -p j2k-transcode --all-features --lib dct8x8_grid_to_2d_97_idct_scratch_path_reuses_spatial_storage -- --nocapture`,
  `cargo xtask stable-api --write`, and `cargo xtask stable-api`.
  The accidental single-block 5/3 helpers
  (`dct8x8_to_dwt53_float_linear` and `idct8x8_then_dwt53_float`) were then
  removed from the stable `j2k-transcode` API. Tests and the DCT53 benchmark
  source now use the retained one-block grid APIs for the same behavior, but no
  benchmark executable was run in this non-benchmark slice. The stable API
  snapshot was regenerated to 543,735 bytes / 1,728 `pub fn` entries, with
  `architecture_policy.rs` guards preventing the source helpers or rendered
  stable API entries from returning. This follow-up was checked with
  `cargo check -p j2k-transcode -p j2k-transcode-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-transcode -p j2k-transcode-metal --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-transcode --all-features --lib --tests`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`, and
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`.
  The `JpegToHtj2kTranscoder` scratch-capacity inspectors were then removed
  from the source public impl. Their reuse checks now live as crate-local unit
  tests that inspect private scratch directly, while integration tests keep
  asserting stateful/stateless output agreement through the public API. The
  rendered stable API snapshot remained 543,735 bytes / 1,728 `pub fn`
  entries because those inherent methods were not present in the snapshot, but
  `architecture_policy.rs` now prevents the source methods or rendered entries
  from returning. This follow-up was checked with
  `cargo check -p j2k-transcode -p j2k-transcode-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-transcode -p j2k-transcode-metal --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-transcode --all-features --lib --tests`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`, and
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`.
  The caller-owned DCT-grid scratch types and `*_with_scratch` transform
  functions were then removed from the rendered `j2k-transcode` public API by
  making them crate-private. Production transcode still reuses those scratch
  buffers internally; external tests now use the public non-scratch wrappers,
  and benchmark sources path-include the internal transform modules so their
  scratch-reuse coverage is preserved without publishing those APIs. The stable
  API snapshot was regenerated to 543,152 bytes / 1,726 `pub fn` entries, and
  `architecture_policy.rs` now prevents the scratch types or rendered
  `*_with_scratch` entries from returning. This follow-up was checked with
  `cargo check -p j2k-transcode -p j2k-transcode-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-transcode -p j2k-transcode-metal --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-transcode --all-features --lib --tests`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`,
  and compile-only `cargo check -p j2k-transcode -p j2k-transcode-metal --all-features --benches`.
  The `Dwt53TwoDimensional<f64>::max_abs_diff` and
  `Dwt97TwoDimensional<f64>::max_abs_diff` assertion helpers were then moved
  behind test-only implementations. Integration tests now use the shared
  `tests/support/dwt_diff.rs` helper, keeping behavior coverage without
  publishing assertion utilities.
  The stable API snapshot was regenerated to 542,898 bytes / 1,724 `pub fn`
  entries, with `architecture_policy.rs` guards preventing the rendered helper
  methods from returning. This follow-up was checked with
  `cargo check -p j2k-transcode -p j2k-transcode-metal -p j2k-transcode-cuda --all-features --lib --tests`,
  `cargo clippy -p j2k-transcode -p j2k-transcode-metal -p j2k-transcode-cuda --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-transcode --all-features --lib --tests`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`,
  and compile-only `cargo check -p j2k-transcode -p j2k-transcode-metal --all-features --benches`.
  The duplicate transform error aliases (`Dct53GridError` and
  `Dct97GridError`) were then collapsed into the single public
  `DctGridError` type already used internally. Public transform functions now
  return `DctGridError` directly, and `architecture_policy.rs` prevents the
  aliases from returning in source or the rendered stable API. The stable API
  snapshot was regenerated to 541,604 bytes / 1,722 `pub fn` entries. This
  follow-up was checked with
  `cargo check -p j2k-transcode -p j2k-transcode-metal -p j2k-transcode-cuda --all-features --lib --tests`,
  `cargo clippy -p j2k-transcode -p j2k-transcode-metal -p j2k-transcode-cuda --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-transcode --all-features --lib --tests`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`,
  and compile-only `cargo check -p j2k-transcode -p j2k-transcode-metal --all-features --benches`.
  The duplicate root re-export `j2k_core::GpuAbi` was then removed while
  preserving the intended public `j2k_core::accelerator::GpuAbi` path used by
  Metal/CUDA support. Workspace users were retargeted to the module path, and
  `architecture_policy.rs` now prevents the duplicate root path from returning in
  the rendered stable API. The stable API snapshot was regenerated to 538,876
  bytes / 1,686 `pub fn` entries. This follow-up was checked with
  `cargo check -p j2k-core -p j2k-metal-support -p j2k-transcode-metal -p j2k-cuda-runtime --all-features --lib --tests`,
  `cargo clippy -p j2k-core -p j2k-metal-support -p j2k-transcode-metal -p j2k-cuda-runtime --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-core --all-features --lib --tests`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`,
  and `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`.
  The CUDA runtime stream/event/preload scaffolding was then narrowed: explicit
  stream creation, event construction, kernel preload metadata, and bundled
  kernel-name inventory now stay crate-internal/test-only while high-level
  timing and copy APIs remain public. `architecture_policy.rs` prevents
  `CudaContext::{create_event,create_stream,preload_kernel_module}`,
  `CudaEvent`, `CudaStream`, `CudaKernelModule`, and `CudaKernelName` from
  returning in the rendered stable API. The stable API snapshot was regenerated
  to 534,503 bytes / 1,675 `pub fn` entries, and the rendered
  `CudaContext` public-function count is now 99. This follow-up was checked with
  `cargo check -p j2k-cuda-runtime -p j2k-cuda -p j2k-jpeg-cuda -p j2k-transcode-cuda --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda-runtime -p j2k-cuda -p j2k-jpeg-cuda -p j2k-transcode-cuda --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-cuda-runtime --all-features --lib --tests`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`,
  and `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`.
  The xtask adoption-stack isolation was checked with
  `cargo check -p xtask --bins --tests`,
  `cargo check -p xtask --all-features --bins --tests`,
  `cargo clippy -p xtask --bins --tests -- -D warnings`,
  `cargo clippy -p xtask --all-features --bins --tests -- -D warnings`,
  `cargo test -p xtask --bins --tests`,
  `cargo test -p xtask --all-features --bins --tests`, and
  `cargo test -p xtask --test repo_lint xtask_adoption_stack_is_feature_gated -- --nocapture`.
  The backend error constructor API shrink, including removal of the accidental
  public `J2kError::adapter_backend` helper, was checked with
  `cargo check -p j2k --all-features --lib --bins --tests`,
  `cargo check -p j2k-metal --all-features --lib --bins --tests`,
  `cargo check -p j2k-cuda --all-features --lib --bins --tests`,
  `cargo clippy -p j2k --all-features --lib --bins --tests -- -D warnings`,
  `cargo clippy -p j2k -p j2k-cuda -p j2k-metal --all-features --lib --bins --tests -- -D warnings`,
  `cargo test -p j2k --all-features --lib --tests`,
  `cargo test -p j2k -p j2k-cuda -p j2k-metal --all-features --lib --tests`,
  `cargo xtask stable-api`, and
  `cargo test -p xtask --test repo_lint -- --nocapture`. The CUDA/Metal native decode error mappers were
  checked with `cargo check -p j2k-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-metal --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-metal --all-features --lib error::tests -- --nocapture`,
  `cargo check -p j2k-cuda --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda --all-features --lib --tests -- -D warnings`, and
  `cargo test -p j2k-cuda --all-features --lib error::tests -- --nocapture`.
  The adaptive-route public API shrink was checked with
  `cargo check -p j2k --all-features --lib --bins --tests`,
  `cargo clippy -p j2k --all-features --lib --bins --tests -- -D warnings`,
  `cargo test -p j2k --all-features --lib --tests`, `cargo xtask stable-api`,
  and
  `cargo test -p xtask --test repo_lint adaptive_route_policy_model_stays_out_of_public_api -- --nocapture`.
  The transcode Auto-threshold rationale guard was checked with
  `cargo check -p j2k-transcode-cuda --all-features --lib --tests`,
  `cargo check -p j2k-transcode-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-transcode-cuda --all-features --lib --tests -- -D warnings`,
  `cargo clippy -p j2k-transcode-metal --all-features --lib --tests -- -D warnings`,
  and
  `cargo test -p xtask --test repo_lint transcode_gpu_auto_threshold_policy_is_documented -- --nocapture`.
  The shared transcode stage-counter guard was checked with
  `cargo test -p xtask --test repo_lint transcode_stage_counters_are_shared_between_gpu_adapters -- --nocapture`.
  The shrink-factor and progressive inspect/decode guards were checked with
  `cargo test -p j2k-native --all-features --lib checked_image_dimensions_reject_shrink_factor_overflow -- --nocapture`,
  `cargo test -p j2k-jpeg --all-features --test inspect inspect_and_decoder_info_agree_for_progressive_fixtures -- --nocapture`,
  `cargo test -p xtask --test repo_lint decode_capability_correctness_regressions_are_guarded -- --nocapture`,
  `cargo clippy -p xtask --all-features --bins --tests -- -D warnings`, and
  full `cargo test -p xtask --test repo_lint -- --nocapture`.
  The fixture-compare type/comparator/gate splits and ratchet were checked with
  `cargo check -p j2k-compare --all-features --lib --bins --tests`,
  `cargo clippy -p j2k-compare --all-features --lib --bins --tests -- -D warnings`,
  `cargo test -p j2k-compare --all-features --lib --tests`,
  `cargo test -p j2k-compare --all-features --test bench_harness fixture_compare_binary_exposes_fair_fixture_matrix -- --nocapture`,
  `cargo test -p xtask --test repo_lint compare_bins_use_library_common_helpers -- --nocapture`,
  `cargo test -p xtask --test repo_lint -- --nocapture`, and
  `cargo clippy -p xtask --all-features --bins --tests -- -D warnings`.
  The follow-up `BatchInputs` move into `fixture_compare/types.rs` lowered
  `fixture_compare.rs` to 2,276 lines and was checked with
  `cargo check -p j2k-compare --all-features --lib --bins --tests`,
  `cargo clippy -p j2k-compare --all-features --lib --bins --tests -- -D warnings`,
  `cargo test -p j2k-compare --all-features --lib --tests -- --nocapture`, and
  full `cargo test -p xtask --test repo_lint -- --nocapture` (131 passed).
  The transcode CPU DWT constant guard was checked with
  `cargo test -p xtask --test repo_lint wavelet_and_idct_constants_use_codec_math_sources -- --nocapture`.
  The component-plane accessor unification was checked with
  `cargo check -p j2k-native -p j2k --all-features --lib --tests`,
  `cargo clippy -p j2k-native -p j2k --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-native -p j2k --all-features --lib --tests`,
  `cargo test -p xtask --test repo_lint component_plane_metadata_accessors_are_shared -- --nocapture`,
  and `cargo xtask stable-api`.
  The JPEG cache FNV-1a digest unification was checked with
  `cargo check -p j2k-core -p j2k-jpeg -p j2k-jpeg-cuda -p j2k-jpeg-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-core -p j2k-jpeg -p j2k-jpeg-cuda -p j2k-jpeg-metal --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-jpeg -p j2k-jpeg-cuda -p j2k-jpeg-metal --all-features --lib --tests`,
  `cargo test -p xtask --test repo_lint jpeg_cache_digests_use_shared_fnv1a64_helpers -- --nocapture`,
  and `cargo xtask stable-api`.
  The CUDA/Metal canonical Huffman derivation guard was checked with
  `cargo test -p xtask --test repo_lint jpeg_metal_huffman_derivation_uses_shared_entropy_canonical_tables -- --nocapture`.
  The shared JPEG color fast-packet accessor guard was checked with
  `cargo test -p xtask --test repo_lint jpeg_fast_packet_accessors_live_in_shared_jpeg_adapter -- --nocapture`.
  The GPU decoder CPU-host facade guard was checked with
  `cargo test -p xtask --test repo_lint gpu_decoder_cpu_host_facades_use_core_blanket_impl -- --nocapture`.
  The HT code-block scalar fallback guard was checked with
  `cargo test -p xtask --test repo_lint ht_code_block_scalar_fallback_lives_in_trait_default -- --nocapture`.
  The const-array-derived repo-lint hardening was checked with
  `cargo test -p xtask --test repo_lint publish -- --nocapture`,
  `cargo test -p xtask --test repo_lint ci_workflow_runs_semver_checks_for_stable_library_crates -- --nocapture`,
  and `cargo clippy -p xtask --all-features --bins --tests -- -D warnings`.
  The semver-visible `TranscodePipelineMap::debug_report` diagnostic string
  helper was removed; tests, examples, and benchmark source now use structured
  `TranscodePipelineMap` fields directly. This public API slice regenerated
  `docs/stable-api-1.0.public-api.txt` to 524,325 bytes / 1,604 `pub fn`
  entries and was checked with `cargo fmt --all --check`,
  `cargo check -p j2k-transcode -p j2k-transcode-metal --all-features --lib --bins --examples --tests`,
  `cargo clippy -p j2k-transcode -p j2k-transcode-metal --all-features --lib --bins --examples --tests -- -D warnings`,
  `cargo test -p j2k-transcode --all-features --test jpeg_to_htj2k -- --nocapture`,
  `cargo test -p j2k-transcode-metal --all-features --test route_report -- --nocapture`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`, and
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`.
  The duplicate public `j2k::adapter::device_plan` path was then made internal
  while preserving the shared request normalizer as root
  `j2k::{DeviceDecodePlan, DeviceDecodeRequest}` exports for first-party GPU
  adapters. This public API slice regenerated
  `docs/stable-api-1.0.public-api.txt` to 523,851 bytes / 1,604 `pub fn`
  entries and was checked with `cargo fmt --all --check`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`,
  `cargo test -p j2k --test device_plan -- --nocapture`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`,
  `cargo check -p j2k -p j2k-cuda -p j2k-metal --all-features --lib --tests`,
  `cargo clippy -p j2k -p j2k-cuda -p j2k-metal --all-features --lib --tests -- -D warnings`,
  and `git diff --check`.
  The CUDA transcode 9/7 batch request APIs were then flattened so
  `CudaDwt97BatchWithPoolRequest` owns `blocks`/`geometry`/`pool` directly and
  `CudaHtj2k97CodeblockBatchWithPoolRequest` owns
  `blocks`/`geometry`/`params`/`pool` directly; the now-duplicate public inner
  `CudaDwt97BatchRequest` and `CudaHtj2k97CodeblockBatchRequest` types were
  removed from the rendered API and guarded from returning. This public API
  slice regenerated `docs/stable-api-1.0.public-api.txt` to 523,540 bytes /
  1,604 `pub fn` entries and was checked with
  `cargo check -p j2k-cuda-runtime -p j2k-transcode-cuda --all-features --lib --tests`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`,
  `cargo clippy -p j2k-cuda-runtime -p j2k-transcode-cuda --all-features --lib --tests -- -D warnings`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`,
  `cargo test -p j2k-cuda-runtime --all-features --lib transcode -- --nocapture`,
  `cargo fmt --all --check`, and `git diff --check`.
  The duplicate public
  `CudaContext::decode_htj2k_codeblocks_cleanup_multi_with_resources_and_pool`
  wrapper was then removed in favor of
  `decode_htj2k_codeblocks_cleanup_multi_with_resources_and_pool_timed(..., false)`,
  which preserves the status readback and execution evidence path. This public
  API slice regenerated `docs/stable-api-1.0.public-api.txt` to 523,213 bytes /
  1,603 `pub fn` entries and was checked with
  `cargo check -p j2k-cuda-runtime --all-features --lib --tests`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`,
  `cargo clippy -p j2k-cuda-runtime --all-features --lib --tests -- -D warnings`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`,
  `cargo test -p j2k-cuda-runtime --all-features --lib empty -- --nocapture`,
  `cargo fmt --all --check`, and `git diff --check`.
  No benchmark executable was run.
  The CUDA HTJ2K single-codeblock encode wrappers were then removed from the
  stable API. The external CUDA kernel parity test now exercises the same
  cleanup encoder through `encode_htj2k_codeblocks` with one job, and the
  runtime resource-backed regression uses
  `encode_htj2k_codeblocks_with_resources` with one job. The dead internal
  single-codeblock launch wrapper, private parameter ABI struct, and byte
  helpers were removed, while the bundled kernel registry entry remains covered
  by existing kernel-manifest tests. This public API slice regenerated
  `docs/stable-api-1.0.public-api.txt` to 522,732 bytes / 1,601 `pub fn`
  entries and was checked with
  `cargo fmt --all --check`,
  `cargo check -p j2k-cuda-runtime -p j2k-cuda --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda-runtime -p j2k-cuda --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-cuda-runtime --all-features --lib htj2k_encode_resources_feed_one_job_batch_encode_when_required -- --nocapture`,
  `cargo test -p j2k-cuda --all-features --test htj2k_cuda_kernels cuda_htj2k_encode_kernel_matches_native_scalar_codeblock_when_required -- --nocapture`,
  `cargo xtask stable-api --write`,
  `cargo xtask stable-api`, and
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`.
  The CUDA execution portions of the two targeted tests were skipped by the
  local `J2K_REQUIRE_CUDA_RUNTIME` gate on this macOS host. No benchmark
  executable was run.
  The CUDA HTJ2K resident encode implicit-pool wrappers
  `encode_htj2k_codeblocks_resident_with_resources` and
  `encode_htj2k_codeblock_regions_resident_with_resources` were then removed
  from the stable API. The public table-upload wrappers now call the explicit
  `_and_pool` APIs internally, production CUDA encode paths pass a
  `CudaBufferPool` explicitly, and benchmark source was updated to the explicit
  API without running benchmark executables. This public API slice regenerated
  `docs/stable-api-1.0.public-api.txt` to 522,057 bytes / 1,599 `pub fn`
  entries and was checked with
  `cargo fmt --all --check`,
  `cargo check -p j2k-cuda-runtime -p j2k-cuda --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda-runtime -p j2k-cuda --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-cuda-runtime --all-features --lib htj2k_encode_tables_feed_resident_region_encode_when_required -- --nocapture`,
  `cargo test -p j2k-cuda --all-features --lib cuda_resident_quantized_subband_feeds_resident_ht_batch_when_runtime_required -- --nocapture`,
  `cargo xtask stable-api --write`,
  `cargo xtask stable-api`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`, and
  `git diff --check`.
  The CUDA execution portions of the two targeted tests were skipped by the
  local `J2K_REQUIRE_CUDA_RUNTIME` gate on this macOS host. No benchmark
  executable was run.
  The CUDA HTJ2K simple decode `with_resources`/`with_resources_and_pool`
  public wrappers and the no-caller cleanup decode wrapper were then removed
  from the stable API. The public `decode_htj2k_codeblocks` table-upload path
  now uses the resource-backed helper internally, and the resource-backed empty
  decode regression remains as a crate-local unit test. This public API slice
  regenerated `docs/stable-api-1.0.public-api.txt` to 521,127 bytes / 1,596
  `pub fn` entries and was checked with
  `cargo fmt --all --check`,
  `cargo check -p j2k-cuda-runtime --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda-runtime --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-cuda-runtime --all-features --lib htj2k_empty_codeblock_decode_with_resources_zero_fills_when_required -- --nocapture`,
  `cargo xtask stable-api --write`,
  `cargo xtask stable-api`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`, and
  `git diff --check`.
  The CUDA execution portion of the targeted runtime test was skipped by the
  local `J2K_REQUIRE_CUDA_RUNTIME` gate on this macOS host. No benchmark
  executable was run.
  The duplicate public CUDA HTJ2K multi-buffer dequantize non-pool wrapper was
  then removed from the stable API in favor of
  `j2k_dequantize_htj2k_codeblocks_multi_device_with_pool`; production already
  used the explicit pool path, and the runtime regression now does the same.
  This public API slice regenerated
  `docs/stable-api-1.0.public-api.txt` to 520,894 bytes / 1,595 `pub fn`
  entries and was checked with
  `cargo fmt --all --check`,
  `cargo check -p j2k-cuda-runtime --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda-runtime --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-cuda-runtime --all-features --lib j2k_dequantize_htj2k_codeblocks_multi_uses_one_dispatch_when_runtime_required -- --nocapture`,
  `cargo xtask stable-api --write`,
  `cargo xtask stable-api`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`, and
  `git diff --check`.
  A first attempted targeted test filter matched zero tests and was discarded;
  the exact test above was then run. Its CUDA execution portion was skipped by
  the local `J2K_REQUIRE_CUDA_RUNTIME` gate on this macOS host. No benchmark
  executable was run.
  The duplicate public CUDA HTJ2K cleanup packetization no-tag wrapper was then
  removed in favor of
  `CudaContext::packetize_htj2k_cleanup_packets_with_tag_state`; tests that
  exercise stateless packetization now pass empty tag-state slices explicitly.
  This public API slice regenerated
  `docs/stable-api-1.0.public-api.txt` to 520,561 bytes / 1,594 `pub fn`
  entries and was checked with
  `cargo fmt --all --check`,
  `cargo check -p j2k-cuda-runtime -p j2k-cuda --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda-runtime -p j2k-cuda --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-cuda --all-features --test htj2k_cuda_kernels cuda_htj2k_packetization_kernel_matches_native_scalar_cleanup_packet_when_required -- --nocapture`,
  `cargo xtask stable-api --write`,
  `cargo xtask stable-api`,
  `cargo test -p xtask --test repo_lint -- --nocapture`, and
  `git diff --check`.
  A first attempted packetization test filter matched zero tests and was
  discarded; the exact cleanup-packet test above was then run. Its CUDA
  execution portion was skipped by the local `J2K_REQUIRE_CUDA_RUNTIME` gate on
  this macOS host. No benchmark executable was run.
  The duplicate public `j2k_core::passthrough` implementation-module path was
  then made private while preserving the root facade exports
  `j2k_core::{CompressedPayloadKind,CompressedTransferSyntax,
  PassthroughCandidate,PassthroughDecision,PassthroughRejectReason,
  PassthroughRequirements}`. The rendered stable API still references those
  root-exported types by their defining module path inside downstream
  signatures, but the public module entry and module-owned functions are gone
  and guarded from returning. This public API slice regenerated
  `docs/stable-api-1.0.public-api.txt` to 513,698 bytes / 1,592 `pub fn`
  entries, with `j2k-core` down to 179 rendered public functions, and was
  checked with `cargo fmt --all --check`,
  `cargo check -p j2k-core -p j2k -p j2k-jpeg --all-features --lib --tests`,
  `cargo clippy -p j2k-core -p j2k -p j2k-jpeg --all-features --lib --tests -- -D warnings`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`, and
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`.
  No benchmark executable was run.
  The duplicate public `j2k_core::row_sink` and `j2k_core::scratch`
  implementation-module paths were then made private while preserving root
  facade exports `j2k_core::{RowSink,ScratchPool}`. This removed their
  duplicate module-owned methods from the rendered API while keeping downstream
  root-path usage intact. This public API slice regenerated
  `docs/stable-api-1.0.public-api.txt` to 513,089 bytes / 1,589 `pub fn`
  entries, with `j2k-core` down to 176 rendered public functions, and was
  checked with `cargo fmt --all --check`,
  `cargo check -p j2k-core -p j2k -p j2k-jpeg --all-features --lib --tests`,
  `cargo clippy -p j2k-core -p j2k -p j2k-jpeg --all-features --lib --tests -- -D warnings`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`, and
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`.
  No benchmark executable was run.
  The duplicate public `j2k_core::pixel`, `j2k_core::sample`, and
  `j2k_core::scale` implementation-module paths were then made private while
  preserving root facade exports
  `j2k_core::{PixelFormat,PixelLayout,Sample,SampleType,Downscale}`. This
  removed the duplicate public module entries and `Sample` trait associated
  constants from the rendered API while keeping existing root-path usage
  intact. This public API slice regenerated
  `docs/stable-api-1.0.public-api.txt` to 510,244 bytes / 1,589 `pub fn`
  entries, with `j2k-core` still at 176 rendered public functions by the
  prefix-count method used above, and was checked with
  `cargo check -p j2k-core -p j2k -p j2k-jpeg --all-features --lib --tests`,
  `cargo clippy -p j2k-core -p j2k -p j2k-jpeg --all-features --lib --tests -- -D warnings`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`, and
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`.
  No benchmark executable was run.
  The duplicate public `j2k_core::error` implementation-module path was then
  made private while preserving root facade exports for `BufferError`,
  `CodecError`, `InputError`, `NotImplemented`, `Unsupported`,
  `AdapterErrorKind`, `AdapterErrorParts`, and the adapter classification
  helpers. Public impl/signature lines in downstream crates may still mention
  the defining private module path in the generated snapshot, but the public
  module entry and module-owned functions are gone and guarded from returning.
  This public API slice regenerated `docs/stable-api-1.0.public-api.txt` to
  506,932 bytes / 1,579 `pub fn` entries, with `j2k-core` down to 166 rendered
  public functions by the prefix-count method used above, and was checked with
  `cargo check -p j2k-core -p j2k -p j2k-jpeg -p j2k-cuda -p j2k-jpeg-cuda -p j2k-metal -p j2k-jpeg-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-core -p j2k -p j2k-jpeg -p j2k-cuda -p j2k-jpeg-cuda -p j2k-metal -p j2k-jpeg-metal --all-features --lib --tests -- -D warnings`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`, and
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`.
  No benchmark executable was run.
  The duplicate public `j2k_core::{backend,batch,context,device,traits,types}`
  implementation-module paths were then made private while preserving the root
  facade exports for backend selection, batch jobs, context/cache contracts,
  device request validation, codec traits, and metadata/geometry types.
  `j2k_core::accelerator` stays public because `GpuAbi` is an intentional
  namespaced ABI contract used by first-party GPU crates. Generated downstream
  impl/signature lines may still mention defining private module paths for
  root-exported types, but the public module entries and module-owned functions
  are gone and guarded from returning. This public API slice regenerated
  `docs/stable-api-1.0.public-api.txt` to 479,554 bytes / 1,503 `pub fn`
  entries, with `j2k-core` down to 98 rendered public functions by the
  prefix-count method used above, and was checked with
  `cargo check -p j2k-core -p j2k -p j2k-jpeg -p j2k-cuda -p j2k-jpeg-cuda -p j2k-metal -p j2k-jpeg-metal -p j2k-transcode -p j2k-transcode-metal -p j2k-transcode-cuda -p j2k-tilecodec --all-features --lib --tests`,
  `cargo clippy -p j2k-core -p j2k -p j2k-jpeg -p j2k-cuda -p j2k-jpeg-cuda -p j2k-metal -p j2k-jpeg-metal -p j2k-transcode -p j2k-transcode-metal -p j2k-transcode-cuda -p j2k-tilecodec --all-features --lib --tests -- -D warnings`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`, and
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`.
  No benchmark executable was run.
  The JPEG baseline GPU encode adapter helpers
  `validate_jpeg_baseline_gpu_encode_tile`,
  `jpeg_baseline_gpu_encode_params`,
  `jpeg_baseline_gpu_encode_tile_plan`,
  `jpeg_baseline_gpu_encode_batch_plan`,
  `jpeg_baseline_gpu_entropy_capacity_bytes`,
  `jpeg_baseline_entropy_capacity_bytes`,
  `jpeg_baseline_sampling_for`, `same_source_buffer_batch_end`, and the
  baseline validation helpers were then made crate-private/private while
  preserving the public shared GPU encode trait/types and
  `encode_jpeg_baseline_gpu_{tile,batch}` entrypoints used by CUDA and Metal
  adapters. This public API slice regenerated
  `docs/stable-api-1.0.public-api.txt` to 477,368 bytes / 1,493 `pub fn`
  entries, with `j2k-jpeg` down to 277 rendered public functions by the
  prefix-count method used above, and was checked with
  `cargo check -p j2k-jpeg -p j2k-jpeg-cuda -p j2k-jpeg-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg -p j2k-jpeg-cuda -p j2k-jpeg-metal --all-features --lib --tests -- -D warnings`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`, and
  `cargo test -p xtask --test repo_lint jpeg_gpu_encode_host_orchestration_uses_shared_adapter_helper -- --nocapture`.
  No benchmark executable was run.
  The unused public CUDA runtime wrapper
  `CudaContext::upload_htj2k_decode_resources` was then made private; callers
  either use `decode_htj2k_codeblocks` for the simple table-upload path or the
  explicit table-resource reuse path used by first-party adapters at that
  point. A later public API slice hides those reusable resource APIs from the
  rendered 1.0 inventory while preserving first-party adapter access. This public
  API slice regenerated `docs/stable-api-1.0.public-api.txt` to 477,144 bytes /
  1,492 `pub fn` entries, with `j2k-cuda-runtime` down to 220 rendered public
  functions by the prefix-count method used above, and was checked with
  `cargo check -p j2k-cuda-runtime -p j2k-cuda --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda-runtime -p j2k-cuda --all-features --lib --tests -- -D warnings`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`, and
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`.
  No benchmark executable was run.
  The duplicate public `j2k::adapter::encode_stage` module path was then made
  internal while preserving the encode-stage contracts at the `j2k` root
  facade for first-party GPU crates and downstream adapters. This public API
  slice regenerated `docs/stable-api-1.0.public-api.txt` to 475,938 bytes /
  1,492 `pub fn` entries and was checked with
  `cargo check -p j2k -p j2k-metal -p j2k-cuda -p j2k-transcode --all-features --lib --tests`,
  `cargo clippy -p j2k -p j2k-metal -p j2k-cuda -p j2k-transcode --all-features --lib --tests -- -D warnings`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`,
  `cargo fmt --all --check`, and `git diff --check`.
  No benchmark executable was run.
  The unused public CUDA surface profiling convenience
  `Surface::download_into_profiled` was then removed; callers use
  `Surface::download_into`, while profiling remains available through the
  explicit CUDA profile report APIs. This public API slice regenerated
  `docs/stable-api-1.0.public-api.txt` to 475,817 bytes / 1,491 `pub fn`
  entries, with `j2k-cuda` down to 117 rendered public functions by the
  prefix-count method used above, and was checked with
  `cargo check -p j2k-cuda --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda --all-features --lib --tests -- -D warnings`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`, and
  `cargo fmt --all --check`. The broad non-benchmark workspace
  `cargo check --workspace --all-features --lib --bins --examples --tests` and
  `cargo clippy --workspace --all-features --lib --bins --examples --tests -- -D warnings`
  also pass after this slice, along with `cargo run -p xtask -- unsafe-audit`,
  `cargo xtask panic-surface`, `cargo xtask semver`, `cargo deny check`, and
  `cargo machete`.
  No benchmark executable was run.
  The CUDA grayscale HTJ2K plan-builder profiling hooks were then made
  crate-private. The former integration plan-shape tests were moved to
  crate-local tests, and the two decode-kernel parity tests that needed the
  internal flat plan moved with them; the remaining public CUDA kernel
  integration tests stay in `tests/htj2k_cuda_kernels.rs`. This public API
  slice regenerated `docs/stable-api-1.0.public-api.txt` to 474,811 bytes /
  1,487 `pub fn` entries, with `j2k-cuda` down to 113 rendered public
  functions by the prefix-count method used above, and was checked with
  `cargo check -p j2k-cuda --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-cuda --all-features --lib htj2k_plan_tests -- --nocapture`,
  `cargo test -p j2k-cuda --all-features --test htj2k_cuda_kernels -- --nocapture`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`, and
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`.
  No benchmark executable was run.
  The duplicate JPEG adapter packet-builder convenience wrappers
  `build_fast420_packet_for_decoder`, `build_fast422_packet_for_decoder`,
  `build_fast444_packet_for_decoder`, and `build_gray_packet_for_decoder` were
  then removed from the semver-visible API. First-party JPEG Metal callers now
  combine `j2k_jpeg::adapter::decoder_bytes(decoder)` with the byte-slice
  packet builders directly. This public API slice regenerated
  `docs/stable-api-1.0.public-api.txt` to 474,085 bytes / 1,483 `pub fn`
  entries, with `j2k-jpeg` down to 273 rendered public functions by the
  prefix-count method used above, and was checked with
  `cargo check -p j2k-jpeg -p j2k-jpeg-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg -p j2k-jpeg-metal --all-features --lib --tests -- -D warnings`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`,
  `cargo test -p j2k-jpeg --all-features --test fast_packet -- --nocapture`,
  `cargo test -p j2k-jpeg-metal --all-features --lib routing::tests -- --nocapture`,
  and
  `cargo test -p j2k-jpeg-metal --all-features --lib viewport::tests -- --nocapture`.
  No benchmark executable was run.
  The JPEG decoder custom-alpha convenience wrappers
  `Decoder::decode_rgba8_into_with_alpha` and
  `Decoder::decode_region_rgba8_into_with_alpha` were then removed from the
  semver-visible API. The public `PixelFormat::Rgba8` path still provides the
  supported default alpha behavior, and crate-local decoder tests now cover the
  underlying custom-alpha full-image and region `OutputFormat::Rgba8 { alpha }`
  paths directly. This public API slice regenerated
  `docs/stable-api-1.0.public-api.txt` to 473,746 bytes / 1,481 `pub fn`
  entries, and was checked with
  `cargo check -p j2k-jpeg --all-features --lib --tests`,
  `cargo test -p j2k-jpeg --all-features --lib decode_into_output_format_writes_custom_rgba_alpha -- --nocapture`,
  `cargo test -p j2k-jpeg --all-features --lib decode_region_output_format_writes_custom_rgba_alpha -- --nocapture`,
  `cargo clippy -p j2k-jpeg --all-features --lib --tests -- -D warnings`,
  `cargo test -p j2k-jpeg --all-features --test decode_into -- --nocapture`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`, and
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`.
  No benchmark executable was run.
  The redundant JPEG decoder native-region scratch wrapper
  `Decoder::decode_region_into_with_scratch` was then made crate-private.
  The one external production caller in `j2k-jpeg-metal` now calls
  `decode_region_scaled_into_with_scratch(..., Downscale::None)` explicitly,
  and crate-local trait/tile decode glue keeps using the private helper. This
  public API slice regenerated `docs/stable-api-1.0.public-api.txt` to 473,515
  bytes / 1,480 `pub fn` entries, with `j2k-jpeg` down to 272 rendered public
  functions by the prefix-count method used above, and was checked with
  `cargo check -p j2k-jpeg -p j2k-jpeg-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg -p j2k-jpeg-metal --all-features --lib --tests -- -D warnings`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`,
  and `cargo test -p j2k-jpeg-metal --all-features --lib -- --nocapture`.
  No benchmark executable was run.
  The public `j2k_jpeg::adapter::JpegColorFastPacket` trait and its
  semver-visible trait-impl accessors were then removed. Public fast-packet
  structs still expose their fields and packet builders, while CUDA uses a
  private concrete-plan macro and Metal uses its existing private
  `FastSubsampledPacket` bound for generic fast-path code. This regenerated
  `docs/stable-api-1.0.public-api.txt` to 462,262 bytes / 1,368 `pub fn`
  entries, with `j2k-jpeg` down to 160 rendered public functions. Checks:
  `cargo check -p j2k-jpeg -p j2k-jpeg-cuda -p j2k-jpeg-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg -p j2k-jpeg-cuda -p j2k-jpeg-metal --all-features --lib --tests -- -D warnings`,
  `cargo xtask stable-api --write`, `cargo xtask stable-api`,
  `cargo test -p xtask --test repo_lint jpeg_fast_packet_accessors_stay_out_of_public_api -- --nocapture`,
  `cargo test -p j2k-jpeg --all-features --test fast_packet -- --nocapture`,
  `cargo test -p j2k-jpeg-cuda --all-features --lib -- --nocapture`, and
  `cargo test -p j2k-jpeg-metal --all-features --lib fast -- --nocapture`.
  No benchmark executable was run.
  The experimental CUDA JPEG chunked-entropy diagnostic facade was then marked
  `#[doc(hidden)]`, matching the existing policy for adapter/benchmark support
  exports that must compile but should not enter the 1.0 stable API inventory.
  Repo lint now forbids
  `j2k_jpeg_cuda::Codec::diagnose_tile_rgb8_chunked_entropy_with_session`
  from returning in `docs/stable-api-1.0.public-api.txt`. This regenerated the
  snapshot to 461,991 bytes / 1,367 `pub fn` entries, with `j2k-jpeg-cuda` down
  to 55 rendered public functions. Checks:
  `cargo public-api -p j2k-jpeg-cuda --all-features -sss --color never`,
  `cargo fmt --all --check`, `cargo xtask stable-api --write`,
  `cargo xtask stable-api`, `cargo check -p j2k-jpeg-cuda --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg-cuda --all-features --lib --tests -- -D warnings`,
  and
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`.
  No benchmark executable was run.
  The underlying experimental CUDA runtime chunked-entropy diagnostic cluster
  was then hidden from the 1.0 inventory as well:
  `CudaJpegChunkedEntropy{Config,Plan,Report}`,
  `CudaJpegEntropy{Sync,Overflow}State`, and
  `CudaContext::diagnose_jpeg_420_entropy_self_sync`. These remain available
  to first-party diagnostics and compile-only benchmark support but are no
  longer presented as stable 1.0 API. The snapshot was regenerated to 457,645
  bytes / 1,358 `pub fn` entries, with `j2k-cuda-runtime` down to 211 rendered
  public functions. Checks:
  `cargo public-api -p j2k-cuda-runtime --all-features -sss --color never`,
  `cargo fmt --all --check`, `cargo xtask stable-api --write`,
  `cargo xtask stable-api`,
  `cargo check -p j2k-cuda-runtime -p j2k-jpeg-cuda --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda-runtime -p j2k-jpeg-cuda --all-features --lib --tests -- -D warnings`,
  and
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`.
  No benchmark executable was run.
  The CUDA runtime pool-trace, NVTX/timing wrapper, and copy-kernel upload
  support surfaces were then hidden from the 1.0 inventory:
  `CudaBufferPool::take_with_trace`, `CudaBufferPoolTakeTrace`,
  `CudaContext::copy_with_kernel`, `CudaContext::with_nvtx_range`, and
  `CudaContext::time_default_stream_named_us{,_if}`. The test-only pooled i16
  upload helpers `CudaBufferPool::upload_i16{,_pinned}` were hidden as a
  follow-up. They remain callable by
  first-party CUDA adapter crates, but no longer appear in the stable API
  snapshot. This regenerated the snapshot to 455,845 bytes / 1,351 `pub fn`
  entries, with `j2k-cuda-runtime` down to 204 rendered public functions.
  Checks: `cargo xtask stable-api --write`, `cargo public-api -p j2k-cuda-runtime --all-features -sss --color never`,
  `cargo fmt --all --check`, `cargo xtask stable-api`,
  `cargo check -p j2k-cuda-runtime -p j2k-cuda -p j2k-jpeg-cuda --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda-runtime -p j2k-cuda -p j2k-jpeg-cuda --all-features --lib --tests -- -D warnings`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`,
  and `cargo clippy -p xtask --all-features --test repo_lint -- -D warnings`.
  No benchmark executable was run.
  The CUDA runtime queued-execution support surface was then hidden from the
  1.0 inventory: `CudaQueuedExecution`, `CudaQueuedHtj2kCleanup`,
  `CudaContext::j2k_inverse_dwt_batch_device_enqueue_with_pool`,
  `CudaContext::j2k_inverse_dwt_batch_sequence_enqueue_with_pool`,
  `CudaContext::decode_htj2k_codeblocks_cleanup_multi_enqueue_with_resources_and_pool`,
  and `CudaContext::j2k_dequantize_queued_htj2k_cleanup_with_pool`. These APIs
  remain available for first-party CUDA scheduling and cleanup plumbing, but
  no longer appear in the rendered stable API. This regenerated the snapshot to
  454,041 bytes / 1,342 `pub fn` entries, with `j2k-cuda-runtime` down to 195
  rendered public functions. Checks: `cargo xtask stable-api --write`,
  `cargo xtask stable-api`, `cargo fmt --all --check`,
  `cargo check -p j2k-cuda-runtime -p j2k-cuda --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda-runtime -p j2k-cuda --all-features --lib --tests -- -D warnings`,
  and
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`.
  No benchmark executable was run.
  The raw CUDA HTJ2K packetization ABI surface was then hidden from the 1.0
  inventory: `CudaHtj2kPacketization{Packet,Subband,SubbandTagState,TagNodeState,Block,Status,StageTimings}`,
  `CudaHtj2kPacketizedTile`, and
  `CudaContext::packetize_htj2k_cleanup_packets_with_tag_state`. These types
  and the kernel entrypoint remain available to first-party `j2k-cuda`
  packetization paths and parity tests, but no longer appear in the rendered
  stable API. This regenerated the snapshot to 449,423 bytes / 1,336 `pub fn`
  entries, with `j2k-cuda-runtime` down to 189 rendered public functions.
  Checks: `cargo xtask stable-api --write`, `cargo xtask stable-api`,
  `cargo fmt --all --check`,
  `cargo check -p j2k-cuda-runtime -p j2k-cuda --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda-runtime -p j2k-cuda --all-features --lib --tests -- -D warnings`,
  and
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`.
  No benchmark executable was run.
  The raw CUDA transcode runtime surface was then hidden from the 1.0
  inventory: reversible/9/7 band structs, 9/7 batch geometry and request
  structs, 9/7 quantize params, host/device code-block band structs, and the
  direct `CudaContext::j2k_transcode_*` runtime entrypoints. They remain
  source-visible to first-party CUDA transcode glue and runtime tests, but no
  longer appear in `docs/stable-api-1.0.public-api.txt`. This regenerated the
  snapshot to 442,734 bytes / 1,330 `pub fn` entries, with
  `j2k-cuda-runtime` down to 183 rendered public functions. Checks:
  `cargo xtask stable-api --write`, `cargo xtask stable-api`,
  `cargo fmt --all --check`,
  `cargo check -p j2k-cuda-runtime -p j2k-transcode-cuda --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda-runtime -p j2k-transcode-cuda --all-features --lib --tests -- -D warnings`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`,
  and
  `cargo test -p j2k-cuda-runtime -p j2k-transcode-cuda --all-features --lib --tests -- --nocapture`.
  No benchmark executable was run.
  The raw CUDA J2K encode/decode runtime ABI surface was then hidden from the
  1.0 inventory: direct deinterleave/forward-DWT/MCT/quantize/IDWT/store
  `CudaContext::j2k_*` entrypoints plus the `CudaJ2k*` job/result structs,
  resident DWT outputs, DWT level shape, and local batch-timing structs. They
  remain source-visible to first-party CUDA decode/encode/transcode glue and
  tests, but no longer appear in the rendered stable API. This regenerated the
  snapshot to 421,361 bytes / 1,269 `pub fn` entries, with
  `j2k-cuda-runtime` down to 122 rendered public functions. Checks:
  `cargo xtask stable-api --write`, `cargo xtask stable-api`,
  `cargo fmt --all --check`,
  `cargo check -p j2k-cuda-runtime -p j2k-cuda -p j2k-transcode-cuda --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda-runtime -p j2k-cuda -p j2k-transcode-cuda --all-features --lib --tests -- -D warnings`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`,
  `cargo test -p j2k-cuda-runtime -p j2k-cuda -p j2k-transcode-cuda --all-features --lib --tests -- --nocapture`,
  and `git diff --check`. No benchmark executable was run.
  The JPEG adapter full device-plan/checkpoint planning surface was then hidden
  from the 1.0 inventory: `DeviceDecodePlan`, `DeviceComponentPlan`,
  `DeviceCheckpoint`, `build_device_plan`, and `summarize_device_batch`.
  `DeviceBatchSummary` remains visible because it is embedded in
  `JpegCapabilityReport`. This regenerated the snapshot to
  419,610 bytes / 1,267 `pub fn` entries, with `j2k-jpeg` down to
  158 rendered public functions. Checks:
  `cargo xtask stable-api --write`, `cargo xtask stable-api`,
  `cargo fmt --all --check`,
  `cargo check -p j2k-jpeg -p j2k-jpeg-cuda -p j2k-jpeg-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg -p j2k-jpeg-cuda -p j2k-jpeg-metal --all-features --lib --tests -- -D warnings`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`,
  `cargo test -p j2k-jpeg --all-features --lib --tests -- --nocapture`,
  and `git diff --check`. No benchmark executable was run.
  The JPEG fast-packet raw ABI surface was then hidden from the 1.0 inventory:
  `FastPacketError`, `TableKind`, `JpegCanonicalHuffmanTable`,
  `JpegEntropyCheckpointV1`, `JpegFast{420,422,444}PacketV1`,
  `JpegGrayPacketV1`, `JpegHuffmanTable`, the byte-slice packet builders, and
  `JpegHuffmanTable::derive_canonical`. These remain source-visible to
  first-party CUDA/Metal adapters and tests. This regenerated the snapshot to
  411,333 bytes / 1,260 `pub fn` entries, with `j2k-jpeg` down to
  151 rendered public functions. Checks: `cargo xtask stable-api --write`,
  `cargo xtask stable-api`, `cargo fmt --all --check`,
  `cargo check -p j2k-jpeg -p j2k-jpeg-cuda -p j2k-jpeg-metal --all-features --lib --tests`,
  `cargo clippy -p j2k-jpeg -p j2k-jpeg-cuda -p j2k-jpeg-metal --all-features --lib --tests -- -D warnings`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`,
  `cargo test -p j2k-jpeg --all-features --test fast_packet -- --nocapture`,
  and `git diff --check`. No benchmark executable was run.
  The raw CUDA JPEG runtime ABI surface was then hidden from the 1.0 inventory:
  `CudaJpegRgb8DecodePlan`, `CudaJpegRgb8Sampling`,
  `CudaJpegEntropyCheckpoint`, `CudaJpegHuffmanTable`,
  `CudaJpegBaselineEncodeFormat`, `CudaJpegBaselineEncodeParams`,
  `CudaJpegBaselineEncodeHuffmanTable`,
  `CudaJpegBaselineEntropyEncode{Job,BatchJob}`, and direct
  `CudaContext::{decode_jpeg_rgb8_owned,decode_jpeg_rgb8_owned_into,
  encode_jpeg_baseline_entropy,encode_jpeg_baseline_entropy_batch}` entrypoints.
  These remain source-visible to the first-party JPEG CUDA adapter. This
  regenerated the snapshot to 402,839 bytes / 1,253 `pub fn` entries, with
  `j2k-cuda-runtime` down to 115 rendered public functions. Checks:
  `cargo xtask stable-api --write`, `cargo xtask stable-api`,
  `cargo fmt --all --check`,
  `cargo check -p j2k-cuda-runtime -p j2k-jpeg-cuda --all-features --lib --tests`,
  `cargo clippy -p j2k-cuda-runtime -p j2k-jpeg-cuda --all-features --lib --tests -- -D warnings`,
  `cargo test -p xtask --test repo_lint accidental_test_and_adapter_internals_stay_out_of_public_api -- --nocapture`,
  `cargo test -p j2k-cuda-runtime -p j2k-jpeg-cuda --all-features --lib --tests -- --nocapture`,
  and `git diff --check`. No benchmark executable was run in this slice; the
  later 2026-07-07 performance-evidence pass is recorded in the live status
  above and in `docs/benchmark-evidence.md`.

## 1. Verified-intact areas (do not re-audit without new evidence)

- **Untrusted-input panic surface is clean.** ~95% of the workspace's
  `.expect(`/`.unwrap()` counts sit inside `#[cfg(test)]` modules. Production
  decode paths (marker parsing in `crates/j2k-native/src/j2c/codestream.rs`,
  entropy decode, IDWT/MCT, public APIs in `crates/j2k/src`) contain no
  attacker-reachable panic site found by either pass. Allocation sizing on
  untrusted dimensions uses `checked_mul` plus a 512 MiB cap
  (`crates/j2k-native/src/lib.rs:266-290`, `crates/j2k-native/src/j2c/tile.rs:264-280`,
  `crates/j2k-jpeg/src/decoder.rs:49`). Production `panic!` count: zero.
- **Env-var documentation parity holds.** All code-read `J2K_*` vars appear in
  `docs/env-vars.md` and vice versa; phantom vars are ban-listed by name in
  `xtask/tests/repo_lint.rs` and the lint passes.
- **Adapter error classification is unified.** `CodecError` and the
  `adapter_error_is_*` helpers live in `crates/j2k-core/src/error.rs:104-164`;
  all four GPU adapters delegate to them.
- **Waived mirrored-twin families remain validly waived.** The three waivers in
  `engineering/mirrored-twin-unification.md` (Extended12 vs Progressive12 entropy
  stage, NEON dual vs top_only kernels, native IDWT f32 vs i64) were re-checked
  and hold.
- **`cargo xtask panic-surface` is honestly ratcheted** at 17/17 unwraps and
  98/106 expects with zero `#[allow(clippy::unwrap_used/expect_used)]`
  escapes. It was not wired into CI at audit time; the current working tree
  wires it into both `ci()` and the CI workflow.

## 2. Broken gates (fix before any other work)

All reproduced on the working tree, 2026-07-04:

1. `cargo check --workspace --all-targets --all-features` fails: E0616 —
   tests read private fields `htj2k_tile_attempts`/`htj2k_tile_dispatches` of
   `CudaEncodeStageAccelerator` (`crates/j2k-cuda/src/encode.rs` test module vs
   `crates/j2k-cuda/src/encode/stage.rs:55`).
2. `cargo clippy --workspace --all-targets --all-features -- -D warnings`
   fails first on `derivable_impls` at
   `crates/j2k-core/src/accelerator.rs:40`. When that lint is bypassed at the
   command line, `j2k-jpeg` immediately exposes split-file wildcard imports in
   `crates/j2k-jpeg/src/decoder/{extended12.rs,lossless_helpers.rs,color_convert.rs,core_traits.rs}`
   and matching wildcard imports from `crates/j2k-jpeg/src/decoder.rs`.
3. `cargo xtask stable-api` fails: `docs/stable-api-1.0.public-api.txt` is
   stale relative to the working tree's public API.
4. `cargo test -p xtask --test repo_lint` fails 1 of 113:
   `public_text_does_not_embed_local_user_home_paths`, tripped by an untracked
   `.tmp-metadata.json` cargo-metadata dump at the repo root (not gitignored)
   that embeds local user-home paths.
5. `cargo run -p xtask -- unsafe-audit` fails: `docs/unsafe-audit.md` is
   missing entries for nine files - all new modules created by the uncommitted
   god-file splits:
   `crates/j2k-jpeg/src/decoder/output_format.rs`,
   `crates/j2k-metal/src/surface.rs`,
   `crates/j2k-metal/src/compute/decode_dispatch.rs`,
   `crates/j2k-metal/src/compute/direct_surface_pack.rs`,
   `crates/j2k-metal/src/compute/forward_transform.rs`,
   `crates/j2k-metal/src/compute/lossless_prepare.rs`,
   `crates/j2k-metal/src/compute/resident_codestream.rs`,
   `crates/j2k-metal/src/compute/resident_tier1.rs`, and
   `crates/j2k-metal/src/compute/tier1_encode.rs`. A separate comparison using
   the xtask predicate (`unsafe ` or `unsafe{`) also found stale rows for
   `crates/j2k-jpeg/src/decoder.rs`, `crates/j2k-metal/src/compute.rs`,
   `crates/j2k-metal/src/compute/direct_execute_impl.rs`, and
   `crates/j2k-metal/src/lib.rs`; the current xtask only enforces missing
   entries, not stale entries.
6. `cargo xtask panic-surface` passes but **runs in no CI workflow** and is
   omitted from the `ci()` meta-task (`xtask/src/main.rs:184-191`). The only
   documented panic ratchet never executes automatically. The same
   "exists-but-never-runs" status applies to `machete`, `clippy-strict`,
   `nextest`, and `j2k-perf-guard`.
7. Targeted clippy, after suppressing only the first two workspace blockers at
   the command line, finds production issues in GPU JPEG adapters:
   `j2k-jpeg-cuda` still calls deprecated first-party `j2k_jpeg::Decoder`
   methods at `crates/j2k-jpeg-cuda/src/decoder.rs:56,110,133,157` instead of
   `DecodeRequest`; `j2k-jpeg-metal` then fails on production warnings in
   `crates/j2k-jpeg-metal/src/abi.rs:383`,
   `crates/j2k-jpeg-metal/src/compute/batch_decode_full.rs`, and
   `crates/j2k-jpeg-metal/src/compute/batch_decode_region.rs`
   (`needless_range_loop`, `too_many_arguments`, `needless_pass_by_value`,
   `question_mark`).

Passing at audit time: `cargo fmt --all --check`, `cargo deny check` (duplicate
`weezl` warning), `cargo machete`, `cargo xtask panic-surface` (manual run).
Current dependency evidence supersedes that warning: `cargo tree -d --workspace`
does not report duplicate `weezl`, `cargo tree --workspace -i weezl` shows only
`weezl v0.1.12`, and `cargo tree --workspace -i block@0.1.6` routes
`metal v0.33.0` through the local `third_party/block-0.1.6-patched` override.

**Delivery risk:** the entire remediation refactor (196 files, +11,337/−71,295
lines, including the CI GPU fail-open fixes in `.github/workflows/`) is
uncommitted on `main` since 2026-07-02. Until committed, the published CI
remains the fail-open version and two days of work have no safety net.

## 3. Confirmed correctness findings

1. **Metal fallback routing no longer depends on substring matching.**
   The original `message.contains("unsupported classic kernel input")` routing
   hazard has been remediated in the working tree. Runtime fallback uses the
   typed `Error::MetalDirectFallback` variant and
   `MetalDirectFallbackReason`; direct-plan fallback matches
   `j2k_native::DecodingError::DirectPlanUnsupported(_)` rather than an error
   string. Regression coverage asserts that message-only unsupported text is
   not enough to trigger fallback.
2. **Backend errors no longer use string-only classification.**
   The original `J2kError::Backend(String)` hole has been remediated in the
   working tree: `crates/j2k/src/error.rs` now uses `BackendError` plus
   failure-mode `BackendErrorKind` values (`Other`, `Truncated`,
   `NotImplemented`, `Unsupported`, `Buffer`, `Validation`,
   `InternalInvariant`), and the `CodecError` impl maps classified backend
   failures into `is_truncated`, `is_unsupported`, `is_not_implemented`, and
   `is_buffer_error`. Existing unclassified native/adapter failures remain
   `Other` unless the callsite can prove a stronger classification.
3. **Unchecked shrink-factor arithmetic is remediated.**
   `crates/j2k-native/src/j2c/codestream.rs` now uses checked shift and
   checked multiply for target-resolution shrink factors, then revalidates the
   derived image dimensions. Unit coverage
   `checked_image_dimensions_reject_shrink_factor_overflow` and repo lint
   `decode_capability_correctness_regressions_are_guarded` pin the behavior.
4. **Inspect/decode capability drift risk (JPEG progressive) is guarded.**
   `Decoder::inspect` still performs lightweight metadata inspection, but
   progressive fixture coverage now asserts `Decoder::new(...).info()` matches
   `Decoder::inspect(...)` for 8-bit RGB, 12-bit grayscale, and 12-bit APP14
   RGB progressive inputs. Repo lint
   `decode_capability_correctness_regressions_are_guarded` pins that coverage.
5. **Divergent mirrored copies** (see Section 5) include one
   behavior-relevant divergence: transcode Auto-routing thresholds differ
   between the CUDA and Metal accelerators. The working tree now documents and
   lint-guards the deliberate policy difference: CUDA offers single transform
   jobs above its floor, while Metal keeps single-job reversible 5/3 and 9/7
   Auto disabled and uses batch-first routing plus a staged 9/7 axis cap.

Verified readback claim: Metal readback and CPU-side row/fill access now go
through checked buffer-length helpers except for raw `contents()` calls inside
the central Metal support/direct-buffer/JPEG-buffer helper modules. Repo lint
`metal_raw_buffer_contents_access_stays_confined_to_checked_helpers` scans the
Metal source crates and fails on new raw access outside that allow-list.

Deprioritized: panics/asserts in `crates/j2k-cuda-runtime/build.rs` are the
conventional build-script failure mechanism and stay; the one hygiene item is
that `J2K_CUDA_OXIDE_ARCH` is passed unvalidated into the `cargo oxide build
--arch` invocation (`crates/j2k-cuda-runtime/build.rs:327`). Optional kernel
projects deliberately fail soft by writing a placeholder PTX stub
(`crates/j2k-cuda-runtime/build.rs:391`); document that behavior where the
features are described.

## 4. Adjudicated non-findings (verified safe; do not re-flag)

1. **`crates/j2k-native/src/reader.rs:173`** — `read_byte().unwrap()` inside
   `read_marker` is locally guaranteed: `peek_byte().ok_or(...)?` on the line
   above proves the byte exists and both use the same `byte_pos` base. The
   second marker byte (the real truncation case) is handled with `ok_or(...)?`.
2. **`crates/j2k-native/src/color.rs:344-420` interleave slicing** — verified
   by empirical repro: an 8×8 three-component image with per-component SIZ
   sampling factors (1,1)/(2,2)/(2,2) was encoded with
   `encode_j2k_lossless_components` and decoded through the interleaved
   `Image::decode_into` path. Result: `Ok`, byte-correct output — component
   planes are upsampled to the reference grid before `interleave_and_convert`,
   so the `[..max_len]` slices see equal-length planes, and the output-iterator
   unwraps are guarded by `validate_interleaved_output_buffer`
   (`crates/j2k-native/src/lib.rs:2229`). Optional hardening: an explicit
   equal-length check at interleave entry.
3. **`crates/j2k-native/src/j2c/segment.rs:50,99,193` progression indexing** —
   safe by construction: `IteratorInput::try_new_with_custom_bounds`
   (`crates/j2k-native/src/j2c/progression.rs:52-82`) clamps POC-marker bounds
   to actual component/layer/resolution counts and rejects empty ranges;
   precinct indices come from `ResolutionTile::num_precincts()` per
   (component, resolution), not from file bytes.

## 5. Duplication inventory (non-test)

Most Rust duplication concentrates in backend adapter/facade layers; shader
duplication is tracked separately because it is larger and less mechanically
comparable. Ranked worst-first; items 1–4 also carry divergence hazards:

1. **JPEG baseline GPU encode adapters are now shared at the host-driver
   boundary.** The CUDA and Metal adapters are down to ~299/314 LOC:
   `crates/j2k-jpeg-cuda/src/encode.rs` vs `crates/j2k-jpeg-metal/src/encode.rs`
   now share backend-neutral validation, GPU ABI parameter planning, entropy
   capacity sizing, same-buffer span detection, single/batch host-loop
   orchestration, and JPEG frame assembly through
   `crates/j2k-jpeg/src/adapter/baseline_encode.rs`. Backend files now provide
   only resident source keys, tile metadata conversion, backend error mapping,
   kernel job construction, and table mapping. Repo lint pins the shared driver
   and the post-driver line ratchets.
2. **Mirrored `J2kEncodeStageAccelerator` trait is now closed.**
   `crates/j2k-types/src/lib.rs` owns `J2kEncodeStageAccelerator` and
   `CpuOnlyJ2kEncodeStageAccelerator`; `j2k` re-exports the neutral contract at
   the root facade while `adapter::encode_stage` remains an internal
   implementation module, and `j2k-native` consumes the same trait. Repo lint
   prevents `j2k-native` from reintroducing a private mirror and prevents the
   duplicate public `j2k::adapter::encode_stage` path from returning.
3. **Transcode accelerator bookkeeping** (~250 LOC):
   `crates/j2k-transcode-cuda/src/lib.rs` vs
   `crates/j2k-transcode-metal/src/lib.rs` — mirrored error enums, dispatch-mode
  enums, threshold constants, counter getters, and the attempt/gate/dispatch/
  recover template. Shared `DctToWaveletStageCounters` and combined HTJ2K
  offer/dispatch accounting now live in `j2k-transcode`; CUDA and Metal also
  share `TranscodeStageDispatchMode` for Auto/Explicit unavailable and
  recoverable-error behavior. Repo lint prevents local mode enums or local
  recover/unavailable helpers from returning. Drift that remains deliberate and
  lint-guarded: Metal-only per-stage thresholds and staged-batch axis cap, plus
  CUDA-only resident HT preencode paths.
4. **`ImageDecode` CPU-host facades are now shared.**
   `j2k-core::CpuBackedImageDecode` owns the blanket host-output
   `ImageDecode` implementation for GPU decoders, and all four CUDA/Metal
   J2K/JPEG decoders implement that hook instead of local host facade copies.
   Device-submit methods remain backend-specific because CUDA validation and
   Metal request/session accounting now intentionally differ; repo lint pins
   the shared CPU-host boundary.
5. **JPEG row-sink cluster is now closed.** The decode `SinkWriter` row adapter
   now lives in `crates/j2k-jpeg/src/decoder/sink_writer.rs`, and bench
   profiling reuses it through a black-box `RowSink`; repo lint pins both
   facts. The old `ComponentWriterAdapter` bridge is gone; `ComponentRowWriter`
   now flows through the blanket `OutputWriter for &mut W` implementation in
   `crates/j2k-jpeg/src/decoder/core_traits.rs`. The Metal viewport-cache row
   writers now share `PlaneRowTarget` in
   `crates/j2k-jpeg-metal/src/compute/viewport_cache.rs`, and repo lint
   prevents the duplicate full-row writer from returning.
6. **Canonical JPEG Huffman derivation is now shared.**
   `crates/j2k-codec-math/src/jpeg.rs` owns the Annex C derivation, the
   `j2k-jpeg` entropy and fast-packet layers delegate to it, and CUDA/Metal
   adapters consume the shared canonical metadata instead of rebuilding
   huffsize/huffcode arrays locally. Repo lint pins the CUDA and Metal paths.
7. **JPEG lossless color decode cluster** (~250 LOC within one file):
   `crates/j2k-jpeg/src/decoder.rs:2533` now shares full-frame RGB/YCbCr
   sampling dispatch by bit depth, and `decoder/lossless_helpers.rs` owns the
   shared color validation helper used by component, sampled, and row-stream
   decode. The component and sampled decode helpers centralize restart handling
   and sample decode, the full-output and row-stream component paths share
   `decode_lossless_color_sample`, sampled MCU component/plane traversal lives
   in `decode_lossless_sampled_color_mcu`, and the sampled output renderer now
   shares one generic 8/16-bit loop behind bit-depth-specific color conversion.
   Lossless RGBA region paths now reuse
   `LosslessRgbRegionFallback` for RGB/YCbCr selection and shared temporary RGB
   scratch-copy handling in `decoder/lossless_region.rs`, including the shared
   full-frame decode and scaled copy executor. `LosslessRestartTracker`
   centralizes restart-marker cadence for full-output, row-stream, component,
   and sampled lossless loops; `Extended12RestartTracker` now centralizes
   restart-marker cadence for sequential 12-bit grayscale/color and plane
   decode loops. Remaining duplication is broader decoder-family routing.
8. **Metal JPEG decode shader scaffolding** (2,296 lines across three decode
   chunks plus a 171-line shared helper chunk):
   `crates/j2k-jpeg-metal/src/shaders_decode_fast444.metal`,
   `crates/j2k-jpeg-metal/src/shaders_decode_fast422_regions.metal`, and
   `crates/j2k-jpeg-metal/src/shaders_decode_fast420.metal` now share
   decode-status initialization through `init_decode_status` in
   `shaders_shared.metal`, the DC-only/full-IDCT branch through
   `idct_block` in `shaders_encode.metal`, and single-image/batch entropy
   declarations and configure calls through `JPEG_ENTROPY_THREAD_VARS`,
   `JPEG_CONFIGURE_ENTROPY_THREAD`, `JPEG_BATCH_ENTROPY_THREAD_VARS`,
   `JPEG_CONFIGURE_BATCH_ENTROPY_THREAD`, and the simple full-image
   decode/idct/deposit path through `decode_idct_deposit_block`. The fast444,
   fast422, and fast420 region/scaled decode/deposit-or-skip paths now route
   through `jpeg_decode_idct_deposit_region_block_or_skip` and
   `jpeg_decode_deposit_scaled_region_block_or_skip` in
   `shaders_decode_helpers.metal`; the fast444, fast422, and fast420
   non-region scaled decode/deposit paths now route through
   `jpeg_decode_deposit_scaled_block`;
   texture batch checkpoint setup now
   routes through `configure_batch_entropy_thread`, and texture repair metadata
   clear setup routes through `jpeg_decode_clear_meta_quad`. YCbCr texture-write
   scaffolding now routes through `jpeg_write_ycbcr_rgba`, with repo-lint guards
   preventing the old direct `rgba_float_ycbcr(` calls in split decode shaders,
   and fast422 texture boundary interpolation now routes through
   `h2v1_boundary_left_from_samples` and
   `h2v1_boundary_right_from_samples`, with repo-lint guards preventing direct
   h2v1 boundary weighted formulas from returning. Fast420 h2v2
   texture-boundary weighted chroma sums now route through
   `h2v2_weighted_sample_sum`, with repo-lint guards preventing the old inline
   sums from returning. Paired fast420 h2v2 horizontal boundary texture writes
   now route through `jpeg_write_h2v2_boundary_pair`, with repo-lint guards
   preventing direct fast420 calls to `h2v2_boundary_{left,right}_from_sums`
   from returning. Fast420 h2v2 horizontal boundary top/bottom repair-row skip
   logic now routes through `jpeg_skip_h2v2_boundary_repair_row`, with
   repo-lint guards preventing the duplicated inline skip conditions from
   returning. Fast420/fast422
   texture-boundary clamped copy-span math now routes through
   `jpeg_clamped_extent(...)`, with repo-lint guards preventing repeated inline
   `min(span, limit - min(origin, limit))` expressions from returning,
   per-kernel `const bool intersects = block_intersects_rect`,
   `const bool mcu_intersects = block_intersects_rect`, and
   `const bool y0_intersects = block_intersects_rect` scaffolding plus the old
   per-texture-kernel checkpoint setup and repeated metadata-clear assignments
   from returning.
   Repo lint now
   keeps the shader chunks on tight per-file line ratchets: `shaders_encode.metal`
   <1,812, `shaders_decode_helpers.metal` <174,
   `shaders_decode_fast420.metal` <966,
   `shaders_decode_fast422_regions.metal` <935, and
   `shaders_decode_fast444.metal` <405. Remaining shader duplication is
   sampling-specific texture row/edge orchestration.
9. **FastPacket accessor trait was removed from the semver-visible API.**
   `crates/j2k-jpeg/src/adapter/fast_packet.rs` keeps the concrete
   `JpegFast{420,422,444}PacketV1` packet structs and byte-slice builders as
   the adapter surface. The former public `JpegColorFastPacket` accessor trait
   and color-family accessor macro were removed; CUDA now builds its owned
   decode plan from concrete packet fields, while Metal keeps equivalent
   methods on its private `FastSubsampledPacket` bound. Repo lint prevents the
   old public trait, facade re-export, and rendered stable API entries from
   returning.
10. **Precomputed DWT forwarding wrappers are now centralized.**
   `crates/j2k-native/src/j2c/encode/precomputed.rs` uses one
   `forward_precomputed_encode_stage_hooks!` macro for the two 5/3 and 9/7
   precomputed-DWT adapters, and documents the unrelated input/color and
   whole-subband/tile hooks left at their trait defaults. Repo lint
   `native_encode_options_and_tile_parts_live_in_focused_modules` pins the
   shared macro and defaulted-hook boundary.
11. **MQ-coder QE spec table is now shared.**
   The original duplicate table was remediated by
   `crates/j2k-native/src/j2c/mq.rs`; both
   `arithmetic_decoder.rs` and `arithmetic_encoder.rs` import
   `super::mq::QE_TABLE`. Repo lint
   `mq_qe_table_is_shared_by_encoder_and_decoder` prevents private copies from
   returning.
12. **Metal batch/submission scaffolding** (~50 LOC):
   `crates/j2k-metal/src/batch.rs:150-214` vs
   `crates/j2k-jpeg-metal/src/batch.rs:137-238`.
13. **Small items are closed:** component-plane accessor macro sharing is
    now single-sourced in `j2k-native` and guarded by repo lint. FNV-1a digest
    helpers are now single-sourced in `j2k-core` and guarded by repo lint. The
    scalar `decode_code_block` fallback lives in the `HtCodeBlockDecoder` trait
    default and is guarded by repo lint. The DWT 9/7 lifting constants in
    `crates/j2k-transcode/src/dct97_2d.rs` now import `j2k-codec-math`, and repo
    lint `wavelet_and_idct_constants_use_codec_math_sources` prevents local
    copies from returning.

Sibling-crate concept duplication (Surface/Codec/Error/Info/output-target type
zoos across the GPU adapter crates) is the structural root of items 1–4 and
should be addressed through shared traits/types in `j2k-core`/`j2k-types`, not
ad-hoc merging.

## 6. God files and structure

The prior split work relocated code but left seams uncut. The current policy is
to tighten each ratchet immediately after a split; the tightest ratchets were
lowered in this sweep (`decoder.rs` <3,985, `j2c/encode.rs` <3,900,
`fixture_compare.rs` <2,295, `j2k-jpeg-metal/src/lib.rs` <930,
`j2k-native/src/lib.rs` <2,260, `j2k-metal/src/compute.rs` <390,
`j2k-transcode/src/jpeg_to_htj2k.rs` <1,770, and
`resident_codestream.rs` <2,785).
Current remaining offenders and seams:

1. `crates/j2k-jpeg/src/decoder.rs` (3,972 lines, ratcheted below 3,985) — public decode API, private
   codec-family renderers, tile free-function API, and routing.
   Scratch/memory-cap math now lives in `decoder/scratch.rs` (156 lines), but
   the full-image and region-scaled lossless output-format dispatch now share
   one helper, and the row-sink adapter now lives in
   `decoder/sink_writer.rs` (84 lines) with bench-profile reuse through a
   black-box `RowSink`; component row output now uses the blanket
   `OutputWriter for &mut W` bridge in `decoder/core_traits.rs`; shared
   lossless color validation, shared full-output/row-stream per-pixel color
   decode, sampled MCU decode, shared restart-marker cadence, and the shared
   sampled output renderer now live in `decoder/lossless_helpers.rs` (741 lines). Lossless RGB/YCbCr region
   fallback routing, full-frame decode reuse,
   scaled copy, and temporary RGBA scratch-copy handling now live in
   `decoder/lossless_region.rs` (157 lines). The deeper remaining duplication
   is broader decoder-family routing.
2. `crates/j2k-native/src/j2c/encode.rs` (3,893 lines, ratcheted below 3,900)
   — encode orchestration
   still mixes typed-component preparation, multi-tile assembly, ROI planning,
   DWT adapter conversion, subband preparation, and Tier-1 dispatch. The
   single-tile implementation and high-bit exact single-tile i64 encode helper
   live in `encode/single_tile.rs`; raw sample width/sign-extension helpers now
   live in `encode/samples.rs` (59 lines), i64 packetization request objects
   and helpers now live in `encode/i64_packetize.rs` (111 lines), and public API
   conversion/deinterleave helpers now live in
   `encode/api_helpers.rs` (100 lines).
3. `crates/j2k-compare/src/fixture_compare.rs` (2,276 lines) — still an entire
   compare product in one file. Manifest parsing is split to
   `fixture_compare/manifest.rs`; TSV report row construction is split to
   `fixture_compare/rows.rs` (286 lines); domain model enums are split to
   `fixture_compare/types.rs` (174 lines), which also owns `BatchInputs`;
   OpenJPH/Kakadu comparator CLI plumbing is split to
   `fixture_compare/comparators.rs` (284 lines);
   publication-gate logic is split to `fixture_compare/gates.rs` (326 lines).
   Remaining seams are corpus / measure.
4. `crates/j2k-jpeg-metal/src/lib.rs` (915 lines) — still large after sibling
   crate splits. The public `Error` type now lives in `error.rs` (105 lines),
   and public session wrappers now live in `session.rs` (484 lines) with crate
   root re-exports. Surface and reusable Metal output types now live in
   `surface.rs` (557 lines), and `JpegTileBatch` now lives in `tile_batch.rs`
   (240 lines). The public `Decoder` wrapper now lives in `decoder.rs` (271
   lines), `Codec` batch implementation and RGB8 batch request types now live
   in `codec_batch.rs` (701 lines), and single-decode request types now live in
   `decode_request.rs` (88 lines), while private root route helpers now share a
   `JpegFastPackets` request bundle, all with crate-root re-exports or
   root-stable public paths. Continue mirroring the sibling's split: routing.
5. `crates/j2k-metal/src/compute/resident_codestream.rs` (2,778 lines) —
   cohesive by name but still built from large batch submitters that run
   validation, table packing, allocation, dispatch, and harvest inline. HT
   cleanup dispatch lives in `resident_codestream/ht_cleanup.rs`, and classic
   profiling labels live in `resident_codestream/classic_labels.rs` (32 lines);
   remaining stages should split similarly.
6. `crates/j2k-native/src/lib.rs` (2,253 lines, ratcheted below 2,260) — much
   smaller after tests, helpers, and HT table/SigProp adapter helpers moved
   out to `ht_adapter.rs` (95 lines), but the crate root still mixes public
   image APIs with reference DSP/block-codec exports. Continue extracting
   focused modules as review pressure requires.

Parameter-struct debt: 9 `allow` attributes containing
`clippy::too_many_arguments`, now enforced by a repo-lint ratchet that counts
multiline and crate-level attributes. One broad native crate-level allow remains
and hides 52 native diagnostics if removed; it should be retired by scoped
request-struct passes rather than treated as resolved. Sampling says most
remaining local suppressions are missing structs (geometry tuples,
buffer+offset pairs, scan-cursor state, request objects that already exist but
are bypassed), multiplied through the mirrored twin families.

Policy for all splits: after each split, tighten the ratchet to just above the
new size — never leave hundreds of lines of headroom.

## 7. Self-enforcement tooling debt

`xtask/tests/repo_lint.rs` (3 lines),
`xtask/tests/repo_lint_support/mod.rs` (840 lines),
`xtask/tests/repo_lint_support/architecture_policy.rs` (597 lines),
`xtask/tests/repo_lint_support/corpus_policy.rs` (157 lines),
`xtask/tests/repo_lint_support/dependency_policy.rs` (81 lines),
`xtask/tests/repo_lint_support/docs_and_workflows_policy.rs` (3,735 lines),
`xtask/tests/repo_lint_support/public_docs_policy.rs` (797 lines),
`xtask/tests/repo_lint_support/release_policy.rs` (187 lines),
`xtask/tests/repo_lint_support/shader_policy.rs` (362 lines), and
`xtask/tests/repo_lint_support/source_policy.rs` (235 lines), and
`xtask/tests/repo_lint_support/workflow_policy.rs` (364 lines; 131 lints total)
are still mostly exact-substring matching. They reliably fail closed on
deleted/moved files, but:

- **Vacuous-green vectors:** negative assertions only ban historical names
  (reintroducing a duplicate under a new name passes); whitespace-embedded
  forbidden patterns silently match nothing after a reformat
  (e.g. the GPU skip-pattern lint bakes in exact 8-space indentation). The
  const-array parser and publishable crate directory helper now fail closed on
  empty parses; apply the same non-empty guard centrally to the remaining
  derived collections and scan sets.
- **Brittleness:** CI YAML content is string-pinned down to embedded script
  internals and an action SHA; exact generic signatures break on a where-clause
  reflow; deprecation-count floors (`>= 16` markers) penalize the intended
  cleanup; the same identifiers are pinned in up to four places.
- **Structure:** ~80–90 lints reduce to four data-shapes
  (file-must-contain / must-not-contain / dir-scan-forbidden / doc-must-mention)
  and should become a data-driven table with one runner enforcing existence and
  non-empty scan sets centrally; keep the genuinely structural lints
  (SHA-256 manifest, cargo-metadata graph diff, public-API snapshot, fuzz
  manifest) as code, split along the file's existing seven modules.
  `assert_contains_all` and `assert_not_contains_all` now centralize several
  file-must-contain and must-not-contain checks with non-empty pattern guards.
  `FilePatternCheck` is the first data-driven file-pattern runner and now owns
  the README codec API, CI docs/benchmark compile-gate, backend surface
  metadata/residency, public-crate release posture, current crate routing,
  support-matrix, reset J2K Metal bench-surface, CUDA trace-export, and
  reusable benchmark-generator rows, plus release-doc version policy,
  j2k-compare package exclusion, staged dependency preflight text,
  public-facade membership, architecture doc classification, unpublished
  tooling crate, adaptive-route API exclusion, OpenHTJ2K corpus notice/license
  coverage, decode-capability shrink-factor/progressive inspect rows, and GPU
  coverage-exclusion/substitute-evidence rows. `PatternCheck` now reuses the
  same pattern-set runner for already-extracted text sections and owns the
  scoped `xtask test` function, CI coverage job, xtask
  nextest/machete/strict-clippy rows, deinterleave/strict-decode policy rows,
  CI miri/fuzz/deny/unsafe-audit metadata rows, GPU policy/workflow rows, CUDA
  Oxide strict-build docs rows, NVIDIA-comparator retirement evidence, JPEG
  fixture split ownership rows, and j2k-compare library/bin helper split rows.
  The next batch also moved native public-contract ownership, hidden native
  adapter exports, JPEG Metal batch/viewport/request API ownership, mirrored
  twin evidence, JPEG decoder upsample-helper evidence, and J2K Metal request
  API routing rows to the same runner. A third batch moved J2K Metal public
  error/surface/session/tile-batch/decoder ownership and batch
  heuristic/CPU-fallback/execution ownership rows to `PatternCheck`, leaving
  line-count and shared-completion-count ratchets as explicit structural
  assertions. The shader split guard batch moved Metal encode-bitstream and
  JPEG Metal shader owned-symbol/shared-helper/monolith-exclusion rows to
  `PatternCheck`, while preserving source-order and file-size ratchets as
  explicit structural assertions. The CUDA encode and JPEG-to-HTJ2K transcode
  ownership guard batch moved API/resident/stage/packetization/HTJ2K and
  options/report/error/batch ownership rows to `PatternCheck`, while preserving
  line-count ratchets as structural assertions. The MQ table/FNV/JPEG
  fast-packet ownership batch moved literal file-policy rows to
  `FilePatternCheck`, while preserving component accessor call-count checks as
  structural assertions. The CPU-backed GPU decoder facade ownership row and
  j2k-metal codec-math dependency row also moved to `FilePatternCheck`.
  The shader-policy split moved the Metal encode-bitstream and JPEG Metal
  shader split/ratchet tests from `docs_and_workflows_policy.rs` into
  `shader_policy.rs`, reducing the largest policy file from 4,451 to 4,094
  lines while keeping the same source-order and line-ratchet assertions.
  This split was checked with `cargo fmt --all --check`,
  `cargo test -p xtask --test repo_lint shader_policy -- --nocapture`,
  `cargo clippy -p xtask --all-features --test repo_lint -- -D warnings`, and
  `cargo test -p xtask --test repo_lint -- --nocapture` (131 passed).
  The workflow-policy split then moved CI miri/fuzz/deny/unsafe-audit metadata,
  GPU path and GPU-validation workflow policy, CUDA Oxide/profile workflow
  policy, and retired NVIDIA-comparator workflow checks from
  `docs_and_workflows_policy.rs` into `workflow_policy.rs`, reducing the largest
  policy file from 4,094 to 3,735 lines while keeping the same file-pattern
  assertions. This split was checked with `cargo fmt --all --check` and
  `cargo test -p xtask --test repo_lint workflow_policy -- --nocapture`
  (13 passed), then with
  `cargo clippy -p xtask --all-features --test repo_lint -- -D warnings`,
  full `cargo test -p xtask --test repo_lint -- --nocapture` (131 passed),
  and `git diff --check`.
  This latest runner migration was checked with `cargo fmt --all`,
  `cargo clippy -p xtask --all-features --test repo_lint -- -D warnings`, and
  `cargo test -p xtask --test repo_lint -- --nocapture` (131 passed).
  `RustSourceScanCheck` now owns the adapter private-module import ban and
  production CUDA nvJPEG ban and fails closed on empty scan definitions or empty
  Rust source directories; continue migrating matching lints to shared runners
  and moving reusable support code out of the policy body.
  The GPU policy docs/lint slice was checked with `cargo fmt --all --check`,
  `cargo clippy -p xtask --all-features --test repo_lint -- -D warnings`, and
  `cargo test -p xtask --test repo_lint gpu_validation_workflow_is_self_hosted_and_explicit -- --nocapture`.
- **CI reality:** `gpu-validation.yml` is `workflow_dispatch` only. The
  executed-test-count floors and per-backend GPU checks run on neither PRs nor
  pushes, and the workflow comments explicitly prohibit adding push or
  `pull_request` triggers without a policy decision. The default CI policy now
  requires a successful manual dispatch before merge for GPU-touching PRs by
  checking `gpu-validation.yml` runs for the PR head SHA; replacing that with
  automatic GPU triggers still needs an approved trigger policy.
  The adoption benchmark/report stack (~6,550 lines, 61% of `xtask/src`) gates
  nothing on merge. It is now behind the opt-in xtask `adoption` feature for
  default builds, with a disabled-command shim and all-feature coverage for the
  explicit adoption path.

## 8. Docs and published-site hygiene

`docs/` is served verbatim as the public site (Pages deploy-from-branch).

- This file and `engineering/mirrored-twin-unification.md` are internal
  engineering records and must stay outside the published `docs/` root.
- `docs/stable-api-1.0.public-api.txt` (632 KB) is 62% of the site payload;
  it is a gate artifact, not site content — relocate if Pages payload matters.
- At audit time, the sitemap was triplicated (`sitemap.xml`, `j2k-sitemap.xml`,
  and `sitemap.txt`). The working tree now keeps only `docs/sitemap.xml`, and
  `docs/robots.txt` registers only that sitemap.
- At audit time, `CHANGELOG.md` had no v0.6.1 entry, no release dates, and
  inconsistent release sections. The working tree now has dated 0.6.0, 0.6.1,
  and 0.6.2 sections.
- At audit time, README embedded a ~170-line benchmark-policy manual duplicating
  `docs/benchmark-corpora.md`. The working tree now points to
  `docs/benchmark-corpora.md` and `docs/benchmark-evidence.md` instead.
- At audit time, `docs/architecture.md`'s crate-classes table and README's
  crate table omitted the published crate `j2k-types`. The working tree now
  includes those rows; keep the repo lint coverage so this does not regress.
- The stale-dashboard claim from the superseded plan ("all repo lints pass" /
  "open findings are zero") has been corrected in this document. Keep this as
  the final audit record and update it in place; do not add another status file
  that can drift from the gates.
- Historic churn context: 76 of 117 markdown files ever added have been
  deleted (plans, specs, handoffs, roadmaps) across three historical project
  renames. This document should be the last standing audit record; update it
  in place rather than adding successor status files.

## 9. Remediation plan

Do not delete passing tests. Update lint ratchets downward as code shrinks.
Run the narrowest affected tests plus repo guardrails per PR.

**Gate policy for this remediation sweep:**

- Routine correctness testing uses `cargo xtask test` or its explicit
  equivalent: `cargo test --workspace --all-features --lib --bins --tests`
  followed by `cargo test --workspace --all-features --doc`, with the
  repository's macOS/Metal exclusions handled by `xtask`.
- Do not run `cargo test --workspace --all-targets --all-features` for this
  sweep. It is not a pure test gate in this repository because it can include
  benchmark executables.
- Benchmark compilation and performance evidence stay explicit:
  `cargo xtask bench-build`, `cargo xtask j2k-perf-guard --quick`, full
  non-quick perf/adoption benchmarks, and `cargo bench` are run only for the
  performance-evidence phase or when a benchmark/performance path is being
  changed.
- Final non-benchmark gates before commit: `cargo fmt --all --check`,
  `cargo check --workspace --all-features --lib --bins --tests`,
  `cargo clippy --workspace --all-features --lib --bins --tests -- -D warnings`,
  `cargo xtask test`, `cargo test -p xtask --test repo_lint`,
  `cargo run -p xtask -- unsafe-audit`, `cargo xtask stable-api`,
  `cargo xtask panic-surface`, `cargo deny check`, `cargo machete`, and
  `cargo xtask semver`.
- Performance/hardware signoff is recorded separately. If hardware or time
  constraints prevent those runs, document the skipped gate and reason instead
  of treating the benchmark as a required test.

**Phase 0 — restore green gates and commit (blocking, mostly complete):**

1. Done: internal docs live under `engineering/`, with repo-lint references
   updated.
2. Done: `.tmp-metadata.json` was removed and `.gitignore` now excludes it.
3. Done: the initial clippy blockers were cleared, including private-field
   test coupling, derivable defaults, split-file wildcard imports, and CUDA
   JPEG adapter migration to `DecodeRequest`. Continue running final
   non-benchmark gates after each new edit set.
4. Done for current public API state: stable API was regenerated and reviewed.
   The current snapshot is 239,689 bytes / 665 `pub fn` entries after the
   accumulated shrink slices: private-module facade cleanup, removed request
   wrappers and deprecated methods, collapsed transcode counter mutators,
   moved test/support-only builders out of public crates, private `j2k-core`
   implementation modules behind root facade exports, private JPEG baseline GPU
   planning helpers, private unused CUDA runtime wrappers, hidden duplicate
   `j2k::adapter::encode_stage` module-path exposure, and hidden duplicate
   `j2k` root encode-stage contract reexport renderings while preserving
   source compatibility and the canonical `j2k-types` contract, plus removal
   of the unused CUDA surface profiled
   download helper and crate-private CUDA grayscale HTJ2K plan-builder
   profiling hooks, plus removal of duplicate JPEG adapter
   `*_packet_for_decoder` wrappers in favor of `decoder_bytes(decoder)` and the
   byte-slice packet builders, plus removal of the custom-alpha JPEG decoder
   convenience wrappers in favor of the existing `PixelFormat::Rgba8`
   default-alpha public path, plus making the redundant JPEG decoder
   native-region scratch wrapper crate-private, plus removal of the public
   JPEG fast-packet accessor trait while preserving source-visible concrete
   fast-packet fields/builders for first-party adapters, plus hiding the raw
   JPEG fast-packet ABI and builders from the rendered 1.0 inventory, plus
   hiding the raw CUDA JPEG runtime decode/encode ABI and direct kernel
   entrypoints from the rendered 1.0 inventory, plus hiding the full JPEG
   adapter device-plan/checkpoint planning API while preserving the public
   `DeviceBatchSummary` capability-report surface. Repo lint guards the accidental
   adapter/module paths and removed helper APIs from returning, including the
   hidden CUDA JPEG chunked-entropy diagnostic facade and runtime diagnostic
   cluster plus the hidden CUDA runtime pool-trace, NVTX/timing, and copy-kernel
   support surfaces plus test-only pooled i16 upload helpers, queued cleanup,
   HTJ2K packetization ABI, and raw CUDA transcode runtime request/band structs
   plus direct `j2k_transcode_*` runtime entrypoints, and raw CUDA J2K
   encode/decode runtime ABI structs plus direct `CudaContext::j2k_*` runtime
   entrypoints, plus hidden Metal lossless report/timing surfaces and JPEG
   Metal resident batch report/preflight helpers, plus hidden JPEG Metal
   viewport helper `decode_viewport_to_surface`, plus hidden root-level
   `j2k-jpeg` `_with_options` decode free-function wrappers, plus hidden
   CUDA/Metal adapter `ImageCodec` impl renderings that exposed private scratch
   defining paths, plus hiding the rendered `j2k_jpeg::adapter` and
   `j2k_jpeg::transcode` module inventories while keeping
   `DeviceBatchSummary` visible through the `j2k_jpeg` root, plus hiding the
   rendered `j2k_transcode::accelerator` compatibility module while defining
   the accelerator contract at the `j2k_transcode` root, plus hiding the
   first-party `idct_blocks_to_signed_samples_rayon` helper from the rendered
   inventory, plus hiding duplicate decoder `ImageDecode`/`ImageDecodeRows`
   trait-adapter impl renderings for `j2k::J2kDecoder` and
   `j2k_jpeg::Decoder`, plus hiding direct HTJ2K CUDA runtime `CudaContext`
   decode/dequantize/encode entrypoints while preserving source-visible access
   for first-party CUDA adapters/tests, plus hiding CUDA HTJ2K backend
   target/result/status/timing impl renderings, duplicate `GpuAbi` primitive/array
   impl renderings, duplicate `J2kCodec`/`JpegCodec` batch-decode impl
   renderings, and duplicate concrete CPU/Rayon accelerator trait-impl
   renderings for `CpuOnlyJ2kEncodeStageAccelerator`,
   `CpuOnlyDctToWaveletStageAccelerator`, and
   `RayonReversibleDwt53Accelerator`, plus duplicate backend accelerator
   trait-impl renderings for `CudaEncodeStageAccelerator`,
   `MetalEncodeStageAccelerator`, `CudaDctToWaveletStageAccelerator`, and
   `MetalDctToWaveletStageAccelerator`, plus hiding first-party native
   direct-device planning, delegated code-block decode, direct-plane reuse
   probing, and coefficient-extraction methods from the rendered inventory
   while preserving source-visible access for the public `j2k` recode facade
   and CUDA/Metal adapters, plus hiding native coefficient-domain
   precomputed/prequantized/preencoded encode helpers from the rendered
   inventory while preserving source-visible access for transcode and recode
   facades, plus hiding native raw accelerator/quantization hooks while
   leaving ordinary native encode, HTJ2K encode, ROI encode, and component
   plane encode APIs rendered, plus hiding generic CUDA runtime kernel-output
   wrapper and typed device-buffer view types from the rendered inventory
   while leaving them source-visible for first-party adapters/runtime code, plus
   hiding raw CUDA HTJ2K code-block job and lookup-table structs from the
   rendered inventory, plus hiding duplicate concrete `CodecError` trait-impl
   method renderings while preserving the visible shared trait contract, plus
   hiding duplicate concrete root/JPEG/tilecodec `CodecContext` and
   `ScratchPool` impl renderings while preserving the visible concrete
   constructors and intended accessors, plus hiding first-party transcode
   dispatch-mode helper methods `recover` and `unavailable` while preserving
   the visible `Auto`/`Explicit` mode enum and `is_auto()` query, plus making
   `JpegOutputBuffer` explicit allocation-cap override helpers private while
   preserving the public default-cap `new`, `with_stride`, `resize`, and
   `resize_with_stride` paths, plus hiding first-party adapter error
   classification helper functions and concrete adapter `AdapterErrorParts`
   impl renderings while keeping the shared trait and enum visible for adapter
   implementations, plus hiding first-party `j2k-core` buffer/allocation helper
   functions from the rendered API while preserving source-visible adapter use
   and keeping `DEFAULT_MAX_HOST_ALLOCATION_BYTES` visible, plus hiding the
   `IndexedBatchResult` alias and first-party `j2k-core` batch/backend helper
   functions from the rendered API while preserving source-visible callers,
   plus hiding source-visible `j2k`/`j2k-jpeg` context-reuse tile helpers from
   the rendered API while preserving ordinary one-shot and batch tile APIs,
   plus hiding JPEG batch-session worker diagnostics from the rendered API.
   Rerun
   `cargo xtask stable-api --write` after any further semver-visible cleanup.
5. Done: `docs/unsafe-audit.md` was refreshed, stale rows were removed, and
   `cargo xtask unsafe-audit` now fails closed on stale documented rows.
6. Done: `panic-surface` is wired into `ci.yml` and the `ci()` meta-task.
7. Deferred until the integrated sweep is ready: commit the working tree,
   including the GPU-validation hardening.
8. Ongoing: keep this document's status synchronized as items land.

**Phase 1 — typed errors and divergence hazards (~days):**

1. Done: Metal substring routing was replaced with typed error
   classification; backend errors now carry failure-mode
   `BackendErrorKind` values instead of `Backend(String)`-only.
2. Done: Metal readback and CPU-side row/fill access now use checked helpers;
   repo lint confines raw `contents()` calls to central helper modules.
3. Mostly done: duplication items 1-4 and canonical Huffman derivation are
   unified or pinned by repo lint; remaining duplication work is tracked in
   Phase 2 shader/JPEG-family routing.
4. Done: checked shrink-factor arithmetic at
   `crates/j2k-native/src/j2c/codestream.rs`, pinned by unit coverage and repo
   lint.
5. Done: inspect/decode agreement tests for JPEG progressive fixtures, pinned
   by repo lint.

**Phase 2 — finish structural work (incremental):**

1. Done for the current sweep: the six tracked god-file/module-shell targets
   are split, including `encode_impl`-adjacent helpers and the repo-lint module
   tree.
2. Done for the current split set: active size ratchets are tight, including
   `decoder.rs <3,985`, `encode.rs <3,900`, and
   `resident_codestream.rs <2,785`.
3. Done for the current sweep: `too_many_arguments` suppression attributes are
   down from the audited baseline of 157 to a corrected current ratchet of 4.
   The ratchet now counts multiline and crate-level allow attributes instead of
   only exact one-line prefixes, and
   `cargo test -p xtask --test repo_lint too_many_arguments_suppressions_stay_below_current_ratchet -- --nocapture`
   passes.
4. Partially done: mechanical duplication items 5-13 are mostly unified or
   guarded. Metal shader entropy setup, simple full-image decode/idct/deposit
   scaffolding, region/scaled decode/deposit-or-skip routing, fast422/fast420
   non-region scaled decode/deposit routing, texture batch checkpoint setup,
   texture repair metadata clearing, and YCbCr texture-write scaffolding are now
   shared and tight-ratcheted. Fast422 texture boundary interpolation now uses
   shared h2v1 boundary helpers, and fast420 h2v2 texture-boundary weighted
   chroma sums and paired horizontal boundary writes now use shared h2v2
   helpers. Remaining shader work is
   sampling-specific texture row/edge orchestration after the shared h2v2
   row-skip extraction. The broader public API shrink is tracked in
   Phase 3 item 6.

**Phase 3 — tooling and docs:**

1. Data-driven lint runner; remaining non-empty guards; broader
   whitespace-normalized matching. In progress: const-array-derived package
   lists now fail closed when empty, publishable package entries are checked
   against real crate directories, all former top-level policy modules are
   split out, `repo_lint_support` holds shared helpers plus a data-driven
   `FilePatternCheck` runner for file-must-contain/must-not-contain rows plus
   `PatternCheck` for already-extracted text sections and command
   dispatch/help rows, and `RustSourceScanCheck` for forbidden Rust-source
   scans including GPU hardware-gate silent-return policy, and normalized
   matching is in use for the
   adoption, HT fallback, and CI permissions guards. The late docs/workflow
   cluster covering deinterleave, decode strictness, CI/GPU workflow policy,
   CUDA Oxide docs, NVIDIA retirement evidence, JPEG fixture ownership, and
   j2k-compare helper ownership has also moved to the shared pattern runner.
   Native public contracts, hidden native adapter exports, JPEG Metal
   batch/viewport/request API ownership, mirrored-twin evidence, JPEG decoder
   upsample-helper evidence, and J2K Metal request API routing are now covered
   by the shared runner as well. J2K Metal public
   error/surface/session/tile-batch/decoder ownership and batch
   heuristic/CPU-fallback/execution ownership rows are now on the shared runner
   too. Metal encode-bitstream and JPEG Metal shader split ownership/shared
   helper rows are now on the shared runner, with source-order and file-size
   ratchets kept as structural checks. CUDA encode module ownership and
   JPEG-to-HTJ2K transcode options/report/error/batch ownership rows are now on
   the shared runner too. Shared MQ table ownership, FNV-1a JPEG cache helper
   ownership, shared JPEG fast-packet accessor ownership, and literal
   component-plane accessor ownership rows are also on `FilePatternCheck`, with
   component macro call-count checks kept explicit. CPU-backed GPU decoder
   facade ownership, the j2k-metal codec-math dependency row, and the xtask
   adoption-stack feature-gate policy are also on `FilePatternCheck`. Release
   publishable-package, publish-workflow, and publish-script coverage
   assertions now use the shared pattern helpers instead of ad hoc package
   loops.
2. Done for policy wiring: `gpu-validation.yml` remains `workflow_dispatch`
   only; CI's `gpu-path-policy` job enforces successful manual CUDA and/or Metal
   `gpu-validation.yml` jobs by PR head SHA for GPU-touching changes; and
   `CONTRIBUTING.md` plus repo-lint document and pin the requirement. Actual
   merge of GPU-touching changes still requires a successful manual dispatch or
   an approved trigger-policy decision. Local self-hosted CUDA validation of
   the current working tree has passed on `jcwal@100.75.125.59` with
   `J2K_REQUIRE_CUDA_RUNTIME=1`, `J2K_REQUIRE_CUDA_OXIDE_BUILD=1`, and the
   focused JPEG hardware-decode gate enabled, but that is not a substitute for
   the documented GitHub workflow-dispatch evidence if policy requires it.
3. Done: sitemap dedup, CHANGELOG backfill (v0.6.1, dates), README policy
   pointer, and architecture/README crate-table fixes including `j2k-types`.
4. Done: the adoption stack is isolated behind xtask's opt-in `adoption`
   feature. Default builds compile only a disabled-command shim; explicit
   `--all-features` checks still cover the adoption path.
5. Dependency debt is addressed in the working tree: duplicate `weezl` versions
   are collapsed to `weezl v0.1.12`, `block v0.1.6` is patched locally through
   `[patch.crates-io]`, and `cargo deny check` passes. Remove the local block
   patch only after `metal` publishes a fixed dependency path.
6. In progress: complete the public API review/shrink before 1.0. The latest
   rendered API evidence is 239,689 bytes and 665 `pub fn` entries after
   removing duplicate fast-packet module exposure, the CUDA test-only DWT
   reshape export, JPEG Metal viewport helper/resident entrypoints, and the
   deprecated `j2k-jpeg`, `j2k-jpeg-metal`, and `j2k-metal` request-wrapper
   methods plus `J2kError::adapter_backend`, converting public CUDA transcode
   batch APIs to named request structs, removing unused CUDA transcode no-pool
   batch wrappers, and making the duplicate public `j2k-jpeg` decoder module
   path plus duplicate public `j2k` view/context/error/scratch module paths and
   duplicate public `j2k-jpeg` info/context/batch-session/capabilities/
   output-buffer/segment/error/encoder module paths plus duplicate public
   `j2k-native` error module path and duplicate public `j2k-transcode`
   transform/oracle module paths internal while preserving root facade exports,
   collapsing the shared transcode stage-counter mutation API to one typed
   `record(event, count)` entry point, moving the prequantized HTJ2K 9/7
   oracle builders to unpublished `j2k-transcode-test-support`, and making the
   reversible 5/3 block-sample helper crate-private, and making DCT-grid
   scratch-capacity inspectors test-only internals, removing
   `JpegToHtj2kTranscoder` scratch-capacity inspectors from the source public
   impl, and making caller-owned DCT-grid scratch types plus `*_with_scratch`
   transform functions crate-private, and making DWT result `max_abs_diff`
   helpers test-only, collapsing duplicate DCT grid error aliases into
   `DctGridError`, and removing the duplicate root `j2k_core::GpuAbi`
   re-export while preserving `j2k_core::accelerator::GpuAbi`, and keeping CUDA
   runtime stream/event/preload scaffolding internal/test-only, and removing
   unused public CUDA HTJ2K untimed decode/dequantize wrappers while preserving
   the steady-state async cleanup enqueue and IDWT paths, and keeping CUDA copy
   kernel device-to-device/cuda-oxide parity helpers private to runtime
   implementation/tests, and removing duplicate public `j2k-jpeg::Decoder`
   passthrough/restart-index accessors now covered by `JpegView`, and removing
   the duplicate public `j2k::J2kDecoder` passthrough accessor now covered by
   `J2kView`, and removing the one-shot public `j2k_jpeg::Decoder::decode_tile`
   row-sink helper now covered by `JpegView::parse`,
   `Decoder::from_view_in_context`, and `decode_rows_with_scratch`, and removing
   duplicate public `j2k_jpeg::Decoder::inspect_with_options` now covered by
   `JpegView::parse_with_options(...).info()`, and removing duplicate public
   `j2k_jpeg::Decoder::new_with_options` now covered by
   `JpegView::parse_with_options(...)` plus `Decoder::from_view(...)`, and
   making duplicate public `j2k_jpeg::Decoder::decode_request_with_scratch`
   private behind `Decoder::decode_request`, and removing duplicate public
   `j2k_jpeg::Decoder` custom-alpha RGBA wrappers while keeping crate-local
   coverage of the underlying output-format alpha paths, and removing duplicate public
   `j2k::J2kDecoder::bytes` now covered by `J2kView::bytes()` while
   first-party CUDA/Metal adapters store their borrowed source bytes internally,
   and removing duplicate public `j2k::J2kDecoder::support_info` now covered by
   `J2kView::support_info()` and `J2kDecoder::inspect_support(...)`, and making
   CUDA runtime pinned-host-buffer and unnamed timing helpers crate-local or
   test-only, and removing the unused CUDA JPEG 4:2:0-specific owned-decode
   facade (`CudaJpeg420Rgb8DecodePlan` and
   `decode_jpeg_420_rgb8_owned*`) while preserving the generic
   `CudaJpegRgb8DecodePlan` path used by the JPEG CUDA adapter, and removing
   unused public CUDA runtime IDWT untimed wrappers
   `j2k_inverse_dwt_single_device_untimed` and
   `j2k_inverse_dwt_batch_device_untimed_with_pool` while preserving the timed
   single-device/batch paths, the pooled steady-state path used by `j2k-cuda`,
   and the async enqueue path, and making CUDA/Metal encode-stage per-stage
   attempt and dispatch getters test-only internals while preserving the public
   consolidated `dispatch_report()` surface, and removing the semver-visible
   `TranscodePipelineMap::debug_report` diagnostic string helper while keeping
   structured pipeline-stage fields available to tests, examples, and benchmark
   sources, and making duplicate public `j2k::adapter::device_plan` internal
   while preserving root `j2k::{DeviceDecodePlan, DeviceDecodeRequest}` facade
   exports for first-party GPU adapters, and flattening CUDA transcode 9/7
   batch `*_WithPoolRequest` shapes so duplicate inner request types no longer
   appear in the public API, and removing the duplicate public CUDA HTJ2K
   cleanup multi non-timed wrapper while preserving the timed/status-returning
   entrypoint, and retiring the duplicate public CUDA HTJ2K single-codeblock
   encode wrappers while preserving one-job batch encode coverage, and removing
   duplicate public CUDA HTJ2K resident encode implicit-pool wrappers while
   preserving table-upload and explicit `_and_pool` resource-reuse APIs, and
   removing duplicate public CUDA HTJ2K simple decode resource wrappers while
   preserving the table-upload decode API and internal resource-backed helper,
   and removing the duplicate public CUDA HTJ2K multi-buffer dequantize
   non-pool wrapper while preserving the explicit caller-pool API, and removing
   the duplicate public CUDA HTJ2K cleanup packetization no-tag wrapper while
   preserving the explicit tag-state packetization API, and hiding the
   duplicate public `j2k_core::passthrough` implementation-module path while
   preserving the root passthrough facade exports, and hiding the duplicate
   public `j2k_core::{row_sink,scratch}` implementation-module paths while
   preserving the root `RowSink` and `ScratchPool` facade exports, and hiding
   the duplicate public `j2k_core::{pixel,sample,scale}` implementation-module
   paths while preserving the root `PixelFormat`/`PixelLayout`,
   `Sample`/`SampleType`, and `Downscale` facade exports, and hiding the
   duplicate public `j2k_core::error` implementation-module path while
   preserving root error/classification facade exports, and hiding duplicate
   public `j2k_core::{backend,batch,context,device,traits,types}`
   implementation-module paths while preserving root shared-contract facade
   exports and the intentional public `j2k_core::accelerator::GpuAbi` path, and
   making JPEG baseline GPU encode plan-building/validation helpers private
   while preserving the public shared adapter trait/types and tile/batch
   entrypoints, and making the unused public CUDA runtime
   `upload_htj2k_decode_resources` wrapper private while preserving the simple
   decode and explicit table-resource upload APIs, and hiding the duplicate
   public `j2k::adapter::encode_stage` module path while preserving root
   encode-stage exports, and removing the unused public
   `j2k_cuda::Surface::download_into_profiled` helper, and making CUDA
   grayscale HTJ2K plan-builder profile hooks crate-private with plan-shape and
   decode-kernel parity tests moved to crate-local coverage, and removing the
   duplicate public JPEG adapter `*_packet_for_decoder` convenience wrappers
   while preserving `decoder_bytes(decoder)` plus the byte-slice packet
   builders for first-party adapters, and making redundant public
   `j2k_jpeg::Decoder::decode_region_into_with_scratch` crate-private while
   callers use `decode_region_scaled_into_with_scratch(..., Downscale::None)`,
   and removing the public `j2k_jpeg::adapter::JpegColorFastPacket` accessor
   trait plus its semver-visible trait-impl methods while keeping concrete
   packet fields/builders and private CUDA/Metal adapter access paths, and
   hiding generic CUDA runtime kernel-output wrapper types
   `CudaKernel{,Batch,ContiguousBatch}Output`, `CudaPooledKernelOutput`, and
   the contiguous output range type from the rendered inventory, and hiding
   typed `CudaDeviceBufferView` wrappers plus `CudaDeviceBuffer::typed_view`
   methods from the rendered inventory, and hiding raw CUDA HTJ2K code-block
   job and lookup-table structs from the rendered inventory.
   Concrete `CodecError` impl method inventories for the root J2K/JPEG crates
   and CUDA/Metal adapter crates are also hidden as duplicate documentation of
   the public `j2k_core::CodecError` trait contract.
   The CUDA runtime pool-trace, NVTX/timing wrapper, and copy-kernel upload
   support surfaces are hidden from the rendered stable API while remaining
   available to first-party adapter crates; the test-only pooled i16 upload
   helpers are hidden as well. CUDA queued-execution and queued-cleanup support
   surfaces are also hidden from the rendered stable API while remaining
   available to first-party CUDA scheduling paths. Raw CUDA HTJ2K packetization
   ABI types and kernel entrypoint are hidden from the rendered stable API while
   remaining available to first-party CUDA packetization paths. Raw CUDA J2K
   encode/decode runtime ABI structs and direct `CudaContext::j2k_*` runtime
   entrypoints are hidden from the rendered stable API while remaining
   available to first-party CUDA decode/encode/transcode glue and tests.
   Encode-stage accelerator surface review is complete for the current sweep:
   individual attempt/dispatch getters are treated as duplicate diagnostics of
   `J2kEncodeDispatchReport` and guarded from returning by repo lint.
   The full JPEG adapter device-plan/checkpoint planning API is also hidden
   from the rendered 1.0 inventory while `DeviceBatchSummary` remains visible
   for `JpegCapabilityReport`. The JPEG fast-packet raw ABI surface and
   byte-slice packet builders are hidden from the rendered 1.0 inventory while
   remaining source-visible to first-party CUDA/Metal adapter code. Raw CUDA
   JPEG runtime decode/encode ABI types and direct `CudaContext` entrypoints
   are hidden from the rendered 1.0 inventory while remaining source-visible to
   `j2k-jpeg-cuda`. The broader `j2k_jpeg::adapter` and
   `j2k_jpeg::transcode` rendered module inventories are hidden as well; the
   source modules remain public for first-party crates and the public
   capability-report summary is rooted at `j2k_jpeg::DeviceBatchSummary`. The
   `j2k_transcode::accelerator` compatibility module is hidden from the
   rendered inventory too; accelerator types, errors, counters, and traits are
   now root-defined while `j2k_transcode::accelerator::*` remains a hidden
   source-compatible path. The adjacent `idct_blocks_to_signed_samples_rayon`
   helper is source-visible to the Metal transcode adapter but hidden from the
   rendered inventory and guarded from returning. Duplicate decoder
   `ImageDecode`/`ImageDecodeRows` trait-adapter impl renderings are hidden for
   `j2k::J2kDecoder` and `j2k_jpeg::Decoder`, leaving their inherent public
   methods as the rendered API. Direct HTJ2K CUDA runtime `CudaContext`
   decode, coefficient-allocation, dequantize, and encode entrypoints are also
   hidden from the rendered 1.0 inventory while remaining source-visible for
   first-party CUDA adapters/tests. CUDA HTJ2K backend target/result/status and
   timing structs are likewise hidden from the rendered inventory while
   remaining source-visible to first-party CUDA adapters/tests. Generic CUDA
   runtime kernel-output wrappers and their contiguous-output range type are
   hidden from the rendered inventory as first-party adapter plumbing, and
   typed device-buffer views are hidden as runtime-internal validation helpers;
   raw CUDA HTJ2K code-block job and lookup-table structs are likewise hidden
   as first-party adapter/runtime plumbing. Concrete `CodecError` impl
   renderings are hidden for first-party error types; the public trait remains
   visible at `j2k_core::CodecError`. `j2k-cuda-runtime` is now down to 38
   rendered public functions. Primitive
   and array `GpuAbi` impl method renderings are hidden from the stable API
   snapshot too; the intentional public contract remains the
   `j2k_core::accelerator::GpuAbi` trait path, and the stray primitive
   pseudo-crate function counts are gone. The `J2kCodec`
   `ImageCodec`/`TileBatchDecode` trait impl method renderings are hidden as
   duplicate documentation of the root batch decode facades; `j2k` is now down
   to 122 rendered public functions. The matching `JpegCodec`
   `ImageCodec`/`TileBatchDecode` trait impl method renderings are hidden as
   duplicate documentation of the JPEG root batch decode facades, with
   `j2k-jpeg` currently at 108 rendered public functions. The concrete CPU-only
   and Rayon accelerator impl method renderings for the J2K encode-stage and
   JPEG-to-HTJ2K transcode traits are also hidden as duplicate documentation
   of public trait contracts, with `j2k-transcode` currently at 41 rendered
   public functions and `j2k-types` down to 21 rendered public functions. The
   CUDA/Metal encode-stage and transcode-stage backend impl method renderings
   are hidden by the same policy, bringing `j2k-cuda` to 53 rendered public
   functions, `j2k-metal` to 56, `j2k-transcode-metal` to 16, and
   `j2k-transcode-cuda` to 2. Native direct-device planning, delegated
   code-block decode, direct-plane reuse probing, and coefficient-extraction
   adapter hooks are hidden from the rendered inventory too, bringing
   `j2k-native` down to 94 rendered public functions while keeping the
   ordinary decode methods visible. Native coefficient-domain
   precomputed/prequantized/preencoded encode helpers are also hidden from the
   rendered inventory, bringing `j2k-native` down to 77 rendered public
   functions while leaving the ordinary `encode`, `encode_htj2k`, ROI, and
   component-plane encode APIs visible. Native raw accelerator encode hooks and
   the quantization-step helper used by transcode internals are hidden as well,
   bringing `j2k-native` down to 74 rendered public functions. First-party
   `j2k-core` buffer/allocation helpers are also hidden from the rendered API,
   followed by first-party `j2k-core` batch/backend helpers, bringing
   `j2k-core` down to 76 rendered public functions while keeping the public
   allocation cap constant visible, and hiding `j2k`/`j2k-jpeg`
   context-reuse tile helpers, bringing `j2k` down to 114 rendered public
   functions and `j2k-jpeg` down to 96, plus hiding JPEG batch-session worker
   diagnostics, bringing `j2k-jpeg` down to 94.
7. Performance evidence recorded separately. On 2026-07-07,
   `cargo xtask bench-build` and
   `cargo xtask j2k-perf-guard --baseline-ref HEAD --quick` passed; the
   quick-guard details are recorded in `docs/benchmark-evidence.md`.
   Hardware-dependent strict CUDA runtime benchmark validation remains a
   Linux/NVIDIA-host item outside this local macOS package check.

**Out of current local package slice:** arbitrary performance tuning unrelated
to correctness and strict CUDA runtime benchmark validation on Linux/NVIDIA
hardware. Benchmark commands are run only during explicit performance-evidence
work or when a change directly targets benchmark/performance behavior.
