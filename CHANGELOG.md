# Changelog

This changelog tracks the current staged release line. Historical phase notes
and stale roadmap entries have been removed from the public documentation set.

## [0.6.2]

- Defines the public codec claim as JPEG 2000 Part 1 codestream support, JP2
  still-image wrapping, HTJ2K Part 15 support, and JPH wrapping, with
  JPX / JPEG 2000 Part 2 extensions explicitly out of scope.
- Adds the public support matrix and `cargo xtask public-support` gate covering
  signed/high-bit samples, wide component counts, progression/tile-part cases,
  packet-marker combinations, ROI maxshift, JP2/JPH metadata, HT refinement
  decode/encode, J2K-to-HTJ2K recode, and publication gates.
- Expands the public `j2k` facade with component-plane decode/encode APIs for
  arbitrary component counts, signedness, sampling, mixed bit depths, and
  native high-bit sample access while keeping Gray/RGB/RGBA convenience paths.
- Adds JP2/JPH wrapper helpers and validation for still-image metadata,
  codestream branding, color boxes, palettes/component mapping, channel
  definitions, BPCC, and ICC preservation paths.
- Completes repo-local HTJ2K cleanup/refinement, multi-layer/rate, JPH, and
  recode self-checks while keeping external OpenJPH/Kakadu evidence as a
  publication gate.
- Removes the cuda-oxide transcode IDCT per-thread local-memory table
  materialization, reducing the self-hosted RTX 4070 SUPER cuda-oxide
  JPEG-to-HTJ2K transcode profile from `40.813 MP/s` before the fix to
  `380.411 MP/s` in the `v0.6.2` validation run.
- Keeps cuda-oxide transcode opt-in and records the CUDA C vs cuda-oxide
  benchmark evidence in the public benchmark documentation.
- Adds `j2k transcode <input.jpg> <output.j2k> --htj2k --lossless-53` to the
  CLI as the first conservative JPEG-to-HTJ2K smoke-test command.
- Refreshes adoption-facing docs with a shorter quickstart, support matrix, and
  current `0.6.x` security/environment-variable line.

## [0.6.0]

- Stages the `j2k` facade release.
- Keeps CPU decode as the portable correctness baseline.
- Treats Runtime backend selection defaults to `Auto` as the public backend
  policy.
- Adds resident Metal and CUDA device memory surfaces for supported adapter
  paths through cuda-runtime integration.
- Uses J2K-owned CUDA kernels for supported CUDA codec stages.
- Requires recorded benchmark evidence before NVIDIA performance claims.
- Consolidates shared J2K encode-stage, CUDA submit, Metal runtime, tilecodec,
  JPEG output, and test-support helpers.

### Breaking API Changes

- Collapses the 26 `j2k-metal` lossless encode/submit permutations
  (`{encode,submit}_lossless_from_{padded_,}metal_buffer{,s}` x
  to-metal/with-report/with-config/to-metal-batch) into three request-based
  entry points: `submit_lossless_batch` (host codestream bytes),
  `submit_lossless_batch_to_metal` (Metal-backed codestreams with batch
  stats), and `encode_lossless_batch_with_report` (host-byte timing
  reports), all taking the new `MetalLosslessEncodeBatchRequest` (`tiles` +
  now-public `MetalEncodeInputStaging` + `MetalLosslessEncodeConfig`). The
  single-tile submit wrapper `SubmittedJ2kLosslessMetalEncode` is removed
  (submit a one-tile batch and take the first result). Single-tile
  `_with_report` callers now route through the batch report entry, which
  may use the resident batch path where the removed wrapper always used
  the per-tile staged pipeline; the report's stage-residency fields can
  differ accordingly.
- Collapses the 24 `j2k-jpeg-metal` `Codec::decode_rgb8_*_with_session`
  batch permutations (full/scaled/region-scaled x bytes/decoders x
  reusable/resizable buffer/textures) into two request-based entry points:
  `decode_rgb8_batch_into_buffer_with_session` and
  `decode_rgb8_batch_into_textures_with_session`, taking the new
  `Rgb8MetalBatchRequest` (`Rgb8MetalBatchSource` + `Rgb8MetalBatchOp`) and
  `MetalBufferBatchTarget`/`MetalTextureBatchTarget` enums. The three hottest
  permutations remain as convenience wrappers with unchanged signatures:
  `decode_rgb8_decoder_batch_into_resizable_metal_textures_with_session` and
  the two region-scaled resizable forms. Unsupported-batch error reasons no
  longer distinguish byte vs decoder sources.
- Introduces the `j2k-types` contract crate: the 49 encode-stage
  job/output/report types shared by `j2k` and `j2k-native`
  are defined once there (both crates re-export them at their existing
  paths), and the `j2k` adapter's `*_from_native`/`*_to_native`
  encode-stage converter functions are removed since both sides now use the
  same types.
- Widens transcode accelerator errors: `DctToWaveletStageAccelerator` methods
  now return `TranscodeStageError` (`Unsupported`/`Backend`/`DeviceUnavailable`)
  instead of `&'static str`, `JpegToHtj2kError::Accelerator` carries the new
  type, and the Metal/CUDA transcode error `as_static_str` funnels are removed
  so backend failures keep their diagnostic detail.
- Renames backend capability detection to compile-time defaults and makes
  facade Metal/CUDA gating symmetric with those compile-time features.
- Renames the JPEG fast packet adapter surface from Metal-specific names to
  backend-neutral `JpegFast*` packet/table/error names.
- Trims and documents the J2K adapter surface, removing legacy preference
  aliases before the facade release.
- Replaces broad facade glob reexports with explicit export lists and adds the
  missing root `TileBatchDecodeDevice` and `TileBatchDecodeSubmit` traits.
- Narrows CUDA runtime root exports to explicit public modules and types.
- Makes `ProfileSummary` drop emission explicit; cloned summaries no longer
  inherit stderr side effects.
- Makes JPEG sampling-factor construction fallible instead of panicking on
  caller input, and adds explicit max-byte JPEG output-buffer constructors for
  callers that need to override the default allocation cap.

### Maturity And Fixes

- Fixes confirmed stale-cache, malformed-input, tile-grid, GPU validator, FFI
  cleanup, unsafe-deposit, and test-helper drift findings from the release
  audit.
- Enforces shared host allocation caps for codec-owned output scratch, aligns
  lightweight J2K inspect SIZ validation with full decode, and caps bounded J2K
  row-decode stripes by bytes as well as rows.
- Stops default CUDA builds from probing PATH `nvcc`; strict GPU validation now
  requires an explicit absolute `NVCC`.
- Strengthens unsafe, fuzz, Miri, and dependency-advisory governance in CI.
- Moves repository policy checks into `cargo xtask repo-lint` and pins public
  API, release-integrity, environment-variable, workflow, and packaging
  invariants there.
- Documents supported `J2K_*` environment variables and removes the
  experiment-only JPEG Metal fast420 split selector from the runtime surface.
- Routes env-gated Metal timing output through `j2k-profile`.
- Adds stricter J2K component-plane validation before output writes and removes
  stale generated-table dead-code suppression.
- Refreshes the stable public API inventory after facade and profile surface
  changes.
