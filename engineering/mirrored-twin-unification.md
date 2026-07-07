# Mirrored-Twin Unification Record

Date: 2026-07-04

This record closes audit item 46. Each audited mirrored-twin family is either
merged behind one implementation or intentionally left separate because the
golden-test comparison axis is behavioral, not syntactic.

## Unified Families

- Native/Metal packet ordering: `j2k-types` owns shared packet descriptors and
  progression ordering helpers; native and Metal call that one implementation.
- CUDA compact HTJ2K planning: `Htj2kCompactPlanJob` backs both compact public
  entry points.
- CUDA classic/HT decoded block copy: `DecodedSubBandBlock` backs both classic
  and HT wrappers.
- Adapter error classification: `j2k-core` owns `AdapterErrorKind`,
  `AdapterErrorParts`, and structural classifier helpers consumed by adapters.
- JPEG Metal Fast444: `FastSubsampledMetal` backs the full-ROI fast444 path.
- Metal direct required-region retain: `RequiredRegionJob` plus
  `retain_jobs_for_required_region` backs classic and HT job vectors.
- Metal direct sub-band group scanning: `prepare_sub_band_groups` backs classic
  and HT grouping; codec-specific group construction remains separate because
  classic carries segment tables and HT does not.
- Metal hybrid region-scaled planning: `RegionScaledColorPlanCache` and the
  shared cached planner back uncached, global-cache, and session-cache callers.
- JPEG sample-width upsample helpers: `UpsampleSample`,
  `upsample_h2v1_sample_at`, and `upsample_h2v2_rows_at` back u8 and u16
  wrappers.
- Metal shader classic batch kernels: split shader chunks now route classic
  code-block batch entry points through `j2k_encode_classic_code_blocks_dispatch`.

## Documented Waivers

- Extended12 versus Progressive12 JPEG decode is not a safe merge target for the
  entropy/DCT stage. Extended12 decodes sequential block pixels from scan bytes,
  while Progressive12 first merges progressive scans and dequantizes progressive
  DCT blocks. The shared surface is the post-DCT output path:
  `Extended12WriteRegion`, `Extended12Plane`, color sampling helpers, and the
  generic u16 upsample helpers are reused by both families.
- NEON `dual` versus `top_only` row-pair kernels are intentionally separate at
  the load/store stage. The top-only path avoids bottom-row loads and writes and
  uses top-row-adjacent fallback behavior for odd tails. The public dispatchers
  remain single entry points that choose the lane shape from the optional bottom
  row.
- Native IDWT f32 versus i64 remains separate at the filter implementation
  level. f32 handles irreversible 9/7 and reversible float fallback; i64 handles
  exact reversible 5/3 for high-bit-depth paths. Shared IDWT geometry and
  required-region propagation live in `j2k-native` helpers consumed by native,
  CUDA direct planning, and Metal direct execution.

## Golden Checks

The following narrow checks pin the unified and waived families without running
hardware-dependent or long corpus suites:

```bash
cargo test -p xtask --test repo_lint metal_direct_required_region_retain_uses_shared_job_helper
cargo test -p xtask --test repo_lint metal_direct_sub_band_group_scan_uses_shared_helper
cargo test -p xtask --test repo_lint metal_hybrid_region_scaled_cache_uses_shared_scope
cargo test -p xtask --test repo_lint jpeg_decoder_upsample_sample_width_twins_use_generic_helpers
cargo test -p xtask --test repo_lint mirrored_twin_unification_record_is_current
cargo test -p j2k-jpeg --test decode_into progressive12_ycbcr420
cargo test -p j2k-jpeg --test decode_into extended12_ycbcr420
cargo test -p j2k-jpeg --test batch session_batch_decode_progressive12_ycbcr420_matches_single_tile_decode
cargo test -p j2k-jpeg --test batch session_batch_decode_extended12_ycbcr420_matches_single_tile_decode
cargo test -p j2k-jpeg --test neon_hot_paths
cargo test -p j2k-native --test component_planes classic_reversible_i64_decode_round_trips_29_bit_native_bytes
cargo test -p j2k --test decode decode_rgb8_codestream_roundtrips_reversible_pixels
```
