# Adaptive J2K / HTJ2K RCA and Gates Design

Date: 2026-05-31

## Purpose

Improve the adaptive JPEG 2000 / HTJ2K GPU route evidence without promoting
Metal or CUDA to the default route prematurely.

The work keeps `ACCELERATED` conservative: CPU remains the default unless a
stage gate and the matching end-to-end gate both pass for the measured workload
shape. Strict device paths remain diagnostic capability proof and must fail
loudly when required.

CUDA work is intentionally last in the implementation sequence. The first
passes tighten shared gate reporting and Metal resident encode RCA, then CUDA
profiling and CUDA RGB/RGBA decode benches are added after those surfaces are
stable.

## Scope

In scope:

- Preserve candidate stage evidence in adaptive route reports when the
  end-to-end gate is missing or failing.
- Add tests that prevent stage candidates from silently becoming default GPU
  routes.
- Add Metal resident HTJ2K encode RCA timing fields for route-composition
  overhead.
- Add facade benchmark rows and harness guards needed for later RGB/RGBA
  512/1024 gate reruns.
- Add CUDA detailed decode profiling and CUDA RGB/RGBA decode benches last.
- Update gate documentation only after measured evidence is available.

Out of scope:

- Promoting Metal or CUDA as an adaptive default route.
- Moving codec logic into adapter crates.
- Treating strict device dispatch as production route approval.
- Treating `Rgba8` CUDA decode rows as evidence for true four-component HTJ2K
  input; the intended first row is RGBA output from a three-component RGB
  codestream.
- Publishing performance claims from gate docs.

## Architecture

The existing crate boundaries stay unchanged:

- `signinum-j2k-native` remains the codec engine.
- `signinum-j2k` owns public J2K / HTJ2K API and adaptive route planning.
- `signinum-j2k-metal` and `signinum-j2k-cuda` remain adapters.
- `signinum` remains the stable facade.

Adaptive reports must distinguish:

- CPU-shaped stages.
- Logical GPU-shaped stages with missing benchmark evidence.
- Logical GPU-shaped stages that pass stage evidence but are blocked by the
  end-to-end route gate.
- Logical GPU-shaped stages that fail stage evidence and require RCA.
- Exact stage/backend reclassifications to CPU after RCA.

Only stages with passing stage evidence and passing end-to-end evidence may be
selected for a device backend in an adaptive route.

## Components

### Planner and Gate Reporting

`crates/signinum-j2k/src/adapter/adaptive_route.rs` will keep enough evidence in
`J2kAdaptiveRouteReport` to show candidate stages even when the route remains
CPU-only because the end-to-end gate is missing or failing.

Tests in `crates/signinum-j2k/tests/adaptive_route.rs` will cover:

- Stage evidence passes but end-to-end evidence is missing.
- Stage evidence passes but end-to-end evidence fails.
- The selected backend remains CPU in both cases.
- RCA reclassification applies only to the exact stage/backend pair.
- Full promotion still requires both stage and end-to-end gates.

### Metal Resident Encode RCA

`crates/signinum-j2k-metal` will add additive resident encode timing fields. The
fields will preserve existing public stats while adding more useful RCA buckets:

- coefficient prep
- fused deinterleave/RCT where applicable
- DWT53
- coefficient extraction
- HT block encode
- packet block prep
- packetization
- codestream assembly
- sync/wait
- host readback where the API materializes bytes

Metal timing labels must be honest about what they measure. `Instant` buckets
represent host planning, submission, and waiting overhead; they are not claimed
as exact GPU execution time unless sourced from GPU timestamp data.

The Metal work must not change `Auto` route behavior. Resident host encode stays
strict/diagnostic until gates prove otherwise.

### Facade Gate Rows

Facade benchmarks will be extended to make later gate reruns possible for
RGB/RGBA 512/1024 HTJ2K encode rows. Harness tests will assert that CPU,
adaptive, and strict-device row names remain present and that strict rows honor
`SIGNINUM_REQUIRE_*_BENCH` behavior.

These rows are evidence capture infrastructure only. They do not approve a
default GPU route by themselves.

### CUDA Decode Profiling

CUDA is implemented after the shared planner, Metal RCA, and facade scaffolding.

The CUDA profile change will preserve existing summary field names and add a
detailed companion report or additive fields that avoid source-breaking public
struct changes where practical. The detailed profile should separate:

- wall total versus stage sum
- table/resource upload
- payload upload
- job/status upload
- status readback
- HT cleanup/refinement attribution
- dequant
- IDWT
- MCT
- store/format conversion
- explicit output download if a profiled download helper is used
- per-stage dispatch counts

Existing fused cleanup/refinement timing must not be double-counted in totals.
If refinement remains fused with cleanup, the report should label refinement as
attribution-only instead of presenting it as an independent kernel duration.

Strict CUDA decode remains CUDA-resident. CPU-staged upload remains available
only through explicitly named CPU-staged APIs.

### CUDA RGB/RGBA Decode Benches

`crates/signinum-j2k-cuda/benches/htj2k_decode.rs` will add RGB8 and RGBA8 rows
after CUDA profiling is in place. Existing Gray8 labels remain unchanged.

Rows to add for each color format:

- full tile
- ROI
- scaled
- ROI-scaled
- tile batch

Fixtures are generated in the benchmark with native HTJ2K encode, matching the
current Gray8 approach. The first RGBA row decodes a three-component RGB
codestream into RGBA output.

CUDA rows must assert strict CUDA residency, not CPU decode plus upload.

## Implementation Order

1. Planner and adaptive route tests.
2. Metal resident encode RCA stats and Metal bench/profile output.
3. Facade gate row scaffolding and bench harness tests.
4. Documentation scaffolding that defines how new evidence will be recorded,
   without adding unmeasured numbers.
5. CUDA detailed decode profiling.
6. CUDA RGB/RGBA decode benches and CUDA bench harness tests.
7. Timed Metal, CUDA, and facade gate runs.
8. Final `docs/adaptive-j2k-gates.md` update with measured decisions.

## Verification

Required local checks before claiming code progress:

```sh
cargo fmt --all
git diff --check
cargo test -p signinum-core --test repo_integrity public_docs_describe_facade_auto_and_cuda_runtime_surface_scope
cargo test -p signinum-j2k --test adaptive_route
cargo test -p signinum --test bench_harness
```

Metal-specific checks when on an Apple Silicon Metal host:

```sh
cargo test -p signinum-j2k-metal resident_lossless_stage_stats_default_to_zero
SIGNINUM_REQUIRE_METAL_BENCH=1 cargo test -p signinum-j2k-metal metal_padded_private_ht_encode_to_metal_buffer_stays_resident -- --nocapture
SIGNINUM_REQUIRE_METAL_BENCH=1 SIGNINUM_J2K_METAL_PROFILE_STAGES=1 cargo bench -p signinum-j2k-metal --bench encode_stages -- --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2
SIGNINUM_REQUIRE_METAL_BENCH=1 cargo bench -p signinum --bench facade --features metal -- facade_j2k_htj2k_encode_backend_speed_matrix --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2
```

CUDA-specific checks, run last on the CUDA runner:

```sh
cargo check -p signinum-j2k-cuda --features cuda-runtime --benches --tests
cargo clippy -p signinum-j2k-cuda --features cuda-runtime --all-targets -- -D warnings
SIGNINUM_REQUIRE_CUDA_RUNTIME=1 SIGNINUM_REQUIRE_CUDA_HTJ2K_STRICT=1 cargo test -p signinum-j2k-cuda --features cuda-runtime --test host_surface
cargo bench -p signinum-j2k-cuda --features cuda-runtime --bench htj2k_decode --no-run
SIGNINUM_REQUIRE_CUDA_BENCH=1 cargo bench -p signinum-j2k-cuda --features cuda-runtime --bench htj2k_decode -- --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2
SIGNINUM_REQUIRE_CUDA_BENCH=1 cargo bench -p signinum --bench facade --features cuda-runtime -- facade_j2k_htj2k_encode_backend_speed_matrix --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2
```

## Risks

- Adding fields to public structs can be source-breaking for external struct
  literals. Prefer additive companion reports or accessors where practical.
- Host-side `Instant` timing can be mistaken for GPU execution timing. Field
  names and docs must distinguish wall/host overhead from GPU event duration.
- Bench rows can silently skip without `SIGNINUM_REQUIRE_*_BENCH`; signoff runs
  must set the require variables.
- RGB/RGBA CUDA decode rows may still lose because current color ROI paths are
  route-expensive. Losing rows remain RCA evidence, not default-route approval.
- Documentation must not mix Apple Metal host CPU timings with CUDA runner
  timings as if they came from the same machine.
