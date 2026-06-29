# Metal Codec Acceleration Plan

This is the single working plan for expanding Apple Metal acceleration in the
J2K codec workspace. It replaces older fragmented roadmap notes and keeps the
scope codec-only: JPEG 2000 / HTJ2K decode, encode, transcode, residency,
backend routing, benchmarks, and publication evidence.

## Goal

Make J2K the obvious Rust codec choice for JPEG 2000 / HTJ2K on Apple Silicon:
CPU remains the correctness baseline, strict Metal requests fail clearly when a
shape is unsupported, and `Auto` uses Metal only for shapes with parity,
resident-path, and benchmark evidence.

The adoption-facing promise should be:

- fast Rust JPEG 2000 Part 1 and HTJ2K Part 15 decode, encode, and transcode
- conservative, predictable CPU fallback in `Auto`
- strict Metal errors for unsupported explicit device requests
- resident Metal output APIs where keeping bytes on-device is the win
- reproducible benchmark and parity evidence against external corpora

## Non-Goals

- Do not make viewer, DICOM UI, or WSI application behavior the codec roadmap.
- Do not claim full end-to-end Metal coverage until every stage in that route
  is resident and benchmarked.
- Do not widen `Auto` routing for coverage optics.
- Do not silently fall back for strict Metal requests.
- Do not optimize tiny, irregular, or transfer-bound workloads unless benchmark
  evidence shows a real win.
- Do not make browser image support a primary adoption story.

## Implementation Rules

- Reuse existing functions, traits, route planners, buffers, reports, and test
  helpers before adding new ones.
- Do not recreate a function that already exists under another module. Search
  first, then extend the existing implementation when ownership boundaries
  allow it.
- Match the repo's current structure. Public facade work belongs in `j2k`,
  native codec hooks belong in `j2k-native`, Metal runtime and kernels belong
  in Metal adapter/support crates, and benchmark/adoption automation belongs in
  existing bench or `xtask` surfaces.
- Keep architecture decisions centralized. High-level backend policy, public API
  shape, cross-crate boundaries, and route semantics must be decided in this
  plan or by the primary maintainer before implementation spreads across crates.
- Prove wins over CPU before widening `Auto`. A Metal path needs parity,
  dispatch accounting, benchmark evidence, and an explanation of why transfer
  and dispatch overhead do not erase the gain.
- Keep strict device behavior logically consistent. `BackendRequest::Metal`
  must mean supported Metal execution or a structured error, never silent CPU
  fallback.
- Prefer resident paths when residency is the reason Metal wins. If the route
  reads back to host after every stage, treat that as a likely benchmark
  blocker until proven otherwise.
- Keep code maintainable. Small targeted extensions beat broad special-case
  branches; shared behavior should live behind existing traits or local helper
  APIs, not ad hoc duplication.
- Keep failure modes explicit. Unsupported shapes, capacity failures, missing
  hardware, malformed inputs, and fallback decisions must surface clear errors
  or route-report reasons.
- Use Codex Spark subagents only when they make the work safer or faster, such
  as parallel codebase reconnaissance, independent benchmark-result review, or
  focused test gap analysis. The primary agent must give each subagent explicit
  scope, verify its findings against the code, and retain final responsibility
  for architecture, route policy, and correctness decisions.
- Treat subagent output as input, not authority. Merge only findings that are
  supported by source references, tests, benchmarks, or reproducible commands.

## Current Baseline

The workspace already has these Metal surfaces:

- `j2k-metal`: JPEG 2000 / HTJ2K decode surfaces and encode-stage adapters.
- `j2k-jpeg-metal`: Metal JPEG decode and baseline encode adapters.
- `j2k-transcode-metal`: JPEG-to-J2K/HTJ2K transform-stage acceleration.
- `j2k-metal-support`: shared Metal device, queue, library, and route helpers.

The current posture is correct and should remain:

- CPU is the correctness baseline.
- `BackendRequest::Auto` may return CPU output.
- `BackendRequest::Metal` is strict.
- GPU routes must be parity-covered and benchmark-gated.
- Public docs should say "Metal-accelerated stages" unless the route is fully
  resident end to end.

## Priority Order

1. Metal J2K/HTJ2K decode to resident surfaces.
2. Metal HTJ2K encode for large 8-bit Gray/RGB tiles and images.
3. Resident JPEG-to-HTJ2K transcode pipeline.
4. Auto routing and public facade ergonomics.
5. External benchmark/adoption evidence.
6. Broader format compatibility and edge-case widening.

## Workstream A: J2K / HTJ2K Decode

### Objective

Make Metal decode useful from the public codec surface for the workloads most
likely to adopt J2K: large tiled imagery, region decode, scaled decode, and
resident output.

### A1. Decode Support Matrix

Build and maintain a table covering:

- codec: classic J2K, JP2, HTJ2K, JPH
- operation: full image, region, scaled, region+scaled, batch
- output: `Gray8`, `Rgb8`, `Rgba8`, `Gray16`, `Rgb16`
- coding path: classic Tier-1, HT cleanup, HT refinement
- transform: reversible 5/3, irreversible 9/7
- color transform: none, RCT, ICT
- destination: host bytes, Metal buffer, Metal texture where applicable
- request mode: CPU, Auto, strict Metal
- status: supported, strict-rejected, Auto-CPU, planned

Current matrix:

| Codec/container | Operation | Pixel formats | Strict Metal status | Auto status | Destination | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| Classic J2K codestream | Full image | `Gray8`, `Gray16`, `Rgb8`, `Rgba8`, `Rgb16` | Supported and parity-tested | CPU until benchmark-gated | Metal buffer for strict, host bytes for Auto/CPU | `crates/j2k-metal/tests/device.rs` full grayscale/color strict tests |
| HTJ2K codestream | Full image | `Gray8` | Supported and parity-tested | CPU until benchmark-gated | Metal buffer for strict, host bytes for Auto/CPU | `crates/j2k-metal/tests/device.rs` HT grayscale strict tests |
| Classic J2K and HTJ2K codestream | Region | `Gray8` | Supported and parity-tested | CPU with fallback report | Metal buffer for strict, host bytes for Auto/CPU | `crates/j2k-metal/tests/device.rs` region strict tests and Auto report tests |
| Classic J2K and HTJ2K codestream | Scaled | `Gray8`; classic J2K `Rgb8` | Supported and parity-tested | CPU with fallback report | Metal buffer for strict, host bytes for Auto/CPU | `crates/j2k-metal/tests/device.rs` scaled strict/session tests and Auto report tests |
| Classic J2K and HTJ2K codestream | Region+scaled | `Gray8`, `Rgb8`, `Rgba8`, `Rgb16` | Supported and parity-tested | CPU with fallback report | Metal buffer for strict, host bytes for Auto/CPU | `crates/j2k-metal/tests/device.rs` region+scaled strict tests |
| Classic J2K and HTJ2K codestream | Full, region, scaled, region+scaled | `Rgba16` | Strict-rejected before launch | CPU when explicitly requested through CPU/Auto | Host bytes for CPU/Auto | `crates/j2k-metal/tests/device.rs` Rgba16 strict rejection tests |
| Classic J2K and HTJ2K codestream | Batch full and batch region+scaled | `Gray8`, `Gray16`, `Rgb8`, `Rgba8`, `Rgb16` | Implemented selectively; needs refreshed benchmark matrix | CPU unless batch Metal is explicitly requested | Metal buffer or host bytes by request | `crates/j2k-metal/src/batch.rs`, benchmark refresh pending |
| JP2 and JPH wrappers | Full, region, scaled, region+scaled | `Gray8`, `Rgb8`, `Rgba8`, `Gray16`, `Rgb16` | Planned until wrapper-specific parity rows are added | CPU | Host bytes | No public Metal claim yet |
| Any supported codec/container | Texture output | all formats | Planned | CPU/host or buffer-only today | Not exposed for J2K decode | No public Metal texture claim yet |

Acceptance:

- Matrix lives in this file or a linked generated artifact.
- Every public Metal decode claim maps to a row.
- Unsupported strict Metal rows have explicit error text tests.

### A2. Full Decode To Resident Metal Surface

Stabilize and document strict Metal full decode for:

- classic grayscale 8-bit and 16-bit
- classic RGB 8-bit and 16-bit where color plan support exists
- HTJ2K grayscale and RGB 8-bit
- JPH wrappers where native decode already resolves file metadata correctly

Implementation direction:

- Keep code-block decode hooks and direct-plan execution isolated inside
  `j2k-metal`.
- Prefer resident output when decoded component planes can be reused.
- Fall back to host staging only through explicit CPU-staged APIs, not strict
  Metal decode.

Tests:

- Full strict Metal decode matches CPU bytes for each supported shape.
- `SurfaceResidency::MetalResidentDecode` is asserted for resident routes.
- Strict Metal rejects unsupported pixel formats before launching kernels.

### A3. Region And Scaled Decode

Prioritize `decode_region_scaled_to_device` because it is the most useful codec
primitive for large images.

Implementation direction:

- Expand direct grayscale/color plan coverage only when ROI and scale geometry
  can be represented without extra host transfers.
- Cache prepared plans for repeated same-image/same-shape calls.
- Preserve structured fallback reasons for unsupported direct plans.

Tests:

- Region decode parity against CPU.
- Scaled decode parity against CPU.
- Region+scaled decode parity against CPU.
- Repeated plan cache tests assert reduced planning work.
- Auto stays CPU until benchmark evidence justifies route widening.

### A4. Batched Tile Decode

Make same-shaped batch decode a first-class codec benchmark target.

Implementation direction:

- Group compatible requests by codec, dimensions, pixel format, wrapper kind,
  transform, and output residency.
- Submit compatible batches through one Metal session.
- Keep distinct/irregular batches CPU unless measured otherwise.

Tests:

- Mixed compatible batches preserve output order.
- Unsupported item failure is surfaced per request.
- Batch errors preserve strict Metal error categories.

Benchmarks:

- batch size `1,16,64,256,1024`
- full decode
- region+scaled decode
- resident buffer output
- host readback output

## Workstream B: HTJ2K Encode

### Objective

Make Metal HTJ2K encode the clearest Apple Silicon win for large 8-bit
Gray/RGB inputs while keeping classic J2K and marker-heavy shapes conservative.

### B1. Encode Support Matrix

Track encode support by:

- codec: classic J2K, HTJ2K
- quality: lossless, lossy
- components: Gray, RGB, RGBA only if alpha semantics are explicit
- bit depth: 8, 16, high-bit CPU-only
- transform: none, RCT 5/3, ICT 9/7
- DWT levels
- code block size
- progression order
- marker requests: TLM, PLT, PLM, PPM, PPT, SOP, EPH
- output: host codestream, Metal buffer codestream
- route: stage-by-stage, resident batch, CPU

Acceptance:

- Existing stage dispatches remain documented.
- Resident paths and stage-only paths are clearly separated.
- Public docs never imply unsupported end-to-end Metal encode.

### B2. Resident Lossless HTJ2K RGB8 / Gray8

Prioritize large-tile lossless HTJ2K because it has the cleanest adoption
story and existing acceleration pieces.

Implementation direction:

- Keep large Auto gate conservative.
- Support `Gray8` and `Rgb8`.
- Preserve RCT, DWT 5/3, HT cleanup encode, and CPU packetization until
  resident packetization is benchmarked.
- Prefer `submit_lossless_batch_to_metal` for resident codestream outputs.

Acceptance:

- Auto large tiles dispatch Metal stages.
- Small tiles remain CPU in Auto.
- Strict Metal requires expected device stages or returns unsupported.
- Round-trip validation passes against CPU decode.

### B3. Resident Packetization And Codestream Assembly

Move packetization and codestream assembly to Metal only for shapes where it
beats CPU after accounting for setup and readback.

Implementation direction:

- Keep CPU packetization as oracle.
- Add resident packetization only when packet descriptors can be bounded.
- Keep marker-heavy shapes CPU until parity and capacity tests are complete.
- Maintain memory-budget and in-flight tile controls.

Tests:

- Packet bytes match CPU packetization.
- Marker requests reject or route CPU explicitly when unsupported.
- Capacity failures produce structured errors.
- Conservative retry paths do not hide correctness failures.

Benchmarks:

- resident host codestream output
- resident Metal buffer codestream output
- host readback time
- packetization-only time
- codestream assembly time
- memory budget sensitivity

### B4. Lossy HTJ2K 9/7 Encode

Prioritize only after lossless resident path is stable.

Implementation direction:

- Use Metal ICT, forward 9/7 DWT, and quantization where benchmarked.
- Keep rate allocation and marker-heavy packetization CPU until evidence says
  otherwise.
- Report output size, rate target, PSNR where available, and speed.

Acceptance:

- Lossy CPU and Metal-stage outputs decode successfully.
- Quality report remains stable.
- Auto route requires end-to-end win, not only stage win.

### B5. Classic J2K Encode

Treat classic J2K Metal encode as selective acceleration, not the flagship.

Implementation direction:

- Keep coefficient-prep Auto gates at benchmark-backed dimensions.
- Do not automatically dispatch classic Tier-1 in host-output Auto until
  end-to-end evidence is positive.
- Keep resident classic work experimental until packet and token-pack routes are
  stable across representative corpora.

Acceptance:

- Stage parity tests remain strong.
- Strict Metal errors identify missing required stages.
- Auto does not regress small-image encode.

## Workstream C: JPEG-to-HTJ2K Transcode

### Objective

Reduce host/device transfers across JPEG decode, coefficient projection,
HTJ2K encode, packetization, and output so batch transcode becomes an obvious
codec-level Apple Silicon feature.

### C1. Pipeline Residency Map

Make the transcode path report:

- JPEG entropy decode residency
- DCT coefficient residency
- DCT-grid to DWT projection residency
- quantization residency
- HT block encode residency
- packetization residency
- codestream assembly residency
- host/device transfer count and byte count
- CPU fallback points and reasons

Acceptance:

- Every timed transcode report includes the residency map.
- Auto declines unsupported segments with structured reasons.
- Strict Metal fails if a required segment is unsupported.

### C2. Resident Handoff Types

Add explicit handoff types between crates instead of forcing host materialization.

Candidate types:

- JPEG Metal decoded/coefficient packet handles
- DCT-grid resident buffers
- projected DWT subband resident buffers
- prequantized HTJ2K component buffers
- encoded HT code-block resident buffers
- packetized codestream resident buffers

Rules:

- Types must carry dimensions, sampling, bit depth, signedness, color, and
  buffer lifetime clearly.
- Types must not expose invalid raw pointers or unchecked offsets.
- Host readback must be explicit.

### C3. Batch Auto Thresholds

Current single-job transcode thresholds should stay conservative. Focus on
same-geometry batches first.

Benchmarks:

- batch sizes `1,16,32,64,256`
- 224x224, 256x256, 512x512, 1024x1024 tiles
- 4:2:0, 4:2:2, 4:4:4 JPEG sources where supported
- lossless 5/3 and lossy 9/7 HTJ2K targets
- host output and resident output

Acceptance:

- Auto thresholds are justified by benchmark rows.
- Distinct geometry batch controls remain CPU unless measured.
- Thresholds are encoded as policy constants with tests.

## Workstream D: JPEG Metal Input Adapter

### Objective

Keep JPEG Metal support selective, but make accepted baseline paths excellent
for codec transcode and benchmark workloads.

### D1. Fast Packet Coverage

Maintain strict support for:

- baseline 4:2:0
- baseline 4:2:2
- baseline 4:4:4
- `Gray8`, `Rgb8`, `Rgba8` output

Do not widen to progressive, CMYK, YCCK, or exotic marker shapes until strict
error coverage and benchmarks exist.

### D2. Auto Routing

Single-image JPEG decode should remain CPU unless cold-start and warm-session
benchmarks show a repeatable win.

Auto candidates:

- coalesced tile batches
- region+scaled batch decode
- resident texture/buffer output
- transcode batches that consume resident output

Acceptance:

- Auto route changes cite benchmark group names.
- Accepted and rejected fast-packet planning overhead is measured.
- Small restart-coded tile batches stay CPU unless evidence changes.

## Workstream E: Public API And Ergonomics

### Objective

Make acceleration discoverable from stable public APIs without forcing normal
users into internal stage accelerators.

### E1. Route Reports

Expose enough route detail that users can answer:

- Did Metal run?
- Which stages dispatched?
- Why did Auto choose CPU?
- Was output resident or host-backed?
- How many host/device transfers happened?

Acceptance:

- Public reports avoid leaking unstable internal kernel names.
- Reports distinguish final output backend from stage dispatch backend.
- `Auto` CPU decisions are explainable.

### E2. Facade Helpers

Add or improve high-level helpers for common codec calls:

- decode to resident Metal surface
- region+scaled decode to resident Metal surface
- encode lossless HTJ2K with Auto acceleration
- transcode JPEG to HTJ2K with Auto acceleration
- submit batch decode/encode/transcode work with reusable Metal session

Rules:

- Keep `j2k` as the primary user-facing crate.
- Adapter crates may expose advanced session/resident APIs.
- Strict device requests must never silently change backend.

### E3. Examples

Add minimal examples that are codec-only:

- decode JP2/JPH to host bytes with Auto
- decode J2K/JPH to Metal surface with strict Metal
- region+scaled decode with Auto and route report
- lossless HTJ2K encode with Auto and dispatch report
- JPEG-to-HTJ2K batch transcode with residency report

Each example should:

- use generated or tiny fixture data where possible
- print route report fields
- avoid app/viewer concerns
- document expected CPU fallback on non-macOS hosts

Current examples:

- `crates/j2k-metal/examples/decode_route_report.rs` prints Auto decode
  fallback reason, output residency, and strict Metal decode behavior for a
  generated HTJ2K region+scaled decode.
- `crates/j2k-metal/examples/htj2k_encode_auto_report.rs` prints final encode
  backend and per-stage dispatch counts for generated lossless HTJ2K RGB8 Auto
  encode.
- `crates/j2k-metal/examples/resident_encode_buffer.rs` submits a strict Metal
  lossless HTJ2K encode to a Metal-backed codestream buffer and validates the
  produced bytes through the CPU decoder.
- `crates/j2k-transcode-metal/examples/jpeg_to_htj2k_route_report.rs` runs
  JPEG-to-HTJ2K through the Metal route facade and prints requested backend,
  selected transform backend, output backend, structured fallback reason,
  transfer bytes, and the transcode pipeline residency map.

## Workstream F: Benchmark And Evidence

### Objective

Make performance claims publishable and trusted.

### F1. Required Benchmark Matrix

Maintain benchmark rows for:

- CPU baseline
- strict Metal where supported
- Auto
- host output
- resident output
- batch sizes `1,16,64,256,1024`
- generated smoke fixtures
- external natural-image fixtures
- external conformance/interoperability fixtures
- domain-style large tiled fixtures where license permits

### F2. External Corpus Policy

Public adoption claims require manifest-backed external rows. Generated
fixtures are development evidence, not adoption evidence.

Required corpus categories:

- ISO conformance vectors where available
- OpenJPEG data
- OpenJPH / HTJ2K fixtures
- natural RGB/gray source images materialized to J2K/HTJ2K
- large-image tiles from non-private sources where possible

Report:

- corpus name
- license status
- source command
- input hash
- codec/container
- comparator availability
- skipped rows and reasons

### F3. Performance Report Requirements

Every Metal performance claim must include:

- hardware model
- OS version
- Metal version where available
- git revision and dirty state
- exact command
- input source and manifest
- batch size
- output residency
- host readback policy
- CPU thread policy
- comparator versions
- skipped rows
- route dispatch counts
- publication eligibility status

### F4. Regression Gates

Add or preserve narrow gates:

- `cargo test -p j2k-metal --all-targets`
- `cargo test -p j2k-jpeg-metal --all-targets`
- `cargo test -p j2k-transcode-metal --all-targets`
- `cargo test -p j2k --all-targets`
- `cargo xtask public-support`
- `cargo xtask adoption-report`

Hardware gates:

- ignored Metal encode auto-routing benchmark
- JPEG Metal compare benchmark
- transcode Metal benchmark
- adoption benchmark with `--require-metal` on Apple Silicon runners

## Workstream G: Correctness, Safety, And Compatibility

### Objective

Keep acceleration from weakening codec correctness or API trust.

### G1. Strict Error Policy

Strict Metal must return structured errors for:

- unsupported pixel format
- unsupported component count
- unsupported bit depth
- unsupported color transform
- unsupported marker request
- unsupported packet shape
- unsupported geometry
- unavailable Metal runtime
- memory/capacity budget failure

### G2. Parity Policy

Every new Metal stage needs:

- CPU oracle parity test
- malformed or unsupported input test where applicable
- explicit strict rejection test
- Auto fallback test
- dispatch-count assertion

### G3. Memory Safety Policy

For resident buffers:

- validate offsets and lengths before launch
- validate `u32` ABI conversions
- validate row pitch and output shape
- avoid unchecked host-readable assumptions
- keep unsafe blocks localized and audited

### G4. High-Bit And Exotic Shape Policy

High-bit, signed, sampled, component-mapped, palette, and metadata-heavy paths
should stay CPU until there is a clear Metal implementation plan and external
parity coverage.

## Milestones

### Milestone 0: Plan And Cleanup

- Remove old fragmented roadmap docs.
- Keep one codec-only Metal plan.
- Ensure no public docs reference deleted roadmap files.

Exit criteria:

- `docs/roadmap/metal-codec-acceleration-plan.md` is the only roadmap plan.
- Broken internal links are fixed or absent.

### Milestone 1: Decode Matrix And Strict Errors

- Add decode support matrix.
- Add missing strict Metal rejection tests.
- Add route report fields for decode residency and fallback reason.

Exit criteria:

- Full/region/scaled/region+scaled support is documented.
- Every strict unsupported decode shape fails before kernel launch.

### Milestone 2: Resident Decode Benchmarks

- Add/refresh resident decode benchmarks.
- Include host readback and no-readback rows.
- Add external manifest support where missing.

Current status:

- `crates/j2k-metal/tests/metal_decode_benchmark.rs` records CPU, strict Metal
  resident/no-readback, and strict Metal readback rows.
- `cargo xtask adoption-benchmark --metal` runs the Metal decode benchmark and
  summarizes its rows in `summary.json`.
- Generated rows collected on June 28, 2026 on Apple M4 Pro showed CPU faster
  than strict Metal for the sampled full and region+scaled decode shapes, so
  `Auto` decode remains CPU.
- External manifest-backed decode rows are supported through
  `J2K_METAL_DECODE_INPUT_DIRS` and `J2K_METAL_DECODE_MANIFEST`, but must still
  be run before any public Metal decode speed claim.

Exit criteria:

- Benchmark report can justify which decode shapes Auto may use.
- Small and distinct/irregular controls are measured.

### Milestone 3: HTJ2K Encode Auto Polish

- Stabilize large Gray8/RGB8 HTJ2K lossless Auto route.
- Improve route reports for stage dispatches.
- Clarify final host output backend versus accelerated stages.

Current status:

- `EncodedJ2k.backend` reports the backend that satisfied the encode contract;
  host-output Auto can still report `Cpu` while `dispatch_report` records Metal
  stage work.
- `crates/j2k-metal/examples/htj2k_encode_auto_report.rs` prints final backend
  and per-stage dispatch counts for generated lossless HTJ2K RGB8 Auto encode.
  On June 28, 2026, this example produced a CPU final backend with five Metal
  stage dispatches on Apple M4 Pro, so the user-visible route distinction is now
  documented and runnable.
- Existing tests cover small Auto CPU fallback, large HTJ2K RGB8/Gray8 stage
  dispatches, and resident private-input paths in `crates/j2k-metal/src/encode/tests.rs`.

Exit criteria:

- Users can call a high-level API and see why Metal did or did not run.
- Large HTJ2K encode rows have reproducible evidence.

### Milestone 4: Resident Encode Output

- Strengthen `submit_lossless_batch_to_metal`.
- Add examples for resident codestream output.
- Benchmark packetization/codestream assembly residency.

Current status:

- `crates/j2k-metal/examples/resident_encode_buffer.rs` covers the public
  `submit_lossless_batch_to_metal` path for a generated Gray8 strict HTJ2K
  resident codestream buffer on macOS.
- Existing resident encode tests cover host-output and Metal-buffer output
  parity, packetization use, codestream assembly use, memory budget behavior,
  and host readback timing fields in `crates/j2k-metal/src/encode/tests.rs`.

Exit criteria:

- Resident buffer output has parity, capacity, and benchmark evidence.
- Host readback cost is explicit.

### Milestone 5: Transcode Residency

- Add resident handoff types.
- Connect supported JPEG Metal batch output to Metal transcode stages.
- Connect transcode output to HTJ2K encode inputs without unnecessary readback.

Current status:

- `TranscodeTimingReport` and `TranscodePipelineMap` now expose visible
  host/device transfer counts and byte counts for staged 9/7 Metal transcode
  pack/upload and readback boundaries.
- `j2k-transcode-metal` exposes `jpeg_to_htj2k_with_metal_route` and
  `jpeg_to_htj2k_batch_with_metal_route` for CPU, Auto Metal, and strict Metal
  routing. Auto returns structured CPU fallback reasons; strict Metal uses the
  explicit accelerator and fails if no Metal dispatch satisfies successful
  work.
- `crates/j2k-transcode-metal/benches/dct97.rs` includes same-geometry batch
  Rayon CPU, Auto Metal, and explicit Metal benchmark cases. When
  `J2K_TRANSCODE_METAL_PROFILE_STAGES` is set, profile rows label the actual
  selected transform processor, keep CPU and Metal variants under a stable
  workload context, and include host-to-device/device-to-host transfer counts,
  byte counts, and resident DCT/DWT handoff descriptor counts.
- A focused same-geometry batch row was collected on June 28, 2026 on Apple M4
  Pro for generated 224 x 224 sRGB/YBR 4:2:0 JPEG tiles, batch size 128:
  Rayon CPU 86.581-89.594 ms, Auto Metal 57.334-58.776 ms, strict Metal
  58.053-65.069 ms. The profiled run emitted 93 CPU rows labeled
  `request=cpu path=cpu transform_processor=cpu`, 118 Auto rows labeled
  `request=metal_auto path=auto transform_processor=metal`, and 118 strict rows
  labeled `request=metal_explicit path=metal transform_processor=metal`; all
  three route variants used `context=srgb_ybr420_224_batch_128`.
- `j2k-transcode` now exposes backend-neutral resident handoff descriptors:
  `ResidentBufferRef`, `ResidentJpegDctGrid`, `ResidentDwtSubband`, and
  `ResidentCodestreamBuffer`. Constructors validate buffer ranges, dimensions,
  sampling, bit depth, stride, backend requirements, and codestream capacity
  without exposing raw pointers.
- Staged 9/7 Metal transcode validates `ResidentJpegDctGrid` descriptors for
  uploaded DCT block buffers and `ResidentDwtSubband` descriptors for resident
  DWT band buffers before readback or quantization. The pipeline map reports
  the validated handoff counts.
- Buffer-backed Metal HTJ2K/J2K encode output exposes a device-memory range,
  and `j2k-transcode-metal` converts it into a `ResidentCodestreamBuffer`
  descriptor with allocation, capacity, and backend validation.
- The broader transcode batch matrix and distinct-geometry controls are still
  pending before making wider Auto claims.

Exit criteria:

- Pipeline map shows fewer host/device transfers.
- Batch JPEG-to-HTJ2K transcode has Metal and CPU comparison rows.

### Milestone 6: Auto Routing Expansion

- Widen Auto only for routes that passed prior benchmark gates.
- Add policy constants and tests for thresholds.
- Keep strict device behavior unchanged.

Exit criteria:

- Auto improvements are explainable by public benchmark artifacts.
- No small-shape regressions.

### Milestone 7: Adoption Bundle

- Run full adoption benchmark bundle on Apple Silicon.
- Include external manifest-backed decode and encode corpora.
- Publish concise benchmark report.
- Update docs pages with exact support language.

Exit criteria:

- `adoption-report` says Metal claims are publication eligible.
- Public docs avoid unsupported end-to-end claims.

## Suggested Issue Breakdown

1. Create decode Metal support matrix.
2. Add strict Metal decode rejection tests for unsupported formats.
3. Add route-report fields for decode fallback reason.
4. Benchmark full decode host versus resident output.
5. Benchmark region+scaled decode host versus resident output.
6. Benchmark same-shaped batch decode across batch sizes.
7. Add external corpus manifests to Metal decode benches.
8. Stabilize large HTJ2K RGB8 Auto encode report.
9. Add Gray8 large HTJ2K Auto encode benchmark row.
10. Add resident HTJ2K encode host-readback benchmark row.
11. Add resident HTJ2K encode Metal-buffer output benchmark row.
12. Add packetization parity tests for resident encode route.
13. Add packetization capacity failure tests.
14. Add codestream assembly residency benchmark.
15. Add JPEG-to-HTJ2K transcode residency report fields.
16. Add resident DCT-grid handoff type.
17. Add resident projected-DWT handoff type.
18. Add resident prequantized HTJ2K component type.
19. Add batch transcode benchmark with same geometry.
20. Add batch transcode benchmark with distinct geometry control.
21. Add Auto threshold tests for transcode batch routing.
22. Add codec-only Metal decode example.
23. Add codec-only HTJ2K encode Auto example.
24. Add codec-only JPEG-to-HTJ2K transcode example.
25. Update public docs pages with final support wording.

## Definition Of Done

This plan is complete when:

- Metal decode has clear public support rows and strict rejection rows.
- Large HTJ2K encode has public Auto ergonomics and benchmark evidence.
- JPEG-to-HTJ2K transcode can keep supported batch stages resident.
- `Auto` Metal routes are benchmark-backed and tested.
- Strict Metal requests never silently fall back.
- External adoption benchmark reports are publication eligible.
- Public docs describe exactly what is accelerated and what remains CPU.
