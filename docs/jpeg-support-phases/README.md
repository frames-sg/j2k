# JPEG Support Expansion Phases

This document is the working roadmap for broadening Signinum JPEG support
without weakening the current WSI-oriented fast paths. It covers the requested
CPU-first tranche:

- A: 8-bit sequential CMYK/YCCK decode and progressive ROI/scaled decode
- B: 12-bit extended/progressive JPEG decode
- C: lossless SOF3 decode

Arithmetic-coded, hierarchical, and differential JPEG remain explicit future
work outside this tranche. They should continue to report structured
unsupported errors until a separate entropy and conformance plan exists.

## Current Baseline

`signinum-jpeg` currently has strong support for common 8-bit Huffman JPEG:

- baseline sequential and extended sequential 8-bit CPU decode
- grayscale, YCbCr, and APP14 RGB CPU decode
- initial 8-bit sequential CMYK/YCCK CPU conversion to `Rgb8`/`Rgba8`
- ROI, scaled, region-scaled, tile-batch, batch-session, and reusable host
  output paths for sequential WSI-shaped work
- progressive 8-bit full-image, ROI, scaled, and region-scaled CPU decode via
  full progressive coefficient assembly and output projection
- parser metadata for 12-bit extended/progressive and lossless SOF3
- initial 12-bit extended sequential and progressive grayscale full-image/ROI/
  scaled/region-scaled CPU decode to `Gray16` or expanded `Rgb16`, plus
  initial 12-bit APP14 RGB 4:4:4 and YCbCr 4:4:4/4:2:2/4:2:0 full-image/
  ROI/scaled/region-scaled CPU decode to `Rgb16`; other 12-bit subsampled
  color support and stronger non-constant 12-bit oracle fixtures remain open
- initial lossless SOF3 8-bit grayscale full-image/ROI/scaled/region-scaled CPU
  decode to `Gray8` and 16-bit grayscale decode to `Gray16` for predictors
  1-7, including restart-coded grayscale streams, plus 8-bit APP14 RGB decode
  to `Rgb8`, including restart-coded APP14 RGB streams; YCbCr/16-bit color
  layouts and row output remain open

`signinum-jpeg-metal` currently accelerates selected 8-bit YCbCr fast packet
shapes:

- resident RGB8 buffer/texture output for fast 4:2:0, 4:2:2, and 4:4:4
- full, scaled, and region-scaled batch shapes with caller-owned reusable
  output
- cached decoder batch preflight and resident viewport helpers

These are not universal JPEG claims. They are the starting point for parity and
measured acceleration work.

## Global Rules

1. CPU decode parity lands before Metal acceleration for any new JPEG class.
2. CPU parity must have committed fixtures, reference output, and focused API
   tests before benchmark or backend routing changes.
3. Metal routes must be resident routes. Do not satisfy explicit Metal requests
   by CPU-decoding and uploading pixels under the same API.
4. `BackendRequest::Auto` may only choose Metal for benchmark-approved shapes.
   Unsupported or unproven shapes stay CPU-backed.
5. Capability reports must distinguish:
   - parser recognition
   - CPU decode support
   - Metal resident support
   - benchmark-approved Auto eligibility
6. Docs must avoid saying "all JPEGs" until arithmetic, hierarchical, and
   differential JPEG have a separate accepted plan and implementation.

## Phase 0: Docs, Fixtures, And Capability Audit

Purpose: make existing claims honest before broadening decode support.

Required work:

- Update support docs to state current JPEG limits and link to this roadmap.
- Add or extend capability tests for current support and rejection reasons:
  initial CMYK/YCCK CPU support, `Extended12`, `Progressive12`, `Lossless`,
  progressive ROI/scaled CPU support, arithmetic SOFs, hierarchical SOFs, and
  differential SOFs.
- Build or import small conformance fixtures for A/B/C:
  - CMYK baseline and YCCK baseline
  - progressive 8-bit with ROI/scaled reference output
  - 12-bit extended sequential
  - 12-bit progressive
  - lossless SOF3 predictor cases
- Record the reference source per fixture. Prefer libjpeg-turbo where it
  supports the class; otherwise record another deterministic oracle and the
  exact command/tool version.

Exit criteria:

- Docs describe current limits and the target phases without contradiction.
- Every unsupported A/B/C input has a structured test proving the current
  rejection before implementation starts.

## Phase A1: 8-Bit Sequential CMYK/YCCK CPU Decode

Purpose: support common 8-bit sequential four-component JPEGs on CPU.
12-bit four-component interactions stay out of this phase and are handled only
after the Phase B precision work is stable.

Implementation requirements:

- Keep parser APP14 handling as the source of truth for CMYK vs YCCK.
  Status: initial APP14 CMYK/YCCK fixtures use parser-owned color metadata.
- Add CPU output conversion to `Rgb8` and `Rgba8` for 8-bit CMYK/YCCK inputs.
  Status: `Rgb8` full-image, ROI, scaled, region-scaled, and batch-session
  coverage has landed for 8-bit sequential APP14 CMYK/YCCK fixtures. `Rgba8`
  full-image and ROI coverage has landed; `decode_rows` and region component
  row output are covered for the supported RGB output shape. Scaled `Rgba8`
  remains unsupported by the public output-format policy.
- Preserve clear behavior for unsupported direct CMYK output unless
  `signinum-core` gains a public CMYK pixel format.
- Add row, full-image, ROI, scaled, region-scaled, and batch tests where the
  existing API shape supports them.
  Status: full-image, ROI, scaled, region-scaled, and batch-session coverage
  has landed for the supported RGB/RGBA output shapes. Row output has landed
  for the supported RGB row surfaces. Subsampled four-component fixtures and
  malformed coverage remain open.
- Add capability reasons that distinguish "recognized but CPU unsupported"
  from "supported on CPU but not Metal resident".
  Status: capability reports now mark supported CMYK/YCCK CPU RGB8/RGBA8
  shapes while keeping Metal/CUDA rejected.

Metal follow-up candidate:

- A resident CMYK/YCCK conversion and pack kernel can win when CPU entropy/IDCT
  already produces component planes or when a batch can keep conversion/store
  on device. It should not be routed automatically until measured.

Exit criteria:

- CPU `Rgb8`/`Rgba8` output matches the fixture oracle within documented
  tolerance.
- Existing YCbCr/RGB fast paths do not regress.

## Phase A2: Progressive ROI/Scaled CPU Decode

Purpose: make progressive 8-bit support usable for viewport and WSI-like
access patterns.

Implementation requirements:

- Reuse existing progressive coefficient assembly as the correctness path.
  Status: landed for 8-bit progressive RGB8 output.
- Add ROI, scaled, and region-scaled output from assembled progressive
  coefficients.
  Status: landed as output projection after full progressive coefficient
  assembly; this is not entropy-stage ROI skipping or reduced-IDCT scaling.
- Avoid pretending the entropy stage is ROI-skip capable unless restart or scan
  structure makes that demonstrably correct.
  Status: current implementation decodes the full progressive entropy stream.
- Add batch-session coverage for progressive inputs where output jobs are
  independent.
  Status: landed for scaled and region-scaled session jobs against single-tile
  decode.
- Expose capability metadata so higher layers know progressive ROI/scaled is
  CPU-supported but not yet Metal-resident.
  Status: landed for supported 8-bit progressive RGB8 CPU shapes while Metal
  remains rejected until a measured resident path is implemented.

Metal follow-up candidate:

- CPU assembles progressive coefficients; Metal consumes coefficient planes for
  IDCT, color conversion, ROI/scale, and resident store. This is the first
  plausible Metal progressive shape because it avoids GPU progressive entropy
  decoding.

Exit criteria:

- Progressive full, ROI, scaled, and region-scaled CPU output match the oracle
  for committed fixtures.
- Existing progressive full-image tests still pass.

## Phase B1: 12-Bit Extended Sequential CPU Decode

Purpose: support 12-bit DCT sequential JPEG for medical and diagnostic inputs.

Implementation requirements:

- Add 12-bit quantized coefficient and IDCT output handling without truncating
  internal precision.
  Status: initial scalar full-image/ROI/scaled/region-scaled grayscale
  `Gray16`/`Rgb16`, APP14 RGB 4:4:4 `Rgb16`, and YCbCr
  4:4:4/4:2:2/4:2:0 `Rgb16` paths have landed for SOF1 12-bit streams
  without restart markers.
- Prefer `Gray16` and `Rgb16` output for native precision.
  Status: `Gray16` and expanded grayscale `Rgb16` are available for the initial
  grayscale path, and direct APP14 RGB 4:4:4 plus YCbCr
  4:4:4/4:2:2/4:2:0 `Rgb16` is available for the initial color path; other
  12-bit subsampled color remains open.
- Make any 12-bit-to-8-bit output an explicit documented conversion path, not
  an implicit default.
  Status: 12-bit-to-8-bit output stays unsupported.
- Add ROI, scaled, region-scaled, and batch coverage after full-image parity.
  Status: ROI, scaled, and region-scaled output have landed for the initial
  grayscale, APP14 RGB 4:4:4, and YCbCr 4:4:4/4:2:2/4:2:0 paths;
  session-batch coverage has landed for the initial color paths.
- Update `JpegOutputBuffer` and capability reporting for 16-bit output formats.
  Status: capability reporting marks full-image/ROI/scaled/region-scaled
  `Extended12` grayscale `Gray16`/`Rgb16`, APP14 RGB 4:4:4 `Rgb16`, and
  YCbCr 4:4:4/4:2:2/4:2:0 `Rgb16` CPU-eligible.

Metal follow-up candidate:

- Metal can plausibly win on 12-bit IDCT, color conversion, and `Rgb16`/`Gray16`
  store for batches. It needs separate kernels or specialization from the
  current 8-bit RGB8 resident path.

Exit criteria:

- 12-bit extended sequential CPU decode matches the reference oracle for
  full-image and API-shaped outputs.
  Status: partially met for the committed all-zero grayscale fixture and
  channel-distinct APP14 RGB 4:4:4 and YCbCr 4:4:4/4:2:2/4:2:0 fixtures
  covering full-image/ROI/scaled/region-scaled `Gray16`/`Rgb16` output as
  applicable.
- 8-bit paths do not share precision state that can corrupt current behavior.

## Phase B2: 12-Bit Progressive CPU Decode

Purpose: complete the non-lossless 12-bit DCT path.

Implementation requirements:

- Extend progressive coefficient assembly to 12-bit precision.
  Status: landed for initial grayscale SOF2 coefficient assembly to native
  `Gray16` or expanded `Rgb16` output, APP14 RGB 4:4:4 `Rgb16` output, and
  YCbCr 4:4:4/4:2:2/4:2:0 `Rgb16` output.
- Support full-image first, then ROI/scaled through the same output machinery
  used by Phase A2 and B1.
  Status: full-image, ROI, scaled, and region-scaled output have landed for
  the initial grayscale, APP14 RGB 4:4:4, and YCbCr 4:4:4/4:2:2/4:2:0 paths
  through full progressive coefficient assembly and 12-bit block projection.
- Keep capability reasons separate for unsupported progressive 12-bit shapes
  during partial implementation.
  Status: capability reporting marks grayscale `Progressive12`
  `Gray16`/`Rgb16`, APP14 RGB 4:4:4 `Rgb16`, and YCbCr
  4:4:4/4:2:2/4:2:0 `Rgb16` CPU-eligible while keeping Metal/CUDA rejected;
  other 12-bit progressive subsampled color output remains unsupported.

Metal follow-up candidate:

- Same as Phase A2/B1: CPU entropy/coefficient assembly first, then Metal IDCT
  and output stages after benchmarks.

Exit criteria:

- 12-bit progressive full and viewport-shaped CPU outputs match oracle data.
  Status: partially met for the committed all-zero grayscale SOF2 fixture and
  channel-distinct APP14 RGB 4:4:4 and YCbCr 4:4:4/4:2:2/4:2:0 fixtures
  covering full-image/ROI/scaled/region-scaled `Gray16`/`Rgb16` output as
  applicable. Non-constant external-oracle fixtures and other 12-bit
  subsampled color remain open.

## Phase C1: Lossless SOF3 CPU Decode

Purpose: support non-DCT lossless JPEG used by older medical pipelines.

Implementation requirements:

- Implement the SOF3 predictor pipeline separately from DCT decode.
  Status: initial predictors 1-7 full-image/ROI/scaled/region-scaled grayscale
  `Gray8`/`Gray16` pipeline has landed and is separate from DCT/IDCT decode.
- Start with common predictor selections and sample precisions represented by
  committed fixtures.
  Status: predictors 1-7, 8-bit and 16-bit grayscale predictor-sensitive
  fixtures have landed.
- Support `Gray8`/`Gray16` and RGB variants only when the component and sample
  model is understood.
  Status: `Gray8` is supported for the initial 8-bit grayscale shape and
  `Gray16` is supported for the initial 16-bit grayscale shape; `Rgb8` is
  supported for the initial 8-bit APP14 RGB 4:4:4 shape.
- Add predictor-specific tests, restart-marker tests, malformed-stream tests,
  and row behavior where the predictor dependencies allow it.
  Status: predictor-specific positive coverage has landed for predictors 1-7,
  including ROI/scaled/region-scaled output; restart-coded grayscale coverage
  and APP14 RGB color/restart coverage have landed. Malformed streams, row
  output, YCbCr/16-bit color, and broader precision coverage remain open.
- Keep unsupported predictors as `UnsupportedPredictor` or a more specific
  structured error.
  Status: unsupported predictor values return `UnsupportedPredictor` during
  decode setup and capability inspection.

Metal follow-up candidate:

- Lossless predictor decode is branchy and dependency-heavy. Metal should only
  be considered after CPU benchmarks show a repeated batch shape where resident
  store or row-parallel predictor work can win.

Exit criteria:

- SOF3 CPU decode matches oracle output for committed predictor fixtures.
  Status: partially met for the committed predictors 1-7 8-bit and 16-bit
  grayscale fixtures across full-image/ROI/scaled/region-scaled output.
- DCT decode code remains isolated from lossless predictor logic.
  Status: met for the initial predictors 1-7 path.

## Phase D: Routing And Public API Hardening

Purpose: make the new CPU support visible without over-promising GPU support.

Required work:

- Extend `JpegCapabilityReport` so unsupported, CPU-supported, Metal-candidate,
  and Auto-approved states are explicit for A/B/C.
- Add public docs for each support class and output format.
- Add stable API snapshots only after the route vocabulary is finalized.
- Keep `signinum-jpeg-metal` rejection reasons precise for A/B/C until each
  Metal follow-up lands.

Exit criteria:

- Higher layers can route A/B/C inputs without duplicating marker, color-space,
  precision, or SOF logic.

## Metal Promotion Phases

Metal work starts only after the matching CPU phase exits.

### M0: Measurement Harness

- Add CPU baseline, CPU warm-session, and Metal candidate benches for each
  class before writing promotion logic.
- Include resident buffer and texture rows where downstream WSI viewers would
  consume Metal output directly.
- Record host, command, input source, and comparator availability in docs when
  publishing a claim.

### M1: CMYK/YCCK Resident Conversion

- Candidate work: color transform plus pack/store into caller-owned buffer or
  texture.
- Promote only for batches where CPU decode plus Metal conversion beats CPU
  conversion by a meaningful margin or removes a downstream upload.

### M2: Progressive Coefficient-To-Pixel Resident Output

- Candidate work: Metal IDCT, upsample, ROI/scale, and store from CPU-assembled
  progressive coefficients.
- Do not move progressive entropy decode to Metal in this tranche.

### M3: 12-Bit Resident IDCT And Store

- Candidate work: 12-bit IDCT and `Gray16`/`Rgb16` resident output.
- Requires explicit 16-bit buffer/texture output contracts before routing.

### M4: Lossless Resident Predictor Experiments

- Candidate only after CPU SOF3 parity and benchmarks expose a batch shape worth
  accelerating.
- Keep Auto disabled until the win is repeatable on Apple Silicon.

## Stale-Docs Sweep Rules

When changing JPEG support, update these documents in the same branch:

- `docs/support-matrix.md`: public support scope and backend limits
- `docs/wsi-decode-api.md`: route/capability behavior
- `docs/parity.md`: fixture and oracle requirements
- `docs/architecture.md`: crate ownership and active areas
- `docs/bench.md`: benchmark commands and publication policy
- `docs/release.md`: release-state and backend-routing claims
- `docs/wsi-dicom-passthrough.md`: fallback transcode and passthrough wording
- `docs/dct-to-htj2k-notes.md`: coefficient-domain transcode scope
- `docs/stable-api-1.0.md`: only when public API changes

Historical measurement logs may stay historical, but new sections must say when
older rows are superseded and must not be reused as current support claims.
