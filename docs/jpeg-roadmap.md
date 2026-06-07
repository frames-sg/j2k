# JPEG Roadmap

This document is the working map for making `signinum-jpeg` a production-grade
WSI and medical JPEG runtime. It records what comes next. Current support
claims remain in [`support-matrix.md`](support-matrix.md), detailed phase notes
remain in [`jpeg-support-phases`](jpeg-support-phases/README.md), and benchmark
rules remain in [`bench.md`](bench.md).

## Target JPEG Set

The target decode set is deliberately narrower than "all JPEG":

- SOF0 baseline 8-bit Huffman JPEG.
- WSI sampling shapes: 4:4:4, 4:2:2, and 4:2:0.
- DQT, DHT, SOS, DRI, and restart marker handling.
- APP14 RGB, CMYK, and YCCK.
- SOF1 12-bit extended sequential.
- SOF2 progressive.
- SOF3 lossless.

Arithmetic-coded JPEG, hierarchical JPEG, and differential JPEG stay out of
scope for decode. They must keep returning explicit structured unsupported
errors and capability rejection reasons.

## Current Checkpoint

The CPU decoder has broad coverage for the target set, including session batch
runtime, reusable output buffers, route introspection, 8-bit APP14 RGB/CMYK/YCCK
conversion, 12-bit SOF1/SOF2 RGB/YCCK/CMYK paths, SOF3 grayscale and sampled
color paths, and structured rejection for unsupported shapes.

The remaining CPU work is not another decoder rewrite. It is evidence,
edge-case hardening, and routing contract cleanup:

- Broaden malformed four-component fixtures beyond the current non-divisible
  sampling rejection case.
- Add external-oracle or real-tile 12-bit fixtures for the supported SOF1/SOF2
  color shapes.
- Audit capability reports against every target class and every explicit
  out-of-scope class.
- Keep libjpeg-turbo as an oracle/comparator only, not as a Signinum decode
  backend.
- Keep nvJPEG out of production JPEG decode planning; CUDA JPEG work uses
  Signinum-owned kernels only.

## Phase 1: Close CPU Evidence Gaps

Goal: prove CPU decode parity for the target set with fixtures that exercise
real routing and output contracts.

Required work:

- Build a requirement checklist from the target JPEG set and map each item to
  tests, fixtures, capability reports, and public docs.
- Add malformed fixtures for duplicate tables, bad table selectors, bad scan
  component maps, invalid restart order, truncated entropy, conflicting APP14
  metadata, and unsupported SOF classes.
- Add non-constant external-oracle fixtures where available for 12-bit SOF1 and
  SOF2 RGB, YCbCr, CMYK, and YCCK.
- Verify full, ROI, scaled, region-scaled, row, and session-batch behavior for
  every supported class where that API shape is meant to apply.
- Ensure unsupported classes never fall back into generic decode attempts after
  capability rejection.

Exit evidence:

- Focused `signinum-jpeg` decode and capability tests for every checklist item.
- `cargo test -p signinum-jpeg --no-fail-fast`.
- `cargo test -p signinum-core --test repo_integrity --no-fail-fast`.
- Updated `support-matrix.md`, `wsi-decode-api.md`, and `parity.md` only where
  public claims changed.

## Phase 2: Harden WSI Batch Runtime

Goal: make the CPU path predictable for repeated viewport reads before moving
more work to device adapters.

Required work:

- Keep `JpegBatchSession` as the preferred hot-loop API for many-tile decode.
- Add benchmark rows for 16, 64, and 256 tile batches across full, scaled, and
  region-scaled RGB/RGBA output.
- Compare one-shot batch, warm session, and warm session with reused
  `JpegOutputBuffer`s.
- Add allocation counters or profile rows that show scratch/output reuse is
  actually happening.
- Keep first-error index behavior, output ordering, `DecodeOptions`, and
  `TileBatchOptions` stable.

Exit evidence:

- Benchmark commands and input sources recorded in `bench.md`.
- Batch/session tests that compare reused output buffers with fresh outputs
  byte-for-byte.
- No public benchmark claim without host, command, input, comparator, skipped
  rows, and revision metadata.

## Phase 3: Metal Only Where It Can Win

Goal: promote resident Metal JPEG work only for shapes where CPU parity is
settled and a resident path is architecturally likely to beat CPU/session decode
for WSI batches.

Initial Metal candidates:

- SOF0 8-bit Huffman 4:2:0, 4:2:2, and 4:4:4 RGB/RGBA output.
- APP14 RGB/CMYK/YCCK only after CPU parity and color-conversion tests are
  strong enough to serve as the oracle.
- Session-aware many-tile APIs with caller-owned Metal buffer or texture output.
- No CPU materialization unless the caller explicitly asks for CPU pixels.

Promotion rules:

- `Auto` routing stays CPU until a benchmarked resident Metal route wins on
  the target Apple Silicon machine and the route has parity tests against CPU.
- Explicit Metal requests fail loudly for unsupported JPEG classes.
- Metal docs must distinguish resident decode from CPU-decode-then-upload.

Exit evidence:

- Metal resident parity tests against CPU bytes for each promoted shape.
- Apple Silicon benchmark rows showing when the route wins and when it does
  not.
- Capability reports that expose Metal eligibility without re-parsing marker
  logic in callers.

## Phase 4: CUDA Owned-Kernel Maintenance

Goal: keep CUDA JPEG decode owned by Signinum and scoped to shapes that have
working kernels and clear evidence.

Required work:

- Maintain strict rejection for ROI, scaled, Gray8, and non-RGB8 CUDA JPEG
  requests until owned kernels support them.
- Keep 4:2:0, 4:2:2, and 4:4:4 full-tile RGB8 paths matched against CPU
  download bytes.
- Continue using cached packet/table state and caller/device-owned output
  buffers.
- Do not add nvJPEG back into production planning.

Exit evidence:

- CUDA runtime tests gated by the existing CUDA-required environment policy.
- Capability reports that say why unsupported CUDA requests are rejected.
- Benchmarks that compare owned CUDA kernels against CPU/session and any
  archived vendor-baseline data without depending on vendor decode at runtime.

## Stop Conditions

Do not mark the JPEG goal complete until current evidence proves:

- Every target JPEG class has CPU decode tests for the supported API surfaces.
- Every out-of-scope JPEG class has explicit structured errors and capability
  rejection tests.
- Session and reusable-output behavior is verified under batch workloads.
- Metal routes are promoted only with parity tests and benchmark evidence.
- Public docs state current support without relying on obsolete phase notes.
