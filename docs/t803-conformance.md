# ISO/IEC 15444-4 / ITU-T T.803 Conformance

Status: **Part 1 and selected Part 15 conformance published for 0.14.0**

Conformance stated for release `0.14.0`:

- `j2k` CPU decoder:
  - JPEG 2000 Part 1: **Profile-1 Cclass-1, Profile-1 Cclass-1HF, and the
    Annex G JP2 reader.**
  - HTJ2K Part 15: **DS1-HM Cclass-1h, MMAGB 15** (including the DS1-HT,
    DS0-HM, and DS0-HT subsets), **Cclass-1HFh, MMAGB 20**, and the **Annex G
    JPH reader at MMAGB 15**.
- `j2k-cuda` and `j2k-metal`: separate results for the same Part 1 and Part 15
  points. Each case records which stages ran on the CPU and which on the GPU.
  No case runs entirely on the GPU.

The test harness implements ISO/IEC 15444-4:2024 / ITU-T T.803 v3. Part 4
defines how to test JPEG 2000 conformance against reference outputs; it is not
a codestream format or a performance benchmark. The statements above are based
on the five reports attached to the
[v0.14.0 release](https://github.com/frames-sg/j2k/releases/tag/v0.14.0), all
generated from the same release commit with no development-only features.

## Scope

The GPU adapters may run some stages on the CPU; such cases are labelled
`hybrid`. Annex G JP2 color and component normalization currently runs on the
CPU for both GPU adapters. JPX / Part 2 is out of scope, except for the
JP2-compatible JPX input Annex G requires. T.803 v3 has no Cclass-2h test set,
so there is no Cclass-2h conformance statement; the project's own
Cclass-2h-scale resource and boundary tests are not a substitute.

The project does not describe J2K as "fully Part 1 compliant" or "fully
Part 15 compliant". Every Profile/Cclass/MMAGB statement refers to published
reports for one release commit.

## 0.14.0 results

All five reports on the [0.14.0 release](https://github.com/frames-sg/j2k/releases/tag/v0.14.0)
pass all 160 selected decoder cases with no skips, and their encoder checks
pass: 90 Part 1 decoder and JP2 cases, and 70 Part 15 cases (60 codestream
comparisons plus the ten Annex G JPH families). For each BSET, the harness
picks the largest BMAGB that does not exceed the stated MMAGB. The CPU reports
are from Linux x86-64, macOS arm64, and Windows x86-64. CUDA ran on an NVIDIA
GeForce RTX 4070 SUPER and Metal on an Apple M4.

Both GPU reports have 81 hybrid cases and 79 CPU-only cases out of 160; none
runs entirely on the GPU. Parsing runs on the CPU in every case; hybrid cases
run the supported Tier-1, dequantization, IDWT, MCT, color/output, and transfer
stages on the GPU. The stage labels come from dispatch counters recorded
during the run. The release verifier checked the reports for the tagged source
after [hosted validation](https://github.com/frames-sg/j2k/actions/runs/43311353177)
and [hardware validation](https://github.com/frames-sg/j2k/actions/runs/43311370162)
passed.

The report also decodes, with the vendored OpenJPEG 2.5.3, every selected
codestream whose COD enables the multi-component transform and whose SIZ has
more than four components (`p0_13.j2k`, with 257 components), and compares
component metadata and samples before any T.803 normalization.

Releases 0.9.0, 0.11.0, 0.11.2, 0.12.0, 0.13.0, and 0.13.1 passed the same 160
cases; their reports are attached to their GitHub releases. The 0.13.0 release
has no CUDA report: it was published under a
[CUDA waiver](../xtask/release-evidence/cuda-waiver-0.13.0.md).

## Running the tests

The official corpus is downloaded only from the URL and archive hash pinned in
`corpus/j2k-conformance/t803-v3.toml`:

```bash
cargo xtask t803 fetch
cargo xtask t803 run --iut cpu --suite all
cargo xtask t803 run --iut cuda --suite all
cargo xtask t803 run --iut metal --suite all
```

`fetch` rejects unapproved redirects, changed archive or file hashes, unsafe
archive entries, duplicate paths, unexpected case names, and oversized input.
The copyrighted corpus stays under `target/t803/`; only the JSON/Markdown
reports and hashes are kept.

Add `--development` when running on a dirty tree. Development runs use the same
corpus and comparisons, but the release verifier rejects their reports.
Release runs on a clean commit omit the flag.

Each backend is verified separately. A CPU statement needs the three CPU
operating-system reports; a CUDA or Metal statement needs only that backend's
hardware report. A missing GPU report does not affect the CPU statement:

```bash
cargo xtask t803 verify --scope cpu --candidate-sha "$RC_SHA" \
  --report path/to/cpu-linux.json \
  --report path/to/cpu-macos.json \
  --report path/to/cpu-windows.json
cargo xtask t803 verify --scope cuda --candidate-sha "$RC_SHA" \
  --report path/to/cuda.json
cargo xtask t803 verify --scope metal --candidate-sha "$RC_SHA" \
  --report path/to/metal.json
```

`--scope all` verifies all five reports together. The tag-publish workflow
uses it for every release, except one with a reviewed CUDA waiver, which it
verifies with `--scope cpu` and `--scope metal`.

Verification requires all 160 selected Part 1/Part 15 decoder and JP2/JPH cases
with no skips, every report passing, matching source and corpus hashes, and the
right implementation, platform, and backend for each report. A report that
labels a CPU-assisted case as GPU-only is rejected.

## Encoder tests

The CPU, CUDA, and Metal Annex F implementation statements and the pairwise
and boundary test matrix are in `corpus/j2k-conformance/`. The pinned T.804
OpenJPEG decoder decodes 55 of the 56 CPU cases and all 35 CUDA and Metal
cases. OpenJPEG 2.5.3 rejects HT code-blocks with RGN, so the CPU HT+RGN case
is decoded by the separately pinned OpenHTJ2K decoder; it is an
interoperability check, not a T.804 result. A successful reference decode is
the Annex D legality check, and lossless output must also match the source
exactly. Lossy rate and PSNR checks are the project's own quality checks.

Encoder testing is informative under T.803 and is not part of decoder
conformance. Every encoder case reports which stages ran on the GPU and which
fell back to the CPU.

T.803 does not test robustness, security, or performance; those are covered by
fuzzing, security review, and benchmarks.

T.803 also does not test coding styles set in tile-part headers. Eighteen of
its 94 codestreams change a component's wavelet with a main-header COC, but
none has a tile-part COD or COC, so a decoder can pass every selected case
while ignoring tile-part overrides, as `j2k` did until
[#295](https://github.com/frames-sg/j2k/issues/295).
`crates/j2k-native/tests/tile_header_markers.rs` covers tile-part COD, COC,
QCD and QCC overrides, their precedence, and tiles that differ in wavelet,
decomposition levels and multi-component transform, against the decodes of
the codestreams they were made from. Two-tile codestreams built the same way
from OpenJPEG encodes decode identically in `j2k` and OpenJPEG 2.5.4.
