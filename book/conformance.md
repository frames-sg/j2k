# Conformance

JPEG 2000 has an official conformance test suite: ISO/IEC 15444-4, also
published as ITU-T T.803. It is a set of codestreams and reference images,
with error tolerances, grouped into profiles and compliance classes. A decoder
conforms to a class when it decodes every test case of that class within its
tolerance. The suite doesn't test speed, robustness, or security.

We run the current edition, ISO/IEC 15444-4:2024 (T.803 v3), on every
release, for the CPU decoder and for the CUDA and Metal backends.

## What J2K covers

The `j2k` CPU decoder conforms to:

- **JPEG 2000 Part 1:** Profile-1 Cclass-1 and Profile-1 Cclass-1HF, and the
  Annex G JP2 reader.
- **HTJ2K (Part 15):** DS1-HM Cclass-1h at MMAGB 15, which includes the
  DS1-HT, DS0-HM, and DS0-HT subsets; Cclass-1HFh at MMAGB 20; and the Annex G
  JPH reader at MMAGB 15.

`j2k-cuda` and `j2k-metal` are tested against the same classes, and their
results are reported separately.

Not covered:

- JPX and other Part 2 extensions are out of scope, except the JP2-compatible
  JPX input that Annex G requires. See [Format support](format-support.md).
- T.803 v3 has no test set for Cclass-2h, so J2K makes no Cclass-2h
  statement. Our own tests at Cclass-2h scale don't replace one.

We don't describe J2K as "fully Part 1 compliant" or "fully Part 15
compliant". Each statement above names a profile and class and rests on the
reports for one release.

## Results

The reports for each release are attached to it on [GitHub
Releases](https://github.com/frames-sg/j2k/releases). For the latest release:

| Backend | Platform | Cases passed | Where the cases ran |
| --- | --- | --- | --- |
| CPU | Linux x86-64, macOS arm64, Windows x86-64 | 160 of 160 on each | CPU |
| CUDA | NVIDIA GeForce RTX 4070 SUPER | 160 of 160 | 81 partly on the GPU, 79 on the CPU |
| Metal | Apple M4 | 160 of 160 | 81 partly on the GPU, 79 on the CPU |

The 160 cases are 90 Part 1 cases (decoder and JP2) and 70 Part 15 cases: 60
codestream comparisons and the ten Annex G JPH families. No case was skipped.
For each Part 15 test set, the suite picks the largest BMAGB that does not
exceed the class's MMAGB.

On a GPU backend, parsing always runs on the CPU, and so does Annex G JP2
color and component handling. In the 81 hybrid cases, the GPU runs the
stages it supports among block decoding, dequantization, the inverse wavelet
transform, the color transform, and output. No case runs entirely on the GPU.
Which stages ran where comes from dispatch counters recorded during the run.

The reports also cross-check one codestream with OpenJPEG: `p0_13.j2k`, which
has 257 components and uses the multi-component transform, is decoded by both
and its component metadata and samples compared before any T.803
normalization.

## Encoder checks

T.803 treats encoder testing as informative, so these results are not part of
the conformance statement. The implementation statements for the CPU, CUDA,
and Metal encoders, and the test matrix, are in `corpus/j2k-conformance/`.

Every encoded codestream is decoded by the OpenJPEG decoder pinned for ITU-T
T.804, which serves as the Annex D legality check: it decodes 55 of the 56 CPU
cases and all 35 CUDA and Metal cases. OpenJPEG rejects HTJ2K code blocks with
RGN markers, so the one remaining CPU case, HTJ2K with ROI, is decoded by a
pinned OpenHTJ2K build instead. That is an interoperability check, not a T.804
result. Lossless output must match the source exactly. Rate and PSNR checks on
lossy output are our own. Every encoder case records which stages ran on the
GPU and which on the CPU.

## Gaps in the suite

T.803 has no codestream that sets a coding style in a tile-part header. Of its
94 codestreams, 18 change a component's wavelet with a COC marker in the main
header, but none has a tile-part COD or COC. A decoder can pass every case
while ignoring tile-part overrides. J2K's own tests cover tile-part COD, COC,
QCD, and QCC overrides, their precedence, and tiles that differ in wavelet,
decomposition levels, and multi-component transform. Two-tile codestreams
built from OpenJPEG encodes decode identically in J2K and OpenJPEG 2.5.4.

Robustness and security are covered by fuzzing, and speed by the benchmarks
in [Performance](performance.md).

## Running the suite

The test codestreams are copyrighted, so they aren't in the repository.
`fetch` downloads them from the URL pinned in
`corpus/j2k-conformance/t803-v3.toml`, checks every hash, and unpacks them
under `target/t803/`:

```bash
cargo xtask t803 fetch
cargo xtask t803 run --iut cpu --suite all
cargo xtask t803 run --iut cuda --suite all    # Linux with an NVIDIA GPU
cargo xtask t803 run --iut metal --suite all   # macOS
```

`--suite` also accepts `part1` or `part15`. Reports, in JSON and Markdown, go
to `target/t803/reports/`. `run` refuses a working tree with uncommitted
changes or untracked files unless you add `--development`, which marks the
reports as development runs.

How the runner works and how release reports are verified is described in
the [contributor book](../contributor-book/testing.md).
