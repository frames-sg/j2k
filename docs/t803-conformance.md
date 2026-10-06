# ISO/IEC 15444-4 / ITU-T T.803 Conformance

Status: **Part 1 and selected Part 15 conformance published for 0.12.0**

Conformance stated for release `0.12.0`:

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
[v0.12.0 release](https://github.com/frames-sg/j2k/releases/tag/v0.12.0), all
generated from the same release commit with no development-only features.

## Scope

| Implementation under test | Result | Where stages run |
| --- | --- | --- |
| `j2k` CPU | Part 1 Profile-1 Cclass-1, Profile-1 Cclass-1HF, Annex G JP2 reader; Part 15 DS1-HM Cclass-1h at MMAGB 15, Cclass-1HFh at MMAGB 20, Annex G JPH reader at MMAGB 15. | CPU. |
| `j2k-cuda` | Same Part 1 and Part 15 points. | Each case reports parsing, Tier-1, transforms, output, and transfers as CPU, CUDA, or not used. |
| `j2k-metal` | Same Part 1 and Part 15 points. | Each case reports parsing, Tier-1, transforms, output, and transfers as CPU, Metal, or not used. |

The GPU adapters may run some stages on the CPU; such cases are labelled
`hybrid`. Annex G JP2 color and component normalization currently runs on the
CPU for both GPU adapters. JPX / Part 2 is out of scope, except for the
JP2-compatible JPX input Annex G requires. T.803 v3 has no Cclass-2h test set,
so there is no Cclass-2h conformance statement; the project's own
Cclass-2h-scale resource and boundary tests are not a substitute.

The project does not describe J2K as "fully Part 1 compliant" or "fully
Part 15 compliant". Every Profile/Cclass/MMAGB statement refers to published
reports for one release commit.

## 0.12.0 results

All five reports on the [0.12.0 release](https://github.com/frames-sg/j2k/releases/tag/v0.12.0)
pass all 160 selected decoder cases (90 Part 1, 70 Part 15) with no skips,
and their encoder checks pass. The CPU reports are from Linux x86-64,
macOS arm64, and Windows x86-64. CUDA ran on an NVIDIA GeForce RTX 4070 SUPER
and Metal on an Apple M4.

Both GPU reports have 81 hybrid cases and 79 CPU-only cases out of 160;
none runs entirely on the GPU. The release verifier checked the reports
for the tagged source after [hosted validation](https://github.com/frames-sg/j2k/actions/runs/37427079958)
and [hardware validation](https://github.com/frames-sg/j2k/actions/runs/37431440941)
passed.

## 0.11.2 results

The [0.11.2 release](https://github.com/frames-sg/j2k/releases/tag/v0.11.2)
has all five JSON reports and their Markdown versions for commit
`75a3e0618e1963d8403e4edad0fa95ee1c217ec1`. Each report passes all 160
selected decoder cases (90 Part 1, 70 Part 15) with no skips. The CPU reports
are from Linux x86-64, macOS arm64, and Windows x86-64. CUDA ran on an NVIDIA
GeForce RTX 4070 SUPER and Metal on an Apple M4.

Both GPU reports have 81 hybrid cases and 79 CPU-only cases out of 160; none
runs entirely on the GPU. The release verifier checked the reports after the
[hosted validation](https://github.com/frames-sg/j2k/actions/runs/36353712175)
and [hardware validation](https://github.com/frames-sg/j2k/actions/runs/36355267186)
passed.

## 0.11.0 results

All five reports on [release 0.11.0](https://github.com/frames-sg/j2k/releases/tag/v0.11.0)
are for commit `09d746a7b040258eb5dd505b44384eec0152a8b9` and pass all 160
selected decoder cases with no skips. The CPU reports are from Linux x86-64,
macOS arm64, and Windows x86-64, and the CUDA and Metal reports from real
hardware. The release verifier checked them after the hosted and GPU release
workflows passed.

## 0.9.0 results

The macOS arm64, Linux x86-64, and Windows x86-64 (MSVC) CPU reports on
[release 0.9.0](https://github.com/frames-sg/j2k/releases/tag/v0.9.0) each pass
all **160 selected cases with no skips** at commit
`b197f01ab4b9271f1cbc36921755a5b9d588bd5a`: 90 Part 1 decoder/JP2 cases and
70 Part 15 decoder/JPH cases. The 70 Part 15 cases are 60 codestream
comparisons plus the ten Annex G JPH families. For each BSET, the harness picks
the largest BMAGB that does not exceed the stated MMAGB.

Both GPU reports also pass 160/160, with 81 hybrid and 79 CPU-only cases and
none entirely on the GPU. Part 1 has 48 hybrid and 42 CPU-only cases; Part 15
has 33 and 37. CUDA ran on an NVIDIA GeForce RTX 4070 SUPER and Metal on an
Apple M4 Pro. Parsing ran on the CPU in every case; hybrid cases ran the
supported Tier-1, dequantization, IDWT, MCT, color/output, and transfer stages
on the GPU. The totals happen to match, but the per-case dispatch counts differ
between backends (for example, CUDA records uploaded bytes and Metal records
host-input counts). The stage labels come from dispatch counters recorded
during the run.

The CPU Annex D/F encoder matrix passes 56/56 cases. 55 are decoded by the
pinned T.804 OpenJPEG decoder. OpenJPEG 2.5.3 rejects the required HT+RGN case,
so that case is decoded by a separately pinned OpenHTJ2K decoder instead. The
CUDA and Metal matrices each pass 35/35 through OpenJPEG; CUDA has 34 hybrid
cases and one CPU-only case, Metal 33 hybrid and two CPU-only. Encoder results
are informative and are not part of the decoder conformance statement.

All five reports are for the commit above. The release assets include JSON and
Markdown versions listed in `SHA256SUMS`; the schema-7 JSON SHA-256 hashes are:

| Implementation/platform | Report SHA-256 |
| --- | --- |
| CPU/macOS arm64 | `9638c41d41842e99f385ea71fd8c83791416d512233a5fed43081c9964d5092b` |
| CPU/Linux x86-64 | `509502eddf48ebe5d77d614234746693be30e1c537f1890af67edf4593de64eb` |
| CPU/Windows x86-64 MSVC | `57569214409450b27edb8ba1634e4e2fc6778f52370ef4b98a0ed94b093c3a9a` |
| CUDA/Linux x86-64, RTX 4070 SUPER | `84c52044254278bb45e48664d94765d8fd44bdd07fc6b03d610fa983f5944039` |
| Metal/macOS arm64, M4 Pro | `2206dd313ad5f10a16652ecc8b08e5c9f4a5f72eb0b31be361411a0462644d0c` |

### The `c1-c0p0-13` fix

`c1-c0p0-13` used to fail because of a bug in the test harness, not the
decoder. The codestream has 257 components and uses the reversible component
transform. T.803 B.2.5 compares Cclass-0 output before the inverse MCT, so the
first component's reference value is 1; the Cclass-1 component-0 reference
after the inverse RCT is 0. The harness guessed whether MCT was used from the
display colorspace, which is unknown for 257 components. It now reads the COD
transform flag and transform kind with the codestream inspector, reconstructs
the pre-MCT component for Cclass-0, and reports the MCT stage from the same
data. A 257-component regression test covers this.

The report now also decodes, with a second decoder, every selected codestream
whose COD enables MCT and whose SIZ has more than four components. For
`p0_13.j2k`, J2K and the vendored OpenJPEG 2.5.3 produced identical component
metadata and samples for all 257 components before any T.803 normalization;
both output hashes are
`a01808e0cbf14288274188c8bebb5ef8c2aa46304eca964a2ac71bed1713c1fd`.
OpenJPEG CLI 2.5.4 also wrote 257 PGX components with no sample differences;
the SHA-256 of the concatenated one-sample components was
`54acfbfedc4d8da40f76f275e1a98f10af8ef1fb9fb39e5a67a00aabcbe6597c`.

`p0_13.j2k`, `c0p0_13.pgx`, and `c1p0_13-0.pgx` are byte-identical in ITU's
current attachment, ISO's 2024 electronic insert, and ITU's 2002 suite. No
corpus mapping, hash, dimension, precision, signedness, reduction, crop,
tolerance, or comparison arithmetic was changed to make the case pass.

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

`--scope all` verifies all five reports together; the tag-publish workflow uses
it for releases that include Part 15.

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
