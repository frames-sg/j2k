# Benchmark Results

This page records benchmark commands, results, and the machines they ran on.
When a harness writes JSON or CSV output, that output is authoritative. Other
docs link here instead of copying numbers. Older results are in Git history.

## Scope

What the codec supports is listed separately in
[`docs/public-support.md`](public-support.md).

Speed comparisons against other codecs need an external benchmark bundle:

```bash
cargo run -p xtask --features adoption -- adoption-report --run-dir target/j2k-adoption-benchmark/full
```

`adoption-report` refuses an incomplete bundle and lists what is missing.
Speed comparisons must use the manifest-backed external rows, not the
generated in-repo fixtures.

## CPU JPEG safe-SIMD development run - 2026-08-12

The CPU JPEG safe-SIMD refactor was measured before and after on the same Apple
M4 Pro host running macOS 26.5.2 build `25F84`, with Rust/Cargo 1.96.0 and LLVM
22.1.2. Criterion used 95% confidence, 50 samples, a three-second warm-up, and
a ten-second measurement. Benchmark JPEGs were generated deterministically
before timing; setup decoded every case to validate geometry and output
checksums. The commands were:

```bash
cargo bench -p j2k-jpeg --bench decode_cpu -- --save-baseline safe-simd-before
cargo bench -p j2k-jpeg --features bench-internals --bench micro -- --save-baseline safe-simd-before
cargo bench -p j2k-jpeg --bench decode_cpu -- --baseline safe-simd-before
cargo bench -p j2k-jpeg --features bench-internals --bench micro -- --baseline safe-simd-before
```

The final end-to-end decode comparison was:

| Case | Before 95% interval | After 95% interval | Criterion change interval |
| --- | ---: | ---: | ---: |
| Gray8 512 x 512 | 1.2084–1.2649 ms | 1.1740–1.1822 ms | -12.903% to -5.046% |
| RGB8 512 x 512 4:4:4 | 2.7072–2.8042 ms | 2.6190–2.6301 ms | -5.201% to -2.813% |
| RGB8 512 x 512 4:2:2 | 2.4961–2.5115 ms | 2.4447–2.4634 ms | -2.473% to -1.302% |
| RGB8 512 x 512 4:2:0 | 1.9161–1.9243 ms | 1.8769–1.8839 ms | -2.651% to -1.946% |
| RGB8 257 x 263 4:2:0 | 508.92–510.58 us | 500.94–502.03 us | -1.652% to -1.149% |
| RGB8 512 x 512 4:2:0 rows | 1.9360–1.9420 ms | 1.8981–1.9022 ms | -2.403% to -1.966% |

No affected microbenchmark exceeded the 2% regression criterion; every final
95% confidence interval was lower than the saved baseline. This includes general and
bottom-half-zero NEON IDCT, 255/256/cropped 4:2:0 row pairs, gray and planar-RGB
rows, and 255/256/unaligned YCbCr rows. An intermediate build did confirm
1.7–3.3% regressions in the color decode cases at doubled measurement time.
Generated-code inspection traced those to per-chunk bounds checks and a lost
4:2:0 helper inline. Safe array chunking plus a fixed ten-byte overlapping-load
leaf restored the code shape; safe dispatch was not reverted.

These runs were on an uncommitted working tree, not a release commit. Some
unrelated microbenchmarks also got faster, so the size of each improvement is
partly host state and code layout; what the run shows is that AArch64 had no
regression beyond the threshold. Native x86-64 results follow.

### Native Windows AVX2 validation

The same refactor was measured natively with the MSVC target on Windows 11 Pro
10.0.22631, an AMD Ryzen 7 5800X3D with AVX2, and Rust/Cargo 1.96.0. The Linux
VM on that host was used only as the remote transport; every test and timed
process was a native `x86_64-pc-windows-msvc` executable. Baseline and
candidate used the same benchmark sources, `fearless_simd 0.5.0`, shared build
settings, and deterministic inputs. The final paired run pinned each process
to the same logical processor and used High process priority. Criterion used
the same 95% confidence, 50 samples, three-second warm-up, and ten-second
measurement settings. The sole inconclusive 4:4:4 case was repeated with a
20-second measurement as required.

| Case | Before 95% interval | After 95% interval | Criterion change interval |
| --- | ---: | ---: | ---: |
| Gray8 512 x 512 | 1.8990–1.9206 ms | 1.8712–1.8975 ms | -2.192% to -0.449% |
| RGB8 512 x 512 4:4:4 (20 s repeat) | 4.7098–4.7265 ms | 4.7236–4.7483 ms | +0.003% to +0.495% |
| RGB8 512 x 512 4:2:2 | 4.2607–4.2973 ms | 4.1895–4.2214 ms | -2.295% to -1.171% |
| RGB8 512 x 512 4:2:0 | 4.8575–4.8962 ms | 4.7957–4.8259 ms | -1.825% to -0.862% |
| RGB8 257 x 263 4:2:0 | 1.2906–1.3418 ms | 1.2740–1.2827 ms | -4.832% to -0.826% |
| RGB8 512 x 512 4:2:0 rows | 4.8291–4.8550 ms | 4.7672–4.8029 ms | -1.646% to -0.717% |

Every affected AVX2 microbenchmark also cleared the 2% upper-bound criterion.
The upper confidence bounds were +0.329% for general AVX2 IDCT, -0.829% for
4:2:0 upsampling, -0.445% to -1.065% for full row pairs, +1.000% for cropped
4:2:0, +0.662% for gray rows, +1.366% for planar RGB rows, and +0.176% or
better for full, tail, and unaligned YCbCr rows.

An additional unchanged reduced 2x2 scalar-IDCT diagnostic measured a
confirmed +5.119% to +5.971% shift at an absolute candidate time of
11.780–11.859 ns. Baseline and candidate source and generated x86-64 function
bodies are identical (235 instructions), so the shift comes from code placement
and caching, not extra work. That function is not part of the SIMD change, and
the end-to-end decode results above show no regression. Like the AArch64 runs,
these were on an uncommitted working tree.

### `fearless_simd` 0.7 upgrade validation - 2026-08-13

The staged 0.10.0 dependency graph resolves `fearless_simd 0.7.0` in the
workspace and every affected fuzz lockfile. `cargo xtask release-cpu` passed
with that version on the Apple M4 Pro AArch64 host, the Linux x86-64 VM, and
the native Windows x86-64 MSVC host described above.

The dependency is AArch64-only for `j2k-jpeg`. In an adjacent
version-isolation control from one otherwise unchanged intermediate source
tree on Windows, changing only the workspace requirement and lockfile between
0.5.0 and 0.7.0 caused Cargo to reuse the exact benchmark executable (SHA-256
`2e37a5ffa851964a5e3ca4fafb6f819372071ba8f033e94bb83dda297d51ea80`).
A 20-second 4:4:4 rerun of that identical binary still moved by -1.916% to
-1.069%, so the earlier apparent x86 regressions were run-to-run noise.

On AArch64, normalized inspection of the unstripped release-benchmark output
found identical instruction bodies for the compared JPEG NEON IDCT, row
conversion, and 4:2:0 kernels under 0.5.0 and 0.7.0; relocation targets,
constant-pool offsets, and whole-binary placement were excluded from that
comparison. The host was too busy (WindowServer and other interactive processes) for a
clean 0.7.0 timing run; repeated Criterion attempts had large scheduling
outliers. The 0.5.0 timings above are still the latest AArch64 numbers, and a
quiet rerun on the same host is needed before saying the 0.7.0 upgrade has no
AArch64 regression.

## How `Auto` routing thresholds are chosen

`BackendRequest::Auto` uses generated routing tables checked into the
repository; it does not calibrate on the user's machine. A workload cell is
sent to the GPU only if the CPU, hybrid, and any GPU-only route all produce
identical bytes on the same pinned external input, the hybrid median is at
least 10% faster than every alternative, and its Criterion 95% confidence
interval does not overlap theirs.

CUDA and Metal collect all six required operations with their production APIs:
full decode, ROI decode, scaled decode, batch decode, lossless encode, and lossy
encode. Each result records the commit SHA, manifest SHA-256, hardware and
driver, route label, output SHA-256, and Criterion ID. Routes where parsing,
entropy decoding, output, or codestream assembly still run on the CPU are
labelled `hybrid`.

After a hardware run, check the raw results against the manifest and the
Criterion estimates:

```bash
cargo xtask auto-routing verify \
  --evidence target/gpu-benchmark/auto-routing/evidence.json \
  --external-manifest "$J2K_AUTO_ROUTING_MANIFEST" \
  --criterion-root target/criterion \
  --out target/gpu-benchmark/auto-routing/verified.json
```

The verifier makes every routing decision itself; benchmark input cannot ask
for one. It rejects missing operations or cases, route or output mismatches,
unsafe Criterion paths, unsupported confidence levels, changed estimate files,
and commit or platform mismatches. The output hash covers the raw results, the
manifest, and every estimate used.

Accepted results and the workloads they route to the GPU are listed in
`docs/routing-promotion-evidence.json`. Regenerate the Rust routing tables with
`cargo xtask promotion-codegen`; CI runs `cargo xtask promotion-codegen
--check` to catch stale tables. The generator checks the manifest schema,
backend, SHA-256 values, that all six operations were measured, workload
identity, limits, and duplicates. A route with no accepted result stays on the
CPU.

A local two-input Metal smoke on August 4, 2026 exercised the pipeline but was
not a representative corpus. No cells were routed to the GPU: the
decode routes were slower, and the measured lossless and lossy encode medians
were only about 4.2% and 7.8% faster than CPU. No `Auto` threshold was changed
from that diagnostic.

### Local Metal lossless routing run - 2026-09-24

This run replaced the Metal decode cells; the combined run below has since
re-measured them. Its corpus is local, not a
publication corpus: 512x512 tiles from GDC whole-slide DICOM files, decoded with
`opj_decompress`, stitched into 256x256, 640x480, 1024x1024, and 2048x2048 RGB
images and 640x480 and 2048x2048 gray images, then re-encoded as Part 1
lossless (OpenJPEG), HTJ2K lossless codestreams (OpenJPH `-reversible true`),
and HTJ2K lossless JPH files (Grok). There are 22 cases (14 decode, 8 encode);
manifest SHA-256
`217d700a9f882c4cd63b76aed250de92124d74a0bf476d33855dbbe857d6befe`.
The run used `release-bench`, candidate `c1cc3ff1` with a dirty tree, and an
Apple M4 Pro (16 GPU cores) on macOS 27.0. The verifier accepted all 76 cells
and routed 33 to the GPU. The verified artifact SHA-256 is
`69987ce1cee902e4ef664964250059021ef15b59e0828802098fbe84d192df77`.

Lossy inputs were excluded from this run. `tests/auto_routing_parity.rs` found
that Metal 9/7 decode differed from the CPU by one code value in roughly 10
samples per million on every third-party 9/7 file, while every 5/3 file matched.
Because GPU routing requires identical bytes, the earlier 9/7 Metal cells
(`metal_part1` lossy repeated, `metal_part15` lossy repeated and half-scale)
were removed. The cause was found and fixed the same day; see "Local Metal
routing run with lossy inputs" below.

The resulting Metal cells cover lossless inputs only. Repeated batches of 16 run
on Metal for HTJ2K RGB8 from 256x256 (JPH from 640x480), HTJ2K Gray8 from
640x480, Part 1 RGB8 from 640x480, and Part 1 Gray8 from 2048x2048.
Single full-image decodes run on Metal for HTJ2K from 640x480, for RGB8 and
Gray8 and for codestreams and JPH, when the source components match the output
format. Part 1 single-image decodes stay on the CPU because Metal measured 105%
to 1400% slower. ROI and half-scale decodes qualified only for HTJ2K Gray8 at 2048x2048;
Metal Auto does not route those operations for gray sources, so they stay on the CPU.
The lossy RGB8 encode threshold is now 2048x2048 pixels, where HTJ2K and Part 1
lossy encode measured 54% and 38% faster; at 640x480 they were within 4% of the CPU.

### Local Metal routing run with lossy inputs - 2026-09-24

The 9/7 mismatch was in output rounding, not in the wavelet transform. The CPU
rounds each centered sample to the nearest integer, ties to even, and only then
adds the unsigned level shift, as OpenJPEG does (`opj_lrintf(value) +
dc_level_shift`). The Metal store, pack, and inverse-colour-transform kernels
added the shift first and rounded half up. Adding 128 in f32 drops one bit of
precision, so a value just below `k + 0.5` became a tie and rounded up: every
mismatch was Metal one code value high, at a sample whose shifted CPU value was
exactly `k + 0.5`. The fix makes Metal integer output round before the shift and
compute the inverse ICT with the CPU's fused expressions. Three CPU paths
disagreed with the CPU full decode in the same way (region decode, row
streaming, and `CpuBatchDecoder`, which built integer output from unrounded
component planes), as did the CPU inverse ICT's non-SIMD tail. They now round
the same way, so a CPU region decode equals the crop of the full decode.

The first run of this corpus also exposed an unrelated Metal bug: the repeated
Part 1 batch path decoded into a recycled scratch buffer without zero-filling
code blocks that have no coding passes, so a lossy batch decoded after a
lossless one of the same geometry kept stale coefficients (39% of bytes wrong
for `gray8-2048x2048-part1-lossy`). That path now zero-fills like the
single-image, grouped, and distinct-batch paths.

The corpus is the lossless run's 22 cases plus 12 lossy decode files made from
the same stitched sources: Part 1 lossy (OpenJPEG) and HTJ2K lossy codestreams
(OpenJPH) for every RGB and gray size. There are 34 cases (26 decode, 8 encode);
manifest SHA-256
`dd6363374dccf8d6b34656f90b9557b5a01f8582b94f5014e3165dc61f1248ec`.
`tests/auto_routing_parity.rs` found no mismatch in any full, ROI, or half-scale
decode, and the bench's own parity check passed for every CPU, Metal, and Auto
route. The run used `release-bench`, candidate `c1cc3ff1` with a dirty tree,
and the same Apple M4 Pro on macOS 27.0. The verifier accepted all 124 cells and
routed 49 to the GPU. The verified artifact SHA-256 is
`66f9d83f932efb7df6cab2de0048849f09343da00915e22794b0056b29f2aa5f`; its cells
are recorded under the `metal_local_combined` source.

Every lossless cell re-measured to the same thresholds as the lossless run. The
new lossy cells are:

- Repeated batches of 16: HTJ2K RGB8 from 256x256 (46% faster at 256x256),
  Part 1 RGB8 from 1024x1024 (56%; 640x480 measured 10% slower), and HTJ2K and
  Part 1 Gray8 from 2048x2048 (62% and 18%; HTJ2K Gray8 at 640x480 was only 8%
  faster, below the 10% bar).
- Single full-image decodes: HTJ2K RGB8 and Gray8 from 640x480 (44% and 37%
  faster at 640x480, 77% and 78% at 2048x2048).

Part 1 lossy single-image decodes stay on the CPU (66% to 2400% slower on
Metal). No half-scale cell qualified: HTJ2K lossy RGB8 half-scale measured 3%
to 176% slower, so the earlier `metal_part15` half-scale cell stays removed.
ROI and half-scale qualified only for HTJ2K Gray8, which Metal Auto does not
route. The lossy RGB8 encode threshold stays at 2048x2048 pixels (Part 1 and
HTJ2K 36% and 57% faster there).

### External CUDA routing development run - 2026-08-05

The uninterrupted CUDA matrix used the same 12 external cases from
`uclouvain/openjpeg-data` commit
`39524bd3a601d90ed8e0177559400d23945f96a9` and manifest SHA-256
`f07072f5d0313c0249e2df5df2310cd5c6c5a4b3414a933537fabf2362d2065c`.
It ran all 36 cells with Cargo `release-bench`, Criterion 0.95 confidence
intervals, ten samples, a one-second warm-up, and a three-second target on an
AMD Ryzen 7 5800X3D with an NVIDIA GeForce RTX 4070 SUPER, driver 596.49,
Linux x86-64, and CUDA 13.2.

The verifier accepted every cell and routed 18 decode cells to the GPU, using
the measured output sizes as thresholds. RGB8 reversible uses CUDA for full
output from 256 x 149, ROI output from 128 x 74, and half-scale output from
1296 x 972. RGB8 irreversible uses CUDA for full output from 640 x 480 and ROI
or half-scale output from 320 x 240. Gray8 reversible uses CUDA only for full
output from 640 x 480. Gray8 irreversible uses CUDA for full output from
3323 x 891, ROI output from 1661 x 445, and half-scale output from
1662 x 446. Repeated-input batches of 16 use the full-image thresholds.

These thresholds apply only to raw Part 1 codestreams with the measured source
component/output-format pair. They do not cover JP2 color normalization, other scale factors, HTJ2K, higher depths, RGBA, distinct-input
batches, unmeasured operations, smaller output work, or shapes below either
measured dimension. All 12 encode cells stayed on CPU because CUDA-assisted
encode was slower than CPU in this end-to-end matrix.

The verified artifact's internal SHA-256, recorded beside the thresholds, is
`ded1eb045f9673e5bbe64dc873be3ba227ecb61ec11b6c9ad53653dbcc993f44`.
The raw results file SHA-256 is
`ad0b434dbd64f669d58054f4a25f9272f741bf2a689f6f70816d98fe87c02e61`;
the serialized verified file SHA-256 is
`a565d47f81ed32588e551167a91c0df68daff3d7ebd6ff8588fbe4d8ab27ac79`.
Every route produced the same output SHA-256 before timings were compared.
None of these APIs has a GPU-only route, so the comparison is CPU against
hybrid.

The run was on an uncommitted working tree based on commit
`6400fcd4c9f8cf9708563d62411eadf158f94282`. The full matrix must be rerun on
the release commit before these numbers are published as release results.

### External Metal routing development run - 2026-08-04

The full routing matrix was then run against 12 external decode/encode cases
from `uclouvain/openjpeg-data` commit
`39524bd3a601d90ed8e0177559400d23945f96a9`. The external manifest SHA-256 is
`f07072f5d0313c0249e2df5df2310cd5c6c5a4b3414a933537fabf2362d2065c`.
The run used Cargo `release-bench`, Criterion 0.95 confidence intervals, ten
samples, a one-second warm-up, and a three-second target measurement on an
Apple M4 Pro with a 16-core GPU and 48 GB RAM, macOS 26.5.2 build `25F84`, and
Metal compiler `32023.883`.

The verifier accepted all 36 workload cells and routed four to the GPU. Times
below are Criterion medians; each selected hybrid interval was entirely below
the CPU interval.

| Cell | CPU median | Hybrid median | Speedup |
| --- | ---: | ---: | ---: |
| Repeated RGB8 irreversible decode, 640 x 480, batch 16 | 68.364 ms | 43.232 ms | 36.762% |
| Repeated Gray8 irreversible decode, 3323 x 891, batch 16 | 265.409 ms | 149.856 ms | 43.538% |
| Repeated RGB8 reversible decode, 2592 x 1944, batch 16 | 2429.216 ms | 179.528 ms | 92.610% |
| RGB8 irreversible encode, 2592 x 1944 | 813.508 ms | 716.208 ms | 11.961% |

The verified artifact's internal SHA-256 is
`162a47f7a96b2be88abebc100aab672513af04895532863fa1a293660546f879`.
The raw results file SHA-256 is
`3b2ffad6fe3ebb42e2182612946a5c87ebf0b267e25c38bfb5c9d07c11aa6e7d`.
These hashes are recorded next to the thresholds in the routing code.

The batch rows reuse one encoded input 16 times, so they measure repeated
decoding of the same image, not batches of different images. `Auto` therefore
uses the GPU only for repeated Part 1 Gray8/RGB8 requests in the measured
reversible/irreversible classes and sizes.
In this Part 1 matrix, single-image, ROI, scaled, HTJ2K, higher-depth, signed,
RGBA, and unmeasured lossless/lossy cells stayed on CPU. Lossless encode and
Gray8 lossy encode also stayed on CPU; the latter measured only a 6.8%
improvement. The Part 15 matrix below adds only the HTJ2K/JPH cells it lists.

The run was on an uncommitted working tree based on commit
`6400fcd4c9f8cf9708563d62411eadf158f94282`. The full matrix must be rerun and
reverified on the release commit before these numbers are published as release
results.

### Official Part 15 routing development runs - 2026-08-07

CUDA and Metal used the same three external T.803 workloads: raw HTJ2K
`p0_04` BSET 12, JPH file 1 BSET 12, and JPH file 10 component 0 as the encode
source. The schema-2 manifest SHA-256 is
`422f40e4086b53e43f2338f97b468c869257cc23ec4899432cf019585782d48e`.
Each backend ran ten end-to-end cells covering full, ROI, scaled, and repeated
batch decode plus lossless and lossy encode. Every CPU/hybrid pair produced an
identical output hash. Neither API has a GPU-only route, so these measure the
hybrid routes.

| Backend and measured host | Cells `Auto` routes to the GPU | Hybrid speedup versus CPU | Verified artifact SHA-256 |
| --- | --- | --- | --- |
| CUDA, RTX 4070 SUPER, driver 596.49, WSL2 Linux 5.15.153.1 | Raw HT and JPH full, ROI, half-scale, and repeated batch decode (8/10) | 70.39% to 91.41% | `77370c83710ebf578139ad0bfa2608ffad989d83faec8d5eee213691290c0088` |
| Metal, M4 Pro 16-core GPU, macOS 26.5.2 build `25F84` | Raw HT half-scale, raw HT repeated batch, and JPH repeated batch decode (3/10) | 23.42% to 76.91% | `cfa66686d053bb3e2d4c8756abaf84aab65d8505a635795cefd38de53573c1f5` |

CUDA lossless and lossy encode were respectively 528.23% and 305.10% slower
than CPU. Metal lossless and lossy encode were respectively 240.00% and 58.54%
slower. Those four cells stay on the CPU. Metal raw/JPH full and ROI decode and
JPH half-scale decode also stay on the CPU because they did not meet the
routing rule. The thresholds apply only to the measured codec, container,
format, dimensions, operation, and repeat count, not to other HTJ2K/JPH
workloads.

The CUDA raw/serialized-report SHA-256 values are `0b4ac7a08e4adba0e9ed9a7983a4fb3cd9ae0cc81a2ab1162e604f13a785065a`
and `faacdfa190d0c18acdf9c24a584e285606c37af8dcd15cc649dd6eac2eaa8a10`.
The corresponding Metal values are `5a66373022b8df18decb3e60d255464f76e6bee3195e90489b61c8366fdec459`
and `ffe6697176de2c2ac716615f5b96476d610c04715fdad339857c716fb960ef06`.

Both reports record base commit `f92646d0e6f0d0ef6c1e60b60beaad29da1afd3b`
plus uncommitted changes. The full matrix must be rerun on the release commit
before these numbers are published as release results.

### Metal HTJ2K host-output encode matrix - 2026-08-08

The host-output encode route was measured separately because the Part 15
inputs above are too small for GPU coefficient preparation and HT Tier-1 to pay
for Metal setup. The
schema-2 manifest SHA-256 is
`452080d2b0611f67246450f58479354803469cc0c13d10df4a4ac866e03f90c3`.
It contains deterministic `j2k-test-support` Gray8 and RGB8 PNM inputs at
512 x 512, 1,024 x 1,024, and 2,048 x 2,048, plus two official T.803 HT/JPH
decode anchors. The lossless encode cells use Metal coefficient preparation
and HT Tier-1 with CPU packetization and final host codestream output.

Before timing, every hybrid and `Auto` codestream matched the CPU codestream
byte for byte, and each lossless codestream decoded exactly to its source PNM.
The batch-16 rows run the same host-output encode 16 times on one Metal
session. They measure that route, not the fully resident Metal-buffer batch
API.

| Format and size | Single CPU | Single hybrid | Single speedup | Batch-16 CPU | Batch-16 hybrid | Batch speedup | Fixed `Auto` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Gray8 512 x 512 | 2.057 ms | 8.850 ms | -330.23% | 32.847 ms | 137.666 ms | -319.12% | CPU |
| RGB8 512 x 512 | 5.479 ms | 9.490 ms | -73.20% | 88.428 ms | 156.564 ms | -77.05% | CPU |
| Gray8 1,024 x 1,024 | 9.409 ms | 15.746 ms | -67.35% | 152.592 ms | 236.271 ms | -54.84% | CPU |
| RGB8 1,024 x 1,024 | 25.724 ms | 16.200 ms | 37.02% | 412.913 ms | 254.853 ms | 38.28% | hybrid |
| Gray8 2,048 x 2,048 | 43.365 ms | 17.107 ms | 60.55% | 698.504 ms | 269.195 ms | 61.46% | hybrid |
| RGB8 2,048 x 2,048 | 120.778 ms | 20.623 ms | 82.93% | 1,980.952 ms | 351.347 ms | 82.26% | hybrid |

These are Criterion medians from ten samples, a one-second warm-up, a
three-second target measurement, and 95% confidence intervals. A shape was
routed to Metal only when both its single and batch-16 results were more than
10% faster and the hybrid interval did not overlap the CPU interval. There is
no GPU-only host-output route, so the comparison is CPU against hybrid. The
thresholds cover only the six measured shape/format combinations.

The SHA-256 of the decision artifact recorded next to the routing cells is
`c8defb820b55a99e94acdd5849b4597bce0a1718fd7e0d2bc0aa926bc0e130d4`.
After enabling only the qualifying cells, the full 26-cell matrix reran with
no `Auto` output mismatch; its verified artifact SHA-256 is
`c98f11c0b2a2a96853953ceee7ea672e0e5044bdb8abbd397c8c36eb82fe53b8`.
The raw results and serialized verified report SHA-256 values are
`19c1793e3647db44e01903cd619a4da0d70ab5a2b55ab53b3227c67544112090`
and `a1f898a545bfc3c29e4fbbf5197b602b59920e1c27d50c2cd85af44e168de370`.

This run used an Apple M4 Pro with a 16-core GPU and 48 GB RAM, macOS 26.5.2
build `25F84`, and Metal compiler `32023.883`. It records base commit
`f92646d0e6f0d0ef6c1e60b60beaad29da1afd3b` and dirty-worktree identity
`a84f4107ab943540a0951abefc670065b29b9809430074d5255d8bec1cf2b021`
across 2,748 tracked and untracked source paths. The matrix must be rerun on
the release commit before these numbers are published as release results.

## Older results

Older regression runs, migration comparisons, abandoned experiments, and
uncommitted-tree probes are in Git history. They are out of date and should
not be quoted.
