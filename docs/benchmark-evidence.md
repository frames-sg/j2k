# Benchmark Evidence

This document records published benchmark commands, measurements, and
environment details. JSON and CSV artifacts remain the source of truth when
they are produced by a benchmark harness.

This is the narrative owner for current benchmark hosts, commands, results, and
performance qualifications. The workspace README, architecture, and integration
guides link here instead of copying mutable measurements. Superseded diagnostics
remain in Git history rather than accumulating in this page.

## Publication Status

The codec support boundary is tracked separately in
[`docs/public-support.md`](public-support.md): JPEG 2000 Part 1 codestreams,
JP2 still-image files, HTJ2K Part 15 codestreams, and JPH still-image files.
That repo-local support gate is separate from performance reporting.

This page is the current public benchmark evidence note. Broader adoption-facing
speed reports require an external adoption benchmark bundle:

```bash
cargo run -p xtask --features adoption -- adoption-report --run-dir target/j2k-adoption-benchmark/full
```

The `adoption-report` subcommand must require a completed
external bundle and identify any missing evidence. Generated repo-local
fixtures and passing codec self-checks remain implementation evidence; use
manifest-backed external rows for adoption-facing speed reports.

## Fixed Auto-routing promotion evidence

`BackendRequest::Auto` uses committed thresholds; it does not calibrate on a
user's machine. A hybrid workload cell may be promoted only after CPU, hybrid,
and any supported strict-device route produce identical bytes on the same
manifest-pinned external input. Its end-to-end median must be at least 10%
faster than every competitor and its Criterion 95% confidence interval must not
overlap any competing interval.

CUDA and Metal collect all six required operations with their production APIs:
full decode, ROI decode, scaled decode, batch decode, lossless encode, and lossy
encode. Route evidence records the exact candidate SHA, manifest SHA-256,
hardware and driver identity, execution label, output SHA-256, and Criterion ID.
These adapters currently disclose CPU-assisted routes as `hybrid`; they do not
claim a strict device-native route when parsing, entropy decode, output, or
codestream assembly still runs on CPU.

After a hardware run, verify the raw evidence against the exact manifest and
Criterion estimates:

```bash
cargo xtask auto-routing verify \
  --evidence target/gpu-benchmark/auto-routing/evidence.json \
  --external-manifest "$J2K_AUTO_ROUTING_MANIFEST" \
  --criterion-root target/criterion \
  --out target/gpu-benchmark/auto-routing/verified.json
```

The verifier derives each promotion decision; benchmark input cannot request a
promotion. It rejects missing operations or external cases, route/output
mismatches, unsafe Criterion paths, unsupported confidence levels, changed
estimate files, and candidate/platform inconsistencies. The verified artifact
hash covers the raw evidence, manifest, and every referenced estimate.

A local two-input Metal smoke on August 4, 2026 exercised the pipeline but was
not a representative external release corpus. It promoted zero cells: the
decode routes were slower, and the measured lossless and lossy encode medians
were only about 4.2% and 7.8% faster than CPU. No `Auto` threshold was changed
from that diagnostic.

### External CUDA routing development run - 2026-08-05

The uninterrupted CUDA matrix used the same 12 external cases from
`uclouvain/openjpeg-data` commit
`39524bd3a601d90ed8e0177559400d23945f96a9` and manifest SHA-256
`f07072f5d0313c0249e2df5df2310cd5c6c5a4b3414a933537fabf2362d2065c`.
It ran all 36 cells with Cargo `release-bench`, Criterion 0.95 confidence
intervals, ten samples, a one-second warm-up, and a three-second target on an
AMD Ryzen 7 5800X3D with an NVIDIA GeForce RTX 4070 SUPER, driver 596.49,
Linux x86-64, and CUDA 13.2.

The verifier accepted every cell and promoted 18 decode cells. The fixed
policy uses the measured output-work thresholds. RGB8 reversible promotes full
output at 256 x 149, ROI output at 128 x 74, and half-scale output at 1296 x
972. RGB8 irreversible promotes full output at 640 x 480 and ROI or half-scale
output at 320 x 240. Gray8 reversible promotes only full output at 640 x 480.
Gray8 irreversible promotes full output at 3323 x 891, ROI output at 1661 x
445, and half-scale output at 1662 x 446. Qualified repeated-input batches use
the measured full-image thresholds at count 16.

The policy applies only to raw Part 1 codestreams with the measured source
component/output-format pair. It does not extrapolate to JP2 color
normalization, other scale factors, HTJ2K, higher depths, RGBA, distinct-input
batches, unmeasured operations, smaller output work, or shapes below either
measured dimension. All 12 encode cells stayed on CPU because CUDA-assisted
encode was slower than CPU in this end-to-end matrix.

The verified artifact's internal SHA-256, recorded beside the thresholds, is
`ded1eb045f9673e5bbe64dc873be3ba227ecb61ec11b6c9ad53653dbcc993f44`.
The raw evidence file SHA-256 is
`ad0b434dbd64f669d58054f4a25f9272f741bf2a689f6f70816d98fe87c02e61`;
the serialized verified file SHA-256 is
`a565d47f81ed32588e551167a91c0df68daff3d7ebd6ff8588fbe4d8ab27ac79`.
Every compared route produced the same output SHA-256 before timing results
were considered. No strict device-native route exists for these public
surfaces, so the competitive comparison was CPU versus the truthfully labelled
hybrid route.

This was a dirty-tree development run whose recorded candidate SHA is the base
`6400fcd4c9f8cf9708563d62411eadf158f94282`. It supports the fixed policy but
is not exact-release-SHA evidence; the complete matrix must be rerun after
candidate freeze before publication.

### External Metal routing development run - 2026-08-04

The full routing matrix was then run against 12 external decode/encode cases
from `uclouvain/openjpeg-data` commit
`39524bd3a601d90ed8e0177559400d23945f96a9`. The external manifest SHA-256 is
`f07072f5d0313c0249e2df5df2310cd5c6c5a4b3414a933537fabf2362d2065c`.
The run used Cargo `release-bench`, Criterion 0.95 confidence intervals, ten
samples, a one-second warm-up, and a three-second target measurement on an
Apple M4 Pro with a 16-core GPU and 48 GB RAM, macOS 26.5.2 build `25F84`, and
Metal compiler `32023.883`.

The verifier accepted all 36 workload cells and promoted four. Times below are
Criterion medians; each promoted hybrid interval was wholly below the CPU
interval.

| Cell | CPU median | Hybrid median | Speedup |
| --- | ---: | ---: | ---: |
| Repeated RGB8 irreversible decode, 640 x 480, batch 16 | 68.364 ms | 43.232 ms | 36.762% |
| Repeated Gray8 irreversible decode, 3323 x 891, batch 16 | 265.409 ms | 149.856 ms | 43.538% |
| Repeated RGB8 reversible decode, 2592 x 1944, batch 16 | 2429.216 ms | 179.528 ms | 92.610% |
| RGB8 irreversible encode, 2592 x 1944 | 813.508 ms | 716.208 ms | 11.961% |

The verified artifact's internal SHA-256 is
`162a47f7a96b2be88abebc100aab672513af04895532863fa1a293660546f879`.
The raw evidence file SHA-256 is
`3b2ffad6fe3ebb42e2182612946a5c87ebf0b267e25c38bfb5c9d07c11aa6e7d`.
These hashes are the evidence anchors recorded beside the fixed thresholds in
the routing code.

The batch rows intentionally reuse one encoded input 16 times. They establish
decode-once/repeated-output routing only; they are not distinct-image batch
throughput claims. `Auto` therefore promotes only repeated Part 1 Gray8/RGB8
requests in the measured reversible/irreversible classes and size ranges.
In this Part 1 matrix, single-image, ROI, scaled, HTJ2K, higher-depth, signed,
RGBA, and unmeasured lossless/lossy cells stayed on CPU. Lossless encode and
Gray8 lossy encode also stayed on CPU; the latter measured only a 6.8%
improvement in the canonical profile. The later Part 15 matrix below qualifies
only its explicitly listed HTJ2K/JPH cells.

This was a dirty-tree development run whose recorded candidate SHA is the base
`6400fcd4c9f8cf9708563d62411eadf158f94282`, not an exact release candidate.
It supports implementation of the pending fixed policy, but it is not formal
release evidence. The complete matrix must be rerun and reverified from the
exact clean candidate SHA before publication.

### Official Part 15 routing development runs - 2026-08-07

CUDA and Metal used the same three external T.803 workloads: raw HTJ2K
`p0_04` BSET 12, JPH file 1 BSET 12, and JPH file 10 component 0 as the encode
source. The schema-2 manifest SHA-256 is
`422f40e4086b53e43f2338f97b468c869257cc23ec4899432cf019585782d48e`.
Each backend ran ten end-to-end cells covering full, ROI, scaled, and repeated
batch decode plus lossless and lossy encode. Every CPU/hybrid pair produced an
identical output hash. Neither public surface has a strict device-only route,
so these are hybrid product-path measurements, not device-native claims.

| Backend and measured host | Qualified fixed `Auto` cells | Hybrid speedup versus CPU | Verified artifact SHA-256 |
| --- | --- | --- | --- |
| CUDA, RTX 4070 SUPER, driver 596.49, WSL2 Linux 5.15.153.1 | Raw HT and JPH full, ROI, half-scale, and repeated batch decode (8/10) | 70.39% to 91.41% | `77370c83710ebf578139ad0bfa2608ffad989d83faec8d5eee213691290c0088` |
| Metal, M4 Pro 16-core GPU, macOS 26.5.2 build `25F84` | Raw HT half-scale, raw HT repeated batch, and JPH repeated batch decode (3/10) | 23.42% to 76.91% | `cfa66686d053bb3e2d4c8756abaf84aab65d8505a635795cefd38de53573c1f5` |

CUDA lossless and lossy encode were respectively 528.23% and 305.10% slower
than CPU. Metal lossless and lossy encode were respectively 240.00% and 58.54%
slower. Those four cells remain CPU-routed. Metal raw/JPH full and ROI decode
and JPH half-scale decode also remain CPU-routed because they did not meet the
promotion rule. Fixed thresholds apply only to the measured codec, container,
format, dimensions, operation, and repeated-count cells; they do not
extrapolate to other HTJ2K/JPH workloads.

The CUDA raw/serialized-report SHA-256 values are `0b4ac7a08e4adba0e9ed9a7983a4fb3cd9ae0cc81a2ab1162e604f13a785065a`
and `faacdfa190d0c18acdf9c24a584e285606c37af8dcd15cc649dd6eac2eaa8a10`.
The corresponding Metal values are `5a66373022b8df18decb3e60d255464f76e6bee3195e90489b61c8366fdec459`
and `ffe6697176de2c2ac716615f5b96476d610c04715fdad339857c716fb960ef06`.

Both reports record base commit
`f92646d0e6f0d0ef6c1e60b60beaad29da1afd3b`, but they include uncommitted
candidate changes. They justify the development policy and demonstrate real
GPU-assisted wins, but are not exact-clean-SHA release evidence. The full
matrix must be rerun after candidate freeze before publication.

### Metal HTJ2K host-output encode matrix - 2026-08-08

The production-equivalent host-output route was measured separately because
the official Part 15 anchor above does not represent the image sizes where
resident coefficient preparation and HT Tier-1 can amortize Metal setup. The
schema-2 manifest SHA-256 is
`452080d2b0611f67246450f58479354803469cc0c13d10df4a4ac866e03f90c3`.
It contains deterministic `j2k-test-support` Gray8 and RGB8 PNM inputs at
512 x 512, 1,024 x 1,024, and 2,048 x 2,048, plus two official T.803 HT/JPH
decode anchors. The lossless encode cells use Metal coefficient preparation
and HT Tier-1 with CPU packetization and final host codestream output.

Before timing, every hybrid and `Auto` codestream matched the CPU codestream
byte for byte, and each lossless codestream decoded exactly to its source PNM.
The batch-16 rows repeat the same production-equivalent host-output operation
16 times while reusing the accelerator. They are throughput observations for
that route, not the fully resident Metal-buffer batch API.

| Format and size | Single CPU | Single hybrid | Single speedup | Batch-16 CPU | Batch-16 hybrid | Batch speedup | Fixed `Auto` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Gray8 512 x 512 | 2.057 ms | 8.850 ms | -330.23% | 32.847 ms | 137.666 ms | -319.12% | CPU |
| RGB8 512 x 512 | 5.479 ms | 9.490 ms | -73.20% | 88.428 ms | 156.564 ms | -77.05% | CPU |
| Gray8 1,024 x 1,024 | 9.409 ms | 15.746 ms | -67.35% | 152.592 ms | 236.271 ms | -54.84% | CPU |
| RGB8 1,024 x 1,024 | 25.724 ms | 16.200 ms | 37.02% | 412.913 ms | 254.853 ms | 38.28% | hybrid |
| Gray8 2,048 x 2,048 | 43.365 ms | 17.107 ms | 60.55% | 698.504 ms | 269.195 ms | 61.46% | hybrid |
| RGB8 2,048 x 2,048 | 120.778 ms | 20.623 ms | 82.93% | 1,980.952 ms | 351.347 ms | 82.26% | hybrid |

These are Criterion medians from ten samples, a one-second warm-up, a
three-second target measurement, and 95% confidence intervals. The verifier
promoted a shape only when both its single and batch-16 observations exceeded
the 10% threshold and the hybrid interval did not overlap the CPU interval.
No strict-device host-output route is available, so this comparison is CPU
versus the truthfully labelled hybrid product route. The fixed policy does not
extrapolate beyond the six measured shape/format combinations.

The pre-policy decision artifact SHA-256 recorded beside the routing cells is
`c8defb820b55a99e94acdd5849b4597bce0a1718fd7e0d2bc0aa926bc0e130d4`.
After enabling only the qualified cells, the complete 26-cell matrix reran
without an `Auto` parity failure; its verified artifact SHA-256 is
`c98f11c0b2a2a96853953ceee7ea672e0e5044bdb8abbd397c8c36eb82fe53b8`.
The post-policy raw evidence and serialized verified report SHA-256 values are
`19c1793e3647db44e01903cd619a4da0d70ab5a2b55ab53b3227c67544112090`
and `a1f898a545bfc3c29e4fbbf5197b602b59920e1c27d50c2cd85af44e168de370`.

This run used an Apple M4 Pro with a 16-core GPU and 48 GB RAM, macOS 26.5.2
build `25F84`, and Metal compiler `32023.883`. It records base commit
`f92646d0e6f0d0ef6c1e60b60beaad29da1afd3b` and dirty-worktree identity
`a84f4107ab943540a0951abefc670065b29b9809430074d5255d8bec1cf2b021`
across 2,748 tracked and untracked source paths. This is current-tree
development evidence, not exact-clean-SHA release evidence. The matrix must be
rerun after candidate freeze before publication.

## Historical diagnostics

Historical local regression runs, implementation-migration comparisons,
rejected experiments, and dirty-tree throughput probes remain available in Git
history.
They are not current publication evidence and must not be reused as release or
adoption claims.
