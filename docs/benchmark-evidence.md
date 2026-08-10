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
Single-image, ROI, scaled, HTJ2K, higher-depth, signed, RGBA, and unmeasured
lossless/lossy cells stay on CPU. Lossless encode and Gray8 lossy encode also
stay on CPU; the latter measured only a 6.8% improvement in the canonical
profile.

This was a dirty-tree development run whose recorded candidate SHA is the base
`6400fcd4c9f8cf9708563d62411eadf158f94282`, not an exact release candidate.
It supports implementation of the pending fixed policy, but it is not formal
release evidence. The complete matrix must be rerun and reverified from the
exact clean candidate SHA before publication.

## Historical diagnostics

Historical local regression runs, implementation-migration comparisons,
rejected experiments, and dirty-tree throughput probes remain available in Git
history.
They are not current publication evidence and must not be reused as release or
adoption claims.
