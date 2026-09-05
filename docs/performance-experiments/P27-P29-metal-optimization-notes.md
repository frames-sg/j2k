# P27–P29: Metal batch and lossy encode experiments

Measured on 2026-09-04, in place on `feat/htj2k-encode-candidates` at
`23f17707e86e230718c5691e8ba3ae98a1b2ff30`, with a dirty working tree.
Pre-existing changes included the P20 bounded decode path and the Metal runtime
and shader module split. No public Auto routing policy or allocation cap changed.

## Decisions

- **P27 rejected:** 16 MiB chunks for large 9/7 decode batches regressed the
  representative 1024 RGB batch-16 workload. Restored the pre-experiment production
  files exactly, verified by SHA-256. Retained additional large, distinct-input
  CPU pixel parity tests, including odd dimensions and batches of 6, 9 and 11.
- **P28 retained:** keep the lossy input transform, forward 9/7, quantization and
  HT code blocks on the GPU through one completion boundary. Reuse the existing
  Metal packetizer afterward to preserve the explicit device encode contract.
  Final timings and scope are below and in `P28-metal-resident-lossy.json`.
- **P29 small-chunk speed hypothesis rejected; capacity fix retained:** smaller
  global chunks slowed Classic encoding. The existing scheduler now splits only
  when the requested chunk would exceed the existing Tier-1 allocation cap.

## P27: large decode batches

The prototype replaced per-image reconstruction above 20 MiB with consecutive
chunks containing at most `max(1, floor(16 MiB / plane_bytes))` images. It adjusted
all four subband instance offsets and the destination plane offset, then reused
P20's batched interleave and original lift kernels. Smaller stages retained P20's
whole-batch route. No shader arithmetic changed.

The primary repeated comparison used the same executable in two processes:
control, then candidate. Values are Criterion mean wall times in milliseconds;
brackets are its rounded 95% confidence bounds, not ranges across repetitions.

| 1024 RGB, batch 16 | Control | 16 MiB chunks | Change |
| --- | ---: | ---: | ---: |
| Resident output | 40.109 [40.052, 40.163] | 41.177 [41.080, 41.268] | 2.66% slower |
| Host readback | 43.692 [43.596, 43.776] | 44.583 [44.500, 44.664] | 2.04% slower |

Repeating in reverse process order confirmed the direction: resident
40.273 [40.216, 40.343] versus 41.029 [40.939, 41.150] ms; readback
43.887 [43.798, 44.002] versus 44.484 [44.279, 44.723] ms. The eight-cell primary
matrix includes 128 and 640 RGB batch-16 plus a 512 single-image control. Every
output byte matched native CPU decode and both arms' output hashes matched.

An initial 15-sample pair appeared faster, but unchanged controls drifted by
roughly 10–17% across those early runs. That pair is not the promotion evidence.
The repeat-order check was necessary; no cache-size claim is inferred from the
16 or 20 MiB constants. The retained structural regression asserts the actual
P20 per-image fallback sequence counts (7, 10, 12), as well as exact pixels.

## P28: resident lossy preparation and HT encoding

The former staged path downloaded floating-point subbands after forward 9/7,
uploaded them for quantization, downloaded quantized coefficients, and uploaded
those for HT coding. The new path keeps the component planes, transform scratch
and gathered coefficients in pooled private buffers until HT completion. The
new gather kernel calls the existing precise quantizer; forward 9/7 command
encoding is shared with the staged path. HT coding uses the existing kernel.

The compressed block outputs are still read back to construct packet metadata.
The existing Metal packetizer then runs, followed by native codestream assembly.
Thus this is a resident **transform-through-HT** path, not an entirely resident
codestream pipeline. Metal packetization remains a material cost.

An earlier prototype used scalar packetization and measured approximately 4.9–7.6x
speedups. Those numbers do **not** describe the retained explicit device route:
the full facade suite showed that `RequireDevice` requires actual Metal
packetization. The final implementation dispatches it and records real stage
counters. Its final A/B timings replace the earlier numbers in the P28 record.

| Workload | Staged Metal, ms [95% CI] | Resident path, ms [95% CI] | Less time |
| --- | ---: | ---: | ---: |
| 128x128, RGB, batch 1 | 10.0840 [9.9466, 10.2260] | 5.5524 [5.5454, 5.5577] | 44.9% |
| 512x512, gray, batch 1 | 24.7950 [24.6800, 24.9000] | 21.1280 [21.0290, 21.2600] | 14.8% |
| 512x512, RGB, batch 1 | 54.8040 [54.3110, 55.2120] | 45.8720 [45.8040, 45.9490] | 16.3% |
| 640x480, RGB, batch 1 | 63.6140 [63.4620, 63.7690] | 53.2540 [53.1450, 53.3730] | 16.3% |
| 1024x1024, RGB, batch 1 | 187.6000 [187.2900, 187.9200] | 171.9200 [171.7200, 172.1300] | 8.4% |
| 512x512, RGB, batch 16 | 881.5700 [877.9500, 884.6500] | 733.2900 [733.0500, 733.5200] | 16.8% |

Eligibility is unsigned 8-bit gray or RGB with ICT, origin-zero full-resolution
components, square 32/64 code blocks, legal decomposition levels, and at most
16 million pixels. The caller's subband quantization parameters and progression
order are preserved. Unsupported jobs retain the staged fallback. Partial-stage
accelerators and public Auto policy retain their prior routes. The new private
layout helper shares band geometry with lossless encoding without applying the
lossless small-image level-selection policy to lossy inputs.

The CAP magnitude bound is derived from each encoded block's maximum magnitude
and its actual decomposition level. A prototype that used level zero produced a
one-byte CAP mismatch; exact scalar codestream tests caught it and it was fixed.

The whole-tile ABI has no quality-layer byte targets. Regression tests first
showed that a single-layer byte-budget request could incorrectly enter that hook.
The shared native eligibility predicate now declines such requests before
calling either host or resident whole-tile accelerators. Budget allocation remains
on the native staged path. The portable native test covers both hook variants;
the Metal integration test compares the complete budgeted output with CPU-only
encoding. No ABI extension or rate-control policy change was needed.

Correctness coverage includes patterned gray/RGB, even and odd dimensions,
singletons and single-axis zero-level inputs, 0–4 decomposition levels, 32/64
blocks, nonuniform subband quantization, unsupported metadata, constant input
values 0/128/255, all five progression orders, and independent native decode.
Exact codestream comparisons are against `CpuOnlyJ2kEncodeStageAccelerator`;
no epsilon or pixel tolerance is used. Existing explicit-device facade tests
also assert real deinterleave, ICT, transform, HT and packetization dispatches.
The combined input/ICT counter now reflects the fused kernel that actually runs.

## P29: Classic allocation-aware scheduling

The existing 1024 RGB batch-16 request needed a 720,420,864-byte Tier-1 buffer,
exceeding the existing 536,870,912-byte per-allocation cap. The new regression
reproduced that typed allocation error before the scheduler change.

Classic chunk planning now sums the existing tight Tier-1 capacity estimator,
preserves input order, and takes the largest contiguous chunk permitted by both
the requested tile count and `min(device.maxBufferLength, 512 MiB)`. It reuses the
existing submission, completion, retry and buffer ownership machinery. A single
oversized tile still reaches the original typed error. Other allocation checks
remain in force; this is not a bound on total GPU memory or a claim that every
possible large input now fits.

For this 1024 RGB fixture, each tile needs 45,026,304 Tier-1 bytes, so the maximum
chunk is 11 tiles (495,289,344 bytes), followed by 5. The regression verifies all
16 outcomes, identical codestreams and exact independent CPU decode. The prior
benchmark skip for this geometry has been removed.

All nine cells below ran in one process, in size order and then requested chunk
order 8, 4, 16. Values are means with 95% bounds in milliseconds.

| RGB batch 16 | Requested 8 | Requested 4 | Requested 16 |
| --- | ---: | ---: | ---: |
| 512x512 | 54.625 [54.471, 54.766] | 95.548 [95.375, 95.729] | 44.905 [44.840, 44.971] |
| 640x480 | 67.080 [66.904, 67.254] | 116.54 [116.39, 116.66] | 55.390 [55.220, 55.556] |
| 1024x1024 | 205.60 [205.23, 205.97] | 269.25 [268.55, 269.94] | 208.74 [208.35, 209.19], actual 11+5 |

No speedup is claimed over the formerly unsupported 1024/16 configuration.
Automatic 11+5 is slightly slower than the manual 8+8 workaround in this run.
P29's JSON records the rejected global four-tile policy, using requested 16 as
baseline where it already fit and manual 8 as baseline at 1024. Smaller chunks
are not a general throughput improvement. The small-geometry production choice
remains 16; those timings are not a before/after gain from the capacity fix.

## Workloads and measurement limits

Hardware: Apple M4 Pro, 12 CPU cores, 16 GPU cores, 48 GiB RAM; macOS 26.5.2
build 25F84 / Darwin 25.5.0, Metal 4; Apple Metal compiler 32023.883;
Rust 1.96.0 / LLVM 22.1.2. The `release-bench` profile uses fat LTO and one codegen
unit. Each final cell requests 20 samples, 2-second warm-up and a 5-second
measurement target with flat sampling. Slow batch cells extend the target to
collect at least one full operation per sample. Fixture preparation, CPU oracle
checks and cold pipeline compilation are outside timed loops.

P27 uses distinct HT irreversible RGB codestreams generated from `patterned_rgb8`
with the first pixel varied per image. Input hashes frame the codestreams with
little-endian u64 lengths; output hashes frame decoded images the same way.
Timings include GPU completion; resident and explicit readback are separate cells.

P28 uses `patterned_gray8`/`patterned_rgb8`, three DWT levels, guard bits 2 and
64x64 HT cleanup blocks. Its batch-16 workload is sixteen sequential public
encodes through a reused accelerator, with each first pixel varied. Inputs and
outputs are framed with little-endian u64 lengths. Timings include host pixels
to completed host codestreams. Every fixture is encoded twice, compared exactly
with scalar bytes and independently decoded before timing.

P29 repeats one initialized private RGB input 16 times. Input SHA-256 identifies
that one raw image (no framing); output SHA-256 frames all 16 codestreams with
little-endian u64 lengths. Timings start with the input already resident and end
with completed resident codestreams; readback and every-image native decode are
outside timing. All chunk choices produce identical encoded bytes.

These are synthetic workloads on one M4 Pro, not production image-corpus or
cross-Apple-GPU evidence. No per-cell GPU timestamps, counters, occupancy, traffic,
cache misses or peak-memory measurements were collected. Source-derived buffer
capacities are distinguished from measured hardware memory usage. Other system
activity and process order can bias timings; confidence intervals describe each
run's samples, not all possible machines or workloads. CUDA was not changed by
P27–P29; the prior RTX 4070 SUPER launch experiment is recorded in P26.

## Reproduction and retained artifacts

From the repository root:

```sh
J2K_REQUIRE_METAL_BENCH=1 cargo bench --profile release-bench -p j2k-metal --bench decode_stages -- metal_idwt97_geometry_distinct --sample-size 20 --warm-up-time 2 --measurement-time 5
J2K_REQUIRE_METAL_BENCH=1 cargo bench --profile release-bench -p j2k-metal --bench transform_stages -- metal_lossy_resident --sample-size 20 --warm-up-time 2 --measurement-time 5
J2K_REQUIRE_METAL_BENCH=1 cargo bench --profile release-bench -p j2k-metal --bench resident_packetization -- metal_classic_chunks --sample-size 20 --warm-up-time 2 --measurement-time 5
```

A/B processes invoked the just-built executable directly with `--bench`, avoiding
intervening Cargo rebuilds. Temporary controls were removed from production:
`J2K_METAL_DISABLE_IDWT97_CHUNKS=1` selected P20, unset selected the rejected P27
prototype; `J2K_METAL_DISABLE_RESIDENT_LOSSY=1` selected staged encoding, unset
selected the new whole-tile path. Reproduction of those historical arms needs
the removed patch or equivalent temporary source changes, not merely an
environment variable. The ignored local `target/metal-next` contains raw logs,
`baseline.json` source hashes, `p27-prototype.patch`, and `p28-control.patch`.
They are local artifacts; the versioned JSON records preserve the estimates,
confidence bounds and per-cell input/output hashes.

Measured executable SHA-256:

- P27 `decode_stages`: `7bc656b560193623d4d89be50bf3633a6e4d47a8fe9f61bc2cb59622f03f7317`
- P28 final `transform_stages`: `15745da91e16ee8995be2937fef79917479778be8b16fb54bdea7ab8c7a79ff4`
- P29 `resident_packetization`: `79957c555ef8f64179df0548c9905c39a3ff9371bbeeff8aee5e7326da1baf9f`

P27 primary logs are `p27-repeat-control.log` and `p27-repeat-candidate.log`;
reverse-order logs are `p27-reverse-candidate.log` and `p27-reverse-control.log`.
P28 final logs are `p28-final-control.log` and `p28-final-candidate.log`.
P29's final nine-cell matrix is `p29-cap-aware.log`. Earlier scalar-packetizer
numbers in `p28-matched-control.log`/`p28-candidate.log` are superseded.

## Verification

Executed successfully on the final production source:

```sh
J2K_REQUIRE_METAL_RUNTIME=1 cargo test --profile gpu-quick -p j2k-metal --lib --bins --tests --examples --all-features -- --test-threads=1
J2K_REQUIRE_METAL_RUNTIME=1 cargo test --profile gpu-quick -p j2k-metal --lib --all-features -- --ignored --skip metal_irreversible_idwt_gpu_capture --test-threads=1
cargo test --profile gpu-quick -p j2k-native --lib single_tile::tests::whole_tile -- --test-threads=1
cargo test --profile gpu-quick -p j2k-native --lib -- --test-threads=1
cargo clippy --profile gpu-quick -p j2k-metal -p j2k-native --lib --all-features -- -D warnings
cargo clippy --profile gpu-quick -p j2k-metal --bins --tests --benches --examples --all-features -- -D warnings -A clippy::disallowed_methods -A clippy::disallowed_macros
cargo fmt -p j2k-metal -p j2k-native -- --check
cargo bench --profile release-bench -p j2k-metal --bench transform_stages --bench resident_packetization --no-run
cargo xtask repo-lint
cargo xtask gpu-experiment validate docs/performance-experiments/P27-metal-idwt97-chunks.json
cargo xtask gpu-experiment validate docs/performance-experiments/P28-metal-resident-lossy.json
cargo xtask gpu-experiment validate docs/performance-experiments/P29-metal-classic-small-chunks.json
git diff --check
```

The Metal suite passed 519 tests (26 marked ignored in that invocation); the
separate hardware invocation passed 21 of the ignored library tests. Native's
full library suite passed 657, with 2 marked ignored, including all 6 whole-tile
boundary tests. The interactive GPU capture test and four ignored integration
benchmarks were not run. Repository lint passed all 101 checks. Clippy passed
with warnings denied; the non-library invocation permits the repository's test
assertion macros and methods. No production lint suppression was introduced.

The final release-bench executables were also run in Criterion `--test` mode for
`metal_lossy_resident` and `metal_resident_packetization`, including exact scalar
lossy codestream probes and Classic 1024/batch-16 output validation. Production
experiment switches are absent. Final diff/status review preserved unrelated
working-tree changes, and LSP inventory showed no remaining servers.

Red/green evidence is retained locally: `p27-red.log`, `p29-red.log` and
`p29-green.log`, `p28-native-budget-red.log`/`p28-native-budget-green.log`, and
`p28-budget-red.log`/`p28-final-tests.log`. The whole Metal suite subsequently
exposed the packetization contract and fused-input diagnostic differences;
production dispatch and its exact counter assertions were corrected before the
final successful suite. The new benchmark modules were added to the existing
strict source inventory rather than relaxing that assertion.

Cross-platform compilation and performance on other Apple GPUs were not tested.
The pre-existing Classic97 RGB decode discrepancy described in P20 remains
outside these changes; the new lossy encode oracle comparisons all pass exactly.
