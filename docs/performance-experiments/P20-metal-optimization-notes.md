# Metal kernel optimization results — 2026-09-04

The retained performance change batches the existing 9/7 inverse-transform stages across compatible distinct images. It preserves the lift arithmetic, scaling constants, barriers, buffer ownership and public APIs. A separate correctness fix preserves completed high-frequency bands when a multi-level forward 9/7 transform has only one active axis.

## Measured result

Apple M4 Pro, 16 GPU cores, 48 GiB RAM, macOS 26.5.2, Rust 1.96.0, Metal 32023.883; `release-bench`. The final paired runs used one executable, 20 samples per cell, two-second warm-up and a five-second requested collection window. Criterion extended some windows to collect all samples. Values below are Criterion point estimates in milliseconds per entire batch, not per image.

| Image / batch | Resident before → after | Readback before → after |
|---|---:|---:|
| 128×128 / 16 | 8.628 → 5.940 | 8.753 → 5.989 |
| 640×480 / 16 | 16.823 → 14.674 | 18.083 → 15.799 |
| 1024×1024 / 16 | 46.014 → 43.384 | 50.073 → 46.833 |
| 512×512 / 1 | 7.208 → 7.138 | 7.193 → 7.170 |

Batch-16 reductions are approximately 31%, 13%, and 5–7%, respectively. The single-image route is unchanged and its variation is within the noise threshold. See [the promoted record](P20-metal-idwt97-batching.json) for confidence intervals and hashes.

Each batch uses HT irreversible RGB8, three decomposition levels, 64×64 code blocks, and independently allocated codestreams. The first source pixel varies across otherwise identical `patterned_rgb8` images. The benchmark asserts that every codestream hash is distinct and compares every output byte with native CPU decode. This synthetic corpus tests dispatch geometry, not production-image diversity.

Decode timing starts from a prepared batch, excluding parsing, preparation and initial upload. Resident timing waits for completion without copying image pixels to the CPU; readback timing includes the final pixel copy. The original repeated-input benchmark is retained as a separate workload: it can decode once and broadcast, so it does not measure distinct-image batching.

## Why batching is bounded

The new dispatcher interleaves the stacked bands and dispatches the original ten scale/lift stages with image index in the grid's third dimension. Eligible reconstruction stages issue one sequence for the batch instead of one per image. The new structural regression checks sequence counts independently of batch size and verifies CPU pixels.

Single images and reconstruction stages whose total output exceeds 20 MiB use the existing per-image dispatcher. This is an empirical working-set limit, not a hardware cache-capacity claim. Unbounded batching increased the 1024×1024 resident point estimate from 43.281 to 56.764 ms in the initial experiment. At that geometry the final stage holds 64 MiB of output; the bounded implementation batches smaller levels while preserving the original final-stage route. No resource safety limit was relaxed.

Only one GPU was measured. Other GPUs, four-component throughput, alternative block sizes, large whole-slide inputs, ROI throughput, cold pipeline compilation, occupancy, spills and hardware memory traffic remain unmeasured. Earlier runs showed background/DVFS variation, particularly on unchanged controls; small improvements were not accepted on that evidence alone.

## Candidates removed after measurement

| Record | Candidate | Decision |
|---|---|---|
| [P20 unbounded](P20-metal-idwt97-unbounded.json) | Batch every inverse level | Replaced by the bounded route after a large-image regression. |
| [P21](P21-metal-ht-command-coalescing.json) | Reduce HT submission count from seven to two | Rejected: 512×512 improved, 128×128 regressed. |
| [P22](P22-metal-ht-small-state.json) | Specialize width ≤64 HT context arrays | Rejected: no consistent total-time gain; larger encode regressed. |
| [P23](P23-metal-fdwt97-compact-grids.json) | Launch only active forward-lift parity | Rejected: fewer invocations did not improve full encode. |
| [P24](P24-metal-idwt97-compact-grids.json) | Launch only active inverse-lift parity | Rejected: mixed geometry results, including a readback regression. |
| [P25](P25-metal-idwt97-threadgroups.json) | Use 128-thread inverse groups | Rejected: larger-image and small-image readback regressions. |

All temporary selection switches, specialized HT pipelines, compact parity grids and threadgroup helpers were removed. Expanded exact HT scalar/batch tests and profiling-versus-unprofiled codestream checks remain.

## Reproduction

Current production candidate:

```sh
J2K_REQUIRE_METAL_BENCH=1 cargo bench --profile release-bench -p j2k-metal --bench decode_stages -- metal_idwt97_geometry_distinct --sample-size 20 --warm-up-time 2 --measurement-time 5
```

The historical same-binary control added a cached `J2K_METAL_DISABLE_BATCHED_IDWT97` check to `SubmissionContext::submit_idwt` and ORed it into `use_single`. With that temporary harness restored, run the command above with the variable set to `1` and `--save-baseline final-idwt-control`; repeat with it unset and `--baseline final-idwt-control`. The final treatment invoked the just-built Criterion executable directly with `--bench`, preventing any intervening source edit from triggering a rebuild. Its SHA-256 is recorded in P20. The switch is deliberately absent from production.

Other experiment commands:

```sh
J2K_REQUIRE_METAL_BENCH=1 cargo bench --profile release-bench -p j2k-metal --bench resident_packetization -- 'metal_resident_packetization.*/ht' --sample-size 30 --measurement-time 10 --warm-up-time 3
J2K_REQUIRE_METAL_BENCH=1 cargo bench --profile release-bench -p j2k-metal --bench transform_stages -- 'metal_fdwt97/(stage_1024x768_l3|encode_gray8_512_l3)'
J2K_REQUIRE_METAL_BENCH=1 cargo bench --profile release-bench -p j2k-metal --bench decode_stages -- metal_idwt97_geometry_distinct --sample-size 10 --warm-up-time 1 --measurement-time 3
```

The JSON records describe the removed candidate routes and historical A/B variables. HT scratch comparisons hold coalesced submission fixed; the command-coalescing comparison holds generic HT scratch fixed. Encode timing starts with private input already uploaded and excludes final codestream readback. Classic 1024×1024 batch-16 benchmarking hit the existing 512 MiB per-allocation cap (720,420,864 bytes requested); the expanded benchmark explicitly omits that unsupported cell and retains HT at that size.

## Correctness and verification

The new batched regression failed before implementation (16 inverse sequences versus two for an eight-image, two-level grayscale batch) and passes afterward. It covers Classic/HT, grayscale/RGB, batches one/eight, and even/odd dimensions.

The expanded forward-transform reference test exposed a pre-existing multi-level single-axis ping-pong bug. The fix copies completed bands before the buffer swap. Every coefficient bit now matches the scalar reference for 64×48, 1×7, 7×1, 2×3, 31×33, 65×49 and 513×257. Original cases and exact assertions remain.

A separate diagnostic Classic97 RGB fixture at 131×65 showed an existing one-byte CPU/Metal difference at index 10413 (Metal 250, CPU 251), reproduced with the original per-image path. That broader parity limitation remains unresolved; no tolerance was introduced to hide it.

Verification commands:

```sh
cargo fmt -p j2k-metal -- --check
cargo clippy --profile gpu-quick -p j2k-metal --lib --all-features -- -D warnings
cargo clippy --profile gpu-quick -p j2k-metal --bins --tests --benches --examples --all-features -- -D warnings -A clippy::disallowed_methods -A clippy::disallowed_macros
J2K_REQUIRE_METAL_RUNTIME=1 cargo test --profile gpu-quick -p j2k-metal --lib --bins --tests --examples --all-features -- --test-threads=1
J2K_REQUIRE_METAL_RUNTIME=1 cargo test --profile gpu-quick -p j2k-metal --lib --all-features -- --ignored --skip metal_irreversible_idwt_gpu_capture --test-threads=1
```

All five commands above completed successfully. The normal suite passed 509 tests, and the explicit ignored-test run passed all 21 required hardware cases: 530 tests total. Optional GPU capture and four separate benchmark/performance-guard harnesses were not included in that count. Both Clippy checks, formatting and the final diff whitespace check passed. All seven P20–P25 JSON records passed `cargo xtask gpu-experiment validate <record>`.

`cargo xtask repo-lint` passed 100 of 101 checks. Its remaining failure is `suppressions_stay_in_reviewed_device_generation_scopes`: the concurrently edited `crates/j2k-metal/src/encode/tests/stats_inflight.rs:108` adds an unreviewed `deprecated` allowance. That change is outside this optimization and remains intact. The experiment-variable documentation check passes after recording the removed switches in `docs/env-vars.md`.

Unrelated pre-existing and concurrent edits in other code, including encode profiling stats and native compact97 encoding, are preserved. The crate formatter only adjusted layout in the concurrent stats edits. No whole-workspace release or cross-hardware claim is made.
