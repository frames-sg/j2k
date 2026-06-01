# Adaptive J2K / HTJ2K Routing Gates

This file records checked-in gate decisions for adaptive J2K / HTJ2K routing.
It is not a public speed claim. Public claims still require a benchmark report
per `docs/bench.md`.

## Gate Policy

- Default route: adaptive accelerated.
- Strict Metal/CUDA rows: capability proof only.
- A device stage is eligible for default routing only when the optimized device
  stage beats optimized CPU by at least `10% + Criterion noise`.
- A full route is eligible only when the end-to-end adaptive row also beats
  CPU-only for the same workload shape.
- A logical GPU stage that loses its gate remains blocked until RCA either
  fixes optimization debt or reclassifies that exact shape as CPU-shaped.

## Evidence Recording Template

New adaptive gate reruns must record:

- Commit SHA.
- Host, OS, architecture, CPU/GPU, memory, driver/runtime versions.
- Rust compiler version.
- Exact command and required environment variables.
- CPU-only Criterion interval.
- Adaptive Criterion interval.
- Strict device Criterion interval when available.
- Stage Criterion intervals for every device-shaped stage under consideration.
- Gate decision: `approved`, `candidate`, `blocked`, or `reclassified-cpu`.
- RCA reason for every blocked logical GPU-shaped stage.

Do not copy numbers into this file from a different host class. Apple Metal
numbers and CUDA runner numbers are separate evidence sets.

## CUDA HTJ2K Decode RCA Profile Gate

Use the CUDA profile gate when decode rows are blocked and the next decision
needs transfer, launch, stage, trace, or route-composition evidence. The
profile gate is opt-in on the self-hosted CUDA workflow:

```sh
gh workflow run gpu-validation.yml \
  --ref <branch> \
  -f run-linux-ci=false \
  -f run-metal-validation=false \
  -f run-nvidia-baseline=false \
  -f run-timed-benchmarks=false \
  -f run-cuda-htj2k-decode-profile=true
```

The workflow step runs:

```sh
SIGNINUM_REQUIRE_CUDA_RUNTIME=1 \
SIGNINUM_REQUIRE_CUDA_HTJ2K_STRICT=1 \
SIGNINUM_REQUIRE_CUDA_BENCH=1 \
SIGNINUM_J2K_PROFILE_STAGES=summary \
SIGNINUM_J2K_CUDA_TRACE="$(pwd)/target/cuda_htj2k_decode_trace.json" \
samply record --save-only -o target/cuda_htj2k_decode_samply.json.gz -- \
  cargo bench -p signinum-j2k-cuda --bench htj2k_decode \
  --features cuda-runtime,cuda-profiling -- \
  --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2 \
  2>&1 | tee target/cuda_htj2k_decode_profile.log
```

`SIGNINUM_J2K_PROFILE_STAGES=1` may replace `summary` when the RCA needs the
full stage profile. The workflow uploads
`cuda-htj2k-decode-rca-profile` with:

- `target/cuda_htj2k_decode_profile.log`
- `target/cuda_htj2k_decode_trace.json`
- `target/cuda_htj2k_decode_samply.json.gz`
- `target/cuda_htj2k_decode_samply_status.txt`
- `target/criterion`

Upload runs under `always()`, so failed profile attempts still leave the log,
Chrome trace, `samply` status, and partial Criterion evidence for triage.
When the runner cannot lower `/proc/sys/kernel/perf_event_paranoid` because
passwordless sudo and direct sysctl writes are unavailable, the status file
records `samply_status=blocked`; the step still runs the CUDA decode bench with
stage summaries and CUDA trace export. A `samply` CPU profile is only present
when the runner permits Linux perf sampling. Use an absolute trace path because
Cargo may run the bench binary from the package directory.
Treat these artifacts as internal RCA evidence, not public speed claims.

## Direct NVIDIA nvJPEG2000 Decode Comparator Gate

Use `run-nvidia-baseline=true` when a CUDA decode decision needs direct
comparison against NVIDIA's installed library rather than only signinum CPU vs
signinum CUDA route timing. The workflow runs both the existing
JPEG -> HTJ2K transcode comparator and direct HTJ2K decode comparator:

```sh
gh workflow run gpu-validation.yml \
  --ref <branch> \
  -f run-linux-ci=false \
  -f run-metal-validation=false \
  -f run-nvidia-baseline=true \
  -f run-timed-benchmarks=false \
  -f run-cuda-htj2k-decode-profile=false
```

The direct decode command is:

```sh
SIGNINUM_REQUIRE_NV_BASELINE_BUILD=1 \
cargo run --release -p signinum-nvidia-baseline \
  --features nvjpeg2000 --bin decode_compare -- \
  --jpeg-dir crates/signinum-nvidia-baseline/benchtiles/pancreas \
  --warmup 2 \
  --iterations 10 \
  --min-inputs 2 \
  --json target/decode_compare.json \
  --csv target/decode_compare.csv
```

The `nvidia-baseline-comparison` artifact must include:

- `target/transcode_compare.json`
- `target/transcode_compare.csv`
- `target/decode_compare.json`
- `target/decode_compare.csv`

Record the direct decode JSON/CSV rows alongside Criterion decode rows when
reviewing CUDA HTJ2K decode gates.

## 2026-06-01 Local CPU / Metal Profiling Checkpoint

Evidence:

- Base commit before this change set: `7100366`.
- Scope: local working tree with Metal Auto ROI+scaled batch classification,
  known-repeated hybrid color batching, resident timestamp-query dedupe, and
  resident HTJ2K external-validation final-wait coalescing.
- Supersedes the older `2026-05-31 Metal Resident HTJ2K Encode RCA Rerun`
  rows below only for overlapping local Metal rows on this working tree.
- Host: MacBook Pro `Mac16,8`, macOS 26.5 build 25F71, Darwin 25.5.0,
  arm64, Apple M4 Pro 12-core CPU, 16-core GPU, 48 GiB RAM, Metal 4.
- Rust: `rustc 1.88.0 (6b00bc388 2025-06-23)`
- Commands:
  - `samply record --save-only -o target/j2k_cpu_public_api_samply.json.gz -- cargo bench -p signinum-j2k --bench public_api -- --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2`
  - `SIGNINUM_J2K_METAL_PROFILE_STAGES=1 SIGNINUM_GPU_ROUTE_PROFILE=summary samply record --save-only -o target/metal_facade_samply.json.gz -- cargo bench -p signinum --bench facade --features metal -- facade_j2k_htj2k_encode_backend_speed_matrix --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2`
  - `SIGNINUM_J2K_METAL_PROFILE_STAGES=1 SIGNINUM_GPU_ROUTE_PROFILE=summary samply record --save-only -o target/metal_htj2k_encode_samply.json.gz -- cargo bench -p signinum-j2k-metal --bench encode_stages -- --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2`
  - `SIGNINUM_J2K_METAL_PROFILE_STAGES=1 SIGNINUM_GPU_ROUTE_PROFILE=summary cargo bench -p signinum-j2k-metal --bench encode_stages -- j2k_metal_htj2k_rpcl_rgb8_512_batch --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2 2>&1 | tee target/metal_htj2k_encode_profile.log`
  - `xcrun xctrace record --template 'Time Profiler' --output target/apple-traces/j2k_cpu_public_api_time_profile.trace --time-limit 45s --no-prompt --target-stdout - --launch -- $(command -v cargo) bench -p signinum-j2k --bench public_api -- j2k_public_cpu_encode_matrix --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2`
  - `xcrun xctrace record --template 'Time Profiler' --output target/apple-traces/metal_htj2k_encode_time_profile.trace --time-limit 60s --no-prompt --target-stdout - --launch -- $(command -v cargo) bench -p signinum-j2k-metal --bench encode_stages -- j2k_metal_htj2k_rpcl_rgb8_512_batch --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2`
  - `xcrun xctrace record --template 'Metal System Trace' --output target/apple-traces/metal_htj2k_encode_metal_system.trace --time-limit 60s --no-prompt --target-stdout - --launch -- $(command -v cargo) bench -p signinum-j2k-metal --bench encode_stages -- j2k_metal_htj2k_rpcl_rgb8_512_batch --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2`
  - `cargo instruments -t time -p signinum-j2k --bench public_api --profile bench -o target/apple-traces/cargo_instruments_j2k_cpu_public_api_time.trace --time-limit 45000 --no-open -- j2k_public_cpu_encode_matrix --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2`
  - `cargo instruments -t time -p signinum-j2k-metal --bench encode_stages --profile bench -o target/apple-traces/cargo_instruments_metal_htj2k_encode_time.trace --time-limit 60000 --no-open -- j2k_metal_htj2k_rpcl_rgb8_512_batch --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2`
  - `cargo instruments -t 'Metal System Trace' -p signinum-j2k-metal --bench encode_stages --profile bench -o target/apple-traces/cargo_instruments_metal_htj2k_encode_metal_system.trace --time-limit 60000 --no-open -- j2k_metal_htj2k_rpcl_rgb8_512_batch --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2`
  - `SIGNINUM_J2K_METAL_PROFILE_STAGES=1 SIGNINUM_GPU_ROUTE_PROFILE=summary cargo instruments -t time -p signinum-j2k-metal --bench encode_stages --profile bench -o target/apple-traces/cargo_instruments_metal_htj2k_encode_time_after_wait.trace --time-limit 60000 --no-open -- j2k_metal_htj2k_rpcl_rgb8_512_batch --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2`
  - `SIGNINUM_J2K_METAL_PROFILE_STAGES=1 SIGNINUM_GPU_ROUTE_PROFILE=summary cargo instruments -t 'Metal System Trace' -p signinum-j2k-metal --bench encode_stages --profile bench -o target/apple-traces/cargo_instruments_metal_htj2k_encode_metal_system_after_wait.trace --time-limit 60000 --no-open -- j2k_metal_htj2k_rpcl_rgb8_512_batch --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2`
  - `cargo instruments -t time -p signinum-j2k --bench public_api --profile bench -o target/apple-traces/cargo_instruments_j2k_cpu_public_api_time_after_wait.trace --time-limit 45000 --no-open -- j2k_public_cpu_encode_matrix --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2`
  - `SIGNINUM_J2K_METAL_PROFILE_STAGES=1 SIGNINUM_GPU_ROUTE_PROFILE=summary cargo bench -p signinum-j2k-metal --bench encode_stages -- j2k_metal_htj2k_rpcl_rgb8_512_batch --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2 2>&1 | tee target/metal_htj2k_encode_profile_after_wait.log`
  - `SIGNINUM_J2K_METAL_PROFILE_STAGES=1 SIGNINUM_GPU_ROUTE_PROFILE=summary samply record --save-only -o target/metal_htj2k_encode_after_wait_samply.json.gz -- cargo bench -p signinum-j2k-metal --bench encode_stages -- j2k_metal_htj2k_rpcl_rgb8_512_batch --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2`
- Local artifacts:
  - `target/j2k_cpu_public_api_samply.json.gz`
  - `target/metal_facade_samply.json.gz`
  - `target/metal_htj2k_encode_samply.json.gz`
  - `target/metal_htj2k_encode_profile.log`
  - `target/apple-traces/j2k_cpu_public_api_time_profile.trace`
  - `target/apple-traces/metal_htj2k_encode_time_profile.trace`
  - `target/apple-traces/metal_htj2k_encode_metal_system.trace`
  - `target/apple-traces/cargo_instruments_j2k_cpu_public_api_time.trace`
  - `target/apple-traces/cargo_instruments_metal_htj2k_encode_time.trace`
  - `target/apple-traces/cargo_instruments_metal_htj2k_encode_metal_system.trace`
  - `target/apple-traces/cargo_instruments_j2k_cpu_public_api_time_after_wait.trace`
  - `target/apple-traces/cargo_instruments_metal_htj2k_encode_time_after_wait.trace`
  - `target/apple-traces/cargo_instruments_metal_htj2k_encode_metal_system_after_wait.trace`
  - `target/metal_htj2k_encode_profile_after_wait.log`
  - `target/metal_htj2k_encode_after_wait_samply.json.gz`
- CPU public API `samply` metadata has `symbolicated=false`; record the
  artifact existence only, not hotspot conclusions.
- `cargo-instruments` was installed after the direct `xctrace` run and
  succeeded when invoked from a TTY. The non-TTY attempt failed because
  `xctrace` could not write the launched target's stdout path.
- CUDA runner profiling is intentionally last for this slice; the local CPU and
  CPU/Metal gates above are the active evidence set before launching more CUDA
  workflow work.

End-to-end facade gate:

| Shape | CPU-only | Adaptive | Strict Metal | Decision |
| --- | ---: | ---: | ---: | --- |
| RGB8 512 HTJ2K encode | `4.4459 ms .. 4.8904 ms` | `4.5165 ms .. 4.8961 ms` | `34.123 ms .. 34.444 ms` | `blocked`: adaptive does not clear `10% + noise`; strict Metal loses |
| RGB8 1024 HTJ2K encode | `20.365 ms .. 22.130 ms` | `20.352 ms .. 21.915 ms` | `128.410 ms .. 128.823 ms` | `blocked`: adaptive does not clear `10% + noise`; strict Metal loses |
| RGBA8 512 HTJ2K encode | `5.7268 ms .. 6.1588 ms` | `5.7657 ms .. 6.0689 ms` | `39.148 ms .. 39.746 ms` | `blocked`: adaptive does not clear `10% + noise`; strict Metal loses |
| RGBA8 1024 HTJ2K encode | `26.998 ms .. 28.794 ms` | `26.820 ms .. 28.057 ms` | `150.024 ms .. 152.239 ms` | `blocked`: adaptive does not clear `10% + noise`; strict Metal loses |

Stage Criterion evidence:

| Stage / Shape | CPU | Metal | Gate |
| --- | ---: | ---: | --- |
| RCT 512 | `94.00 us .. 98.97 us` | `210.25 us .. 259.44 us` | `reclassified-cpu` |
| RCT 1024 | `366.76 us .. 396.14 us` | `503.98 us .. 601.05 us` | `reclassified-cpu` |
| RCT 2048 | `1.6281 ms .. 1.6972 ms` | `1.8508 ms .. 2.0250 ms` | `reclassified-cpu` |
| DWT 512 | `1.0664 ms .. 1.1011 ms` | `238.00 us .. 272.89 us` | `candidate` |
| DWT 1024 | `5.4354 ms .. 5.8022 ms` | `652.74 us .. 727.42 us` | `candidate` |
| DWT 2048 | `26.650 ms .. 29.742 ms` | `2.7258 ms .. 2.9753 ms` | `candidate` |
| HT code blocks, 192 | `7.6594 ms .. 8.4861 ms` | `3.0452 ms .. 3.1798 ms` | `candidate` |
| HT code blocks, 768 | `30.101 ms .. 31.390 ms` | `6.3754 ms .. 7.0652 ms` | `candidate` |

Encode-path evidence:

| Route / Shape | Criterion interval | Gate |
| --- | ---: | --- |
| CPU classic RGB8 512 | `14.133 ms .. 28.089 ms` | Baseline for classic only |
| CPU HTJ2K RGB8 512 | `5.0183 ms .. 5.3946 ms` | Baseline |
| Auto host Metal-buffer HTJ2K RGB8 512 | `3.8664 ms .. 4.5909 ms` | `candidate` only; facade gate still required |
| Resident strict Metal RGB8 512 | `180.400 ms .. 181.283 ms` | `blocked` |
| CPU classic RGB8 1024 | `61.366 ms .. 73.021 ms` | Baseline for classic only |
| CPU HTJ2K RGB8 1024 | `24.294 ms .. 27.732 ms` | Baseline |
| Auto host Metal-buffer HTJ2K RGB8 1024 | `12.403 ms .. 14.262 ms` | `candidate` only; facade gate still required |
| Resident strict Metal RGB8 1024 | `399.080 ms .. 407.915 ms` | `blocked` |
| Resident strict Metal RPCL RGB8 512 batch 16 | `103.246 ms .. 105.282 ms` | `blocked` |
| Resident strict Metal RPCL RGB8 512 batch 64 | `126.772 ms .. 130.034 ms` | `blocked` |
| Resident strict Metal RPCL RGB8 512 batch 128 | `142.631 ms .. 147.487 ms` | `blocked` |

Post final-wait coalescing evidence:

The external-validation resident HTJ2K batch path now waits once on the last
chunk command buffer before harvesting all chunk status buffers in order. The
CPU-roundtrip validation path deliberately keeps per-chunk waits so validation
readback can overlap later GPU chunks.

| Route / Shape | Criterion interval | Criterion comparison | Gate |
| --- | ---: | --- | --- |
| Resident strict Metal RPCL RGB8 512 batch 16 | `98.229 ms .. 98.496 ms` | `-1.3895%`, within noise | `blocked` |
| Resident strict Metal RPCL RGB8 512 batch 64 | `118.77 ms .. 119.85 ms` | `-3.2704%`, improved | `blocked` |
| Resident strict Metal RPCL RGB8 512 batch 128 | `134.00 ms .. 138.49 ms` | `-2.1376%`, no statistically significant change | `blocked` |

Resident RCA profile rows:

Raw `sync_wait_us` profile fields are converted to milliseconds in this table.

| Tile count | Code blocks | Coefficient prep observed | Command encode observed | Sync wait observed | RCA |
| ---: | ---: | ---: | ---: | ---: | --- |
| 16 | 3072 | `194 us .. 3.493 ms`, median `218 us` | HT median `6 us`, packet prep `2 us`, packetization `2 us`, assembly `1 us` | `101.071 ms .. 130.149 ms`, median `102.377 ms` | sync/wait dominates |
| 64 | 12288 | `652 us .. 1.633 ms`, median `697 us` | HT median `6 us`, packet prep `2 us`, packetization `2 us`, assembly `1 us` | `123.365 ms .. 136.864 ms`, median `124.729 ms` | sync/wait dominates |
| 128 | 24576 | `1.222 ms .. 2.395 ms`, median `1.331 ms` | HT median `6 us`, packet prep `2 us`, packetization `2 us`, assembly `1 us` | `137.544 ms .. 163.802 ms`, median `139.585 ms` | sync/wait dominates |

Apple-native trace cross-check:

| Tool / Template | Target slice | Trace artifact | Criterion interval observed during trace |
| --- | --- | --- | --- |
| `xctrace` / `Time Profiler` | CPU public encode matrix | `target/apple-traces/j2k_cpu_public_api_time_profile.trace` | RGB8 512 classic `9.5540 ms .. 11.605 ms`; HTJ2K `4.3153 ms .. 4.8038 ms`; classic roundtrip `17.714 ms .. 20.453 ms` |
| `xctrace` / `Time Profiler` | Resident Metal RPCL RGB8 512 batch | `target/apple-traces/metal_htj2k_encode_time_profile.trace` | batch 16 `98.159 ms .. 99.739 ms`; batch 64 `118.99 ms .. 121.96 ms`; batch 128 `133.74 ms .. 137.27 ms` |
| `xctrace` / `Metal System Trace` | Resident Metal RPCL RGB8 512 batch | `target/apple-traces/metal_htj2k_encode_metal_system.trace` | batch 16 `98.769 ms .. 100.82 ms`; batch 64 `121.70 ms .. 124.93 ms`; batch 128 `137.53 ms .. 141.23 ms` |
| `cargo-instruments` / `Time Profiler` | CPU public encode matrix | `target/apple-traces/cargo_instruments_j2k_cpu_public_api_time.trace` | trace captured; wrapper reported all three filtered rows `Success` |
| `cargo-instruments` / `Time Profiler` | Resident Metal RPCL RGB8 512 batch | `target/apple-traces/cargo_instruments_metal_htj2k_encode_time.trace` | trace captured; wrapper reported batch 16/64/128 rows `Success` |
| `cargo-instruments` / `Metal System Trace` | Resident Metal RPCL RGB8 512 batch | `target/apple-traces/cargo_instruments_metal_htj2k_encode_metal_system.trace` | trace captured; wrapper reported batch 16/64/128 rows `Success` |
| `cargo-instruments` / `Time Profiler` | CPU public encode matrix, after final-wait coalescing | `target/apple-traces/cargo_instruments_j2k_cpu_public_api_time_after_wait.trace` | trace captured; wrapper reported all three filtered rows `Success` |
| `cargo-instruments` / `Time Profiler` | Resident Metal RPCL RGB8 512 batch, after final-wait coalescing | `target/apple-traces/cargo_instruments_metal_htj2k_encode_time_after_wait.trace` | trace captured; stage sync wait still dominates: 16 `112.444 ms`, 64 `134.916 ms`, 128 `155.133 ms` |
| `cargo-instruments` / `Metal System Trace` | Resident Metal RPCL RGB8 512 batch, after final-wait coalescing | `target/apple-traces/cargo_instruments_metal_htj2k_encode_metal_system_after_wait.trace` | trace captured; stage sync wait still dominates: 16 `110.352 ms`, 64 `136.013 ms`, 128 `166.538 ms` |

Decision:

- Keep Metal HTJ2K encode default routing `blocked` for RGB8/RGBA8 512/1024.
- Keep DWT and HT code-block Metal kernels as GPU-shaped `candidate` stages.
- Reclassify standalone RGB RCT as CPU-shaped for the measured 512/1024/2048
  rows until a fused path clears the stage gate.
- Keep resident strict Metal codestream assembly `blocked`; final-wait
  coalescing trims redundant host waits for external-validation chunks but does
  not change the gate. The measured command encode buckets are microsecond-scale
  while `sync_wait_us` remains roughly `97 ms .. 166 ms` across after-change
  Criterion and `cargo-instruments` runs.

## 2026-06-01 CUDA HTJ2K Decode Table-Cache Rerun

Evidence:

- Commit: `9673693`
- Workflow:
  <https://github.com/frames-sg/signinum/actions/runs/26736318698>
- Result: success
- Branch: `codex/cuda-quality-ht-rewrite`
- Supersedes no CUDA decode gates; this rerun verifies the per-session HTJ2K
  decode table cache and keeps the older RCA as the baseline comparison.
- CUDA `samply` status: `blocked`, `perf_event_paranoid=2`, passwordless sudo
  unavailable, and `/proc/sys/kernel/perf_event_paranoid` not writable.
- Artifacts:
  - `/tmp/signinum-all-cuda-gates-26736318698-artifacts/cuda-htj2k-decode-rca-profile/cuda_htj2k_decode_profile.log`
  - `/tmp/signinum-all-cuda-gates-26736318698-artifacts/cuda-htj2k-decode-rca-profile/cuda_htj2k_decode_trace.json`
  - `/tmp/signinum-all-cuda-gates-26736318698-artifacts/cuda-htj2k-decode-rca-profile/cuda_htj2k_decode_samply_status.txt`
  - `/tmp/signinum-all-cuda-gates-26736318698-artifacts/nvidia-baseline-comparison/decode_compare.csv`
  - `/tmp/signinum-all-cuda-gates-26736318698-artifacts/nvidia-baseline-comparison/decode_compare.json`

CUDA HTJ2K decode rerun delta:

| Check | Prior `7100366` / run `26733997559` | Rerun `9673693` / run `26736318698` | Gate |
| --- | ---: | ---: | --- |
| Full tile gray8 512, strict CUDA | `198.30 ms .. 201.11 ms` | `199.95 ms .. 202.18 ms` | `blocked` |
| Full tile RGB8 512, strict CUDA | `205.15 ms .. 210.26 ms` | `212.17 ms .. 216.27 ms` | `blocked` |
| Tile batch gray8 batch 8, strict CUDA | `253.64 ms .. 257.56 ms` | `251.26 ms .. 257.55 ms` | `blocked` |
| Tile batch RGB8 batch 8, strict CUDA | `308.00 ms .. 310.91 ms` | `307.97 ms .. 314.51 ms` | `blocked` |
| NVIDIA comparator tile 00000, Signinum strict CUDA | `167.037675 ms` | `166.064961 ms` | `blocked` |
| NVIDIA comparator tile 00000, nvJPEG2000 wall | `0.840529 ms` | `0.780679 ms` | comparator still proves runner headroom |
| NVIDIA comparator tile 00001, Signinum strict CUDA | `168.412187 ms` | `168.674478 ms` | `blocked` |
| NVIDIA comparator tile 00001, nvJPEG2000 wall | `0.801779 ms` | `0.798889 ms` | comparator still proves runner headroom |

CUDA profile summary delta:

| Path | Prior | Rerun | RCA |
| --- | --- | --- | --- |
| CUDA HTJ2K decode | `count=665`, `stage_sum_us_avg=20680`, `wall_total_us_avg=80036`, `table_upload_us_avg=469` | `count=625`, `stage_sum_us_avg=20287`, `wall_total_us_avg=75628`, `table_upload_us_avg=364` | table uploads improved, but not enough to move the gate |
| Stage hotspots | `ht_cleanup_us_avg=13713`, `h2d_us_avg=3634`, `idwt_us_avg=1658` | `ht_cleanup_us_avg=13303`, `h2d_us_avg=3583`, `idwt_us_avg=1681` | strict route/session/sync overhead still dominates |

Decision:

- Keep CUDA HTJ2K decode default routing `blocked` for every measured full,
  ROI, scaled, ROI-scaled, and batch decode shape.
- Keep the per-session HTJ2K table cache because it removes repeat setup work
  without changing correctness, but treat it as an incremental cleanup rather
  than the wall-time RCA fix.
- A real CUDA `samply` CPU profile remains blocked until the self-hosted runner
  can set `kernel.perf_event_paranoid <= 1` or use passwordless sudo.

## 2026-06-01 CUDA HTJ2K Decode RCA and NVIDIA Baseline Addendum

Evidence:

- Commit: `7100366`
- Workflow:
  <https://github.com/frames-sg/signinum/actions/runs/26733997559>
- Result: success
- Branch: `codex/cuda-quality-ht-rewrite`
- Supersedes no encode rows; this is an RCA addendum to the CUDA decode and
  NVIDIA baseline evidence.
- CUDA `samply` status: `blocked`, `perf_event_paranoid=2`, passwordless sudo
  unavailable, and `/proc/sys/kernel/perf_event_paranoid` not writable.
- Artifacts:
  - `/tmp/signinum-all-cuda-gates-26733997559-artifacts/cuda-htj2k-decode-rca-profile/cuda_htj2k_decode_profile.log`
  - `/tmp/signinum-all-cuda-gates-26733997559-artifacts/cuda-htj2k-decode-rca-profile/cuda_htj2k_decode_trace.json`
  - `/tmp/signinum-all-cuda-gates-26733997559-artifacts/cuda-htj2k-decode-rca-profile/cuda_htj2k_decode_samply_status.txt`
  - `/tmp/signinum-all-cuda-gates-26733997559-artifacts/nvidia-baseline-comparison/decode_compare.csv`
  - `/tmp/signinum-all-cuda-gates-26733997559-artifacts/nvidia-baseline-comparison/decode_compare.json`

CUDA HTJ2K decode profile gate:

| Decode Shape | CPU | CUDA | Gate |
| --- | ---: | ---: | --- |
| Full tile gray8 512 | `3.0688 ms .. 3.0862 ms` | `198.30 ms .. 201.11 ms` | `blocked` |
| Full tile RGB8 512 | `8.7321 ms .. 8.8589 ms` | `205.15 ms .. 210.26 ms` | `blocked` |
| Full tile RGBA8 512 | `10.896 ms .. 10.943 ms` | `211.20 ms .. 214.93 ms` | `blocked` |
| ROI gray8 256 | `1.9687 ms .. 2.0143 ms` | `197.74 ms .. 201.99 ms` | `blocked` |
| ROI RGB8 256 | `5.4650 ms .. 5.5199 ms` | `209.05 ms .. 214.01 ms` | `blocked` |
| ROI RGBA8 256 | `5.4854 ms .. 5.5260 ms` | `208.73 ms .. 213.34 ms` | `blocked` |
| Scaled gray8 256 | `652.04 us .. 653.83 us` | `192.99 ms .. 198.82 ms` | `blocked` |
| Scaled RGB8 256 | `2.0228 ms .. 2.0470 ms` | `197.41 ms .. 200.20 ms` | `blocked` |
| Scaled RGBA8 256 | `2.0675 ms .. 2.0862 ms` | `196.98 ms .. 200.12 ms` | `blocked` |
| ROI-scaled gray8 128 | `331.89 us .. 332.94 us` | `199.34 ms .. 200.17 ms` | `blocked` |
| ROI-scaled RGB8 128 | `975.39 us .. 988.13 us` | `200.83 ms .. 203.56 ms` | `blocked` |
| ROI-scaled RGBA8 128 | `990.97 us .. 1.0121 ms` | `198.95 ms .. 200.73 ms` | `blocked` |
| Tile batch gray8 batch 8 | `24.571 ms .. 24.609 ms` | `253.64 ms .. 257.56 ms` | `blocked` |
| Tile batch RGB8 batch 8 | `87.860 ms .. 89.131 ms` | `308.00 ms .. 310.91 ms` | `blocked` |
| Tile batch RGBA8 batch 8 | `91.525 ms .. 91.917 ms` | `298.31 ms .. 301.23 ms` | `blocked` |

CUDA profile summary:

| Path | Count | Internal timing summary | RCA |
| --- | ---: | --- | --- |
| CUDA HTJ2K decode | 665 | `stage_sum_us_avg=20680`, `wall_total_us_avg=80036`, `ht_cleanup_us_avg=13713`, `h2d_us_avg=3634`, `idwt_us_avg=1658` | device stages do not explain the full wall-time floor; route/session/sync overhead still dominates |
| CPU decode | 31252 | `total_us_avg=1355`, `codeblock_us_avg=870`, `idwt_us_avg=278`, `store_us_avg=45` | CPU remains the active default route for measured shapes |

Direct NVIDIA nvJPEG2000 comparator:

| Fixture | Signinum CPU | Signinum strict CUDA | nvJPEG2000 wall | nvJPEG2000 GPU | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `nvidia_htj2k:tile_00000.jpg` | `4.006072 ms` | `167.037675 ms` | `0.840529 ms` | `0.720896 ms` | `blocked`: Signinum CUDA route loses to CPU and NVIDIA |
| `nvidia_htj2k:tile_00001.jpg` | `3.801020 ms` | `168.412187 ms` | `0.801779 ms` | `0.691200 ms` | `blocked`: Signinum CUDA route loses to CPU and NVIDIA |

Decision:

- Keep CUDA HTJ2K decode default routing `blocked` for every measured full,
  ROI, scaled, ROI-scaled, and batch decode shape.
- Keep direct NVIDIA comparator evidence internal; it proves the RTX 4070
  runner can decode the same generated HTJ2K inputs quickly through
  nvJPEG2000, so the current Signinum CUDA loss is implementation/route debt,
  not a runner capability limit.
- A real CUDA `samply` CPU profile remains blocked until the runner can set
  `kernel.perf_event_paranoid <= 1` or use passwordless sudo.

## 2026-05-31 Metal Resident HTJ2K Encode RCA Rerun

Evidence:

- Commit: `65b3921`
- Supersedes the older same-date Metal RGB8 HTJ2K encode evidence below for
  overlapping RGB8 512/1024 shapes.
- Host: MacBook Pro `Mac16,8`, macOS 26.5 build 25F71, Darwin 25.5.0,
  arm64, Apple M4 Pro 12-core CPU, 16-core GPU, 48 GiB RAM, Metal 4.
- Rust: `rustc 1.88.0 (6b00bc388 2025-06-23)`
- Commands:
  - `SIGNINUM_REQUIRE_METAL_BENCH=1 SIGNINUM_J2K_METAL_PROFILE_STAGES=1 cargo bench -p signinum-j2k-metal --bench encode_stages -- --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2`
  - `SIGNINUM_REQUIRE_METAL_BENCH=1 cargo bench -p signinum --bench facade --features metal -- facade_j2k_htj2k_encode_backend_speed_matrix --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2`

End-to-end facade gate:

| Shape | CPU-only | Adaptive | Strict Metal | Decision |
| --- | ---: | ---: | ---: | --- |
| RGB8 512 HTJ2K encode | `4.3245 ms .. 4.6258 ms` | `4.3713 ms .. 4.8654 ms` | `33.590 ms .. 33.780 ms` | `blocked`: adaptive does not clear `10% + noise`; strict Metal loses |
| RGB8 1024 HTJ2K encode | `19.664 ms .. 20.856 ms` | `19.600 ms .. 20.727 ms` | `125.84 ms .. 127.29 ms` | `blocked`: adaptive does not clear `10% + noise`; strict Metal loses |
| RGBA8 512 HTJ2K encode | `5.6062 ms .. 5.8995 ms` | `5.5989 ms .. 5.9073 ms` | `38.693 ms .. 39.058 ms` | `blocked`: adaptive does not clear `10% + noise`; strict Metal loses |
| RGBA8 1024 HTJ2K encode | `25.861 ms .. 27.024 ms` | `26.211 ms .. 27.613 ms` | `146.24 ms .. 146.90 ms` | `blocked`: adaptive does not clear `10% + noise`; strict Metal loses |

Stage Criterion evidence:

| Stage / Shape | CPU | Metal | Gate |
| --- | ---: | ---: | --- |
| RCT 512 | `88.956 us .. 92.536 us` | `211.84 us .. 241.96 us` | `reclassified-cpu` |
| RCT 1024 | `380.71 us .. 395.41 us` | `671.99 us .. 954.58 us` | `reclassified-cpu` |
| RCT 2048 | `1.6764 ms .. 1.7454 ms` | `2.0384 ms .. 2.4412 ms` | `reclassified-cpu` |
| DWT 512 | `1.0081 ms .. 1.0408 ms` | `233.08 us .. 250.68 us` | `candidate` |
| DWT 1024 | `5.0448 ms .. 5.2933 ms` | `639.35 us .. 852.23 us` | `candidate` |
| DWT 2048 | `25.844 ms .. 28.346 ms` | `2.9085 ms .. 3.3051 ms` | `candidate` |
| HT code blocks, 192 | `7.0936 ms .. 7.3332 ms` | `2.9461 ms .. 2.9980 ms` | `candidate` |
| HT code blocks, 768 | `28.695 ms .. 29.393 ms` | `5.9062 ms .. 6.1836 ms` | `candidate` |

Encode-path evidence:

| Route / Shape | Criterion interval | Gate |
| --- | ---: | --- |
| CPU classic RGB8 512 | `12.996 ms .. 15.587 ms` | Baseline for classic only |
| CPU HTJ2K RGB8 512 | `4.6573 ms .. 5.2017 ms` | Baseline |
| Auto host Metal-buffer HTJ2K RGB8 512 | `3.9372 ms .. 4.4751 ms` | `candidate` only; facade gate still required |
| Resident strict Metal RGB8 512 | `181.32 ms .. 181.77 ms` | `blocked` |
| CPU classic RGB8 1024 | `53.524 ms .. 61.601 ms` | Baseline for classic only |
| CPU HTJ2K RGB8 1024 | `22.038 ms .. 24.093 ms` | Baseline |
| Auto host Metal-buffer HTJ2K RGB8 1024 | `11.661 ms .. 12.877 ms` | `candidate` only; facade gate still required |
| Resident strict Metal RGB8 1024 | `391.10 ms .. 392.37 ms` | `blocked` |
| Resident strict Metal RPCL RGB8 512 batch 16 | `101.03 ms .. 101.45 ms` | `blocked` |
| Resident strict Metal RPCL RGB8 512 batch 64 | `123.43 ms .. 124.06 ms` | `blocked` |
| Resident strict Metal RPCL RGB8 512 batch 128 | `139.15 ms .. 140.17 ms` | `blocked` |

Resident RCA profile rows:

Raw `sync_wait_us` profile fields are converted to milliseconds in this table.

| Tile count | Code blocks | Coefficient prep observed | Command encode observed | Sync wait observed | RCA |
| ---: | ---: | ---: | ---: | ---: | --- |
| 16 | 3072 | `226 us .. 740 us`, median `327 us` | HT median `8 us`, packet prep `3 us`, packetization `2 us`, assembly `1 us` | `99.872 ms .. 103.931 ms`, median `100.316 ms` | sync/wait dominates |
| 64 | 12288 | `660 us .. 1.644 ms`, median `1.062 ms` | HT median `8 us`, packet prep `3 us`, packetization `2 us`, assembly `1 us` | `120.499 ms .. 136.677 ms`, median `121.361 ms` | sync/wait dominates |
| 128 | 24576 | `1.337 ms .. 2.636 ms`, median `2.145 ms` | HT median `8 us`, packet prep `4 us`, packetization `2 us`, assembly `1 us` | `134.258 ms .. 161.596 ms`, median `135.291 ms` | sync/wait dominates |

Decision:

- Keep Metal HTJ2K encode default routing `blocked` for RGB8/RGBA8 512/1024.
- Keep DWT and HT code-block Metal kernels as GPU-shaped `candidate` stages.
- Reclassify standalone RGB RCT as CPU-shaped for the measured 512/1024/2048
  rows until a fused path clears the stage gate.
- Keep resident strict Metal codestream assembly `blocked`; the measured command
  encode buckets are tiny compared with resident sync/wait time.

RCA:

- Root cause class: resident synchronization / route-composition overhead.
- Evidence: isolated DWT and HT code-block rows are faster on Metal, but strict
  resident end-to-end encode and the facade gate rows lose badly.
- The new resident profile rows narrow the loss: HT command encoding,
  packet-block prep, packetization, and codestream assembly command encoding are
  microsecond-scale, while `sync_wait_us` is roughly `100 ms .. 162 ms`.
- Next optimization target: reduce resident sync boundaries and codestream
  completion waits before reconsidering default Metal encode routing.

## 2026-06-01 CUDA J2K / HTJ2K Measured Runner Rerun

Evidence:

- Commit: `47b8869`
- Workflow:
  <https://github.com/frames-sg/signinum/actions/runs/26729235302>
- Result: success
- Supersedes:
  - `2026-05-31 CUDA HTJ2K Decode RGB/RGBA Rerun Status`
  - `2026-05-31 CUDA J2K / HTJ2K` for overlapping CUDA rows
- Runner: self-hosted `Cuda`, machine `PC`, Linux WSL2 x86_64,
  NVIDIA GeForce RTX 4070, 12282 MiB GPU memory. Host RAM was not
  reported by the workflow diagnostics.
- CUDA driver/toolkit: NVIDIA-SMI `595.71.05`, driver `596.49`,
  driver-supported CUDA compatibility `13.2`, `nvcc` release `13.2` /
  `V13.2.78`.
- Rust: `rustc 1.88.0 (6b00bc388 2025-06-23)`
- Commands:
  - `gh workflow run gpu-validation.yml --ref codex/cuda-quality-ht-rewrite -f run-timed-benchmarks=true -f run-linux-ci=false -f run-metal-validation=false -f run-nvidia-baseline=false`
  - `SIGNINUM_REQUIRE_CUDA_BENCH=1 cargo bench -p signinum-jpeg-cuda --bench device_decode --features cuda-runtime -- --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2`
  - `SIGNINUM_REQUIRE_CUDA_BENCH=1 cargo bench -p signinum-j2k-cuda --bench encode_stages --features cuda-runtime -- --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2`
  - `SIGNINUM_REQUIRE_CUDA_BENCH=1 cargo bench -p signinum-j2k-cuda --bench htj2k_decode --features cuda-runtime -- --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2`
  - `SIGNINUM_REQUIRE_CUDA_BENCH=1 cargo bench -p signinum-j2k-cuda --bench htj2k_encode --features cuda-runtime -- --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2`
  - `SIGNINUM_REQUIRE_CUDA_BENCH=1 cargo bench -p signinum --bench facade --features cuda-runtime -- --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2`
- Note: no `signinum_profile` rows were collected in this workflow because the
  profile environment variables were not enabled.

End-to-end facade gate:

| Shape | CPU-only | Adaptive | Strict CUDA | Decision |
| --- | ---: | ---: | ---: | --- |
| RGB8 512 HTJ2K encode | `17.128 ms .. 17.176 ms` | `17.135 ms .. 17.240 ms` | `20.627 ms .. 20.893 ms` | `blocked`: adaptive does not clear `10% + noise`; strict CUDA loses |
| RGB8 1024 HTJ2K encode | `81.531 ms .. 81.750 ms` | `81.736 ms .. 82.118 ms` | `44.913 ms .. 45.240 ms` | `blocked`: adaptive does not clear `10% + noise`; strict CUDA is capability proof only |
| RGBA8 512 HTJ2K encode | `22.346 ms .. 22.484 ms` | `22.486 ms .. 22.644 ms` | `23.516 ms .. 23.678 ms` | `blocked`: adaptive does not clear `10% + noise`; strict CUDA loses |
| RGBA8 1024 HTJ2K encode | `109.29 ms .. 110.51 ms` | `109.21 ms .. 110.13 ms` | `54.013 ms .. 55.068 ms` | `blocked`: adaptive does not clear `10% + noise`; strict CUDA is capability proof only |

CUDA stage evidence:

| Stage / Shape | CPU | CUDA | Gate |
| --- | ---: | ---: | --- |
| RCT 512 | `1.0781 ms .. 1.0969 ms` | `2.2946 ms .. 2.3064 ms` | `reclassified-cpu` |
| RCT 1024 | `6.6661 ms .. 6.7054 ms` | `7.4541 ms .. 7.5116 ms` | `reclassified-cpu` |
| RCT 2048 | `21.258 ms .. 21.408 ms` | `18.572 ms .. 18.945 ms` | `candidate` |
| DWT 5/3 512 | `3.3216 ms .. 3.3384 ms` | `1.1314 ms .. 1.1873 ms` | `candidate` |
| DWT 5/3 1024 | `19.464 ms .. 19.604 ms` | `2.6517 ms .. 2.6870 ms` | `candidate` |
| DWT 5/3 2048 | `83.616 ms .. 83.935 ms` | `12.798 ms .. 13.272 ms` | `candidate` |
| Quantize 512 | `607.72 us .. 623.63 us` | `1.0540 ms .. 1.0645 ms` | `reclassified-cpu` |
| Quantize 1024 | `2.4449 ms .. 2.4589 ms` | `2.4856 ms .. 2.5152 ms` | `reclassified-cpu` |
| Quantize 2048 | `9.8383 ms .. 10.086 ms` | `8.2107 ms .. 8.3656 ms` | `candidate` |

CUDA HTJ2K decode evidence:

| Decode Shape | CPU | CUDA | Gate |
| --- | ---: | ---: | --- |
| Full tile gray8 512 | `4.4753 ms .. 4.5281 ms` | `175.27 ms .. 179.17 ms` | `blocked` |
| Full tile RGB8 512 | `12.160 ms .. 12.500 ms` | `183.24 ms .. 186.91 ms` | `blocked` |
| Full tile RGBA8 512 | `12.343 ms .. 12.500 ms` | `184.80 ms .. 190.72 ms` | `blocked` |
| ROI gray8 256 | `3.2846 ms .. 3.3169 ms` | `179.73 ms .. 180.92 ms` | `blocked` |
| ROI RGB8 256 | `8.7233 ms .. 9.0811 ms` | `185.23 ms .. 189.43 ms` | `blocked` |
| ROI RGBA8 256 | `8.6874 ms .. 8.7639 ms` | `187.56 ms .. 192.22 ms` | `blocked` |
| Scaled gray8 256 | `975.59 us .. 1.0116 ms` | `175.62 ms .. 178.04 ms` | `blocked` |
| Scaled RGB8 256 | `2.9459 ms .. 2.9876 ms` | `175.83 ms .. 179.25 ms` | `blocked` |
| Scaled RGBA8 256 | `3.0093 ms .. 3.0730 ms` | `175.61 ms .. 179.20 ms` | `blocked` |
| ROI-scaled gray8 128 | `556.99 us .. 561.12 us` | `177.46 ms .. 180.86 ms` | `blocked` |
| ROI-scaled RGB8 128 | `1.5998 ms .. 1.6148 ms` | `179.07 ms .. 180.65 ms` | `blocked` |
| ROI-scaled RGBA8 128 | `1.6122 ms .. 1.6547 ms` | `175.52 ms .. 178.58 ms` | `blocked` |
| Tile batch gray8 batch 8 | `35.926 ms .. 36.346 ms` | `231.53 ms .. 236.35 ms` | `blocked` |
| Tile batch RGB8 batch 8 | `112.70 ms .. 113.53 ms` | `278.29 ms .. 286.08 ms` | `blocked` |
| Tile batch RGBA8 batch 8 | `119.66 ms .. 120.24 ms` | `277.32 ms .. 281.46 ms` | `blocked` |

CUDA HTJ2K encode micro evidence:

| Route / Shape | CPU | CUDA | Gate |
| --- | ---: | ---: | --- |
| Host-input gray8 512 | `5.6775 ms .. 5.7053 ms` | `10.516 ms .. 10.623 ms` | `blocked` |
| Cleanup blocks 64 host-staged | `4.5718 ms .. 4.6181 ms` | `4.2067 ms .. 4.2873 ms` | `blocked`: does not clear `10% + noise` |
| Cleanup blocks 64 resident | `4.5718 ms .. 4.6181 ms` | `1.9976 ms .. 2.0369 ms` | `candidate` |
| Strided cleanup blocks 64 resident | `4.5333 ms .. 4.5893 ms` | `2.0047 ms .. 2.0471 ms` | `candidate` |

Decision:

- Keep CUDA HTJ2K encode default routing `blocked` for RGB8/RGBA8 512/1024.
- Keep strict CUDA facade rows as capability proof only; they do not approve the
  default route while the adaptive rows fail the end-to-end gate.
- Keep DWT 5/3, RCT 2048, quantize 2048, resident cleanup, and resident
  strided cleanup as CUDA candidate stages.
- Reclassify standalone RCT 512/1024 and quantize 512/1024 as CPU-shaped until
  batching or fusion clears the stage gate.
- Keep every measured CUDA HTJ2K decode shape blocked, including RGB/RGBA full
  tile, ROI, scaled, ROI-scaled, and batch rows.

RCA:

- Root cause class: transfer/synchronization and route-composition overhead.
- Evidence: DWT and some large/resident encode stages clear the stage gate, but
  facade adaptive rows are essentially CPU-only and do not clear the default
  route gate.
- Strict CUDA 1024 encode rows are faster than CPU, but the current adaptive
  route does not compose those wins into an approved default path.
- Decode evidence shows a fixed CUDA route floor around `175 ms .. 192 ms` for
  single-tile decode shapes, and batch decode remains slower than CPU. That
  points at synchronization/session/launch or route-composition overhead rather
  than RGB/RGBA output format cost alone.
- Next optimization target: profile CUDA HTJ2K decode transfer, launch, block
  decode, inverse transform, and output-surface completion, then rerun facade
  gates after adaptive routing can use the strict CUDA encode wins.

## 2026-05-31 Metal RGB8 HTJ2K Encode

Evidence:

- Commit: `03072f3`
- Host: Apple M4 Pro, macOS 26.5, arm64, 48 GiB RAM
- Rust: `rustc 1.88.0 (6b00bc388 2025-06-23)`
- Commands:
  - `SIGNINUM_REQUIRE_METAL_BENCH=1 cargo bench -p signinum --bench facade --features metal -- facade_j2k_htj2k_encode_backend_speed_matrix --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2`
  - `SIGNINUM_REQUIRE_METAL_BENCH=1 cargo bench -p signinum-j2k-metal --bench encode_stages -- --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2`

End-to-end facade gate, RGB8 512 HTJ2K encode:

| Route | Criterion interval | Gate |
| --- | ---: | --- |
| CPU-only | `4.0650 ms .. 4.4699 ms` | Baseline |
| Adaptive | `4.0348 ms .. 4.3704 ms` | Not approved: does not clear `10% + noise` |
| Strict Metal | `33.291 ms .. 33.373 ms` | Blocked |

Stage evidence:

| Stage / Shape | CPU | Metal | Gate |
| --- | ---: | ---: | --- |
| RCT 512 | `121.25 us .. 142.88 us` | `232.44 us .. 254.48 us` | Blocked |
| RCT 1024 | `696.98 us .. 780.85 us` | `561.95 us .. 660.15 us` | Candidate |
| RCT 2048 | `3.0957 ms .. 3.5684 ms` | `2.3259 ms .. 3.0497 ms` | Candidate |
| DWT 512 | `1.6079 ms .. 2.1227 ms` | `253.24 us .. 271.22 us` | Candidate |
| DWT 1024 | `5.9109 ms .. 6.1584 ms` | `734.18 us .. 852.77 us` | Candidate |
| DWT 2048 | `33.533 ms .. 37.552 ms` | `3.1058 ms .. 3.4894 ms` | Candidate |
| HT code blocks, 192 | `7.4078 ms .. 7.8666 ms` | `2.9065 ms .. 2.9403 ms` | Candidate |
| HT code blocks, 768 | `33.143 ms .. 36.053 ms` | `5.4646 ms .. 7.0437 ms` | Candidate |

Encode-path evidence:

| Route / Shape | Criterion interval | Gate |
| --- | ---: | --- |
| CPU classic RGB8 512 | `24.641 ms .. 34.037 ms` | Baseline for classic only |
| CPU HTJ2K RGB8 512 | `6.0536 ms .. 6.7296 ms` | Baseline |
| Auto host Metal-buffer HTJ2K RGB8 512 | `4.4763 ms .. 5.4878 ms` | Candidate only; facade gate still required |
| Resident strict Metal RGB8 512 | `164.71 ms .. 167.04 ms` | Blocked |
| CPU HTJ2K RGB8 1024 | `25.373 ms .. 27.290 ms` | Baseline |
| Auto host Metal-buffer HTJ2K RGB8 1024 | `14.552 ms .. 16.585 ms` | Candidate only; facade gate still required |
| Resident strict Metal RGB8 1024 | `381.73 ms .. 389.97 ms` | Blocked |
| Resident strict Metal RPCL RGB8 512 batch 16 | `132.82 ms .. 146.38 ms` | Blocked |
| Resident strict Metal RPCL RGB8 512 batch 64 | `132.95 ms .. 134.99 ms` | Blocked |
| Resident strict Metal RPCL RGB8 512 batch 128 | `191.22 ms .. 425.55 ms` | Blocked |

Decision:

- Do not approve Metal as a default end-to-end RGB8 HTJ2K encode route.
- Do not approve strict/resident Metal codestream assembly as a production
  default route.
- Keep DWT and HT code-block Metal kernels as GPU-shaped candidates, because
  the isolated stage gates are strong.
- Keep 512-pixel RGB RCT CPU-shaped until a fused deinterleave+RCT path proves
  otherwise.

RCA:

- Root cause class: missing residency / route-composition overhead.
- Evidence: isolated Metal DWT and HT block coding are faster, but resident
  end-to-end encode is far slower than CPU. The loss is not explained by the
  math kernels themselves.
- Likely debt:
  - resident path still pays expensive host-visible codestream assembly or
    synchronization boundaries;
  - 512 RGB RCT is too small as a standalone Metal kernel and needs fusion with
    deinterleave or CPU routing;
  - batch resident encode is not scaling with tile count, which points to
    serialized packet/codestream assembly or a fixed synchronization bottleneck.

Next optimization target:

1. Profile resident Metal encode with per-stage timing around coefficient prep,
   HT block encode, packetization, codestream assembly, and host readback.
2. Keep DWT and HT block coding device-shaped; fix the route around them.
3. Route 512 RGB RCT to CPU unless deinterleave+RCT fusion clears the stage
   gate.
4. Do not promote any Metal encode path to adaptive default until the facade
   RGB8 512/1024 rows clear both stage and end-to-end gates.

## 2026-05-31 CUDA J2K / HTJ2K

Evidence:

- Commit: `03072f3`
- Workflow:
  <https://github.com/frames-sg/signinum/actions/runs/26724404569>
- Result: success
- Runner: self-hosted `Cuda`, Linux WSL2 x86_64, NVIDIA GeForce RTX 4070,
  CUDA driver `596.49`, CUDA toolkit `13.2`
- Rust: `rustc 1.88.0 (6b00bc388 2025-06-23)`
- Command family:
  `cargo bench ... --features cuda-runtime -- --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2`

End-to-end facade gate, RGB8 512 HTJ2K encode:

| Route | Criterion interval | Gate |
| --- | ---: | --- |
| CPU-only | `16.329 ms .. 16.502 ms` | Baseline |
| Adaptive | `16.384 ms .. 16.498 ms` | Not approved: does not clear `10% + noise` |
| Strict CUDA | `24.488 ms .. 25.455 ms` | Blocked |

Additional facade rows:

| Route / Shape | Criterion interval | Gate |
| --- | ---: | --- |
| CPU classic RGB8 512 | `26.214 ms .. 26.761 ms` | Baseline for classic only |
| Adaptive classic RGB8 512 | `23.755 ms .. 24.720 ms` | Not approved: no strict stage evidence tied to a full default route |
| CPU HTJ2K RGB8 512 | `16.763 ms .. 16.969 ms` | Baseline |
| Adaptive HTJ2K RGB8 512 | `16.321 ms .. 16.550 ms` | Not approved: does not clear `10% + noise` |

CUDA stage evidence:

| Stage / Shape | CPU | CUDA | Gate |
| --- | ---: | ---: | --- |
| Quantize 512 | `608.70 us .. 620.43 us` | `1.1167 ms .. 1.1276 ms` | Blocked |
| Quantize 1024 | `2.4445 ms .. 2.4663 ms` | `2.5569 ms .. 2.6097 ms` | Blocked |
| Quantize 2048 | `9.9479 ms .. 10.119 ms` | `8.5020 ms .. 8.5778 ms` | Candidate |
| HTJ2K full-tile decode gray8 512 | `4.5257 ms .. 4.5723 ms` | `186.14 ms .. 189.64 ms` | Blocked |
| HTJ2K ROI decode gray8 256 | `3.3849 ms .. 3.4354 ms` | `187.67 ms .. 190.30 ms` | Blocked |
| HTJ2K scaled decode gray8 256 | `981.06 us .. 996.69 us` | `178.88 ms .. 182.60 ms` | Blocked |
| HTJ2K ROI-scaled decode gray8 128 | `557.62 us .. 564.64 us` | `179.42 ms .. 182.76 ms` | Blocked |
| HTJ2K tile-batch decode gray8 batch 8 | `36.486 ms .. 37.378 ms` | `237.69 ms .. 243.50 ms` | Blocked |
| HTJ2K host-input encode gray8 512 | `5.7177 ms .. 5.7524 ms` | `12.400 ms .. 12.558 ms` | Blocked |
| HTJ2K cleanup host-staged blocks 64 | `4.5650 ms .. 4.6138 ms` | `4.4103 ms .. 4.5447 ms` | Not approved: does not clear `10% + noise` |
| HTJ2K cleanup resident blocks 64 | `4.5650 ms .. 4.6138 ms` | `2.3109 ms .. 2.3566 ms` | Candidate |
| HTJ2K strided resident cleanup blocks 64 | `4.5431 ms .. 4.5926 ms` | `2.2882 ms .. 2.3770 ms` | Candidate |

Decision:

- Do not approve CUDA as a default end-to-end RGB8 HTJ2K encode route.
- Keep strict CUDA rows as capability proof for now.
- Keep resident HTJ2K cleanup/code-block work as GPU-shaped candidate stages.
- Keep 2048 quantization as a CUDA candidate stage only; smaller quantization
  rows remain CPU-shaped until batching/fusion changes the gate result.
- Do not use the gray8 CUDA decode rows as WSI RGB routing evidence. They are
  useful RCA smoke rows, and every measured CUDA decode shape is currently
  blocked.

RCA:

- Root cause class: transfer/sync and route-composition overhead.
- Evidence: resident CUDA cleanup kernels beat CPU strongly, while host-staged
  cleanup and full host-input encode lose or fail the `10% + noise` gate. The
  device kernels are not enough; the default route needs resident inputs,
  fewer synchronization boundaries, and a full facade win.
- Decode evidence is worse: every measured CUDA HTJ2K decode route loses by a
  large margin, including ROI/scaled rows that should be attractive if the
  route were truly resident and batched.

Next optimization target:

1. Add per-stage timing around CUDA HTJ2K decode transfer, launch, block decode,
   inverse transform, and host output copy.
2. Keep resident cleanup/code-block encode on the GPU track, but block the
   host-input and strict end-to-end encode routes.
3. Build RGB/RGBA CUDA decode benches before approving any WSI-shaped default
   decode route.
4. Re-run the facade RGB8/RGBA8 512/1024 gates after residency and batching
   optimizations land.
