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

## 2026-05-31 CUDA HTJ2K Decode RGB/RGBA Rerun Status

Evidence:

- Commit: `05ecdec`
- Host: same Apple Metal host as the Metal rerun; this is not a CUDA runner.
- CUDA runtime: `nvidia-smi` unavailable on this host.
- Rust: `rustc 1.88.0 (6b00bc388 2025-06-23)`
- Commands:
  - `command -v nvidia-smi && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader`
  - `cargo check -p signinum-j2k-cuda --features cuda-runtime --benches --tests`
  - `cargo clippy -p signinum-j2k-cuda --features cuda-runtime --all-targets -- -D warnings`
  - `cargo bench -p signinum-j2k-cuda --features cuda-runtime --bench htj2k_decode --no-run`
  - `cargo bench -p signinum --bench facade --features cuda-runtime --no-run`

Decision:

- No CUDA RGB/RGBA decode gate decision is recorded from this host.
- The CUDA code compiles and lints with `cuda-runtime`, and both relevant bench
  binaries build, but measured CUDA Criterion rows still require a self-hosted
  NVIDIA CUDA runner.
- Do not copy the earlier CUDA gray8 numbers below into the new RGB/RGBA gate.

RCA:

- Not measured in this rerun. Run the planned `SIGNINUM_REQUIRE_CUDA_BENCH=1`
  commands on a CUDA runner before classifying the RGB/RGBA decode rows.

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
