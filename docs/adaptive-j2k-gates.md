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
