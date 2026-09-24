# P30–P31: Metal HT cleanup and 9/7 lifting decode

Measured on 2026-09-22 in place on `main` with a dirty working tree (unrelated
concurrent JPEG Metal, native SIMD, and HT-cleanup-only edits were preserved).
Apple M4 Pro, 16 GPU cores, 48 GiB RAM, Metal 4, Rust 1.96.0. No public API,
Auto routing policy, or allocation cap changed. The tables below are the
original Criterion estimates. JSON records from a fresh same-tree A/B on
2026-09-23, each change measured on its own against the current tree, are
`P30-metal-ht-cooperative-cleanup.json`, `P31-metal-idwt97-fused-lifting.json`
and, for the pool cap, `P35-metal-buffer-pool-cap.json`.

## Decisions

- **P30 promoted:** cleanup-only HT code blocks decode with two kernels instead
  of one thread per block.
- **P31 promoted:** the four horizontal 9/7 lifting steps, and the vertical scale
  plus four vertical lifting steps, each run as one fused dispatch instead of
  one full-plane pass per step.
- **Pool cap promoted (2026-09-23):** each J2K Metal buffer pool now retains an
  eighth of the device's `recommendedMaxWorkingSetSize`, clamped to
  [256 MiB, 1 GiB] and to `maxBufferLength` (`buffer_pool/state.rs`,
  `retained_bytes_for_device`). That is about 680 MB on an 8 GB Mac and 1 GiB
  from 16 GB up. A session owns a private and a shared pool, so it can retain up
  to twice the cap between decodes, until the session is dropped.

## P30: two-kernel HT cleanup decode

The previous kernel ran one thread per code block with 6 KiB of per-thread
scratch. A 512×512 RGB image has about 192 64×64 blocks, so a dispatch
occupied only six SIMD groups. Each lane serially decoded MEL, VLC, and all
4,096 MagSgn samples of its block. Stubbing that kernel removed 82% of GPU time
from the 16×512² HT batch.

`ht_cleanup_simd.metal` splits the work:

1. `j2k_decode_ht_cleanup_vlc_*` runs one thread per block and decodes the
   serial MEL/VLC streams. It writes `(u_q << 16) | inf` for quad (q, r) into
   the block's own output word at column 2q, row 2r, so no scratch allocation is
   needed. For blocks at most 64 samples wide, the previous row's significance
   context stays in registers. The reader advances once per quad pair: a pair
   consumes at most 30 bits (two 7-bit codewords plus a 16-bit UVLC prefix and
   suffix, computed from the tables), and every fetch leaves more than 32 valid
   bits. MEL runs are decoded on demand from a 64-bit buffer. The host packs
   1–32 blocks per SIMD group, aiming for about 96 SIMD groups: packing trades
   lane divergence against resident-group count. A 64×64 block's serial walk
   costs about 0.18 ms on one lane.
2. `j2k_decode_ht_cleanup_magsgn_*` runs one SIMD group per block. Each lane
   owns one quad, derives its MagSgn bit count from the stored VLC word and the
   previous row's exponents, and locates its bits with `simd_prefix_exclusive_sum`
   over a destuffed 8,192-bit threadgroup window that the group refills
   cooperatively.

Validation order, status codes, bit consumption, and every arithmetic
expression mirror `decode_ht_cleanup_common<true>`. SigProp/MagRef jobs and
devices whose SIMD width is not 32 keep the previous kernels.

## P31: fused 9/7 lifting

`dispatch_irreversible97_stages_after_horizontal_scale` previously issued nine
full-plane passes per level. It now issues two. In
`j2k_idwt_irreversible97_horizontal_lift_fused`, each threadgroup owns four
whole rows; in `j2k_idwt_irreversible97_vertical_fused`, each owns a 32-column
strip. Each walks its tiles in order with a 4-sample halo. The halo before a
tile comes from a threadgroup carry of the previous tile's pre-lift samples,
because that tile's device copy was already overwritten; the halo after it has
not been written yet. An earlier prototype with independent overlapping tiles
raced on those halos and failed only at batch 16. The lifting expressions and
edge mirroring are those of the original per-step kernels, which are kept as
the test oracle.

## Correctness

- `cooperative_cleanup_matches_cpu_and_legacy_kernel_bit_for_bit` covers eight
  fixtures: gradient and noise; 8- and 16-bit; 131×67 odd dimensions; 32×32,
  64×64, 128×32 and 256×16 blocks; lossless and 9/7. It compares the
  production dispatcher at 1, 12 and 96 replicas (1-, 8- and 32-lane packing)
  with the CPU oracle and the legacy kernel, bit for bit. Two injected faults
  (ignoring MagSgn unstuffing; dropping one exponent neighbour) each failed it.
- `irreversible97_fused_lifting_matches_full_grid_reference_bits` covers 15
  geometries, including multi-tile shapes with partial final tiles and odd
  origins. Each runs at batch 1, 3 and 16 with both high-pass constants,
  against the full-grid per-step reference. It passed six consecutive runs.
- Every `decode_stages` output hash is unchanged, and the geometry benchmark's
  per-byte comparison with native CPU decode passes at all four geometries.
- Update 2026-09-24: a later fix to 9/7 integer output rounding (see
  `docs/benchmark-evidence.md`, "Local Metal routing run with lossy inputs")
  changes Metal 9/7 output on samples that land exactly on a shifted tie, so
  9/7 output hashes recorded here may not reproduce on the current tree. P30
  and P31 remain output-neutral; the fix is in the final store and inverse
  colour transform, not in HT cleanup or lifting.

## Results

Same-tree A/B, `gpu-quick`. The baseline forced the legacy HT kernel and had
the per-step IDWT; the treatment has both P30 and P31. Criterion mean of 10
samples.

| Workload | Before | After | Change |
|---|---:|---:|---:|
| HT 5/3 RGB 512², batch 16 (broadcast), resident | 6.82 ms | 1.97 ms | −71% |
| HT 9/7 RGB 512², batch 16, resident | 6.62 ms | 1.67 ms | −75% |
| HT 9/7 distinct 128², batch 16, resident | 4.86 ms | 1.33 ms | −73% |
| HT 9/7 distinct 640×480, batch 16, resident | 11.35 ms | 6.08 ms | −46% |
| HT 9/7 distinct 1024², batch 16, resident | 31.74 ms | 28.16 ms | −11% |
| HT 9/7 512², batch 1, resident | 6.05 ms | 1.32 ms | −78% |
| Mixed distinct groups, prepared batch, resident | 3.03 ms | 0.83 ms | −73% |
| Classic 5/3 RGB 512², batch 16 (unchanged route) | 68.79 ms | 68.90 ms | noise |

For 640×480, P30 alone gave 7.28 ms and P31 took it to 6.08 ms. `release-bench`
absolute results after both changes (20 samples) were 1.958 ms, 1.665 ms,
1.313 ms, 6.077 ms, 27.86 ms, 1.302 ms, and 0.822 ms, in table order.

## Measured but not changed

- **Large-batch buffer reuse (now promoted, see Decisions).** At 16×1024² RGB
  the decode needs roughly 450 MB of scratch; with a 256 MiB pool and exact-size
  reuse, buffers were reallocated every decode, and commit-to-GPU-start delay
  was about 10 ms. Same-tree Criterion A/B, `gpu-quick`, 20 samples, 256 MiB vs
  the device-scaled cap (1 GiB on this 48 GB machine):

  | Workload (batch 16 unless noted) | 256 MiB | Scaled | Change |
  |---|---:|---:|---:|
  | HT 9/7 distinct 1024², resident | 28.48 ms | 19.98 ms | −29.8% |
  | HT 9/7 distinct 1024², readback | 31.73 ms | 23.06 ms | −27.3% |
  | HT 9/7 distinct 640×480, resident | 6.08 ms | 6.12 ms | +0.7% (fits either cap) |
  | HT 9/7 512², batch 1, resident | 1.306 ms | 1.307 ms | no change |

  Output hashes are unchanged (before the 2026-09-24 9/7 rounding fix noted
  under Correctness).
- **Unbounded batched IDWT.** Removing P20's 20 MiB bound with the fused kernels
  measured 28.2 → 26.6 ms at 1024² and no change at 640×480.
- **Classic Tier-1** is unchanged at 68.8 ms for 16×512² RGB (about 180 Mcoeff/s),
  one serial MQ decoder per block. Its context formation re-reads eight
  neighbour states per visit from per-thread memory. A flag-word rewrite is
  possible, but serial MQ decoding bounds the GPU at roughly multi-core CPU
  throughput, so the CPU Tier-1 hybrid route remains the stronger default.
- **JPEG one-shot calls.** A fresh `MetalSession` per call costs about 2.2 ms of
  driver-side submission time (16×16: 2.66 ms Metal versus 2.3 µs CPU). A
  retained session measured 0.30–0.74 ms per single decode.
- **JPEG 4:4:4 resident textures** take 2.50 ms for 16×256², versus 0.55 ms
  for 4:2:0. That is 4.5× slower for 2× the samples; 4:4:4 still uses the
  direct texture shader rather than component planes.

## Reproduction

```sh
J2K_REQUIRE_METAL_RUNTIME=1 cargo test --profile gpu-quick -p j2k-metal --lib -- cooperative irreversible97 --test-threads=1
J2K_REQUIRE_METAL_BENCH=1 cargo bench --profile release-bench -p j2k-metal --bench decode_stages -- --sample-size 20 --warm-up-time 2 --measurement-time 5
```
