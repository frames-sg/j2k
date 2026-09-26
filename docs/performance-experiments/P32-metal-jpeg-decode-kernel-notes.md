# P32: Metal JPEG decode kernel (Huffman and IDCT)

Measured on 2026-09-23 in place on `main` (`c1cc3ff1`) with a dirty working
tree; the unrelated in-flight JPEG Metal, CPU JPEG, and J2K Metal edits were
preserved. Apple M4 Pro, 16 GPU cores, macOS 27.0, Rust 1.96.0. No public API,
Auto routing policy, or allocation cap changed. The tables below are harness
medians and Criterion estimates. The JSON record, from a fresh same-tree A/B
with P33 and P34 in both arms, is `P32-metal-jpeg-decode-kernel.json`.

## Decisions

- **Promoted: fused Huffman lookahead tables.** `PreparedHuffman` replaces the
  separate `fast_symbol`/`fast_len` byte tables with one `fast` entry,
  `(code length << 8) | symbol`, and adds `fast_ac`,
  `(value << 8) | (run << 4) | total length`, for AC codes whose code and extra
  bits fit the 9-bit lookahead. A hit decodes run, length, and the extended
  value with one constant-memory load and no `receive_extend`. The canonical
  search for longer codes starts at length 10, because every shorter code has
  a lookahead entry.
- **Promoted: register-resident IDCT.** The column pass loads coefficients as
  `short4` rows and transforms four columns at once as `int4`; the row pass
  works from registers. `decode_idct_deposit_block` stores each row straight to
  the plane, removing the `work[64]` and `pixels[64]` round trips. Every
  `idct_islow` caller (texture, region, scaled, split) uses the same passes.

## Why the results are exact

- The fused AC path runs after `ensure_bits_padded(9)`, as the symbol path
  did. The original would consume the same bits: `receive_extend` counts padded
  bits as buffered, so it never refills or fails when `total <= 9` bits are
  buffered. Error codes are unchanged; `status.position` is unchanged because
  no refill order changed.
- The column pass has no per-column DC shortcut. With all AC terms zero, the
  full expression is `(p0·2¹³ + 2¹⁰) >> 11 = p0 << 2` exactly, and i16 inputs
  cannot overflow it. The row pass keeps its AC-zero shortcut: pass-1 DC values
  can reach about 2²⁰, where the full expression's `p0 << 13` could overflow and
  diverge from the CPU.

## Harness

`compute/tests/decode_kernel_harness.rs`:

- Correctness, run with the normal suite: batch buffers (full, half/quarter/
  eighth scale, region, region-scaled), resident textures (full, half scale),
  and one-shot decodes must match the CPU decoder byte for byte. Fixtures cover
  4:2:0, 4:2:2 and 4:4:4; bench, textured and noise content; quality 50 to 100;
  odd sizes (133×77, 131×67, 130×66); restart intervals; and batch 1 and 16 of
  distinct images.
- Stage profile (`#[ignore]`): probe builds replace `decode_idct_deposit_block`
  with bodies that stop after Huffman decoding, coefficient materialization,
  or the IDCT, and stub the pack kernel. GPU time comes from command-buffer
  timestamps, sampled round-robin across variants after a 0.5 s warm-up so clock
  ramping affects every variant alike. `J2K_JPEG_HARNESS_BASELINE_SHADERS`
  compiles a second shader copy for a same-process A/B.
- Fault injection: consuming one bit too few for 12-bit codes, and a rounding
  change in one IDCT output column, each failed the correctness tests.

## Results

GPU ms, median of 21, batch 16 of distinct images. The baseline is the original
algorithm on the new table layout, so it already includes merging the two
lookahead loads into one; the Criterion A/B below is against HEAD.

| Buffer batch (fused decode + pack) | Baseline | After | Change |
|---|---:|---:|---:|
| 4:2:0 bench q90 512² | 0.811 | 0.692 | −15% |
| 4:2:0 textured q90 512² | 0.688 | 0.561 | −18% |
| 4:2:0 textured q75 512² | 0.520 | 0.409 | −21% |
| 4:2:0 textured q90 256² | 0.257 | 0.220 | −14% |
| 4:2:0 textured q90 1024² | 3.165 | 2.543 | −20% |
| 4:2:2 textured q90 512² | 0.900 | 0.681 | −24% |
| 4:2:0 noise q95 512² | 0.896 | 0.822 | −8% |

| Resident texture batch | Baseline | After | Change |
|---|---:|---:|---:|
| 4:2:0 256² | 0.303 | 0.257 | −15% |
| 4:2:0 512² | 0.740 | 0.612 | −17% |
| 4:2:2 512² | 0.941 | 0.717 | −24% |
| 4:4:4 256² | 1.894 | 1.677 | −11% |
| 4:4:4 512² | 3.856 | 3.219 | −17% |

Stage split for 4:2:0 textured q90 512², before → after (ms): Huffman
0.338 → 0.271, coefficients 0.052 → 0.067, IDCT and store 0.171 → 0.094,
pack 0.120 → 0.118. Huffman is now 61% of the decode kernel and 48% of the
full GPU time.

Criterion, `compare` bench, `gpu-quick`, 20 samples; baseline is HEAD's
`abi.rs` and shaders swapped into the same tree:

| Benchmark | HEAD | After | Change (95% CI) |
|---|---:|---:|---:|
| `wsi_tile_batch_rgb` fast420 256², batch 64 | 6.21 ms | 4.55 ms | −26% [−33, −20] |
| `wsi_tile_batch_rgb` fast420 restart2 256² | 5.33 ms | 4.47 ms | −22% [−29, −14] |
| `wsi_tile_batch_rgb` fast422 256² | 6.50 ms | 5.09 ms | −15% [−22, −8] |
| `wsi_tile_batch_rgb` fast444 256² | 6.67 ms | 6.38 ms | −4% [−7, −1] |
| resident textures batch 16, fast420 256² | 547 µs | 502 µs | −8% [−9, −7] |
| resident textures batch 16, fast420 restart2 | 840 µs | 789 µs | −6% [−7, −5] |
| resident textures batch 16, fast422 256² | 464 µs | 435 µs | −7% [−7, −6] |
| resident textures batch 16, fast444 256² | 2.51 ms | 2.34 ms | −7% [−7, −6] |

## Measured and rejected

Each was bit-exact and A/B'd against the promoted state; none helped, so none
was kept. Every change that added live state to the entropy loop was slower,
which points at register pressure rather than ALU or memory as its limit.

| Candidate | Result |
|---|---|
| `simd_all`-uniform IDCT shortcuts (no divergence) | +1% to +4% |
| DC fast path through the fused table | ±1% |
| 32-bit `hi:lo` bit reader (no 64-bit shifts) | +2% to +5% |
| Huffman tables in `device` instead of `constant` space | +2% to +7% |
| Block-sparse IDCT (skip zero right half, prune zero bottom rows) | +1% to +3% |
| End-of-block and ZRL folded into the fused table | ±1% |
| Entropy word prefetched one refill ahead | +10% to +15% |

### Follow-up: four output pixels per pack thread (rejected)

The 4:2:0 and 4:2:2 RGB batch pack kernels were temporarily changed from two
to four horizontal output pixels per thread, with the host dispatch width
adjusted accordingly. The correctness harness remained byte-exact. Same-tree
Criterion A/B (`gpu-quick`, 20 samples, 64 × 256² tiles, retained session)
measured 4:2:0 at 1.552 ms before and 1.549 ms after (change interval −1.05%
to −0.07%, within Criterion's noise threshold). 4:2:2 measured 1.872 ms before
and 1.862 ms after (change interval −0.72% to +1.77%, no detected change).
The extra loop and reduced thread count did not produce a material end-to-end
gain, so the original two-pixel kernels and dispatch width were restored.
Vector output stores were not part of this experiment.

A 10- or 11-bit lookahead was not built: measured fused-hit rates rise only
from 84–89% at 9 bits to 89–94% at 10 bits and under 1 point more at 11, and
folding EOB (up to 8% of symbols) into the table already showed that fewer
slow-path symbols did not move the time. Tables above 9 bits also exceed the
4 KB `setBytes` limit.

## Defects found (fixed 2026-09-23, after this experiment)

- **CPU 4:2:0 bottom row.** `component_row_triplet`
  (`crates/j2k-jpeg/src/entropy/sequential/emit/upsample.rs`) took the lower
  chroma neighbour of an odd final output row from MCU padding instead of
  replicating the last real chroma row. `djpeg` (libjpeg-turbo) on a 130×66
  4:2:0 image matched Metal byte for byte; the CPU differed in the last row
  only. It affected whole-image outputs whose height is even but not a multiple
  of the scaled MCU height, for example 1080-row images, and ROIs touching the
  bottom edge.
  **Fix:** the triplet takes the stripe's real chroma rows,
  `ceil(stripe_rows / v_ratio)`, and replicates the last one, as libjpeg-turbo's
  `set_bottom_pointers` does. The 12-bit writers had the same clamp to the
  padded plane height and now clamp to `ceil(height / 2)`. The harness no
  longer excludes any row and gained ROIs that touch the bottom edge;
  `ybr420_bottom_rows_match_turbo_when_height_is_not_mcu_aligned` compares
  heights 34, 64, 65, 66, 68, 72, 76 (restart none/1/7, full, RGBA and bottom
  region) against libjpeg-turbo, with contrasting content in the MCU padding.
- **Resident texture batches of about 30 or more decoders failed.** Every
  `Decoder` reported 17,926,448 retained bytes regardless of image size, because
  `retained_allocation_bytes_excluding_cpu_checkpoint_cache` added the fixed
  `MAX_DECODER_CONTEXT_ALLOCATION_BYTES`. 64 decoders requested 1.07 GiB against
  the 512 MiB cap (`AllocationTooLarge`), so the batch-64 and batch-256
  `wsi_tile_batch_rgba_textures` benchmarks failed.
  **Fix:** retained bytes now count only what the decoder owns (100,656 bytes
  for a baseline 4:2:0 tile, mostly prepared Huffman tables). Decoders share one `DecoderContext` (the
  thread-local default or a caller's), so its reserve is charged once per
  budget: decode workspace planning (unchanged), CPU checkpoint growth, and
  device-plan construction. Both benchmarks now complete; with 64 × 256² tiles,
  4:2:0 takes 1.22 ms, 4:2:2 1.34 ms, and 4:4:4 8.77 ms.

## Reproduction

```sh
J2K_REQUIRE_METAL_RUNTIME=1 cargo test --profile gpu-quick -p j2k-jpeg-metal --lib -- decode_kernel_harness
J2K_REQUIRE_METAL_RUNTIME=1 J2K_JPEG_HARNESS_BASELINE_SHADERS=/path/to/baseline-shaders \
  cargo test --profile gpu-quick -p j2k-jpeg-metal --lib -- decode_kernel_stage_profile \
  --include-ignored --nocapture --test-threads=1
```
