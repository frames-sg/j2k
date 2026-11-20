# Metal JPEG decoder

`j2k-jpeg-metal` decodes every JPEG that the CPU decoder in `j2k-jpeg`
accepts, and returns the same bytes in a Metal-resident surface. It has two
routes:

- **Fast-packet kernels** decode baseline 4:2:0, 4:2:2, and 4:4:4 JPEGs
  entirely on the GPU, entropy decoding included.
- **Hybrid decode** runs the serial entropy stage on the CPU and everything
  after it on Metal: IDCT or lossless reconstruction, upsampling, color
  conversion, and packing. nvJPEG's hybrid backend splits the work the same
  way.

## Routing

The routing decision is in `src/routing.rs` and `src/compute/batch_plan.rs`.

An explicit `BackendRequest::Metal` request takes:

- the fast-packet kernels for baseline or extended 8-bit 4:2:0 or 4:2:2 more
  than 4 pixels wide, and 4:4:4, to `Gray8`, `Rgb8`, or `Rgba8`, when the
  `j2k-jpeg` fast-packet builders produce a packet;
- hybrid decode for every other request CPU decode accepts, including 16-bit
  output for 12-bit and lossless JPEG;
- an error, before any kernel launches, for requests CPU decode rejects,
  unsupported output formats or backends, and an unavailable Metal runtime.

`BackendRequest::Auto` is narrower, and stays on the CPU unless a benchmark
showed Metal winning:

- Single-image full, region, scaled, and region-scaled requests use the CPU,
  even when a fast packet exists.
- Full `Rgb8` batches use Metal when there are at least 16 compatible 4:2:0 or
  4:2:2 tiles of at least 256x256 without restart markers, distinct inputs
  included. Dimensions, tables, and checkpoint counts must match across the
  batch.
- Smaller, mixed-table, restart-coded, and scaled tile batches use the CPU.
- JPEGs that need hybrid decode use the CPU, because hybrid decode runs at
  about CPU speed (below).
- Sparse viewports use the CPU for scheduled surface output. Contiguous
  restart-coded fast-packet viewports may use Metal. Reusable resident
  viewport outputs may use direct contiguous decode or resident composition.

With an explicit Metal request, a viewport of JPEGs that need hybrid decode
entropy-decodes each tile once and composes them all through hybrid decode.

Don't widen `Auto` for a workload class unless the matching
[benchmark group](#benchmarks) shows a repeatable win for it.

## Fast-packet batches

- **4:2:0 and 4:2:2 texture batches** run entropy decode and IDCT in one grid
  across the compatible tiles, writing three reusable private component
  planes. A second pass reconstructs chroma, converts color, and writes the
  caller's private RGBA textures. Coefficients are never held for the whole
  image. Sixteen 256x256 4:2:0 tiles need 1.5 MiB of plane storage, which
  later calls reuse.
- **4:4:4** has no subsampled chroma to reconstruct, so it uses a direct
  texture path with its own plane decoder.
- **Packet entropy** is copied straight into reusable shared Metal storage;
  only offsets, lengths, and checkpoints live in host vectors. Data the CPU
  writes is in shared storage, and GPU-only planes and textures are private,
  following Apple's [storage-mode guidance][storage-modes].
- **Scratch.** Each runtime has two scratch slots, each with its own lock, so
  two callers can stage batches at once and any more wait. A lease lasts
  until the command buffer completes and the CPU has read the status. Waiting
  callers wake when a slot comes back, including when its owner panics, and a
  poisoned slot is returned as an error.
- **Pipelines** are built once per Metal device (by registry ID) and kept for
  the life of the process. Sessions share them but own their queues and
  mutable state. A failed initialization isn't cached.
- **Validation happens up front.** Building a packet validates the entropy
  stream, so a malformed Huffman stream fails in the constructor. Checkpoint
  traversal uses the entropy-only `skip_block` and builds no coefficient
  blocks.

[storage-modes]: https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus

### Why component planes

An earlier design decoded straight into subsampled textures, with one encoder
per tile and extra horizontal, vertical, and corner repair passes. The ignored
`texture_scheduling_experiment` test (in `src/compute/texture_tuning.rs`)
runs both designs in one optimized binary on the same distinct tiles and
checks every tile against the CPU. Median batch time for 16 distinct 4:2:0
tiles, Apple M4 Pro, `release-bench` profile, five warm-up and fifteen
measured iterations, September 2026:

| Workload | Direct textures | Component planes |
|---|---:|---:|
| 256x256, no restart | 6.7 ms | 0.52 ms |
| 256x256, restart interval 2 | 12.1 ms | 0.76 ms |
| 512x512, no restart | 7.0 ms | 1.20 ms |
| 512x512, restart interval 4 | 26.7 ms | 1.64 ms |

The public-API Criterion bench (warm 256x256 RGBA texture batch of 16) went
from about 6.6 ms to 0.7 ms without restart markers and from 12 ms to 0.9 ms
with restart interval 2.

- Component planes also won at batch sizes 1, 2, and 4, at 16x16 and 64x64,
  and for 4:2:2 at 256x256 and 512x512, with noisier results on small work.
- Threadgroup widths of 32, 64, 128, and 256 made no consistent difference,
  so the width was left alone.
- Entropy decoding on the GPU into whole-image coefficient scratch followed
  by a separate IDCT pass was slower, and wasn't adopted.
- These are warm-batch throughput numbers from one machine with a normal
  desktop running. They say nothing about the latency of a single JPEG, and
  we don't claim a speedup for fast-packet planning, shared staging, or the
  two-slot pool on their own.

## Hybrid decode

Hybrid decode covers progressive, grayscale, 4:4:0 and other sampling, CMYK
and YCCK, 12-bit, lossless, and narrow JPEGs, and restart-coded tiles the
fast batch kernels decline.

- **DCT JPEG.** The CPU entropy decoder writes dequantized, natural-order
  `i16` blocks for every component straight into a leased shared buffer.
  Progressive scans, and sequential frames whose components are split across
  scans, are accumulated first and dequantized once. Per component,
  `jpeg_hybrid_idct8` (the ISLOW kernels, with reduced 4x4, 2x2, and 1x1
  versions for scaled decode) or `jpeg_hybrid_idct12` (12-bit ISLOW in 64-bit
  integers) writes a private plane. `jpeg_hybrid_pack8` or
  `jpeg_hybrid_pack12` then computes each output pixel: libjpeg-turbo's fancy
  or replicate upsampling at the component's scaled size, color conversion
  (YCbCr, RGB, Adobe-inverted CMYK and YCCK, grayscale luma), and Gray, RGB,
  or RGBA packing.
- **Regions.** For 8-bit sequential scans the CPU decodes only the MCU rows
  and columns the region needs, plus one MCU row of filter margin. It seeks
  with restart markers when there are any, skips entropy outside the window,
  and stops after the last row it needs. Progressive, split-scan, and 12-bit
  regions decode the whole coefficient set.
- **Lossless JPEG.** The CPU writes predictor differences into shared
  storage, scan by scan for frames split across scans. Predictor, seed,
  sample limit, and point transform are per component, since each scan has
  its own. `jpeg_lossless_reconstruct_rows` handles predictor-1 components
  (the DICOM default) with a row-parallel prefix sum seeded from first-column
  samples the CPU computes. Other predictors use `jpeg_lossless_reconstruct`,
  an anti-diagonal wavefront per component. Reconstruction wraps modulo 2^16
  and reseeds at each restart interval, as CPU decode does, and a shared
  status word reports out-of-range samples. `jpeg_lossless_pack8` (up to 8-bit
  precision) and `jpeg_lossless_pack16` point-sample for scaled output,
  replicate subsampled components by their sampling ratios as libjpeg-turbo
  does, and convert YCbCr at the frame precision.
- **Errors.** Malformed streams fail in the same CPU entropy code that CPU
  decode uses. When lossless reconstruction reports an error or an
  out-of-range sample, the CPU decoder's own error is returned. So explicit
  Metal never succeeds where CPU decode fails, and never fails with a
  different error.
- **Composition.** Viewports and mixed RGB8 tile batches without fast packets
  pack several source rectangles into one zero-filled pooled output. Reusable
  batch buffers and RGBA textures decode each tile through hybrid decode, then
  use the GPU copy passes.
- **Memory.** Coefficient, difference, and plane storage are leased from the
  runtime's two scratch slots and kept for the next call; output surfaces come
  from the surface pool. Every allocation stays within the device's
  `maxBufferLength` and J2K's allocation cap, and a larger request returns an
  error. A whole-image progressive or 12-bit coefficient set can exceed the
  cap on very large slides where CPU decode succeeds, which is one more reason
  `Auto` keeps these on the CPU.

`tests/hybrid.rs` compares every route byte for byte with CPU decode. Its
inputs cover the libjpeg-turbo scaled matrix (formats, scales, and regions,
including multi-scan sequential and CMYK/YCCK), the conformance fixtures, the
81-case lossless matrix (predictors 1-7, precisions 2-16, point transforms,
restart intervals, whole-ratio sampling at odd sizes, split scans), the DCMTK
DICOM matrix (also compared with DCMTK's own decode), malformed lossless
streams, region edge sweeps, and windowed regions of 1024x768 images with and
without restart markers.

### Hybrid speed

Hybrid decode is about as fast as CPU decode, because the serial entropy stage
dominates. It exists for coverage and to keep output on the GPU, not for
single-image speed. Median wall time per full decode (parse, entropy, GPU work
and wait), Apple M4 Pro, `release-bench` profile, 20 samples, October 2026:

| Input | CPU | Metal hybrid |
|---|---:|---:|
| Progressive 4:2:0, 2048x1536 | 14.3 ms | 13.5 ms |
| Progressive 4:4:4, 1024x1024 | 8.6 ms | 7.9 ms |
| Baseline grayscale, 2048x2048 | 8.3 ms | 8.8 ms |
| Progressive grayscale, 1024x1024 | 2.8 ms | 3.0 ms |
| Baseline 4:4:0, 1600x1200 | 8.1 ms | 8.1 ms |
| Baseline CMYK, 1024x1024 | 5.7 ms | 5.6 ms |
| 12-bit extended grayscale, 512x512 | 0.85 ms | 0.96 ms |
| 16-bit lossless predictor 1, 512x512 | 0.98 ms | 1.32 ms |
| 16-bit lossless predictor 1, 2048x2500 | 22.3 ms | 25.4 ms |
| 16-bit lossless predictor 7, 512x512 | 0.99 ms | 2.15 ms |
| 8-bit lossless RGB predictor 1, 1024x1024 | 6.2 ms | 6.6 ms |

The bench generates the first six inputs with `jpeg-encoder`. The rest are
synthetic CT- and CR-like phantoms encoded with libjpeg-turbo's `cjpeg`
(`-precision 16 -lossless 1|7`, `-precision 12 -quality 95`, and
`-lossless 1 -rgb`) and passed in through `J2K_BENCH_INPUTS`. Beating the CPU
on these classes needs GPU entropy decoding, or CPU entropy decoding
overlapped with GPU work across tiles. Neither exists yet.

## Benchmarks

`benches/compare.rs` holds the routing benchmarks. It accepts extra JPEG
inputs through `J2K_BENCH_INPUTS`, and groups inputs by dimensions,
restart and checkpoint shape, and fast-packet family so batch results show
how well work coalesces. The groups include likely wins and likely losses on
purpose.

| Group | What it answers |
| --- | --- |
| `decode_rgb` | Cold single full-frame CPU vs explicit Metal: is CPU-first single decode right? |
| `wsi_tile_batch_rgb`, `wsi_tile_batch_scaled_rgb_q4` | Repeated tile batches through CPU, explicit Metal, and `Auto`. |
| `jpeg_distinct_batch_routing` | Distinct 4:2:0 and 4:2:2 inputs at 64x64 and 256x256, plan construction included. |
| `jpeg_metal_mixed_tables` | The cost of submitting several table groups to reusable buffers and textures. |
| `wsi_tile_batch_region_scaled_coalesced_rgb_q4` | Region+scaled batches where Metal can amortize setup. |
| `wsi_tile_batch_region_scaled_distinct_rgb_q4` | The low-coalescing control. Route region+scaled to Metal only if this wins. |
| `wsi_tile_batch_rgba_textures` | Resident texture batches with no host download. |
| `viewer_region_scaled_composite_rgb`, `..._device`, `..._warm`, `..._device_warm` | CPU and hybrid viewports with host and device output, cold and warm. |
| `viewer_contiguous_region_scaled_rgb`, `..._device` | Contiguous viewport controls. |
| `viewer_best_region_scaled_rgb_device`, `viewer_best_region_scaled_composite_rgb_device` | The best resident viewport and composite paths. |
| `jpeg_metal_fast_packet_planning` | Route-discovery cost for accepted and rejected fast-packet families. |

`benches/hybrid.rs` has `jpeg_metal_hybrid_decode`: CPU against explicit
hybrid decode for progressive, grayscale, 4:4:0, CMYK, and any
`J2K_BENCH_INPUTS` file. Move a hybrid class to `Auto` only if it wins here.

## Commands

All of these need an Apple Silicon Mac. The `J2K_REQUIRE_METAL_*` variables
make a missing device fail instead of skip.

```bash
# Hybrid parity suite, and every runtime test.
J2K_REQUIRE_METAL_RUNTIME=1 cargo test --locked -p j2k-jpeg-metal --test hybrid -- --test-threads=1
J2K_REQUIRE_METAL_RUNTIME=1 cargo test --locked -p j2k-jpeg-metal -- --test-threads=1

# Hybrid vs CPU (also checks parity). J2K_BENCH_INPUTS is optional.
J2K_REQUIRE_METAL_BENCH=1 J2K_BENCH_INPUTS=/path/to/jpegs \
  cargo bench --locked -p j2k-jpeg-metal --bench hybrid --profile release-bench -- \
  --sample-size 20 --warm-up-time 2 --measurement-time 4 --noplot

# Component planes vs direct textures, and threadgroup widths (checks parity).
# Add J2K_JPEG_TEXTURE_SMALL_BATCHES=1 for small batches and 4:2:2 controls.
J2K_REQUIRE_METAL_RUNTIME=1 cargo test --locked -p j2k-jpeg-metal --lib --profile release-bench \
  texture_scheduling_experiment -- --ignored --test-threads=1 --nocapture

# Public API: fresh CPU/Metal RGB and warm resident batch-16 textures.
J2K_REQUIRE_METAL_BENCH=1 cargo bench --locked -p j2k-jpeg-metal --bench compare --profile release-bench -- \
  '^(decode_rgb/generated/fast420(_restart2)?_256x256/(cpu|metal)|wsi_tile_batch_rgba_textures/batch16/warm_session_reused_textures/generated/fast420(_restart2)?_256x256)$' \
  --sample-size 10 --warm-up-time 2 --measurement-time 5 --noplot
```

To compare kernel changes against a saved copy of the shaders in one process,
set `J2K_JPEG_HARNESS_BASELINE_SHADERS` and run the ignored
`decode_kernel_stage_profile` test (see
[Environment variables](environment-variables.md#benchmark-harnesses)).
