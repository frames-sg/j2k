# JPEG Metal decode architecture and measurements

How `j2k-jpeg-metal` decodes on the GPU, and the measurements behind the
design. Routing rules are in [routing-benchmarks.md](routing-benchmarks.md).

## Batch decode with the fast-packet kernels

- **4:2:0 and 4:2:2 texture batches** run entropy decode and IDCT in one grid
  spanning the compatible tiles, writing three reusable private component
  planes. A second GPU pass reconstructs chroma, converts color, and writes
  the caller's private RGBA textures. Coefficients are never held in a
  full-image intermediate. Sixteen 256x256 4:2:0 tiles need 1.5 MiB of plane
  storage, which later calls reuse.
- **4:4:4** uses a direct texture path: it has no subsampled chroma to repair,
  and its plane decoder has a different ABI.
- **Packet entropy** is copied straight into reusable shared Metal storage;
  only offsets, lengths, and checkpoint metadata live in host vectors.
  CPU-populated data uses shared storage and GPU-only planes and textures stay
  private, following
  [Apple's storage-mode guidance](https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus).
- **Scratch.** Each runtime has two independently locked scratch slots, so two
  callers can stage batches at once; further callers block. A lease lasts
  through command-buffer completion and the CPU reading the status. Callers
  are woken when a slot is returned, including when its owner unwinds, and a
  poisoned slot is a typed error.
- **Pipelines** are built once per Metal device (by registry ID) and kept for
  the life of the process. Sessions share them but own their queues and
  mutable state. A failed initialization is not cached.
- **Validation is eager.** Building a packet validates the entropy stream, so
  a malformed Huffman stream fails in the constructor. Checkpoint traversal
  uses the entropy-only `skip_block` and builds no coefficient blocks.

### Why component planes

An earlier design decoded straight into subsampled textures, with one encoder
per tile and extra horizontal, vertical, and corner repair passes. The ignored
`texture_scheduling_experiment` test compares the two in one optimized binary
on identical distinct tiles, checking every tile against the CPU. Median batch
time for 16 distinct 4:2:0 tiles on an Apple M4 Pro (macOS 26.5.2, Rust
1.96.0, `release-bench`, September 2026; five warm and fifteen measured
iterations):

| Workload | Direct textures | Component planes |
|---|---:|---:|
| 256x256, no restart | 6.723 ms | 0.517 ms |
| 256x256, restart interval 2 | 12.127 ms | 0.755 ms |
| 512x512, no restart | 6.993 ms | 1.200 ms |
| 512x512, restart interval 4 | 26.683 ms | 1.644 ms |

The public-API Criterion bench (warm 256x256 RGBA texture batch of 16, ten
samples, two-second warm-up, five-second measurement) measured 0.677 ms
without restart markers and 0.914 ms with restart interval 2, against 6.635 ms
and 12.029 ms before.

- Component planes also won at batch sizes 1, 2, and 4, at 16x16 and 64x64,
  and for 4:2:2 at 256x256 and 512x512, with noisier small-workload results.
- Threadgroup widths of 32, 64, 128, and 256 made no consistent difference
  with component planes, so the width was left alone.
- GPU entropy decode into full-image coefficient scratch followed by a
  separate IDCT pass was slower and was not adopted.
- These are warm-batch throughput figures from one machine with the desktop
  not isolated. They are not a latency claim for every JPEG request, and no
  speedup is claimed for fast-packet planning, shared staging, or the two-slot
  pool on their own.

## Hybrid decode

Explicit `BackendRequest::Metal` decode accepts every JPEG and output format CPU
decode accepts, and returns the same bytes in a Metal-resident surface. JPEGs
without a fast packet (progressive, grayscale, 4:4:0 and other sampling, CMYK
and YCCK, 12-bit, lossless, narrow images, restart-coded tiles the fast batch
kernels decline) take the hybrid route: the CPU runs the serial entropy stage and
Metal does the rest. This is the split nvJPEG uses for its hybrid backend.

- **DCT JPEG.** The CPU entropy decoder writes dequantized, natural-order `i16`
  blocks for every component straight into a leased shared buffer; progressive
  scans, and sequential frames whose components are split across scans, are
  accumulated first and dequantized once. Per component,
  `jpeg_hybrid_idct8` (the existing ISLOW kernels, reduced 4×4/2×2/1×1 for
  scaled decode) or `jpeg_hybrid_idct12` (12-bit ISLOW in 64-bit integers) writes
  a private plane. `jpeg_hybrid_pack8`/`jpeg_hybrid_pack12` then compute each
  output pixel with libjpeg-turbo's fancy or replicate upsampling at the
  component's scaled size, color conversion (YCbCr, RGB, Adobe-inverted CMYK and
  YCCK, grayscale luma), and Gray/RGB/RGBA packing.
- **Regions.** For 8-bit sequential scans the CPU decodes only the MCU rows and
  columns the region needs, plus one MCU row of filter margin. It seeks via restart
  markers when present, skips entropy outside the window, and stops after the last
  needed row. Progressive, split-scan, and 12-bit regions decode the whole
  coefficient set.
- **Lossless JPEG.** The CPU writes predictor differences into shared storage,
  scan after scan for frames split across scans. Predictor, seed, sample
  limit, and point transform are per component, since each scan has its own.
  `jpeg_lossless_reconstruct_rows` handles predictor-1 components (the DICOM
  default): it runs a row-parallel prefix sum seeded from CPU-computed
  first-column samples. Other components use `jpeg_lossless_reconstruct`, an
  anti-diagonal wavefront per component. Reconstruction wraps modulo 2^16 and
  reseeds at each restart interval as CPU decode does. A shared status word
  reports out-of-range samples. `jpeg_lossless_pack8` (up to 8-bit precision)
  and `jpeg_lossless_pack16` apply point sampling for scaled output, replicate
  subsampled components by their sampling ratios as libjpeg-turbo does, and
  convert YCbCr at the frame precision.
- **Errors.** Malformed streams fail in the same CPU entropy code CPU decode uses.
  When lossless decode reports an error or an out-of-range sample, the CPU
  decoder's own error is returned, so explicit Metal never succeeds where CPU
  decode fails and never fails with a different error.
- **Composition.** Viewports and heterogeneous RGB8 tile batches without fast
  packets pack several source rectangles into one zero-filled pooled output.
  Reusable batch buffers and RGBA textures decode each tile through the hybrid
  route, then use the GPU copy passes.

Coefficient, difference, and plane storage are leased from the runtime's two
scratch slots and retained for the next call. Output surfaces come from the
surface pool. Every allocation stays within the device's
`maxBufferLength` and the repository's allocation cap, and an oversized request
fails with a typed error. A whole-image progressive or 12-bit coefficient set
can therefore exceed the cap on very large slides where CPU decode succeeds.
`Auto` routing picks the CPU for these classes.

Parity tests (`tests/hybrid.rs`) compare every route byte for byte with CPU
decode. Inputs cover:
- the libjpeg-turbo scaled matrix: formats, scales and regions, including
  multi-scan sequential and CMYK/YCCK cases
- conformance fixtures
- the 81-case lossless matrix: predictors 1–7, precisions 2–16, point
  transforms, restart intervals, whole-ratio sampling at odd sizes, and split
  scans
- the DCMTK DICOM matrix, which is also compared with DCMTK's own decode
- malformed lossless streams and region edge sweeps
- windowed regions of 1024×768 images with and without restart markers

Hybrid decode is about as fast as CPU decode because the serial entropy stage
dominates it. Its purpose is coverage and residency, not single-image speed.
Median wall time per full decode (parse, entropy, GPU work and wait), measured
on an Apple M4 Pro, macOS 27.0.1, Rust 1.99.0, `release-bench` profile, on
October 18, 2026, with 20 samples, 2 s warmup and 4 s measurement:

| Input | CPU | Metal hybrid |
|---|---:|---:|
| Progressive 4:2:0, 2048×1536 | 14.348 ms | 13.465 ms |
| Progressive 4:4:4, 1024×1024 | 8.611 ms | 7.936 ms |
| Baseline grayscale, 2048×2048 | 8.344 ms | 8.774 ms |
| Progressive grayscale, 1024×1024 | 2.847 ms | 2.993 ms |
| Baseline 4:4:0, 1600×1200 | 8.142 ms | 8.103 ms |
| Baseline CMYK, 1024×1024 | 5.724 ms | 5.639 ms |
| 12-bit extended grayscale, 512×512 | 0.848 ms | 0.958 ms |
| 16-bit lossless predictor 1, 512×512 | 0.976 ms | 1.321 ms |
| 16-bit lossless predictor 1, 2048×2500 | 22.326 ms | 25.441 ms |
| 16-bit lossless predictor 7, 512×512 | 0.994 ms | 2.153 ms |
| 8-bit lossless RGB predictor 1, 1024×1024 | 6.187 ms | 6.574 ms |

The first six inputs are generated in the benchmark with `jpeg-encoder`. The
others are synthetic CT/CR-like phantoms encoded with libjpeg-turbo 3.1.4
`cjpeg` (`-precision 16 -lossless 1|7`, `-precision 12 -quality 95`, and
`-lossless 1 -rgb`) and passed through `J2K_BENCH_INPUTS`. Before the
row-parallel predictor-1 kernel, the wavefront took 2.155 ms (512×512) and
35.862 ms (2048×2500) for predictor-1 lossless. Predictor 7 uses the
wavefront. A real speedup over the CPU for these classes needs GPU entropy
decode, or CPU entropy decode overlapped with GPU work across tiles; neither
exists yet.

## Reproduction and checks

```sh
# Hybrid decode against CPU decode (also asserts parity). J2K_BENCH_INPUTS is
# optional and may list extra JPEG files or directories.
J2K_REQUIRE_METAL_BENCH=1 J2K_BENCH_INPUTS=/path/to/medical/jpegs cargo bench --locked -p j2k-jpeg-metal --bench hybrid --profile release-bench -- --sample-size 20 --warm-up-time 2 --measurement-time 4 --noplot

# Hybrid parity suite.
J2K_REQUIRE_METAL_RUNTIME=1 cargo test --locked -p j2k-jpeg-metal --test hybrid -- --test-threads=1

# Same-input implementation and threadgroup comparison (also asserts parity).
J2K_REQUIRE_METAL_RUNTIME=1 cargo test --locked -p j2k-jpeg-metal --lib --profile release-bench texture_scheduling_experiment -- --ignored --test-threads=1 --nocapture

# Small batches and 4:2:2 controls.
J2K_REQUIRE_METAL_RUNTIME=1 J2K_JPEG_TEXTURE_SMALL_BATCHES=1 cargo test --locked -p j2k-jpeg-metal --lib --profile release-bench texture_scheduling_experiment -- --ignored --test-threads=1 --nocapture

# Public API benchmark: fresh CPU/Metal RGB and warm resident batch-16 textures.
J2K_REQUIRE_METAL_BENCH=1 cargo bench --locked -p j2k-jpeg-metal --bench compare --profile release-bench -- '^(decode_rgb/generated/fast420(_restart2)?_256x256/(cpu|metal)|wsi_tile_batch_rgba_textures/batch16/warm_session_reused_textures/generated/fast420(_restart2)?_256x256)$' --sample-size 10 --warm-up-time 1 --measurement-time 2 --noplot

# Final resident-output run with longer sampling.
J2K_REQUIRE_METAL_BENCH=1 cargo bench --locked -p j2k-jpeg-metal --bench compare --profile release-bench -- '^wsi_tile_batch_rgba_textures/batch16/warm_session_reused_textures/generated/fast420(_restart2)?_256x256$' --sample-size 10 --warm-up-time 2 --measurement-time 5 --noplot

# Runtime tests require real Metal instead of accepting an unavailable device.
J2K_REQUIRE_METAL_RUNTIME=1 cargo test --locked -p j2k-jpeg-metal -- --test-threads=1
cargo test -p j2k-jpeg --lib
cargo fmt --check -p j2k-jpeg-metal -p j2k-jpeg
cargo clippy --locked -p j2k-jpeg-metal --lib -- -D warnings
cargo clippy --locked -p j2k-jpeg-metal --tests --benches --bins -- -D warnings -A clippy::disallowed_methods -A clippy::disallowed_macros
```

The non-library Clippy allowances match the repository's Metal check policy
and do not apply to library checks.
