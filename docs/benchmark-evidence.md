# Benchmark Results

This page records benchmark commands, results, and the machines they ran on.
When a harness writes JSON or CSV output, that output is authoritative. Other
docs link here instead of copying numbers. Older results are in Git history.

## Scope

What the codec supports is listed separately in
[`docs/public-support.md`](public-support.md).

Speed comparisons against other codecs need an external benchmark bundle:

```bash
cargo run -p xtask --features adoption -- adoption-report --run-dir target/j2k-adoption-benchmark/full
```

`adoption-report` refuses an incomplete bundle and lists what is missing.
Speed comparisons must use the manifest-backed external rows, not the
generated in-repo fixtures.

## CPU classic Tier-1 decoding - 2026-10-29

Apple M4 Pro (12 CPU cores: 8 performance, 4 efficiency; 48 GB), macOS
27.0.1, Rust 1.99.0 release builds. "Before" is the parent of the classic
Tier-1 commits; "after" adds the MQ read-ahead, bit-set candidate rows, window
shifts, one-shift state selection and 256-slot contexts. A second build of
"before" from its own source copy is the A/A control. One harness was built
against each tree; every round runs each build once in a shuffled order, and
each cell reports the per-round median change against "before" in the same
round, with its range.

Inputs: a 4096 x 2048 RGB8 region of the 18000 x 12000 classic `large.jp2`
(real tissue), encoded once by "before" as classic lossless (5/3, 10.6 MB)
and at 2 bpp (9/7, 2.09 MB); the 18000 x 12000 classic 9/7 `large-plt.jp2`
and `large.jp2` themselves. Full decodes are `J2kDecoder::decode_into` to
RGB8 inside a Rayon pool of 1 or 12 threads. "Region" is one serial 1024 x
1024 `decode_region_into` from the middle of `large-plt.jp2`; "8 concurrent"
is eight threads each decoding a different 1024 x 1024 region of `large.jp2`
serially, timed together. Every build produced byte-identical output in every
cell, encodes included.

| Cell | Before (ms) | A/A | After |
| --- | ---: | ---: | ---: |
| Lossless decode, 1 thread | 1026 | -0.4% [-1.4, +2.3] | -11.2% |
| 2 bpp decode, 1 thread | 260 | +0.8% [-1.8, +2.2] | -8.6% |
| Region, 1 thread | 117 | -1.8% [-2.3, +3.5] | -12.0% |
| Lossless decode, 12 threads | 122 | +0.3% [-7.1, +8.5] | -11.3% [-25.7, -5.9] |
| 2 bpp decode, 12 threads | 37.1 | -2.8% [-17.5, +16.3] | -4.4% [-31.3, +4.8] |
| 8 concurrent regions | 185 | -0.6% [-6.1, +1.1] | -9.2% [-20.3, -2.0] |
| Lossless encode, 12 threads | 353 | +0.3% [-4.0, +4.1] | -3.0% [-14.4, +10.6] |

The single-thread rows are seven rounds at load average 9 to 33, so their
"after" ranges are left out: one round of each was an outlier (+13% to +61%),
and the other six were -6.1% to -14.1%. The multi-thread, concurrent and
encode rows are a second run of nine rounds at load average 8 to 13. Without
the 256-slot contexts, single-thread decodes are 0.7 to 1.8 points less fast. `ArithmeticDecoderContext` is one byte, so those contexts add 237
bytes per decoding context; peak heap of the serial region decode was
unchanged, and multi-thread peaks differed by no more than the A/A builds'
own few kilobytes.

## CPU owned batches of large-image regions - 2026-10-29

Same machine and inputs as the next section, load average 4 to 7. Each sample
decodes one round of four 1024 px output tiles at one reduction through
`CpuBatchDecoder::prepare` plus `decode_prepared` with default options;
"CPU tiles" is one serial `decode_region_scaled_pow2_into` per tile, the four
in parallel. The first sample checks that both produce the same bytes. Seven
samples, median ms. "Before" is `main` at 6ca1bdbf with the same test file.

```bash
J2K_LARGE_REGION_INPUT=<codestream> cargo test --release -p j2k \
  --test owned_batch local_large_region_cpu_batch_characterization \
  -- --ignored --nocapture
```

| Input | Level | CPU batch before | CPU batch now | CPU tiles |
| --- | ---: | ---: | ---: | ---: |
| large-ht.j2c | 0 | `ImageTooLarge` | 28 | 26 |
| large-ht.j2c | 1 | `ImageTooLarge` | 27 | 26 |
| large-ht.j2c | 2 | 48 | 24 | 22 |
| large.jp2 | 0 | `ImageTooLarge` | 152 | 149 |
| large.jp2 | 1 | `ImageTooLarge` | 244 | 229 |
| large.jp2 | 2 | 346 | 317 | 312 |
| large-plt.jp2 | 0 | `ImageTooLarge` | 136 | 133 |
| large-plt.jp2 | 1 | `ImageTooLarge` | 224 | 220 |
| large-plt.jp2 | 2 | 343 | 310 | 306 |

Small slide tiles, where the rule keeps the whole-level plan: 32 level-0
`CMU-1-JP2K-33005` tiles (240 x 240), batched with one worker per core, took
9.8 to 10.0 ms before and after for full requests and 8.8 to 9.0 ms before
against 9.1 to 9.2 ms after for a centred half-size region. Preparation, which
now weighs the windowed copy, accounts for 0.1 ms of that. Windowing those
regions without the rule measured 9.5 to 9.7 ms.

## Metal owned batches of large-image regions - 2026-10-28

Same machine as the next section, load average 5 to 11. Inputs are the
18000 x 12000 RGB codestreams used for the region work: `large-ht.j2c`
(HTJ2K), `large.jp2` and `large-plt.jp2` (classic). Each sample decodes one
round of four 1024 px output tiles at one reduction, the requests a raw
`.j2k` viewer level read makes. "Metal batch" is `MetalBatchDecoder::prepare`
plus `decode_prepared`, output left resident; "CPU tiles" is one serial
`decode_region_scaled_pow2_into` per tile, the four in parallel. Seven
samples, median ms.

```bash
J2K_REQUIRE_METAL_RUNTIME=1 J2K_LARGE_REGION_INPUT=<codestream> \
  cargo test --release -p j2k-metal --lib \
  local_large_region_batch_characterization -- --ignored --nocapture
```

| Input | Level | Metal batch | CPU tiles |
| --- | ---: | ---: | ---: |
| large-ht.j2c | 0 | 18 | 26 |
| large-ht.j2c | 1 | 10 | 25 |
| large-ht.j2c | 2 | 9 | 22 |
| large.jp2 | 0 | 136 | 148 |
| large.jp2 | 1 | 130 | 228 |
| large.jp2 | 2 | 151 | 311 |
| large-plt.jp2 | 0 | 113 | 139 |
| large-plt.jp2 | 1 | 116 | 327 |
| large-plt.jp2 | 2 | 143 | 440 |

Before the change, levels 0 and 1 failed on Metal: their plans requested
2,592,000,000 and 648,000,000 byte coefficient buffers, above the 512 MiB
cap. With window narrowing disabled, level 2 measured 36 ms (HTJ2K) and 165
to 166 ms (classic). A separate check found the Metal batch and the CPU tile
output byte-identical at every level of all three inputs.

## Metal owned batches of subsampled WSI tiles - 2026-10-28

Apple M4 Pro (12 CPU cores, 16-core GPU), macOS 27.0.1, load average 31 to
58 from other work. Inputs are 32 interior tiles per pyramid level extracted
from the openslide test slides `JP2K-33003-1.svs` and `JP2K-33003-2.svs`:
raw 256 x 256 classic codestreams, YCbCr with Cb and Cr at half width
(sampling (1,1), (2,1), (2,1)) and no MCT. Each sample decodes all 32 tiles to
resident `Rgb8` with no host readback. "Owned batch" is
`MetalBatchDecoder::prepare` plus `submit_prepared` and `wait`; "per image" is
one `J2kDecoder::decode_request_to_device_with_session` per tile, the path
callers used while preparation rejected subsampling. Eight samples, ms.

```bash
J2K_REQUIRE_METAL_RUNTIME=1 J2K_SAMPLED_CORPUS=<level{L}-tile{i}.j2k dir> \
  J2K_SAMPLED_COUNT=32 cargo test --release -p j2k-metal --lib \
  local_sampled_color_batch_characterization -- --ignored --nocapture
```

| Slide | Level | Owned batch min | Owned batch median | Per image min | Per image median |
| --- | ---: | ---: | ---: | ---: | ---: |
| JP2K-33003-1 | 0 | 11.4 | 12.9 | 152.0 | 173.7 |
| JP2K-33003-1 | 1 | 15.0 | 19.2 | 155.6 | 288.3 |
| JP2K-33003-1 | 2 | 16.7 | 29.5 | 166.5 | 391.6 |
| JP2K-33003-2 | 0 | 7.4 | 12.1 | 92.8 | 142.6 |
| JP2K-33003-2 | 1 | 10.2 | 14.5 | 77.6 | 132.7 |
| JP2K-33003-2 | 2 | 12.6 | 22.9 | 123.1 | 187.3 |

Every owned batch used one command buffer and returned no indexed or group
errors. The per-image path waits for the GPU after each tile. A separate
check on the same tiles found the owned batch, per-image Metal and CPU batch
outputs byte-identical.

## HTJ2K CPU comparator and WSI development run - 2026-10-20

This run checks the in-process OpenJPH and Grok integrations on pathology
pixels. It is development evidence on one ARM machine, not an overall HTJ2K
codec ranking or a complete `adoption-report` publication bundle. Kakadu was
not available. OpenJPEG is retained as a correctness/context row, not the
HTJ2K performance baseline.

### Hardware and reference builds

- Apple M4 Pro, native `aarch64`, 12 CPU cores (8 performance, 4 efficiency),
  48 GiB RAM; macOS 27.0.1 (26A434). All rows use the CPU.
- j2k 0.12.0 working tree, Rust 1.99.0, Cargo `release-bench`: optimization
  level 3, fat LTO, one codegen unit, debug symbols retained. The measured
  executable was held fixed across rounds while other work continued in the
  checkout.
- OpenJPH 0.31.0, rebuilt by `scripts/prepare-openjph-reference.sh` using
  Apple Clang 21.0.0 and CMake 4.2.1. Its generated CMake cache and compiler
  flags confirm `Release`, `-O3 -DNDEBUG -arch arm64`,
  `OJPH_DISABLE_SIMD=OFF`, and `OJPH_DISABLE_NEON=OFF`.
  **Those settings do not imply a specialized NEON decoder.** In this pinned
  OpenJPH version, the ARM branches in `codeblock_fun::init`, the wavelet
  dispatch, and the color dispatch are empty; they retain the generic C++
  kernels, which the compiler may auto-vectorize. The specialized AVX2 paths
  are selected on x86, not this host. These results establish nothing about
  performance against AVX2 OpenJPH.
- Grok 20.3.2, the installed Homebrew `arm64` bottle, linked in-process. Its
  installation receipt records a Clang build; the bottle's original compiler
  flags were not retained locally, so this is not a source-build/SIMD audit of
  Grok. OpenJPEG context rows use vendored OpenJPEG 2.5.3, not the system CLI.

The October 15 Kodak result of 2.10x versus OpenJPH used a single *batch
worker*. That run's Grok binary had a 12-thread process pool even in the
one-worker row, so it must not be described as an all-codec single-thread
comparison. Its original OpenJPH build directory was no longer present for
inspection. The rebuilt reference and explicit thread policies below
supersede that run for this discussion.

### Timing, threads, and RGB8 equivalence

`jp2k_batch_compare` loads compressed inputs before timing and calls each
library directly. No decoder subprocess, file read, or file write is inside
the timed boundary. Timings include header parsing, decoding, inverse color
transform, conversion/interleaving to owned RGB8, allocation, per-batch outer
worker startup, and destruction. Grok's persistent pool is initialized before
timing. Each decoder retains all final image buffers until the end
of the batch. j2k reuses its context/scratch within a batch worker; reference
wrappers construct a decoder per input. Prepared decoder state is not cached
between timed batches.

For batches with multiple inputs, explicit worker counts keep j2k serial
within each image. A later `perf` investigation found that a one-input batch
incorrectly enabled internal parallelism even with an explicit worker count.
Thus the historical mosaic rows below used j2k's shared Rayon pool; they are
not single-thread comparisons. The x86 optimization follow-up below fixes
this exception and supplies corrected measurements. OpenJPH is serial
within each image; OpenJPEG requests one decoder thread. Grok requests one
decode thread per job, and its process-wide Taskflow pool is now initialized
to the stated worker count before any preflight or timing. Its worker threads
are in addition to the submitting threads, which can wait on that pool; the
worker column is not a count of all OS threads. The one-worker runs use a
one-thread Grok pool. Separate runtime checks confirmed that
`grk_num_workers()` reports 1 and 12 for the respective initializations.
A fresh process is used when changing the pool size.

Every selected input is full-resolution, unsigned RGB8, with three equally
sampled components and reversible color transform. There is no subsampling,
ICC transform, precision reduction, ROI, or lossy rounding in these rows.
The native path writes interleaved `Rgb8`; OpenJPH supplies reconstructed
component lines that the shim clamps/interleaves; Grok supplies reconstructed
planes that its shim clamps/interleaves. The preflight now compares every
available reference against j2k and requires maximum absolute byte
difference **zero**, including Grok and OpenJPEG. A failure prevents timing.
This numerical preflight decodes one input at a time; timed throughput uses
the stated batch concurrency.

The audit also fixed the OpenJPH wrapper's order of operations: `set_planar`
must follow `read_headers`, which can reset the layout for RGB without MCT.
The existing no-MCT RGB fixture exposed this bug; the WSI and Kodak inputs
here use MCT. An unnecessary zero-filled intermediate output and full-image
copy were removed from that shim.

The output contracts are equivalent, but the integration costs are not
identical: both OpenJPH and Grok still pack into a C-owned buffer and copy it
once into a Rust `Vec`; j2k writes directly into its output `Vec`. Thus the
rows measure these complete integrations, not isolated entropy-decoder
throughput or the fastest possible output adapter for each library. Both
external packers are the repository's checked conversion loops.

A separate allocation/copy/free diagnostic (`copy_cost.rs` in the run
directory, compiled with `rustc -O`, one warm-up and 101 samples) measured one
additional full-output copy at 0.415 ms for 32 WSI frames, 0.504 ms for 24
Kodak images, and 0.392 ms for the mosaic. Those warmed-memory measurements
are small relative to the complete decode times, but do not isolate the
packing loops, decoder setup, or scratch reuse. They are not subtracted from
the reported results. Image dimensions alone do not establish that per-image
overhead dominates.

### Inputs and encoding parameters

The primary set contains the 32 largest level-zero tissue frames from
`Pancreas.dicom.cpu-pyramid`, a local WSI DICOM export written on 2026-06-16
with the j2k 0.10-era encoder. It uses HTJ2K lossless RPCL transfer syntax
`1.2.840.10008.1.2.4.202`. These are tissue-heavy selections from one slide,
not a random sample of all tissue and background tiles. The retained
extracted copies set the missing SIZ HT capability bit (`Rsiz=0x4000`) so
OpenJPH accepts them; compressed pixel payloads and the original DICOM were
not modified.

A second set encodes those same reconstructed pixels with OpenJPH 0.31.0.
A larger diagnostic arranges the 32 decoded tiles in filename order in an
8-by-4 mosaic and encodes it as one 4096 x 2048 image with 32 codestream tiles.
The mosaic is constructed from real tissue pixels; it is not a contiguous
slide region or a whole-slide read benchmark. Kodak is supplementary context.
No WSI inputs were skipped. The Kodak directory's 24 `._` metadata sidecars
were counted as unsupported inputs; all 24 actual codestreams were selected.

| Set | Inputs / dimensions | Encoder | Codestream tile size | Code-block | Layers | DWT levels | Progression |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| Original WSI frames | 32 x 512 x 512 RGB8 | j2k 0.10-era WSI writer | 512 x 512 | 64 x 64 | 1 | 3 | RPCL |
| OpenJPH WSI frames | Same 32 RGB8 images | OpenJPH 0.31.0 | 512 x 512 | 64 x 64 | 1 | 3 | RPCL |
| Tiled tissue mosaic | 1 x 4096 x 2048 RGB8 | OpenJPH 0.31.0 | 512 x 512 (32 tiles) | 64 x 64 | 1 | 3 | RPCL |
| Kodak context | 24; 768 x 512 or 512 x 768 RGB8 | Retained earlier j2k materialization; exact historical encoder version unrecorded | Full image | 64 x 64 | 1 | 2 | LRCP |

These parameters were read from SIZ/COD markers in every selected codestream.
All use reversible 5/3, RCT, HT block style `0x40`, and default precinct
partitioning (`Scod=0`). The OpenJPH encoding command for each staged PPM was:

```bash
ojph_compress -i tile.ppm -o tile.j2c \
  -reversible true -colour_trans true -num_decomps 3 \
  -block_size '{64,64}' -prog_order RPCL -tlm_marker true
# For the 4096 x 2048 mosaic, also pass: -tile_size '{512,512}'
```

### Results and limits

Across all 18 runs, every preflight passed exact RGB8 comparison against
OpenJPH, Grok, and OpenJPEG, covering 89 selected codestreams. Times below are
milliseconds for the **entire input set**, not per-image latency. Each cell
is the median of three run medians; parentheses show the lowest and highest
run median.

| Set | Workers | j2k | OpenJPH 0.31.0 | Grok 20.3.2 |
| --- | ---: | ---: | ---: | ---: |
| Original WSI frames, 32 | 1 | 62.58 (57.50–66.81) | 135.77 (131.28–141.74) | 182.41 (175.47–201.09) |
| Original WSI frames, 32 | 12 | 12.70 (12.56–13.58) | 23.75 (23.04–23.79) | 44.86 (35.86–46.46) |
| OpenJPH WSI frames, 32 | 1 | 58.85 (57.95–60.76) | 134.60 (131.48–134.86) | 183.84 (179.43–185.11) |
| OpenJPH WSI frames, 32 | 12 | 14.31 (13.93–17.81) | 25.60 (22.67–27.91) | 48.70 (39.15–53.81) |
| Tiled tissue mosaic, 1 | 1 | 45.59 (40.53–79.51) | 146.02 (140.42–152.96) | 190.11 (179.97–197.44) |
| Kodak context, 24 | 1 | 66.05 (63.63–73.46) | 146.69 (144.15–157.01) | 201.37 (192.71–273.95) |

The worker column is the requested batch-worker count. In this historical
run, j2k's mosaic row also used internal parallelism, as described above.

For the original WSI frames, the median of the within-run speed ratios was
2.17x versus OpenJPH and 3.01x versus Grok with one worker; with 12 workers it
was 1.84x and 3.42x. On OpenJPH-encoded WSI frames those ratios were 2.27x and
3.05x with one worker, and 1.63x and 3.02x with 12 workers. The ratios are
computed per run before taking their median, so they need not equal the
ratio of the table's independently aggregated time columns.

This was a busy shared host: the one-minute load average at configuration
starts ranged from 11.2 to 23.6, and unrelated compression work continued
throughout. The native mosaic median varied almost twofold; its within-run
lead over OpenJPH ranged from 1.92x to 3.46x. Treat these as workload-specific
development observations. An isolated-host Mac run, more independent slides,
other bit depths/lossy profiles, and Kakadu are still needed before making a
broader performance claim. The x86 AVX2 follow-up is recorded below.

### Reproduction

Use a separate output directory and the retained WSI frame directory, or
provide another explicitly described corpus. The local run directory is
`target/htj2k-comparison-20261020`; it contains the input preparation script,
run script, raw TSV samples, and reference build log. No raw slide pixels
are committed with this document.

```bash
scripts/prepare-openjph-reference.sh
export J2K_OPENJPH_SOURCE_DIR="$PWD/target/reference/openjph-0.31.0"
cargo build --profile release-bench -p j2k-compare --bin jp2k_batch_compare

J2K_BATCH_COMPARE_REPEATS=21 J2K_BATCH_COMPARE_THREADS=1 \
J2K_BATCH_COMPARE_MAX_ABS_DIFF=0 RAYON_NUM_THREADS=12 \
  target/release-bench/jp2k_batch_compare "$WSI_DIR" 32
# Repeat with J2K_BATCH_COMPARE_THREADS=12 for tile-batch throughput.
# For the directory containing only the mosaic, use batch size 1 and workers 1.
```

Each process runs one warm-up batch per decoder, then rotates the starting
decoder across 21 timed rounds. Three fresh processes measure each
configuration; corpus/configuration order also rotates between runs.

## HTJ2K CPU comparison on the CUDA host - 2026-10-20

This follow-up uses the same 89 compressed inputs, RGB8 output contract,
correctness gate, and rotating timing protocol as the M4 run above. It tests
CPU decoding on the x86 test machine described below; the GPU is not used.

- AMD Ryzen 7 5800X3D: 8 physical cores, 16 logical CPUs, AVX2/FMA available.
  Ubuntu 24.04.4 under WSL2, kernel
  `5.15.153.1-microsoft-standard-WSL2`, about 47 GiB RAM visible to Linux.
  The installed GPU is an NVIDIA GeForce RTX 4070 SUPER with 12 GiB VRAM.
- j2k 0.12.0 working-tree source export, Rust 1.99.0, `release-bench`, with
  the default `std`, `simd`, and `parallel` features. No extra
  `target-cpu=native` compiler flag was used. The export was built in a
  separate run directory without changing the existing remote source
  snapshot.
- OpenJPH 0.31.0, built with GCC 13.3.0 and CMake 3.30.5, `Release`,
  `-O3 -DNDEBUG`, `OJPH_DISABLE_SIMD=OFF`, and
  `OJPH_DISABLE_AVX2=OFF`. A probe linked to the same static library reported
  CPU extension level 10 (AVX2/FMA) and confirmed that the reversible
  `decode_cb32` function pointer equals `ojph_decode_codeblock_avx2`.
  This verifies runtime dispatch, not merely the build option.
- Grok 20.3.2 from the existing clean tagged source tree, built with GCC
  in `Release` mode with `-O3 -DNDEBUG`. Its pool is explicitly set to 1 or
  16 threads, matching the selected batch-worker setting. OpenJPEG 2.5.3 is
  retained only for correctness/context.

The host was idle when the run began (one-minute load average 0.04).
Each configuration uses three fresh processes, one warm-up per decoder and
21 interleaved timed samples per process. Tile batches use 1 and 16 workers;
the single large tiled image and supplementary Kodak set request one worker.
The original mosaic measurement nevertheless enabled internal Rayon work
because of the single-input scheduling bug described above.
The M4's full-host rows use 12 workers, so those rows compare different
machine capacities, not equal thread counts.

All 18 runs passed the exact RGB8 preflight for every selected codestream
against OpenJPH, Grok, and OpenJPEG. The table reports milliseconds for the
entire input set, as the median of three run medians; parentheses give the
lowest and highest run median.

| Set | Workers | j2k | OpenJPH 0.31.0, AVX2 | Grok 20.3.2 |
| --- | ---: | ---: | ---: | ---: |
| Original WSI frames, 32 | 1 | 196.80 (196.25–197.19) | 116.73 (116.66–119.26) | 246.86 (245.84–247.74) |
| Original WSI frames, 32 | 16 | 32.78 (32.76–33.80) | 17.13 (16.83–17.43) | 45.69 (44.94–46.03) |
| OpenJPH WSI frames, 32 | 1 | 197.74 (197.07–200.77) | 115.18 (112.44–118.64) | 246.57 (242.98–247.59) |
| OpenJPH WSI frames, 32 | 16 | 33.38 (32.76–34.38) | 16.89 (16.78–17.52) | 43.96 (43.92–44.74) |
| Tiled tissue mosaic, 1 | 1 | 313.98 (310.99–314.51) | 120.87 (119.97–121.05) | 248.88 (248.57–249.64) |
| Kodak context, 24 | 1 | 285.09 (285.06–286.29) | 132.05 (130.78–132.19) | 273.71 (271.01–274.58) |

The OpenJPH result reverses on this host. For the original WSI set, OpenJPH
has 1.68x our throughput with one worker and 1.91x with 16 workers. We have
1.26x Grok's throughput with one worker and 1.39x with 16 workers. On the
OpenJPH-encoded WSI set, OpenJPH has 1.71x and 1.94x our throughput, while
we have 1.23x and 1.32x Grok's. These are medians of within-run ratios.

In the original large-image measurement, OpenJPH has 2.60x our
throughput, and Grok has about 1.26x. That row includes the native scheduling
bug and is superseded by the optimization results below. On Kodak, OpenJPH has 2.17x our
throughput and Grok has about 1.05x. The M4 lead therefore does not
generalize to this x86 host, and the Grok comparison depends on whether
the input is an independent tile batch or one larger tiled codestream.

The same separate allocation/copy/free diagnostic measured 8.470 ms for the
32 WSI output buffers, 9.678 ms for the 24 Kodak buffers, and 0.586 ms for
the single mosaic buffer. These serial warmed-memory diagnostics were not
subtracted from the decode times and do not isolate the packing loops.
Both external integrations still include their extra FFI output copy.

The x86 measurements began on an idle host, whereas the earlier M4 run had
unrelated load. Together with different CPUs, compilers, operating systems,
and full-host worker counts, this prevents treating the absolute latency
difference between machines as a controlled measurement of architecture
alone. The within-host codec comparison is the useful result. Kakadu,
additional slides, lossy profiles, and other bit depths remain unmeasured
in this follow-up.

The local results directory is
`target/htj2k-comparison-20261020/cuda`. The remote run is under the existing
`j2k-review-20261019/target/htj2k-comparison-20261020` snapshot. It contains
the narrow source export, the copied codestreams, reference build logs,
AVX2 dispatch probe, run script, and raw TSV samples. Input file I/O remains
outside the timer.

## x86 HTJ2K optimization follow-up - 2026-10-20

Hardware, inputs, reference libraries, build profile, and RGB8 contract are
the same as the CUDA-host CPU run above. The original executable was retained
for comparison. The primary A/B experiment uses three fresh processes per
configuration, one warm-up and 11 rotating decoder samples per process.
Baseline/candidate order alternates between runs. No profiled run contributes
to the timing table.

`perf record -e cycles:u -F 999 --call-graph dwarf,16384` and the existing
stage summaries identified three distinct costs:

- A one-input batch ignored an explicit worker count and used internal
  Rayon parallelism. Entropy work was similar for separate frames and the
  mosaic, but IDWT and store took much longer in the mosaic. Correcting the
  worker policy alone reduced the mosaic from 313.04 to 206.55 ms in a
  seven-sample development comparison; separate frames took 198.36 ms.
- Parallel wavelet strips crossed a Rayon callback boundary without
  re-entering the SIMD target-feature scope. The x86 profile showed calls
  to individual AVX2 arithmetic helpers inside these workers. Each strip
  now establishes that scope. On x86, a 512 x 512 component copy also stays
  serial; larger copies can still use the pool.
- The RGB8 interleaver used scalar rounding calls. The x86 path now packs
  eight pixels at a time with the existing safe SIMD library, preserving
  saturating conversion and ties-away-from-zero rounding. HT magnitude
  extraction also uses the already-validated 25-bit limit to replace split
  shifts with single shifts.

The OpenJPH AVX2 and Metal cleanup/store implementations were inspected as
references. These changes reuse the existing Rust kernels and SIMD library;
they introduce no unsafe code or new dependencies. The new packer and copy
threshold are x86-specific. A preliminary shared-M4 check did not favor the
new packer, so ARM retains its existing packing and copy paths.

Times are milliseconds for the whole input set, reported as the median of
three process medians. Reference columns are from the candidate runs.

| Set | Explicit workers | Original j2k | Optimized j2k | Time reduction | OpenJPH AVX2 | Grok |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original WSI frames, 32 | 1 | 198.09 | 146.49 | 26.1% | 118.75 | 261.43 |
| Original WSI frames, 32 | 16 | 33.27 | 27.06 | 18.7% | 17.76 | 50.24 |
| Tiled tissue mosaic, 1 | 1 | 312.47 | 157.74 | 49.5% | 119.54 | 262.64 |

Candidate process medians ranged from 146.46–146.52 ms, 26.51–27.79 ms, and
156.16–159.31 ms, respectively. The mosaic improvement includes correcting
the original thread-policy bug; it is not a 49.5% kernel-only improvement.
OpenJPH still has about 1.23x our serial WSI throughput, 1.52x with 16
workers, and 1.32x on the mosaic. We remain faster than the measured Grok
integration on these configurations. Grok's candidate-run medians were
somewhat higher than its baseline-run medians (248.36/47.78/249.81 ms);
these comparisons describe complete integrations, including allocation and
packing, rather than isolated library kernels.

Supplementary seven-sample A/B runs measured 200.06 to 147.42 ms for the
32 OpenJPH-encoded WSI frames and 286.11 to 228.49 ms for Kodak. Together
with the primary runs, all 89 codestreams passed exact RGB8 comparison.
For the mosaic with automatic internal parallelism enabled in both
binaries, native time fell from 306.89 to 170.06 ms. Thus the changes also
improve that execution mode, although the explicit serial run remains
faster on this host. These supplementary results are single-process
medians, not three-run aggregates.

The final M4 spot check measured 57.03 to 55.90 ms for the serial original
WSI batch and 37.82 to 37.15 ms for the automatically parallel mosaic.
Both passed exact RGB8 comparison. Shared-host load and only seven samples
per process make this a regression check, not evidence of an ARM speedup.
The follow-up x86 cycle profile still identifies HT cleanup/magnitude
decoding as the largest native hotspot; the RGB8 packing hotspot is much
smaller.

All 18 primary runs passed exact RGB8 preflight against OpenJPH, Grok, and
OpenJPEG. Focused HT decode, wavelet, packing, and scheduling tests passed
on x86 and ARM, including existing scalar/SIMD comparisons for encoded and
arbitrary HT bytes. The worker-count regression failed before the fix and
passed afterward. The new packing test checks every 8-bit half-integer and
its adjacent floats, clipping, nonfinite values, tail pixels, and output
buffer preservation.

Native library and test Clippy checks passed on x86 and ARM, using the
repository's allocation-lint exemptions for test targets. A broader
`j2k` library Clippy invocation remains blocked by the existing missing
`#[must_use]` on `J2kToHtj2kOptions::with_coefficient_only` in `src/recode.rs`.

Raw samples, stage summaries, scripts, and text profiles are in
`target/htj2k-comparison-20261020/cuda/x86-opt`; `perf.data` recordings and
the retained executables remain in the matching remote directory. This
remains a development measurement on one x86 machine and one WSI source,
not a claim about all x86 systems or HTJ2K coding profiles.

## x86 SIMD and CUDA Oxide follow-up - 2026-10-20

This pass compares against the optimized CPU executable in the preceding
section, on the same Ryzen 7 5800X3D. CUDA uses the RTX 4070 SUPER, driver
610.88, Rust CUDA Oxide kernels built for sm_89, and a reused CUDA session.
OpenJPH 0.31.0 uses its AVX2 decoder.

The CPU change handles two quads and final coefficient conversion in eight
SIMD lanes, reusing prefix sums to remove scalar bit-count work. Separate
VLC/MagSgn staging was slower and was reverted. The
[HTJ2K design paper](https://kakadusoftware.com/wp-content/uploads/icip2019.pdf)
and [Intel's AVX optimization report](https://cdrdv2-public.intel.com/839078/J2K%20HTJ2K%20AVX%20Optimizations%20WP_Final_110424.pdf)
informed these experiments.

Primary CPU figures are medians of three process medians, nine samples
per process with rotating codec order and alternating before/after order.
All times below are milliseconds for the complete batch.

| Input | Workers | Previous pass | This pass | Time reduction | OpenJPH |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 original WSI tiles | 1 | 147.61 | 136.63 | 7.4% | 119.18 |
| 32 original WSI tiles | 16 | 27.35 | 26.01 | 4.9% | 16.37 |
| One 32-tile mosaic | 1 | 155.03 | 146.27 | 5.7% | 120.83 |

CPU medians ranged over 135.82–140.71, 24.77–26.23, and 144.73–146.42 ms.
Single-process supplements measured
147.33 to 135.76 ms for OpenJPH-encoded WSI and 231.51 to 216.71 ms for Kodak.

CUDA figures use three fresh processes, three pixel-verified warmups, and
31 timed samples per process. Compressed inputs are loaded before timing;
parsing, planning, upload, decode, output destruction, and host readback
are timed. CUDA context/session initialization, first-use module loading,
and the three warmups are excluded. The compressed codestream upload is
included on every timed decode. CPU also returns RGB8 host output. Profiling
runs are excluded.

| Input | CUDA API | CUDA including readback | OpenJPH, 1 worker |
| --- | --- | ---: | ---: |
| 1 original WSI tile | surface | 0.608 | 3.533 |
| 8 distinct original WSI tiles | surface | 2.318 | 30.697 |
| 32 distinct original WSI tiles | surface | 7.462 | 119.179 |
| 32 distinct original WSI tiles | owned NHWC | 9.335 | 119.179 |
| One 32-tile mosaic | owned NHWC | 10.155 | 120.826 |

For 32 tiles, OpenJPH with 16 workers takes 16.369 ms: CUDA is 2.19x
faster through the surface API and 1.75x through the owned API. Resident
surface output takes 5.158 ms, excluding readback. Owned NHWC includes
grouping and dense-output work; these APIs are not interchangeable costs.
The surface API rejects this mosaic; the owned API supports it. CUDA owned
single-process supplements took 9.581 ms for OpenJPH WSI and 13.173 ms
for Kodak.

These ratios compare one RTX 4070 SUPER with one 8-core/16-thread desktop
CPU. They do not establish a comparison with a larger CPU server or with
another GPU decoder. The nvJPEG2000 comparison and larger-batch measurements
below use separate runs on the same GPU.

Nsight Systems attributes approximately 51% of kernel time to HT symbol
and magnitude decoding, 32% to IDWT, and 17% to fused final transform/store.
Batching already combines work into eight launches for 32 WSI tiles.
Eight symbol jobs per warp, tiled horizontal IDWT, and grouped MEL reads
did not consistently improve timings; all three were reverted. No CUDA
kernel speedup is claimed. The profiler now exposes both APIs, optional
readback, and unsorted samples. This follows
[NVIDIA's transfer batching guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#data-transfer-between-host-and-device).

All 28 CPU and 20 CUDA processes passed exact RGB8 checks, covering the
89 codestreams. After local disk space was freed, the 32 focused HT tests
and native library Clippy also passed on ARM. CUDA-enabled profiler Clippy
passed on x86.

The final M4 Pro regression check used three fresh processes per version,
seven samples per process, alternating version order, and one explicit
worker. Median time changed from 55.89 to 54.35 ms for 32 WSI tiles and
from 62.53 to 59.76 ms for the mosaic. Current process medians ranged over
53.44–54.72 and 59.50–60.67 ms respectively. All 12 processes passed exact
RGB8 comparison. These shared-host measurements show no regression on
the checked workloads; they do not establish a general ARM speedup.

The raw timing and validation logs, scripts, and Nsight trace have been
copied locally to `target/htj2k-comparison-20261020/cuda/round2`, alongside
the final ARM results. The original x86/CUDA files remain in the matching
remote directory under `j2k-review-20261019/target/htj2k-comparison-20261020`.

## CUDA batch scaling, nvJPEG2000, and Burn handoff - 2026-10-20

This pass uses the same Ryzen 7 5800X3D, RTX 4070 SUPER, driver, CUDA Oxide
sm_89 build, and 32 original WSI tiles as the preceding section. Batches
larger than 32 cycle the same 32 distinct tiles; they are not larger
unique-input sets.

### CPU worker scaling

The CPU executable decoded the 32 tiles with 1, 4, 8, and 16 explicit
workers. Every process passed an exact RGB8 preflight
(`J2K_BATCH_COMPARE_MAX_ABS_DIFF=0`) against OpenJPEG, OpenJPH, and Grok. Each
configuration used three fresh processes, five in the A/B comparisons, and
worker and binary order alternated between runs. Times are milliseconds, as
the median of process medians. The host was shared with another session and a
GitHub Actions runner during this pass. Every reported process therefore ran
under a shared lock, started only when no CI job was running, and recorded the
host CPU used by other processes; runs from an earlier overlapping window were
discarded. "Before" is the source export before this pass; "after" adds the
three changes described below.

The first table uses the original protocol: all four decoders alternate within
each process, with nine samples per process. OpenJPH is from the "after"
processes.

| Workers | j2k before | j2k after | OpenJPH 0.31.0, AVX2 |
| ---: | ---: | ---: | ---: |
| 1 | 135.69 | 134.54 | 118.81 |
| 4 | 54.44 | 40.16 | 36.24 |
| 8 | 33.34 | 28.79 | 22.83 |
| 16 | 25.43 | 22.67 | 16.61 |

The second table times one decoder per process (`J2K_BATCH_COMPARE_DECODER`),
with 15 samples per process; the reference preflight still runs.

| Workers | j2k before | j2k after | OpenJPH 0.31.0, AVX2 |
| ---: | ---: | ---: | ---: |
| 1 | 134.98 | 133.03 | 116.20 |
| 4 | 50.19 | 36.55 | 33.77 |
| 8 | 31.51 | 26.85 | 23.02 |
| 16 | 26.67 | 21.04 | 19.63 |

From 1 to 16 workers, j2k now speeds up 6.32x when timed alone (5.06x before)
and 5.93x in the alternating protocol (5.34x before). OpenJPH speeds up 5.92x
alone and 7.16x alternating. Its 16-worker time is 16.61 ms alternating but
19.63 ms alone, because its per-tile allocations reuse memory the other
decoders just freed. The original scaling comparison therefore partly measured
allocator state rather than the decoders.

Root cause. Each 512x512 RGB tile allocated about 5.5 MB of f32 decode
buffers (three component planes, the coefficient buffer, and the IDWT output
and scratch), zero-filled them, and freed them. glibc's dynamic trim and mmap
thresholds returned that memory to the kernel, so the next tile page-faulted
and the kernel zero-filled it again. At 16 workers the batch API took 859
page faults and 19% system time per tile; with the buffers retained (change 2
below) it takes 191 faults, all first writes to the caller's output buffers,
and 5%. As a diagnostic, pinning the glibc thresholds made j2k, with only
change 1, 7–32% faster across worker counts when timed alone. Lock contention
and load imbalance do not explain the gap: it closes when allocator churn is
removed, the tiles are within 3% of one another in compressed size, and worker
start and end skew matched between the batch API and a hand-built equivalent.

A smaller, structural part remains. j2k holds the whole tile as f32 planes,
about 6.7 MB per worker including input and output, while OpenJPH decodes a
few lines at a time. With allocator churn removed, j2k fills 1.4 MB per tile
from DRAM with one worker and 3.0 MB with 16, once 16 working sets exceed the
96 MiB L3; OpenJPH fills 0.09 and 0.38 MB. From one to 16 workers, per-tile
user cycles grow 1.81x for j2k and 1.50x for OpenJPH. The HT cleanup kernel,
which writes each component's coefficients into a full-tile buffer, grows
1.86x and accounts for two thirds of j2k's extra cycles; zero-filling and
copying the planes (memset and memmove) grow 2.3x and account for another
fifth.
Closing that gap needs a banded decode that keeps IDWT, colour transform, and
packing within a few rows; this pass does not attempt it.

Three changes were retained, each introduced test-first:

1. IDWT level and band buffers only grow. Later components and levels reuse
   samples already initialized instead of truncating and zero-filling them
   again, which saves about 2.5 MiB of writes per tile. Alone, this changed
   time by −1.1%, 0.0%, −2.7%, and −2.2% at 1, 4, 8, and 16 workers.
2. `CpuBatchDecoder::decode_tiles_into` decodes caller-owned tiles on the
   session's retained workers. Each worker keeps its decoded-component,
   Tier-1, and IDWT allocations across tiles and calls. When `store` writes
   every plane sample (a full decode with 1x1 sampling, no resolution
   reduction, and no store backend), retained planes are no longer
   zero-filled first. With that re-zeroing, the copy from IDWT output into
   the retained planes took 36% of the process's DRAM fills, and retained
   buffers were 2–7% slower than fresh ones with allocator churn disabled.
   Without it, retained buffers are faster than fresh ones at every worker
   count under default glibc, and within 4% of them with churn disabled. The
   decoder counts the samples `store` writes into planes it did not zero, and
   a tile that leaves any of them unwritten fails with
   `DecodingError::IncompletePlaneStore` instead of exposing stale samples.
3. `decode_tiles_into` keeps one process-wide set of those workers between
   calls, so existing callers benefit without code changes. After a call the
   set is kept only while it retains at most 256 MiB.
   `retained_tile_batch_workspace_bytes` reports it and
   `release_tile_batch_workspaces` frees it; neither waits for a running
   batch, and a release requested during one takes effect when it ends. A
   call made while another holds the set uses per-call workers instead of
   waiting.

Against the export before this pass, the unchanged harness timed alone changed
by −1.3%, −23.5%, −11.2%, and −17.9% at 1, 4, 8, and 16 workers, with no
overlap between the five before and five after processes at 4, 8, or 16
workers. Those runs predate the plane-coverage check and the non-blocking
release in changes 2 and 3. Rerun on the final source with the same protocol,
the changes were −0.8%, −25.0%, −13.1%, and −19.5% (134.62 to 133.61, 50.18
to 37.62, 31.20 to 27.11, and 26.84 to 21.60 ms), again with no overlap at 4,
8, or 16 workers.

Three approaches were rejected. Reusing the workspace only within a batch
gained 12% at four workers but lost 4% serially and gained nothing at 8 or 16,
where each worker decodes four or two tiles. Capping glibc arenas at 16
changed nothing at 16 workers. glibc tunables are process-wide, so a library
cannot set them. The per-component coefficient zero-fill (about 2 MiB per
tile) remains. Removing it needs targeted zeroing of the code blocks each
Tier-1 route skips.

The scripts, probe source (`scaling_probe.rs`), and summaries are in
`target/htj2k-comparison-20261020/cuda/scaling`; raw TSVs remain in the remote
run directory.

### CUDA surface batches above 32 images

The original surface API decoded all colour inputs in one group. The session
retains at most 2,048 batch scratch buffers (2 GiB), and one 32-tile batch
retains about 1,253 of them. Larger batches therefore allocated and freed
uncached scratch on every decode. The earlier 1024-image failure was not a
decode failure: the profile harness validated the batch with one 768 MiB host
readback, which exceeds the 512 MiB default host allocation cap.

The surface API now decodes colour batches in successive 32-image groups on
the same session. It keeps the caller's output order and the scratch cache
limits, aggregates profile reports, and rebases source and code-block error
indices to the whole batch. Each group's outputs share one device allocation,
so `Surface::download_batch_tight` and `download_batch_tight_into` now copy
each adjacent run of surfaces with one transfer instead of one transfer per
surface. The profile harness now validates and reads back into a
zero-initialized host buffer per batch. This matches the nvJPEG2000 harness
and allows batches above the 512 MiB single-allocation cap.

The baseline was built from the same source export with `routing.rs` and
`surface.rs` restored to `HEAD`; both binaries use the updated harness.
nvJPEG2000 0.10.0.49 uses ten pipeline states and streams, interleaved RGB8,
and a per-batch `cudaMalloc` output. It validates its warmups against the
same PPM references. Each decoder, batch size, and readback mode used three
fresh processes, three pixel-verified warmups, and 31 timed samples. Case
order rotated between runs and decoder order rotated between cases. Timed
samples include parsing, compressed-input upload, decode, synchronization,
and per-batch device output allocation and destruction. Context and library
initialization and the warmups are excluded. Times are milliseconds, as the
median of three process medians.

| Images | HEAD resident | Grouped resident | nvJPEG2000 resident | HEAD with readback | Grouped with readback | nvJPEG2000 with readback |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 5.14 | 5.21 | 9.80 | 8.54 | 8.29 | 13.32 |
| 64 | 71.50 | 10.54 | 18.01 | 79.99 | 20.08 | 28.77 |
| 128 | 216.56 | 20.41 | 35.11 | 255.15 | 41.07 | 55.67 |
| 256 | 547.30 | 41.05 | 63.93 | 589.48 | 81.59 | 106.70 |
| 1024 | 2870.22 | 168.49 | 260.07 | 3099.52 | 330.80 | 427.55 |

All 90 processes passed exact RGB8 checks, including every 1024-image run.
Grouped resident cost stays between 0.159 and 0.165 ms per image from 32 to
1024 images, while `HEAD` already takes 13.9x longer for 64 images than for 32.
The grouped path is 6.79x faster than `HEAD` at 64 resident images and 17.0x
faster at 1024. Compared with nvJPEG2000, it has 1.78x the throughput at 32
resident images, 1.55x at 1024, and 1.62x to 1.29x with readback. These ratios
are medians of within-run ratios. Above 32 images, readback adds about
0.16 ms per image for both decoders, so host allocation and pageable copies
dominate that column rather than codec work.

Do not cite these nvJPEG2000 ratios as a general comparison; a follow-up
audit found three limits:
- **Inputs:** all 32 inputs are lossless (reversible 5/3, RCT, three
  decomposition levels, 64×64 HT code blocks). Lossy 9/7 was not timed.
- **Output format:** nvJPEG2000 was configured for interleaved RGB8. A
  screening run suggested that its native planar output is substantially
  faster. That screen overlapped another benchmark on the same host, so it is
  not evidence.
- **Host threads:** j2k plans batches across 16 Rayon threads, while this
  harness submitted nvJPEG2000 work from one host thread.

nvJPEG2000 0.10.0.49 has no batched decode call; this harness used ten decode
states on ten streams. Both decoders read back into pageable host memory. A
fair re-measurement compares planar with planar and interleaved with
interleaved, gives nvJPEG2000 its best host-thread configuration, and adds
lossy inputs. It has not been completed.

At 32 images both binaries run the same unchunked decode. The grouped binary
measured 1.4% slower resident (process medians 5.19–5.53 against
5.14–5.14 ms) and 2.9% faster with readback; this pass does not attribute a
32-image change to the grouping code.

### Owned `CudaBatchDecoder` batches above 32 images

The owned API, which `j2k-ml` uses, has its own execution path and the same
problem: one warm 65-image group made 1,274 device allocations, against 2 for
a warm 32-image group. RGB and RGBA groups larger than 32 images now decode in
successive 32-image chunks into one dense allocation (or into consecutive
ranges of a caller's destination). Each chunk except the last completes before
the next starts, so its scratch returns to the cache, and `submit_batch` now
waits for all but the final chunk. Removing those host waits would need the
batch pool to reuse buffers in stream order; today it reuses a buffer only
after the host has confirmed that the GPU finished with it. Error source indices are already global;
code-block job indices are rebased, including for a failure surfaced by the
final chunk's wait.

The "before" binary is the grouped-surface build above, whose owned path is
unchanged from `HEAD`. Both binaries use the same harness and protocol, with
`J2K_CUDA_PROFILE_API=owned` (NHWC output). Times are milliseconds, as the
median of three process medians.

| Images | Before resident | After resident | Speedup | Before with readback | After with readback |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 6.80 | 6.98 | 0.99x | 10.44 | 10.54 |
| 64 | 65.54 | 13.09 | 4.97x | 75.95 | 24.62 |
| 128 | 203.00 | 25.92 | 7.83x | 228.36 | 46.54 |
| 256 | 547.88 | 53.02 | 10.31x | 573.40 | 92.94 |
| 1024 | 2617.90 | 219.10 | 11.96x | 2885.52 | 385.00 |

All 60 processes passed exact RGB8 checks. Owned resident cost stays between
0.20 and 0.22 ms per image from 32 to 1024 images. At 32 images the code path
is unchanged; the 0.99x ratio is within the process ranges (6.79–6.89 and
6.78–7.05 ms). The owned API remains about 0.05 ms per image slower than the
surface API because it also groups images and writes dense NHWC output.

With both changes, the complete strict j2k-cuda suite passed on the RTX 4070
SUPER with `J2K_REQUIRE_CUDA_RUNTIME=1` and `J2K_REQUIRE_CUDA_OXIDE_BUILD=1`:
300 passed, none failed or were ignored. For the surface API it checks 65-image
output order, warm allocations, and one readback transfer per group; cleanup
and session reuse after a later group's planning failure; and global error
indices for a failure in the second group. For the owned API it checks NHWC and
NCHW 65-image output against the CPU decoder, bounded warm allocations, a
65-image caller destination, and global source/job indices for failures in a
middle chunk and in the final chunk, followed by session reuse. A separate
large-group test covering 8-bit, 12-bit, and signed 16-bit RGBA in both layouts
also passed, as did the 107 `j2k-cuda-runtime` library tests, including
bounds checks for the new destination sub-views.

### Direct CUDA-to-Burn handoff

`CudaDirectBurnDecoder` decodes into Burn-owned CUDA storage and registers that
allocation as the tensor. `CudaUploadBurnDecoder` stages the completed codec
output through host memory and uploads it with Burn. Both use the owned
`CudaBatchDecoder` with NCHW output, so these rows are not comparable with the
surface API's 5.21 ms resident result above.

For these runs only, a benchmark overlay (not committed) loaded the 32
original WSI codestreams into `batch_decode_cuda`; batches of 128 and 256 cycle
those tiles. Before timing, every process checked all 32 distinct RGB8 outputs
from both Burn APIs against `CpuBatchDecoder`. The larger batches use the owned
path whose exact output is checked by the owned API runs and tests above. Each
row ran in its own process, three times with rotating row order. Criterion used
its default warmup and 100 samples; the table gives the median of the three
per-process Criterion medians, in milliseconds. The normalized rows convert the
`U8` tensor to `f32`, divide by 255, and synchronize the Burn device inside
each iteration. These runs include the owned API chunking above.

| WSI tiles, full decode | Owned NCHW codec decode, no Burn | Staged Burn upload and normalization | Direct Burn destination and normalization | Staged / direct |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 6.84 | 18.34 | 8.08 | 2.28x |
| 128 | 25.48 | 85.28 | 28.51 | 2.99x |
| 256 | 53.72 | 162.05 | 55.87 | 2.91x |

Direct process medians ranged over 8.05–8.18, 28.34–28.81, and 55.70–55.90 ms.
The direct route adds 1.2–3.2 ms to the owned codec decode for the Burn
allocation, registration, synchronization, and normalization, and settles at
about 0.22 ms per image. Ratios are medians of within-run comparisons. An
earlier 32-image run, made before the owned API change, measured 6.85, 18.45,
and 8.22 ms for the same three rows.

The strict j2k-ml CUDA tests passed on the RTX 4070 SUPER (19 tests,
including two distinct RGB8 codestreams checked against an independent CPU
oracle with status-only codec readback), and the `cuda_direct` example ran.

The first benchmark build exposed a build bug: nested `cargo oxide build`
invocations inherited the caller's `CARGO_TARGET_DIR`. A newly staged device
project could then be treated as a fresh build from another `OUT_DIR` and
produce no PTX. `j2k-cuda-build-support` now gives each staged project its own
target directory. The staged device crates still have no lockfile, so a fresh
build resolves their crates.io dependencies at build time and contacts the
network even under `--offline`.

Results, scripts, the overlay files, and the copied JSON summaries are in
`target/htj2k-comparison-20261020/cuda/scaling`. Raw per-process outputs,
logs, and executables remain in the matching remote `scaling` directory.

## HTJ2K kernel optimization development run - 2026-10-17

These are incremental measurements against the working tree after rebasing on
`origin/main`, before this optimization pass. Both builds used the same Rust
1.99.0 toolchain on an Apple M4 Pro (12 CPU cores, 16 GPU cores, 48 GB).
They include earlier uncommitted optimizations. They do not establish a
performance ranking against other codecs or hardware.

| Workload and timing boundary | Before | After | Time reduction |
| --- | ---: | ---: | ---: |
| CPU HT cleanup encode, 64 x 64, reused workspace | 28.509 us | 22.891 us | 19.7% |
| CPU whole-image lossless HT encode, Kodak `kodim23`, 768 x 512 RGB8 | 8.070 ms | 7.243 ms | 10.2% |
| CPU whole-image lossy HT encode, same image | 59.439 ms | 58.298 ms | 1.9% |
| Metal GPU decode, 16 distinct 1024 x 1024 RGB8 9/7 images | 8.834 ms | 8.220 ms | 7.0% |
| Metal GPU decode, 16 distinct 640 x 480 RGB8 9/7 images | 2.679 ms | 2.476 ms | 7.6% |

CPU numbers average two alternating baseline/candidate rounds. The cleanup
benchmark also averages two coefficient seeds, with 30 Criterion samples, a
one-second warm-up, and a two-second measurement target. Whole-image encode
uses 15 samples, a half-second warm-up, and a one-second target. The retained
changes batch MEL output and use 64-bit VLC/MagSgn reservoirs with bulk byte
packing. A separate packed-popcount change improves the two dense CPU cleanup
decode microbenchmarks by 0.4% and 1.8%. Those kernel-only numbers precede the
whole-image decode continuation below.

Metal numbers are the median of three alternating rounds, each with a
two-second warm-up and 21 GPU timestamp samples. The existing
`decode_stage_profile` test uses generated distinct-image batches. Its staged
timings attribute about 15–18% less time to MagSgn and about 7% less to IDWT.
The changes expose independent MagSgn reads, replace general power-of-two
scaling with exact exponent adjustment for the validated irreversible case,
and remove unnecessary halo predicates from the fused 9/7 kernel. The GPU
boundary includes cleanup, IDWT, and output stores; it excludes host parsing,
preparation, and readback.

Representative commands:

```bash
cargo bench -p j2k-native --bench tier1_bitplane -- \
  htj2k_cleanup_encode/encode_64x64_reused --sample-size 30 \
  --warm-up-time 1 --measurement-time 2 --nresamples 10000 --noplot
J2K_REQUIRE_METAL_RUNTIME=1 cargo test --profile gpu-quick -p j2k-metal --lib -- \
  engine::tests::decode_stage_profile::decode_stage_profile \
  --exact --include-ignored --nocapture --test-threads=1
```

Full-decode checks covered all 24 lossless Kodak images, the 16 medical
codestreams described below, eight WSI tiles, and four lossy VC08 frames. Alternating
whole-image measurements were mostly unchanged or noisy, so the GPU gains
above apply to the measured 9/7 batches, not to every image or decode API.
Register-only 9/7 kernels, larger paired VLC lookup tables, and wider Metal
encoder reservoirs were tested and rejected when they slowed their workloads.

### CPU whole-image decode continuation

The retained CPU changes write eligible cleanup-only blocks directly as final
floating-point sub-band coefficients, fuse the last horizontal wavelet lift
with its interleaved output store (and 9/7 row scaling), and let LLVM vectorize
native sample packing. The internal direct-output path requires no ROI shift,
a width divisible by four, and the existing SIMD magnitude bound. Other
blocks use the established path; public scalar code-block helpers retain
their transactional error behavior.

The comparison below uses `release-bench` builds on the same M4 Pro and Rust
1.99.0. Both include the same concurrent wavelet scratch accounting/reuse
changes and the earlier optimizations above. Each image has 31 timed samples
after warm-up, in three alternating baseline/candidate rounds. Corpus totals
sum the median latency of each image; they are not parallel batch wall times.
The fresh-decoder boundary includes parsing, decoding, and native output
allocation/packing, with input file I/O outside the timer.

| Workload | Before | After | Time reduction |
| --- | ---: | ---: | ---: |
| 16 medical images, serial, fresh decoders | 83.605 ms | 77.451 ms | 7.4% |
| 16 medical images, automatic threading, fresh decoders | 26.438 ms | 24.370 ms | 7.8% |
| 16 medical images, automatic threading, reused decoders | 26.540 ms | 24.159 ms | 9.0% |
| 58 lossless WSI tiles, serial, fresh decoders | 105.475 ms | 101.689 ms | 3.6% |
| 24 lossless Kodak images, serial, fresh decoders | 64.188 ms | 61.316 ms | 4.5% |
| 4 lossy Kodak images, serial, fresh decoders | 13.235 ms | 12.526 ms | 5.4% |
| MG1, automatic threading, fresh decoder | 8.742 ms | 7.846 ms | 10.2% |
| SC1, automatic threading, fresh decoder | 3.415 ms | 3.149 ms | 7.8% |
| 816-frame mixed series, 12 frame workers | 316.469 ms | 300.276 ms | 5.1% |

Automatic threading on the smaller lossless images was approximately
unchanged: the WSI total fell 1.4%, and the lossless Kodak total rose 0.4%.
The initial four-image lossy automatic-threading result was noisy; a focused
repeat with eight alternating rounds and 81 samples per image reduced their
total from 7.4565 to 7.0305 ms (5.7%). The series row repeats each of the 102
codestreams eight times, decodes serially inside each of 12 frame workers,
and reports the median of three alternating rounds of five timed samples.

The existing `jp2k_dicom_bench` was run with
`J2K_DICOM_BENCH_REPEATS=31`, `RAYON_NUM_THREADS=12`, and
`J2K_DICOM_BENCH_MODE=serial` or `parallel`; reused rows additionally set
`J2K_DICOM_BENCH_REUSE_DECODER=1`. Series rows use mode `series`, repeats `5`,
`J2K_DICOM_BENCH_THREADS=12`, and `J2K_DICOM_BENCH_SERIES_REPEAT=8`.

All 16 medical images matched native-depth reference samples exactly,
including signed 16-bit and unsigned 12-bit cases. Both builds also matched
OpenHTJ2K 0.19.0 and OpenJPH 0.31.0 exactly on all 24 lossless Kodak images
and 58 WSI tiles through `jp2k_batch_compare`. The four lossy Kodak images
had maximum absolute differences of one against both references, unchanged
from the baseline. Native unit coverage includes 1–31-bit direct coefficient
output, midpoint modes, strided guards, fallback eligibility, error recovery,
and bitwise wavelet parity at odd dimensions. Worker-queue and per-worker
scratch-slot scheduling experiments were reverted after large-image
regressions.

## HTJ2K DICOM decode development run - 2026-10-09

This run measured the CPU decode, tile-batch, owned-batch, and Metal changes
listed under 0.13.0 in the changelog, against an export of `8b608120`
(0.12.0 code; later commits changed only links) built the same way. Both used Cargo `release`, Rust 1.99.0, on an
Apple M4 Pro (12 CPU cores, 16 GPU cores, 48 GB) running macOS 27.0. The
candidate was an uncommitted working tree. Each cell is the minimum of three
interleaved baseline/candidate rounds; the host was otherwise busy during
part of the run, so absolute times are noisy, but every comparison ran both
builds back to back.

Inputs:

- 16 HTJ2K codestreams from the `openjphjs` test fixtures of
  `cornerstonejs/codecs` (MIT), encoded by OpenJPH from DICOM sample images:
  CT, MR, NM, XA, CR/DX (RG), secondary capture, mammography (MG1), and
  ultrasound. Eleven have raw reference pixels. The other ten decoded outputs
  matched theirs byte for byte; MG1's reference has a different height from
  its codestream and was not compared.
- 58 tiles (512 x 512 RGB, 8-bit, HTJ2K lossless RPCL, 4 levels) extracted
  from a local WSI DICOM that an older j2k build wrote. That build left
  `Rsiz` bit 14 clear, which OpenJPH rejects, so the extracted copies set it.
- OpenHTJ2K 0.19.0 and OpenJPH 0.31.0 from the pinned reference scripts.

Commands:

```bash
J2K_DICOM_BENCH_REPEATS=21 J2K_DICOM_BENCH_RAW_DIR=<raw-dir> \
  cargo run --release -p j2k-compare --bin jp2k_dicom_bench -- <codestream-dir>
J2K_BATCH_COMPARE_MAX_ABS_DIFF=0 J2K_BATCH_COMPARE_REPEATS=5 [J2K_BATCH_COMPARE_THREADS=1] \
  cargo run --release -p j2k-compare --bin jp2k_batch_compare -- <wsi-tile-dir> 58
J2K_METAL_BENCH_REPEATS=9 \
  cargo run --release -p j2k-metal --example dicom_decode_bench -- <codestream-dir>
open_htj2k_dec -i <file> -o <out>.pgx -iter 21 -num_threads {1,12}
```

`jp2k_dicom_bench` parses each codestream into a new `J2kDecoder` and calls
`decode_native_components`, with `CpuDecodeParallelism::Serial` and then
`Auto`. OpenHTJ2K's `-iter` timing also parses each iteration and writes
its output once. Times are milliseconds.

| File | Size | Serial before | Serial now | OpenHTJ2K 1 thread | `Auto` before | `Auto` now | OpenHTJ2K 12 threads |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CT1 | 512 x 512, 16-bit | 1.61 | 0.85 | 0.79 | 1.31 | 0.45 | 0.76 |
| CT2 | 512 x 512, 16-bit | 1.63 | 0.85 | 0.77 | 1.34 | 0.44 | 0.72 |
| MR1 | 512 x 512, 16-bit | 1.67 | 0.87 | 0.80 | 1.33 | 0.46 | 0.75 |
| NM1 | 256 x 1024, 16-bit | 1.56 | 0.79 | 0.73 | 1.30 | 0.43 | 0.72 |
| MR2 | 1024 x 1024, 16-bit | 6.33 | 3.44 | 2.97 | 4.32 | 1.09 | 1.55 |
| XA1 | 1024 x 1024, 16-bit | 6.29 | 3.31 | 2.85 | 4.11 | 1.02 | 1.58 |
| RG1 | 1841 x 1955, 16-bit | 23.31 | 12.93 | 10.98 | 14.20 | 2.74 | 4.46 |
| SC1 | 2048 x 2487, 16-bit | 31.58 | 18.10 | 15.24 | 18.02 | 3.81 | 5.39 |
| MG1 | 3064 x 4774, 16-bit | 94.69 | 52.19 | 43.39 | 56.02 | 9.57 | 14.60 |
| US1 (RCT) | 640 x 480 RGB, 8-bit | 3.93 | 2.26 | 1.38 | 3.55 | 1.17 | 1.32 |

Decoding all 16 files 8 times each on 12 threads, one frame per thread with
serial decode, took 221 ms before and 133 ms now.

The 58 WSI tiles through `decode_tiles_into` (Rgb8) took 212 ms before and
153 ms now on one worker (OpenHTJ2K 152 ms, OpenJPH 225 ms), and 60 ms before
and 30 ms now with the default worker count (OpenHTJ2K 21 ms, OpenJPH 26 ms).
The external decoders run one tile per thread. In this run j2k still admitted
at most four concurrent decodes and parallelized inside each.

A follow-up run the same day replaced that fixed four with a limit scaled to
host memory (48 on this host, so the default worker count became the 12
threads). Another build was compiling throughout (load average 9 to 13), so it
was compared only with an otherwise identical four-worker build, interleaved,
minimum of three rounds. The 58 WSI tiles took 39.2 ms with four workers and
29.0 ms scaled (OpenHTJ2K 27.6 ms, OpenJPH 36.7 ms in the same run).
`CpuBatchDecoder` batches of 16 copies went from 5.5 to 4.3 ms (CT1), 5.6 to
3.6 ms (MR1), 19.1 to 13.3 ms (US1), 19.5 to 14.1 ms (WSI tile), 85.6 to
75.0 ms (SC1), and 191 to 174 ms (MG1). RG1 to RG3 measured 3 to 11% slower,
within the up to 25% spread of the unchanged single-image decodes in the same
run. Images large enough to fill every core by themselves gain little from
more workers. The CPU batch output equaled the Metal batch output byte for
byte for every file.

`dicom_decode_bench` decodes to `Gray16` or `Rgb8` with `J2kDecoder`
(`decode_into`), with explicit Metal (`decode_request_to_device`), and as
owned batches of 16 copies of the same input with `CpuBatchDecoder` and
`MetalBatchDecoder` (preparation excluded). Metal output was read back outside
the timers and equaled the CPU output byte for byte for every file, single
and batch.

| File | CPU before | CPU now | Metal before | Metal now | CPU batch before | CPU batch now | Metal batch before | Metal batch now |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CT2 | 1.37 | 0.48 | 0.54 | 0.58 | 7.21 | 4.63 | 2.37 | 0.98 |
| MR2 | 4.60 | 1.15 | 1.03 | 0.97 | 27.20 | 13.63 | 1.79 | 1.80 |
| XA1 | 4.38 | 1.06 | 1.43 | 0.84 | 25.87 | 13.22 | 1.84 | 1.70 |
| RG1 | 15.53 | 3.01 | 2.54 | 2.42 | 103.46 | 42.47 | 5.27 | 5.15 |
| SC1 | 20.73 | 3.91 | 2.83 | 2.58 | 133.40 | 57.38 | 7.11 | 7.42 |
| MG1 | 60.28 | 10.78 | 7.55 | 7.29 | 402.47 | 154.47 | 22.94 | 23.07 |
| US1 (RCT) | 2.00 | 1.15 | 1.89 | 1.76 | 20.96 | 12.78 | 1.20 | 1.85 |
| WSI tile | 2.12 | 1.08 | 1.33 | 1.28 | 20.69 | 13.75 | 2.46 | 1.92 |

The Metal batch route did not change; its small-image cells moved between
about 1.0 and 2.4 ms in both builds from run to run. Before this change the
explicit Metal `Gray16` decode of the signed CT1 and CT2 inputs did not match
the CPU (see the changelog), so the "Metal before" CT2 time is for that wrong
output. The batch route was already correct for them.

These results are from a development tree and have not been reproduced on the
release runners. The `Auto` routing tables were measured against the 0.12.0
CPU decoder, so some Metal routes they select may now be slower than the CPU;
rerun the routing suite before relying on them.

## HTJ2K block decoder and streamed wavelet run - 2026-10-09

A second run the same day measured the changes made after the section above:
the vector `MagSgn` step, the fused per-pair VLC and `MagSgn` decode, the
skipped empty quads, the vectorized coefficient conversion, and the streamed
inverse wavelet transform. Same host, toolchain, and inputs as above. Other
work kept the host busy (load average 9 to 14 in these runs, briefly far
higher earlier), so builds were compared interleaved, minimum of five rounds
of 15 repeats per file, rather than against the absolute times above.

`jp2k_dicom_bench`, serial decode, sum over the 16 files, milliseconds. The
first step starts from the build the section above calls "now".

| Step | Serial total | Step change |
| --- | ---: | ---: |
| Skip quads with no significant sample | 143.7 / 148.3 | -3.1% |
| Vector `MagSgn` (two quads per step) | 129.5 / 146.9 | -11.8% |
| Vectorized conversion, no per-block zeroing | 132.6 / 134.6 | -1.5% |
| VLC and `MagSgn` fused per quad pair | 111.2 / 129.7 | -14.3% |
| Streamed single-thread wavelet transform | 108.4 / 113.1 | -4.2% |

Each row shows the new and the previous build from that step's own
interleaved run; load differed between runs, so totals differ between rows.
Most files improved at each step; the few that did not moved by about 2% at
most, within run-to-run noise. Every lossless output still matched its
reference.

The final build, the build before the first step, and OpenHTJ2K 0.19.0
(`-iter 11 -num_threads 1`) measured alternately at one thread under the
same load, minimum of five rounds, milliseconds:

| File | Before | Now | OpenHTJ2K |
| --- | ---: | ---: | ---: |
| MG1 | 63.03 | 42.42 | 47.61 |
| RG1 | 15.70 | 10.69 | 13.68 |
| SC1 | 21.75 | 14.85 | 18.80 |
| MR2 | 4.17 | 2.81 | 3.82 |
| CT1 | 1.08 | 0.70 | 1.09 |
| US1 (RCT) | 2.87 | 1.96 | 2.07 |

Before these steps j2k was 16 to 20% behind OpenHTJ2K on the largest files.
Under `Auto` the 16-file total went from 39.5 to 34.0 ms (MG1 12.7 to 11.1
ms); streaming the parallel wavelet bands as per-thread strips was neutral
there (34.8 against 35.2 ms) and is kept because it shares the serial kernel
and needs a quarter of the scratch or less.

The 58 WSI tiles through `decode_tiles_into` with the default worker count,
interleaved against the build before the memory-scaled worker limit: 43.5 ms
before, 21.3 ms now; OpenHTJ2K took 28.2 and 30.2 ms and OpenJPH 37.1 and
36.6 ms in the same rounds. `CpuBatchDecoder` batches of 16 copies against
that same build: CT1 6.5 to 4.0 ms, US1 17.8 to 10.4 ms, RG1 59.5 to 44.7 ms,
MG1 228 to 177 ms. Explicit Metal decodes did not change, and Metal and CPU
outputs matched byte for byte, single and batch, for every file.

Two experiments were measured and not kept: wavelet bands four times larger
(8.6% slower; the 128 KiB band stays in L1) or half as large (no change),
and allocating fresh buffers from the zeroed allocator instead of filling
them (no measurable change, since freed blocks are recycled and must be
cleared anyway).

### Follow-up measurements, same day

Three more changes were measured and not kept:

- `MagSgn` in 16-bit lanes, both quads in one vector, for blocks whose
  samples code at most 14 bits: 16-file serial total 99.5 against 99.6 ms,
  and US1 (all blocks eligible) 4% slower. With the fused decode the vector
  step already overlaps the VLC chain, and widening results back to 32 bits
  cost what the narrower arithmetic saved.
- Filling large decode buffers from every Rayon thread (`par_extend`)
  instead of `resize`: `Auto` total 33.7 against 39.9 ms, 18% slower.
  Decoding through one reused decoder instead of a new one per frame changes
  `Auto` and serial totals by under 1%, so these fills are memory traffic,
  not first-touch page faults, and one thread's `memset` already runs near
  memory bandwidth. Packing MG1's 14.6 million samples to native bytes alone
  takes 1.47 ms on one thread and 1.26 ms on all of them.
- Reusing the HT payload buffer between one-shot Metal decodes: MG1 6%
  faster, RG1 and SC1 29 to 40% slower, interleaved.

Explicit Metal single-image decode (`J2K_METAL_BENCH_ONLY=metal-single` in
`dicom_decode_bench`), a new `J2kDecoder` per frame against one decoder
reused across frames, interleaved, minimum of five rounds of 21, ms:

| File | New decoder | Reused decoder |
| --- | ---: | ---: |
| CT1 | 0.63 | 0.71 |
| MR2 | 0.99 | 1.07 |
| RG1 | 2.65 | 2.15 |
| SC1 | 2.88 | 2.33 |
| MG1 | 9.04 | 4.90 |
| US1 (RCT) | 1.83 | 1.21 |

The reused decoder keeps its prepared plan and the cached device payloads,
so the difference is per-plan setup, which grows with the coded payload:
MG1's is 12.9 MB. On the host it is copied three times before the GPU reads
it (per-block collection in the native plan, per-group concatenation in
`prepare_ht_sub_band_groups`, which also leaves the plan holding the payload
twice, and the upload into a new shared buffer). The ignored test
`single_image_step_profile` attributes GPU time by encoding one step kind
five times: a 3072 x 4096 16-bit image spends about 1.4 ms in HT decode,
1.1 ms in the inverse wavelet transform, and 0.3 ms in the store; a
512 x 512 one about 0.26, 0.03, and 0.0 ms, the rest being per-decode setup
and submission.

`BackendRequest::Auto` sent US1 (640 x 480 `Rgb8`) to Metal at 1.8 to 2.0
ms while the CPU took 1.4 to 1.5 ms, and kept every `Gray16` image on the
CPU, including SC1 and MG1, where one-shot Metal measured faster under the
same load. The routing tables predate these CPU changes; regenerate them
through the hardware routing run described below before relying on them.

## Metal decode into a caller-owned buffer - 2026-10-09

`J2K_METAL_BENCH_ONLY=metal-into` in `dicom_decode_bench` decodes each file
through one `J2kDecoder` and one `MetalBackendSession`, alternating a
`decode_request_to_device_with_session` into a new surface with a
`decode_into_metal_with_session` into one reused shared buffer of the exact
image size. Same machine and corpus as above, load average 10 to 20 from
other work, minimum over five runs of 21 decodes each, ms. The reused buffer
equaled the surface's bytes in every run.

```bash
J2K_METAL_BENCH_ONLY=metal-into J2K_METAL_BENCH_REPEATS=21 \
  cargo run --release -p j2k-metal --example dicom_decode_bench -- <codestreams>
```

| File | Format | New surface | Reused buffer |
| --- | --- | ---: | ---: |
| CT1 | `Gray16` | 0.51 | 0.47 |
| MR2 | `Gray16` | 0.73 | 0.65 |
| RG1 | `Gray16` | 1.52 | 1.31 |
| SC1 | `Gray16` | 1.89 | 1.63 |
| MG1 | `Gray16` | 4.75 | 4.04 |
| US1 (RCT) | `Rgb8` | 1.19 | 0.60 |

Into a buffer, `Rgb8` goes through the color store that batch destinations
use, not the surface route's color execution. A first version that ran session decodes
outside the session's runtime built a new runtime per call, with empty
buffer pools, and took 9.2 ms for MG1.

## Metal single-image plans with referenced payloads - 2026-10-09

`J2kDecoder` now plans HTJ2K images with code-block payloads referenced into
one copy of the codestream. `J2K_METAL_BENCH_ONLY=metal-single` in
`dicom_decode_bench` (a new decoder per frame, so every frame builds its plan)
compared this build against commit `91d2c412`, both Cargo `release` with line
tables, on the machine and corpus above. Minimum of seven interleaved rounds of
21 decodes, load average about 6, ms:

| File | Format | `91d2c412` | Referenced payloads |
| --- | --- | ---: | ---: |
| CT1 | `Gray16` | 0.79 | 0.70 |
| MR2 | `Gray16` | 0.91 | 0.84 |
| RG1 | `Gray16` | 2.38 | 2.17 |
| SC1 | `Gray16` | 2.58 | 2.40 |
| MG1 | `Gray16` | 7.49 | 6.76 |
| US1 (RCT) | `Rgb8` | 1.33 | 1.34 |

Metal output equaled the CPU output byte for byte for all 16 codestreams,
single and batch. Over 60 MG1 frames through new decoders, the process peaked
at 393 MB of memory footprint (165 MB resident) against 666 MB (423 MB), since
the runtime's HT execution cache now drops entries of dropped plans.

Sampling MG1 frames through new decoders (`sample`, `release-bench` profile)
attributed the remaining main-thread time to waiting on the GPU (66%), writing
payload bytes into new shared buffers (16%), and parsing the codestream into
the plan (11%, mostly tag trees). Command-buffer timestamps put GPU execution
at about 3.9 ms and driver scheduling at 0.6 to 0.8 ms, with or without a
reused decoder. Writing 13 MB in 2.7 KB pieces into a new shared Metal buffer
took 1.0 ms, and 0.31 ms into one already written, so about 0.7 ms of the
upload is first-touch page faults that buffer reuse would remove.

## Metal GPU kernels and HT input heap - 2026-10-09

Same machine and corpus. The ignored test `single_image_step_profile`
(`J2K_STEP_PROFILE_FILES`, `J2K_STEP_PROFILE_INTO`) attributes GPU time by
encoding one step kind five times while decoding into a reused destination.
Before these changes, MG1 spent 1.83 ms in HT decode (0.90 ms of it zeroing
the coefficient arena and 0.76 ms in the MEL/VLC kernel), 1.79 ms in the
inverse wavelet transform, and 0.48 ms in the store. The zero fill wrote one
4-byte word per thread; the two-pass wavelet kernels were compute-bound by
per-sample band selection, and the vertical pass ran one threadgroup per
16-column strip down the whole height. The store is near memory bandwidth;
the 1.2 ms a surface decode attributes to it is mostly the new 29 MB surface.

`J2K_STEP_PROFILE_COMPARE_IDWT` times the transform with the new tile
kernels turned off in the same binary, generated 16-bit images, ms:

| Image | Two-pass IDWT | Tile IDWT | Two-pass total | Tile total |
| --- | ---: | ---: | ---: | ---: |
| 1024 x 1024, 5/3 | 0.19 | 0.10 | 0.61 | 0.53 |
| 3072 x 4096, 5/3 | 1.51 | 0.79 | 3.17 | 2.52 |
| 1024 x 1024, 9/7 | 0.24 | 0.15 | 0.69 | 0.61 |
| 3072 x 4096, 9/7 | 1.86 | 1.31 | 3.57 | 3.00 |

Command-buffer timestamps showed a new decoder's MG1 decode spending about
1 ms more GPU time than a reused one, and 0.2 ms more driver scheduling,
besides 0.7 ms of CPU page faults writing the payload into a new buffer;
allocating HT inputs from a runtime heap removes all three.

`dicom_decode_bench` with `J2K_METAL_BENCH_ONLY=metal-single`, minimum of
five interleaved rounds of 21 decodes against the build before these changes,
load average 12 to 17, ms:

| File | New decoder before | New decoder now | Reused decoder before | Reused decoder now |
| --- | ---: | ---: | ---: | ---: |
| CT1 | 0.75 | 0.64 | 0.51 | 0.43 |
| MR2 | 1.07 | 0.84 | 0.72 | 0.59 |
| RG1 | 2.49 | 1.80 | 1.52 | 1.18 |
| SC1 | 2.67 | 2.10 | 1.92 | 1.48 |
| MG1 | 7.66 | 5.15 | 4.82 | 3.57 |
| US1 (RCT) | 1.78 | 1.20 | 1.18 | 1.00 |

All 16 codestreams still decoded byte-identical to the CPU, single and
batch. Over 60 MG1 frames through new decoders, peak memory footprint was
401 MB against 394 MB before.

## Metal zero layouts, batched tiles, and output heap - 2026-10-10

Same machine and corpus, load average 3 to 6 unless noted.

Code-block jobs cover 98.8% of MG1's HT coefficient arena, 98.3% of SC1's,
95.3% of XA1's, and all of RG1's, MR2's, and CT1's. Zeroing only the rest
took the `single_image_step_profile` zero-fill cost for MG1 from 0.26 ms to
nothing measurable; MG1's GPU time decoding into a reused destination was
2.67 ms, against 4.23 ms before the kernel work of the previous section.

`J2K_METAL_BENCH_ONLY=metal-batch` decodes 16 copies with `MetalBatchDecoder`
into codec-owned output and, interleaved, into one reused caller-owned
buffer (`decode_prepared_group_into`). Before the output heap, owned output
cost twice the decode: MG1 22.7 ms owned against 10.7 ms into a reused
buffer, SC1 7.4 against 3.1, RG1 5.7 against 2.5. With the heap turned off
and on in the same binary, ms:

| Batch of 16 | Owned, new buffers | Owned, heap | Reused buffer |
| --- | ---: | ---: | ---: |
| MG1 | 21.3 to 22.2 | 11.7 to 12.3 | 11.5 to 12.4 |
| SC1 | 6.8 to 7.4 | 3.2 to 4.0 | 3.1 to 3.2 |

Single-image decodes with `J2K_METAL_BENCH_ONLY=metal-single`, minimum of
five interleaved rounds of 21, before and after the output heap, ms:

| File | New decoder before | New decoder after | Reused decoder before | Reused decoder after |
| --- | ---: | ---: | ---: | ---: |
| CT1 | 0.455 | 0.430 | 0.417 | 0.412 |
| MR2 | 0.658 | 0.593 | 0.554 | 0.539 |
| RG1 | 1.499 | 1.376 | 1.122 | 0.991 |
| SC1 | 1.780 | 1.577 | 1.388 | 1.215 |
| MG1 | 4.369 | 3.856 | 3.169 | 2.667 |
| XA1 | 0.665 | 0.614 | 0.549 | 0.538 |
| US1 (RCT) | 1.090 | 1.081 | 0.970 | 1.009 |

Reading HT block coding from the parsed header instead of inspecting the
codestream again, with an iterative tag-tree walk, took a new decoder's MG1
frame from 3.83 to 3.73 ms and changed the others by under 2%. Over 60 MG1
frames through new decoders the process peaked at 407 MB of memory footprint
(401 MB before); repeated 16-image MG1 batches peaked at 1.36 GB, which
includes the 512 MiB output heap. All 16 codestreams decoded byte-identical
to the CPU, single and batch.

Runs while Spotlight indexed the build directory (load average near 30)
varied by up to 5x between repetitions and are not reported.

Running the last wavelet level and the grayscale store as one kernel, so the
level's float plane is never written, then took single-image decodes from
(new decoder, reused decoder, ms): MG1 3.75 and 2.60 to 3.35 and 2.19, RG1
1.30 and 0.95 to 1.20 and 0.85, SC1 1.53 and 1.16 to 1.42 and 1.04, MR2 0.59
and 0.51 to 0.57 and 0.50, CT1 0.43 and 0.39 to 0.42 and 0.38. A test decodes
5/3 and 9/7 images to every gray output kind (`u8`, `u16`, signed `i16`, and
signed 8-bit widened to `u16`) through the fused kernel and requires the CPU's
bytes.

The `metal-batch` batches above hold 16 copies of one codestream, which share
one prepared plan, so the batch decodes once and stores the result 16 times.
`J2K_METAL_BENCH_DISTINCT` builds 16 distinct frames instead (the decoded
image with a per-frame offset, re-encoded as lossless HTJ2K). Fusing the last
level, the component store, and the group pack for such groups, with the
fusion turned off and on in the same binary, minimum of three runs, ms:

| 16 distinct frames | Separate store and pack | Fused |
| --- | ---: | ---: |
| SC1 | 24.6 | 11.9 |
| RG1 | 17.2 | 12.6 |
| MR2 | 3.62 | 2.62 |
| CT2 (512 x 512) | 1.22 | 0.99 |
| MG1 | 117 | 62 |

Sixteen distinct MG1 frames failed as a group error before: their stacked
coefficient arena needed one 936 MB allocation, over the 512 MiB cap. Groups
are now decoded in runs whose arenas fit (nine MG1 frames). Large frames gain
little from batching even so: decoding the 16 MG1 frames in runs of 1, 2, 4,
and 8 took 56, 58, 55, and 61 ms, against about 35 ms for 16 single-image
decodes, because a batch keeps every image's scratch planes (more than 1 GB
here) until it completes, beyond what the buffer pools retain. Small frames
gain the most (MR2: 6.5 ms in runs of 1, 3.0 ms in runs of 8).

## CPU JPEG safe-SIMD development run - 2026-08-12

The CPU JPEG safe-SIMD refactor was measured before and after on the same Apple
M4 Pro host running macOS 26.5.2 build `25F84`, with Rust/Cargo 1.96.0 and LLVM
22.1.2. Criterion used 95% confidence, 50 samples, a three-second warm-up, and
a ten-second measurement. Benchmark JPEGs were generated deterministically
before timing; setup decoded every case to validate geometry and output
checksums. The commands were:

```bash
cargo bench -p j2k-jpeg --bench decode_cpu -- --save-baseline safe-simd-before
cargo bench -p j2k-jpeg --features bench-internals --bench micro -- --save-baseline safe-simd-before
cargo bench -p j2k-jpeg --bench decode_cpu -- --baseline safe-simd-before
cargo bench -p j2k-jpeg --features bench-internals --bench micro -- --baseline safe-simd-before
```

The final end-to-end decode comparison was:

| Case | Before 95% interval | After 95% interval | Criterion change interval |
| --- | ---: | ---: | ---: |
| Gray8 512 x 512 | 1.2084–1.2649 ms | 1.1740–1.1822 ms | -12.903% to -5.046% |
| RGB8 512 x 512 4:4:4 | 2.7072–2.8042 ms | 2.6190–2.6301 ms | -5.201% to -2.813% |
| RGB8 512 x 512 4:2:2 | 2.4961–2.5115 ms | 2.4447–2.4634 ms | -2.473% to -1.302% |
| RGB8 512 x 512 4:2:0 | 1.9161–1.9243 ms | 1.8769–1.8839 ms | -2.651% to -1.946% |
| RGB8 257 x 263 4:2:0 | 508.92–510.58 us | 500.94–502.03 us | -1.652% to -1.149% |
| RGB8 512 x 512 4:2:0 rows | 1.9360–1.9420 ms | 1.8981–1.9022 ms | -2.403% to -1.966% |

No affected microbenchmark exceeded the 2% regression criterion; every final
95% confidence interval was lower than the saved baseline. This includes general and
bottom-half-zero NEON IDCT, 255/256/cropped 4:2:0 row pairs, gray and planar-RGB
rows, and 255/256/unaligned YCbCr rows. An intermediate build did confirm
1.7–3.3% regressions in the color decode cases at doubled measurement time.
Generated-code inspection traced those to per-chunk bounds checks and a lost
4:2:0 helper inline. Safe array chunking plus a fixed ten-byte overlapping-load
leaf restored the code shape; safe dispatch was not reverted.

These runs were on an uncommitted working tree, not a release commit. Some
unrelated microbenchmarks also got faster, so the size of each improvement is
partly host state and code layout; what the run shows is that AArch64 had no
regression beyond the threshold. Native x86-64 results follow.

### Native Windows AVX2 validation

The same refactor was measured natively with the MSVC target on Windows 11 Pro
10.0.22631, an AMD Ryzen 7 5800X3D with AVX2, and Rust/Cargo 1.96.0. The Linux
VM on that host was used only as the remote transport; every test and timed
process was a native `x86_64-pc-windows-msvc` executable. Baseline and
candidate used the same benchmark sources, `fearless_simd 0.5.0`, shared build
settings, and deterministic inputs. The final paired run pinned each process
to the same logical processor and used High process priority. Criterion used
the same 95% confidence, 50 samples, three-second warm-up, and ten-second
measurement settings. The sole inconclusive 4:4:4 case was repeated with a
20-second measurement as required.

| Case | Before 95% interval | After 95% interval | Criterion change interval |
| --- | ---: | ---: | ---: |
| Gray8 512 x 512 | 1.8990–1.9206 ms | 1.8712–1.8975 ms | -2.192% to -0.449% |
| RGB8 512 x 512 4:4:4 (20 s repeat) | 4.7098–4.7265 ms | 4.7236–4.7483 ms | +0.003% to +0.495% |
| RGB8 512 x 512 4:2:2 | 4.2607–4.2973 ms | 4.1895–4.2214 ms | -2.295% to -1.171% |
| RGB8 512 x 512 4:2:0 | 4.8575–4.8962 ms | 4.7957–4.8259 ms | -1.825% to -0.862% |
| RGB8 257 x 263 4:2:0 | 1.2906–1.3418 ms | 1.2740–1.2827 ms | -4.832% to -0.826% |
| RGB8 512 x 512 4:2:0 rows | 4.8291–4.8550 ms | 4.7672–4.8029 ms | -1.646% to -0.717% |

Every affected AVX2 microbenchmark also cleared the 2% upper-bound criterion.
The upper confidence bounds were +0.329% for general AVX2 IDCT, -0.829% for
4:2:0 upsampling, -0.445% to -1.065% for full row pairs, +1.000% for cropped
4:2:0, +0.662% for gray rows, +1.366% for planar RGB rows, and +0.176% or
better for full, tail, and unaligned YCbCr rows.

An additional unchanged reduced 2x2 scalar-IDCT diagnostic measured a
confirmed +5.119% to +5.971% shift at an absolute candidate time of
11.780–11.859 ns. Baseline and candidate source and generated x86-64 function
bodies are identical (235 instructions), so the shift comes from code placement
and caching, not extra work. That function is not part of the SIMD change, and
the end-to-end decode results above show no regression. Like the AArch64 runs,
these were on an uncommitted working tree.

### `fearless_simd` 0.7 upgrade validation - 2026-08-13

The staged 0.10.0 dependency graph resolves `fearless_simd 0.7.0` in the
workspace and every affected fuzz lockfile. `cargo xtask release-cpu` passed
with that version on the Apple M4 Pro AArch64 host, the Linux x86-64 VM, and
the native Windows x86-64 MSVC host described above.

The dependency is AArch64-only for `j2k-jpeg`. In an adjacent
version-isolation control from one otherwise unchanged intermediate source
tree on Windows, changing only the workspace requirement and lockfile between
0.5.0 and 0.7.0 caused Cargo to reuse the exact benchmark executable (SHA-256
`2e37a5ffa851964a5e3ca4fafb6f819372071ba8f033e94bb83dda297d51ea80`).
A 20-second 4:4:4 rerun of that identical binary still moved by -1.916% to
-1.069%, so the earlier apparent x86 regressions were run-to-run noise.

On AArch64, normalized inspection of the unstripped release-benchmark output
found identical instruction bodies for the compared JPEG NEON IDCT, row
conversion, and 4:2:0 kernels under 0.5.0 and 0.7.0; relocation targets,
constant-pool offsets, and whole-binary placement were excluded from that
comparison. The host was too busy (WindowServer and other interactive processes) for a
clean 0.7.0 timing run; repeated Criterion attempts had large scheduling
outliers. The 0.5.0 timings above are still the latest AArch64 numbers, and a
quiet rerun on the same host is needed before saying the 0.7.0 upgrade has no
AArch64 regression.

## How `Auto` routing thresholds are chosen

`BackendRequest::Auto` uses generated routing tables checked into the
repository; it does not calibrate on the user's machine. A workload cell is
sent to the GPU only if the CPU, hybrid, and any GPU-only route all produce
identical bytes on the same pinned external input, the hybrid median is at
least 10% faster than every alternative, and its Criterion 95% confidence
interval does not overlap theirs.

CUDA and Metal collect all six required operations with their production APIs:
full decode, ROI decode, scaled decode, batch decode, lossless encode, and lossy
encode. Each result records the commit SHA, manifest SHA-256, hardware and
driver, route label, output SHA-256, and Criterion ID. Routes where parsing,
entropy decoding, output, or codestream assembly still run on the CPU are
labelled `hybrid`.

After a hardware run, check the raw results against the manifest and the
Criterion estimates:

```bash
cargo xtask auto-routing verify \
  --evidence target/gpu-benchmark/auto-routing/evidence.json \
  --external-manifest "$J2K_AUTO_ROUTING_MANIFEST" \
  --criterion-root target/criterion \
  --out target/gpu-benchmark/auto-routing/verified.json
```

The verifier makes every routing decision itself; benchmark input cannot ask
for one. It rejects missing operations or cases, route or output mismatches,
unsafe Criterion paths, unsupported confidence levels, changed estimate files,
and commit or platform mismatches. The output hash covers the raw results, the
manifest, and every estimate used.

Accepted results and the workloads they route to the GPU are listed in
`xtask/routing-promotion-evidence.json`. Regenerate the Rust routing tables with
`cargo xtask promotion-codegen`; CI runs `cargo xtask promotion-codegen
--check` to catch stale tables. The generator checks the manifest schema,
backend, SHA-256 values, that all six operations were measured, workload
identity, limits, and duplicates. A route with no accepted result stays on the
CPU.

A local two-input Metal smoke on August 4, 2026 exercised the pipeline but was
not a representative corpus. No cells were routed to the GPU: the
decode routes were slower, and the measured lossless and lossy encode medians
were only about 4.2% and 7.8% faster than CPU. No `Auto` threshold was changed
from that diagnostic.

### Local Metal lossless routing run - 2026-09-24

This run replaced the Metal decode cells; the combined run below has since
re-measured them. Its corpus is local, not a
publication corpus: 512x512 tiles from GDC whole-slide DICOM files, decoded with
`opj_decompress`, stitched into 256x256, 640x480, 1024x1024, and 2048x2048 RGB
images and 640x480 and 2048x2048 gray images, then re-encoded as Part 1
lossless (OpenJPEG), HTJ2K lossless codestreams (OpenJPH `-reversible true`),
and HTJ2K lossless JPH files (Grok). There are 22 cases (14 decode, 8 encode);
manifest SHA-256
`217d700a9f882c4cd63b76aed250de92124d74a0bf476d33855dbbe857d6befe`.
The run used `release-bench`, candidate `c1cc3ff1` with a dirty tree, and an
Apple M4 Pro (16 GPU cores) on macOS 27.0. The verifier accepted all 76 cells
and routed 33 to the GPU. The verified artifact SHA-256 is
`69987ce1cee902e4ef664964250059021ef15b59e0828802098fbe84d192df77`.

Lossy inputs were excluded from this run. `tests/auto_routing_parity.rs` found
that Metal 9/7 decode differed from the CPU by one code value in roughly 10
samples per million on every third-party 9/7 file, while every 5/3 file matched.
Because GPU routing requires identical bytes, the earlier 9/7 Metal cells
(`metal_part1` lossy repeated, `metal_part15` lossy repeated and half-scale)
were removed. The cause was found and fixed the same day; see "Local Metal
routing run with lossy inputs" below.

The resulting Metal cells cover lossless inputs only. Repeated batches of 16 run
on Metal for HTJ2K RGB8 from 256x256 (JPH from 640x480), HTJ2K Gray8 from
640x480, Part 1 RGB8 from 640x480, and Part 1 Gray8 from 2048x2048.
Single full-image decodes run on Metal for HTJ2K from 640x480, for RGB8 and
Gray8 and for codestreams and JPH, when the source components match the output
format. Part 1 single-image decodes stay on the CPU because Metal measured 105%
to 1400% slower. ROI and half-scale decodes qualified only for HTJ2K Gray8 at 2048x2048;
Metal Auto does not route those operations for gray sources, so they stay on the CPU.
The lossy RGB8 encode threshold is now 2048x2048 pixels, where HTJ2K and Part 1
lossy encode measured 54% and 38% faster; at 640x480 they were within 4% of the CPU.

### Local Metal routing run with lossy inputs - 2026-09-24

The 9/7 mismatch was in output rounding, not in the wavelet transform. The CPU
rounds each centered sample to the nearest integer, ties to even, and only then
adds the unsigned level shift, as OpenJPEG does (`opj_lrintf(value) +
dc_level_shift`). The Metal store, pack, and inverse-colour-transform kernels
added the shift first and rounded half up. Adding 128 in f32 drops one bit of
precision, so a value just below `k + 0.5` became a tie and rounded up: every
mismatch was Metal one code value high, at a sample whose shifted CPU value was
exactly `k + 0.5`. The fix makes Metal integer output round before the shift and
compute the inverse ICT with the CPU's fused expressions. Three CPU paths
disagreed with the CPU full decode in the same way (region decode, row
streaming, and `CpuBatchDecoder`, which built integer output from unrounded
component planes), as did the CPU inverse ICT's non-SIMD tail. They now round
the same way, so a CPU region decode equals the crop of the full decode.

The first run of this corpus also exposed an unrelated Metal bug: the repeated
Part 1 batch path decoded into a recycled scratch buffer without zero-filling
code blocks that have no coding passes, so a lossy batch decoded after a
lossless one of the same geometry kept stale coefficients (39% of bytes wrong
for `gray8-2048x2048-part1-lossy`). That path now zero-fills like the
single-image, grouped, and distinct-batch paths.

The corpus is the lossless run's 22 cases plus 12 lossy decode files made from
the same stitched sources: Part 1 lossy (OpenJPEG) and HTJ2K lossy codestreams
(OpenJPH) for every RGB and gray size. There are 34 cases (26 decode, 8 encode);
manifest SHA-256
`dd6363374dccf8d6b34656f90b9557b5a01f8582b94f5014e3165dc61f1248ec`.
`tests/auto_routing_parity.rs` found no mismatch in any full, ROI, or half-scale
decode, and the bench's own parity check passed for every CPU, Metal, and Auto
route. The run used `release-bench`, candidate `c1cc3ff1` with a dirty tree,
and the same Apple M4 Pro on macOS 27.0. The verifier accepted all 124 cells and
routed 49 to the GPU. The verified artifact SHA-256 is
`66f9d83f932efb7df6cab2de0048849f09343da00915e22794b0056b29f2aa5f`; its cells
are recorded under the `metal_local_combined` source.

Every lossless cell re-measured to the same thresholds as the lossless run. The
new lossy cells are:

- Repeated batches of 16: HTJ2K RGB8 from 256x256 (46% faster at 256x256),
  Part 1 RGB8 from 1024x1024 (56%; 640x480 measured 10% slower), and HTJ2K and
  Part 1 Gray8 from 2048x2048 (62% and 18%; HTJ2K Gray8 at 640x480 was only 8%
  faster, below the 10% bar).
- Single full-image decodes: HTJ2K RGB8 and Gray8 from 640x480 (44% and 37%
  faster at 640x480, 77% and 78% at 2048x2048).

Part 1 lossy single-image decodes stay on the CPU (66% to 2400% slower on
Metal). No half-scale cell qualified: HTJ2K lossy RGB8 half-scale measured 3%
to 176% slower, so the earlier `metal_part15` half-scale cell stays removed.
ROI and half-scale qualified only for HTJ2K Gray8, which Metal Auto does not
route. The lossy RGB8 encode threshold stays at 2048x2048 pixels (Part 1 and
HTJ2K 36% and 57% faster there).

### External CUDA routing development run - 2026-08-05

The uninterrupted CUDA matrix used the same 12 external cases from
`uclouvain/openjpeg-data` commit
`39524bd3a601d90ed8e0177559400d23945f96a9` and manifest SHA-256
`f07072f5d0313c0249e2df5df2310cd5c6c5a4b3414a933537fabf2362d2065c`.
It ran all 36 cells with Cargo `release-bench`, Criterion 0.95 confidence
intervals, ten samples, a one-second warm-up, and a three-second target on an
AMD Ryzen 7 5800X3D with an NVIDIA GeForce RTX 4070 SUPER, driver 596.49,
Linux x86-64, and CUDA 13.2.

The verifier accepted every cell and routed 18 decode cells to the GPU, using
the measured output sizes as thresholds. RGB8 reversible uses CUDA for full
output from 256 x 149, ROI output from 128 x 74, and half-scale output from
1296 x 972. RGB8 irreversible uses CUDA for full output from 640 x 480 and ROI
or half-scale output from 320 x 240. Gray8 reversible uses CUDA only for full
output from 640 x 480. Gray8 irreversible uses CUDA for full output from
3323 x 891, ROI output from 1661 x 445, and half-scale output from
1662 x 446. Repeated-input batches of 16 use the full-image thresholds.

These thresholds apply only to raw Part 1 codestreams with the measured source
component/output-format pair. They do not cover JP2 color normalization, other scale factors, HTJ2K, higher depths, RGBA, distinct-input
batches, unmeasured operations, smaller output work, or shapes below either
measured dimension. All 12 encode cells stayed on CPU because CUDA-assisted
encode was slower than CPU in this end-to-end matrix.

The verified artifact's internal SHA-256, recorded beside the thresholds, is
`ded1eb045f9673e5bbe64dc873be3ba227ecb61ec11b6c9ad53653dbcc993f44`.
The raw results file SHA-256 is
`ad0b434dbd64f669d58054f4a25f9272f741bf2a689f6f70816d98fe87c02e61`;
the serialized verified file SHA-256 is
`a565d47f81ed32588e551167a91c0df68daff3d7ebd6ff8588fbe4d8ab27ac79`.
Every route produced the same output SHA-256 before timings were compared.
None of these APIs has a GPU-only route, so the comparison is CPU against
hybrid.

The run was on an uncommitted working tree based on commit
`6400fcd4c9f8cf9708563d62411eadf158f94282`. The full matrix must be rerun on
the release commit before these numbers are published as release results.

### External Metal routing development run - 2026-08-04

The full routing matrix was then run against 12 external decode/encode cases
from `uclouvain/openjpeg-data` commit
`39524bd3a601d90ed8e0177559400d23945f96a9`. The external manifest SHA-256 is
`f07072f5d0313c0249e2df5df2310cd5c6c5a4b3414a933537fabf2362d2065c`.
The run used Cargo `release-bench`, Criterion 0.95 confidence intervals, ten
samples, a one-second warm-up, and a three-second target measurement on an
Apple M4 Pro with a 16-core GPU and 48 GB RAM, macOS 26.5.2 build `25F84`, and
Metal compiler `32023.883`.

The verifier accepted all 36 workload cells and routed four to the GPU. Times
below are Criterion medians; each selected hybrid interval was entirely below
the CPU interval.

| Cell | CPU median | Hybrid median | Speedup |
| --- | ---: | ---: | ---: |
| Repeated RGB8 irreversible decode, 640 x 480, batch 16 | 68.364 ms | 43.232 ms | 36.762% |
| Repeated Gray8 irreversible decode, 3323 x 891, batch 16 | 265.409 ms | 149.856 ms | 43.538% |
| Repeated RGB8 reversible decode, 2592 x 1944, batch 16 | 2429.216 ms | 179.528 ms | 92.610% |
| RGB8 irreversible encode, 2592 x 1944 | 813.508 ms | 716.208 ms | 11.961% |

The verified artifact's internal SHA-256 is
`162a47f7a96b2be88abebc100aab672513af04895532863fa1a293660546f879`.
The raw results file SHA-256 is
`3b2ffad6fe3ebb42e2182612946a5c87ebf0b267e25c38bfb5c9d07c11aa6e7d`.
These hashes are recorded next to the thresholds in the routing code.

The batch rows reuse one encoded input 16 times, so they measure repeated
decoding of the same image, not batches of different images. `Auto` therefore
uses the GPU only for repeated Part 1 Gray8/RGB8 requests in the measured
reversible/irreversible classes and sizes.
In this Part 1 matrix, single-image, ROI, scaled, HTJ2K, higher-depth, signed,
RGBA, and unmeasured lossless/lossy cells stayed on CPU. Lossless encode and
Gray8 lossy encode also stayed on CPU; the latter measured only a 6.8%
improvement. The Part 15 matrix below adds only the HTJ2K/JPH cells it lists.

The run was on an uncommitted working tree based on commit
`6400fcd4c9f8cf9708563d62411eadf158f94282`. The full matrix must be rerun and
reverified on the release commit before these numbers are published as release
results.

### Official Part 15 routing development runs - 2026-08-07

CUDA and Metal used the same three external T.803 workloads: raw HTJ2K
`p0_04` BSET 12, JPH file 1 BSET 12, and JPH file 10 component 0 as the encode
source. The schema-2 manifest SHA-256 is
`422f40e4086b53e43f2338f97b468c869257cc23ec4899432cf019585782d48e`.
Each backend ran ten end-to-end cells covering full, ROI, scaled, and repeated
batch decode plus lossless and lossy encode. Every CPU/hybrid pair produced an
identical output hash. Neither API has a GPU-only route, so these measure the
hybrid routes.

| Backend and measured host | Cells `Auto` routes to the GPU | Hybrid speedup versus CPU | Verified artifact SHA-256 |
| --- | --- | --- | --- |
| CUDA, RTX 4070 SUPER, driver 596.49, WSL2 Linux 5.15.153.1 | Raw HT and JPH full, ROI, half-scale, and repeated batch decode (8/10) | 70.39% to 91.41% | `77370c83710ebf578139ad0bfa2608ffad989d83faec8d5eee213691290c0088` |
| Metal, M4 Pro 16-core GPU, macOS 26.5.2 build `25F84` | Raw HT half-scale, raw HT repeated batch, and JPH repeated batch decode (3/10) | 23.42% to 76.91% | `cfa66686d053bb3e2d4c8756abaf84aab65d8505a635795cefd38de53573c1f5` |

CUDA lossless and lossy encode were respectively 528.23% and 305.10% slower
than CPU. Metal lossless and lossy encode were respectively 240.00% and 58.54%
slower. Those four cells stay on the CPU. Metal raw/JPH full and ROI decode and
JPH half-scale decode also stay on the CPU because they did not meet the
routing rule. The thresholds apply only to the measured codec, container,
format, dimensions, operation, and repeat count, not to other HTJ2K/JPH
workloads.

The CUDA raw/serialized-report SHA-256 values are `0b4ac7a08e4adba0e9ed9a7983a4fb3cd9ae0cc81a2ab1162e604f13a785065a`
and `faacdfa190d0c18acdf9c24a584e285606c37af8dcd15cc649dd6eac2eaa8a10`.
The corresponding Metal values are `5a66373022b8df18decb3e60d255464f76e6bee3195e90489b61c8366fdec459`
and `ffe6697176de2c2ac716615f5b96476d610c04715fdad339857c716fb960ef06`.

Both reports record base commit `f92646d0e6f0d0ef6c1e60b60beaad29da1afd3b`
plus uncommitted changes. The full matrix must be rerun on the release commit
before these numbers are published as release results.

### Metal HTJ2K host-output encode matrix - 2026-08-08

The host-output encode route was measured separately because the Part 15
inputs above are too small for GPU coefficient preparation and HT Tier-1 to pay
for Metal setup. The
schema-2 manifest SHA-256 is
`452080d2b0611f67246450f58479354803469cc0c13d10df4a4ac866e03f90c3`.
It contains deterministic `j2k-test-support` Gray8 and RGB8 PNM inputs at
512 x 512, 1,024 x 1,024, and 2,048 x 2,048, plus two official T.803 HT/JPH
decode anchors. The lossless encode cells use Metal coefficient preparation
and HT Tier-1 with CPU packetization and final host codestream output.

Before timing, every hybrid and `Auto` codestream matched the CPU codestream
byte for byte, and each lossless codestream decoded exactly to its source PNM.
The batch-16 rows run the same host-output encode 16 times on one Metal
session. They measure that route, not the fully resident Metal-buffer batch
API.

| Format and size | Single CPU | Single hybrid | Single speedup | Batch-16 CPU | Batch-16 hybrid | Batch speedup | Fixed `Auto` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Gray8 512 x 512 | 2.057 ms | 8.850 ms | -330.23% | 32.847 ms | 137.666 ms | -319.12% | CPU |
| RGB8 512 x 512 | 5.479 ms | 9.490 ms | -73.20% | 88.428 ms | 156.564 ms | -77.05% | CPU |
| Gray8 1,024 x 1,024 | 9.409 ms | 15.746 ms | -67.35% | 152.592 ms | 236.271 ms | -54.84% | CPU |
| RGB8 1,024 x 1,024 | 25.724 ms | 16.200 ms | 37.02% | 412.913 ms | 254.853 ms | 38.28% | hybrid |
| Gray8 2,048 x 2,048 | 43.365 ms | 17.107 ms | 60.55% | 698.504 ms | 269.195 ms | 61.46% | hybrid |
| RGB8 2,048 x 2,048 | 120.778 ms | 20.623 ms | 82.93% | 1,980.952 ms | 351.347 ms | 82.26% | hybrid |

These are Criterion medians from ten samples, a one-second warm-up, a
three-second target measurement, and 95% confidence intervals. A shape was
routed to Metal only when both its single and batch-16 results were more than
10% faster and the hybrid interval did not overlap the CPU interval. There is
no GPU-only host-output route, so the comparison is CPU against hybrid. The
thresholds cover only the six measured shape/format combinations.

The SHA-256 of the decision artifact recorded next to the routing cells is
`c8defb820b55a99e94acdd5849b4597bce0a1718fd7e0d2bc0aa926bc0e130d4`.
After enabling only the qualifying cells, the full 26-cell matrix reran with
no `Auto` output mismatch; its verified artifact SHA-256 is
`c98f11c0b2a2a96853953ceee7ea672e0e5044bdb8abbd397c8c36eb82fe53b8`.
The raw results and serialized verified report SHA-256 values are
`19c1793e3647db44e01903cd619a4da0d70ab5a2b55ab53b3227c67544112090`
and `a1f898a545bfc3c29e4fbbf5197b602b59920e1c27d50c2cd85af44e168de370`.

This run used an Apple M4 Pro with a 16-core GPU and 48 GB RAM, macOS 26.5.2
build `25F84`, and Metal compiler `32023.883`. It records base commit
`f92646d0e6f0d0ef6c1e60b60beaad29da1afd3b` and dirty-worktree identity
`a84f4107ab943540a0951abefc670065b29b9809430074d5255d8bec1cf2b021`
across 2,748 tracked and untracked source paths. The matrix must be rerun on
the release commit before these numbers are published as release results.

## Older results

Older regression runs, migration comparisons, abandoned experiments, and
uncommitted-tree probes are in Git history. They are out of date and should
not be quoted.
