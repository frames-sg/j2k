# Benchmark Results

This page records benchmark commands, results, and the machines they ran on.
When a harness writes JSON or CSV output, that output is authoritative. Other
docs link here instead of copying numbers. It holds current evidence only:
when a run is superseded, remove it. Older results are in Git history.

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

`cargo xtask j2k-perf-guard --lane host` compares CPU Criterion medians with
the baseline ref `29143c8e1f00bbbe9cf5ab37b3cea882f6d52139`
(`DEFAULT_BASELINE_REF` in `xtask/src/perf_guard.rs`).

## Fair decode comparison against OpenJPEG, Grok, OpenJPH, OpenHTJ2K, libjpeg-turbo and Kakadu - 2026-11-02

Rerun after the comparator packing fix listed under "Harness rules", decode
only. j2k at `3d253077` on mini and studio, whose ARM decode code is
identical to the bench branch tip, and at the tip `84b93cd9` on cuda. The
JPEG and Kakadu rows were not rerun: they are copied from the first run, at
`70eaacaf`. The JPEG decode code did not change between the runs, and the
Kakadu rows compare command lines, which use no comparator wrapper.

All ratios are **competitor time ÷ j2k time**: above 1 means j2k is faster,
below 1 means the competitor is faster. The harness changes that made this
comparison fair are listed under "Harness rules". Competitor rows measured
before those changes have been removed from this page, except the nvJPEG2000
rows in the CUDA batch section, which carry their own caveats.

### Summary

- **Apple Silicon (M4 Mac mini, M5 Max Mac Studio):** j2k is the fastest
  decoder in every full-image workload except JPEG at high thread counts.
  - Batch HTJ2K: OpenJPH and Grok are 1.7–2.6× slower, OpenHTJ2K is
    1.2–1.4× slower, and OpenJPEG is 3.7–5.5× slower.
  - Batch Part 1: competitors are 1.2–1.8× slower.
  - 16-bit medical (lossless): on HTJ2K, OpenJPH and Grok are 1.9–2.7×
    slower, OpenHTJ2K is 1.2–1.3× slower and OpenJPEG is 3.6–4.4× slower.
    Part 1 competitors are 1.4–1.5× slower. One set goes the other way:
    OpenHTJ2K is slightly faster on CT slices at 18 threads on studio
    (0.95–0.97).
  - JPEG: j2k is 14–21% faster than libjpeg-turbo at 1 thread, and ties it at
    8–18 threads.
- **x86 (Ryzen 7 5800X3D, WSL2):** most of the lead is gone.
  - 8-bit batch HTJ2K: j2k leads OpenJPH (1.05) and Grok (1.17–1.26) at 1
    thread, but both are faster at 16 threads (OpenJPH 0.82–0.83, Grok
    0.91–0.96). OpenHTJ2K is 1.09–1.21× slower and OpenJPEG 2.0–3.1× slower.
  - j2k loses 16-bit HTJ2K to OpenJPH (0.69–0.74), Grok (0.81–0.86) and
    OpenHTJ2K (0.66–0.73).
  - libjpeg-turbo is up to 20% faster on JPEG. j2k still wins Part 1 by
    1.06–1.32×.
  - The cause is in j2k (see "Where j2k is behind").
- **Region and zoom decode of a 16384² image:**
  - j2k is 1.7–3.2× faster than Grok on HTJ2K and 1.05–1.38× on Part 1
    (geomeans).
  - OpenJPEG 2.5.4 returns wrong pixels for every partial HTJ2K region.
  - OpenHTJ2K returns wrong pixels for windows of tiled images.
  - OpenJPH has no region API.
  - Reduced-resolution requests on tiled images don't scale with threads in
    j2k, and Grok is up to 5.0× faster there.
- **Kakadu 8.4.1** (command line vs command line, whole process, first run):
  the j2k command line is faster in all 16 cases, by 1.06–1.90×.
- **Correctness:** every full-image decode by every library matched j2k:
  bit-exact on lossless, within 2 on 9/7, and exact with libjpeg-turbo.

### Machines

| Host | CPU | Threads tested | OS |
|---|---|---|---|
| mini | Apple M4, 10 cores (4P + 6E) | 1, 2, 4, 8, 10 | macOS; idle |
| studio | Apple M5 Max, 18 cores | 1, 2, 4, 8, 18 | macOS; Spotlight indexing off; other agents' builds paused by a lock |
| cuda | AMD Ryzen 7 5800X3D, 8C/16T (Zen 3, AVX2, no AVX-512) | 1, 2, 4, 8, 16 | Linux under WSL2; cells wait while a GitHub Actions job runs |

### Libraries and builds

Every C and C++ library was built from source at its latest release, with
its own upstream CMake `Release` configuration.

| Decoder | Version | Build |
|---|---|---|
| j2k | `3d253077` (mini, studio); `84b93cd9` (cuda); `70eaacaf` (JPEG and Kakadu rows) | Workspace `release` profile (fat LTO, one codegen unit), default target CPU. `3d253077`'s ARM decode code is identical to `84b93cd9`'s; the commits between them change x86-64 decode code and the encoder. |
| OpenJPEG | 2.5.4 | Upstream static library, linked through a local `[patch.crates-io]` overlay of `openjpeg-sys`. The crate's vendored 2.5.3 is compiled without `NDEBUG`, so it was not used. GCC builds get upstream's `-ffast-math`. |
| Grok | 20.4.24 | Upstream Release |
| OpenJPH | 0.33.0 | Upstream Release. Upstream has AVX2 paths on x86 and no NEON, so Mac runs use its generic C++. |
| OpenHTJ2K | 0.23.3 (speed); 0.23.4 (correctness re-check) | Upstream default `-march=native`. 0.23.4 changes only `set_col_range()` on the reuse path; the tiled-region failure below reproduces on it. |
| libjpeg-turbo | 3.2.1 | Upstream Release (NEON; NASM 3.02 SIMD on x86). Accurate integer IDCT, fancy upsampling, RGB output. |
| Kakadu | 8.4.1 | Vendor universal binary (native arm64), studio only, command line only |

### Harness rules

These apply to every decoder in `jp2k_batch_compare`. The JPEG rows come
from the matched batch in `j2k-jpeg`'s `compare` bench, which also loads
inputs before timing, gives both libraries the same workers and verifies
their output.

- **Same input:** every decoder gets the same in-memory codestream bytes,
  loaded before timing.
- **Pixel check before timing:**
  - Each decoder's output is compared with j2k's: exact for lossless, within
    3 for 9/7.
  - With `J2K_BATCH_COMPARE_KEEP_GOING`, a failure is recorded as a result,
    and that decoder is left out of the cell's timing.
- **Same output work:** every decoder writes once, directly into a buffer
  allocated the same way inside the timed region.
  - Before the first run, the Grok and OpenJPEG wrappers packed output one
    sample at a time, and the Grok, OpenJPH and OpenHTJ2K wrappers made
    extra copies. This inflated Grok by up to 61%, OpenJPEG by up to 39%,
    and OpenJPH and OpenHTJ2K by 4–8%. Both were removed, and the first full
    round, measured before that fix, was discarded.
  - For this rerun, the OpenHTJ2K and OpenJPH shims' remaining per-sample
    conversion, which branched on depth and sign for every sample, was
    replaced by row packers that every C and C++ shim shares
    (`crates/j2k-compare/src/j2k_shim_common.h`). Their allocating decode
    mode, used only by tests, was removed. OpenJPEG's full-resolution pack
    gained a plain clamp loop for unsigned samples of at most 8 bits.
  - Against the first run, OpenHTJ2K's batch HTJ2K ratio at 1 thread fell
    from 2.09–2.12× to 1.34–1.36× on the Macs and from 1.87× to 1.18× on
    cuda. OpenJPH's fell from 2.79–2.82× to 2.51–2.54× on the Macs, and on
    cuda from 1.18× to 1.05× at 1 thread and from 0.93× to 0.82× at 16
    threads. On the Macs, j2k's, Grok's and OpenJPEG's own times moved by at
    most 3% (geomean per corpus).
- **Same thread budget:**
  - The harness sizes j2k's global Rayon pool to `J2K_BATCH_COMPARE_THREADS`
    in-process, before any decode, and exits if `RAYON_NUM_THREADS` is set
    to a different value. These runs set both to the same value.
  - A one-image batch gives the whole budget to each decoder's own
    intra-image threading.
  - A larger batch splits images across workers with the same static
    chunking for everyone, and each worker decodes one image serially.
- **Statistics:**
  - 3 rounds per cell, each in a fresh process. The set order rotates
    between rounds, and the decoder order rotates between repeats inside
    each process.
  - 11 timed repeats per process on the main corpus, 7 on tiles and 16-bit,
    5 on region.
  - A decoder's time is the median over processes of each process's median.
  - The ratio is computed within each process, then the median is taken.
  - The typical spread of a cell's three ratios was 0.8% on the Macs and
    2.4% on cuda.

### Inputs

No input was encoded by j2k. JPEG 2000 sets come from OpenJPH, Kakadu and
OpenJPEG, and JPEG sets from libjpeg-turbo `cjpeg`. All use 5 decomposition
levels (6 for the region image), 64×64 code blocks, RPCL and 1 layer.

| Corpus | Images | Codings |
|---|---|---|
| Main | Kodak 24 (768×512), 32 H&E tiles (512×512), one 4096×2048 tissue mosaic | HTJ2K lossless and Qfactor 90, Part 1 lossless and 2 bpp, each from two encoders; JPEG q90 at 4:2:0 and 4:4:4 |
| Whole-slide tiles (held out) | 64 tissue tiles (512×512) from each of Camelyon17 patients 000, 040 and 080, and from Aperio CMU-1 | Native scanner JPEG; native Kakadu Part 1 (CMU-1-JP2K); re-encoded HTJ2K and Part 1 |
| 16-bit medical (IDC DICOM) | 58 CT slices 512² (12-bit), 2 radiographs 3733×3439, 4 mammograms 2394×3062 (12-bit), 64 MR slices 256² (16-bit) | Lossless HTJ2K and Part 1, Gray16 output |
| Region | One 16384² tissue window (Camelyon17 p000) | HTJ2K lossless and q90, Part 1 lossless and 2 bpp, in three layouts: `tiled1024` (TLM index), `prec256` (one tile, 256² precincts, PLT index) and `untiled` (no index) |

Region requests, at seeded positions on each resolution level's pixel grid:
- One 1024² window at full resolution.
- One 1024² window at 1/4 resolution.
- A 512² thumbnail at 1/32 resolution (the whole image).
- 16 full-resolution windows decoded as a batch.

### Batch decode: many images, each decoded serially

Geomean of competitor/j2k time at 1 thread (1T) and at the host's full
thread count (allT).

Main corpus:

| host | family | OpenJPH 1T | OpenJPH allT | Grok 1T | Grok allT | OpenJPEG 1T | OpenJPEG allT | OpenHTJ2K 1T | OpenHTJ2K allT | libjpeg-turbo 1T | libjpeg-turbo allT |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mini (10T) | HTJ2K | 2.54× | 1.91× | 2.58× | 2.07× | 4.80× | 3.73× | 1.34× | 1.27× | – | – |
| mini (10T) | Part 1 | – | – | 1.44× | 1.36× | 1.60× | 1.53× | 1.39× | 1.29× | – | – |
| mini (10T) | JPEG | – | – | – | – | – | – | – | – | 1.15× | 1.06× |
| studio (18T) | HTJ2K | 2.51× | 1.80× | 2.57× | 2.12× | 4.99× | 4.02× | 1.36× | 1.21× | – | – |
| studio (18T) | Part 1 | – | – | 1.36× | 1.26× | 1.60× | 1.48× | 1.36× | 1.23× | – | – |
| studio (18T) | JPEG | – | – | – | – | – | – | – | – | 1.17× | 0.97× |
| cuda (16T) | HTJ2K | 1.05× | 0.82× | 1.17× | 0.96× | 3.14× | 2.38× | 1.18× | 1.21× | – | – |
| cuda (16T) | Part 1 | – | – | 1.19× | 1.06× | 1.27× | 1.26× | 1.24× | 1.15× | – | – |
| cuda (16T) | JPEG | – | – | – | – | – | – | – | – | 0.92× | 0.83× |

Whole-slide tiles:

| host | family | OpenJPH 1T | OpenJPH allT | Grok 1T | Grok allT | OpenJPEG 1T | OpenJPEG allT | OpenHTJ2K 1T | OpenHTJ2K allT | libjpeg-turbo 1T | libjpeg-turbo allT |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mini (10T) | HTJ2K | 2.02× | 1.68× | 2.06× | 1.78× | 5.17× | 3.81× | 1.36× | 1.25× | – | – |
| mini (10T) | Part 1 | – | – | 1.49× | 1.45× | 1.78× | 1.55× | 1.32× | 1.26× | – | – |
| mini (10T) | JPEG | – | – | – | – | – | – | – | – | 1.14× | 1.05× |
| studio (18T) | HTJ2K | 2.00× | 1.68× | 2.07× | 2.05× | 5.48× | 4.21× | 1.36× | 1.27× | – | – |
| studio (18T) | Part 1 | – | – | 1.30× | 1.29× | 1.81× | 1.52× | 1.28× | 1.24× | – | – |
| studio (18T) | JPEG | – | – | – | – | – | – | – | – | 1.21× | 1.05× |
| cuda (16T) | HTJ2K | 1.05× | 0.83× | 1.26× | 0.91× | 2.79× | 2.02× | 1.21× | 1.09× | – | – |
| cuda (16T) | Part 1 | – | – | 1.20× | 1.12× | 1.32× | 1.26× | 1.18× | 1.21× | – | – |
| cuda (16T) | JPEG | – | – | – | – | – | – | – | – | 1.00× | 0.91× |

On native scanner JPEG tiles alone, the ratios range from 1.00 to 1.30 on
the Macs at 1 thread, and from 0.78 to 1.13 on cuda.

16-bit medical:

| host | family | OpenJPH 1T | OpenJPH allT | Grok 1T | Grok allT | OpenJPEG 1T | OpenJPEG allT | OpenHTJ2K 1T | OpenHTJ2K allT |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mini (10T) | HTJ2K | 2.39× | 1.99× | 2.62× | 2.38× | 4.19× | 3.55× | 1.26× | 1.25× |
| mini (10T) | Part 1 | – | – | 1.47× | 1.41× | 1.52× | 1.46× | 1.44× | 1.38× |
| studio (18T) | HTJ2K | 2.44× | 1.93× | 2.71× | 2.49× | 4.43× | 3.81× | 1.29× | 1.22× |
| studio (18T) | Part 1 | – | – | 1.44× | 1.38× | 1.52× | 1.46× | 1.42× | 1.36× |
| cuda (16T) | HTJ2K | 0.69× | 0.74× | 0.81× | 0.86× | 2.27× | 2.05× | 0.66× | 0.73× |
| cuda (16T) | Part 1 | – | – | 1.20× | 1.11× | 1.27× | 1.21× | 1.32× | 1.24× |

### One large image: intra-image threading (4096×2048 mosaic)

| host | family | OpenJPH 1T | OpenJPH allT | Grok 1T | Grok allT | OpenJPEG 1T | OpenJPEG allT | OpenHTJ2K 1T | OpenHTJ2K allT | libjpeg-turbo 1T | libjpeg-turbo allT |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mini (10T) | HTJ2K | 2.53× | 10.16× | 2.81× | 2.47× | 6.98× | 8.02× | 1.22× | 1.95× | – | – |
| mini (10T) | Part 1 | – | – | 1.40× | 1.60× | 1.83× | 2.19× | 1.37× | 1.53× | – | – |
| mini (10T) | JPEG | – | – | – | – | – | – | – | – | 1.09× | – |
| studio (18T) | HTJ2K | 2.56× | 15.14× | 2.92× | 2.44× | 7.35× | 9.63× | 1.26× | 2.90× | – | – |
| studio (18T) | Part 1 | – | – | 1.38× | 1.35× | 1.82× | 2.49× | 1.33× | 2.15× | – | – |
| studio (18T) | JPEG | – | – | – | – | – | – | – | – | 1.13× | – |
| cuda (16T) | HTJ2K | 1.09× | 4.20× | 1.17× | 1.58× | 3.88× | 4.07× | 1.10× | 2.02× | – | – |
| cuda (16T) | Part 1 | – | – | 1.19× | 1.15× | 1.38× | 1.72× | 1.23× | 1.67× | – | – |
| cuda (16T) | JPEG | – | – | – | – | – | – | – | – | 0.90× | – |

OpenJPH cannot use more than one thread on one image, which accounts for
most of its all-threads ratios. Neither JPEG library threads one image.

### Region and zoom (16384² image)

Geomean against Grok, timed cells:

| Host | HTJ2K, 1T | HTJ2K, allT | Part 1, 1T | Part 1, allT |
|---|---:|---:|---:|---:|
| mini | 3.20 | 2.19 | 1.38 | 1.32 |
| studio | 3.16 | 1.93 | 1.36 | 1.05 |
| cuda | 1.71 | 2.60 | 1.22 | 1.33 |

Cells where Grok is faster:
- Reduced-resolution requests on `tiled1024` at full thread count: Part 1
  thumbnails, and 1/4-resolution windows in both families (0.20–0.78).
- `classic_2bpp_prec256` full-resolution windows, mostly at 1 thread
  (0.80–0.97).
- Near ties (0.98–0.99): HTJ2K `prec256` thumbnails at 1 thread on cuda, and
  lossless HTJ2K `tiled1024` full-resolution windows at 18 threads on studio.

Where OpenJPEG returns correct HTJ2K output (full-image thumbnails), it is
3.9–86× slower than j2k. Where OpenHTJ2K's output is correct, it is 3.0–42×
slower on HTJ2K and 1.8–22× slower on Part 1.

### Kakadu (command line vs command line, studio)

Not rerun: these rows are from the first run, with j2k at `70eaacaf`. Each
run reads the codestream, decodes it and writes a 25 MB PPM, timed by wall
clock over 9 alternating rounds. Process start-up is included for both
tools.

| set (4096×2048 RGB) | threads | j2k CLI ms | Kakadu CLI ms | Kakadu / j2k | max diff vs source, j2k / Kakadu |
|---|---:|---:|---:|---:|---|
| classic_2bpp_kdu | 1 | 159.3 | 209.9 | 1.32× | 47 / 46 |
| classic_2bpp_kdu | 18 | 21.4 | 36.4 | 1.70× | 47 / 46 |
| classic_2bpp_opj | 1 | 163.2 | 208.4 | 1.28× | 25 / 25 |
| classic_2bpp_opj | 18 | 22.7 | 36.1 | 1.59× | 25 / 25 |
| classic_lossless_kdu | 1 | 684.0 | 977.4 | 1.43× | 0 / 0 |
| classic_lossless_kdu | 18 | 64.7 | 68.9 | 1.06× | 0 / 0 |
| classic_lossless_opj | 1 | 685.4 | 978.0 | 1.43× | 0 / 0 |
| classic_lossless_opj | 18 | 63.8 | 68.9 | 1.08× | 0 / 0 |
| ht_lossless_kdu | 1 | 45.1 | 64.3 | 1.43× | 0 / 0 |
| ht_lossless_kdu | 18 | 20.2 | 34.6 | 1.71× | 0 / 0 |
| ht_lossless_ojph | 1 | 44.5 | 65.1 | 1.46× | 0 / 0 |
| ht_lossless_ojph | 18 | 19.1 | 36.3 | 1.90× | 0 / 0 |
| ht_q90_kdu | 1 | 38.3 | 47.2 | 1.23× | 38 / 38 |
| ht_q90_kdu | 18 | 17.8 | 32.8 | 1.84× | 38 / 38 |
| ht_q90_ojph | 1 | 39.6 | 48.8 | 1.23× | 37 / 37 |
| ht_q90_ojph | 18 | 17.6 | 33.5 | 1.90× | 37 / 37 |

### Correctness and unsupported operations

| Decoder | Result |
|---|---|
| All, full image | Lossless (8-bit and 16-bit) is bit-exact with j2k. 9/7 is within 2 (Grok) or 1 (others). JPEG is exact with libjpeg-turbo. |
| OpenJPEG 2.5.4 | Wrong pixels on every partial HTJ2K region decode (difference 127–128), in all layouts. Reproduced with `opj_decompress -d`. Part 1 regions and full-image HTJ2K are correct. |
| OpenHTJ2K 0.23.3 and 0.23.4 | Wrong pixels on windows of tiled images (difference 255). Reproduced with its own `region_decode_bench -verify -fullcap 300`, which exits 2 on both versions. Untiled and precinct layouts are correct. |
| OpenJPH 0.33.0 | No region API |
| j2k | The batch region API rejects reduced-resolution requests that span tiles. `J2kDecoder`, used here, handles them. |

Releasing j2k's retained workspaces before every sample
(`J2K_BATCH_COMPARE_J2K_COLD`) changed j2k's time by −1.9% to +0.7% at 1
thread and by −5.6% to +5.2% at full thread count, so workspace retention
does not explain j2k's lead.

### Where j2k is behind

1. **x86 throughput.** Moving from the M4 to the Ryzen at 1 thread (geomean
   over the batch sets):
   - j2k gets 2.7× slower on HTJ2K and 3.9× slower on 16-bit HTJ2K.
   - OpenJPH gets 1.1–1.4× slower on HTJ2K and 1.13× slower on 16-bit
     HTJ2K. Grok gets 1.2–1.7× and 1.21× slower.

   The cuda build includes `035d3046`, which took the default x86-64
   build's decode loops off per-sample libm calls. Before it, the scalar
   horizontal wavelet filter (`filter_row`), among others, called a software
   `floorf` and an `fmaf` trampoline for every sample, because baseline
   x86-64 has neither instruction; aarch64 has both. In a j2k-only A/B on
   cuda (3 processes of 7 repeats per build, build order rotated), that
   change made MR 1.28× faster at 1 thread and moved every other 1-thread
   case by at most 2%. A `-C target-cpu=native` build (of `70eaacaf`) is
   still faster than the default build of the cuda decode code:

   | Image set | Default ÷ native time, 1 thread |
   |---|---:|
   | CT | 1.31× |
   | Radiographs | 1.46× |
   | Mammograms | 1.53× |
   | MR | 1.04× |
   | Camelyon17 p000 tiles; CT Part 1 | 1.02–1.04× |

   The remaining 16-bit HTJ2K gap is not yet isolated.
2. **Tiled reduced-resolution regions.**
   - Thumbnails of `tiled1024` images take the same time at 1 thread and at
     full thread count.
   - HTJ2K 1/4-resolution windows get slower with more threads on studio and
     cuda (cuda lossless: 22.8 → 38.4 ms).
3. **JPEG at high thread counts and on x86.** libjpeg-turbo ties j2k at 8–18
   threads on the Macs. On cuda, ratios are 0.83–1.00, and as low as 0.73 for
   one set. 4:2:0 is the weaker format. Later JPEG work on `main` is not
   reflected here: x86 16-bit AVX2 kernels and streaming row decode took
   Kodak 4:2:0 on cuda at 1 thread from 0.87× to about 1.09× libjpeg-turbo.

### Caveats

- x86 is one Zen 3 CPU under WSL2. No Intel host, native Linux host or
  AVX-512 host was tested.
- Kakadu was compared only as a command line, with a vendor binary.
- Only one coding family was tested (RPCL, 1 layer, 64×64 blocks). Natural
  images are limited to Kodak 24. GPU decoders and encoding were not
  measured.
- Between the two runs, some times moved more than the wrapper changes
  account for. On cuda, Grok's batch times fell 1–7% and j2k's 8-bit batch
  times at 16 threads rose 3–6% (geomeans per corpus), so part of the drop
  in cuda's all-thread ratios, such as Grok on 8-bit HTJ2K from 1.02–1.03 to
  0.91–0.96, comes from drift or the j2k build rather than the wrappers. On
  the Macs, some OpenHTJ2K and OpenJPEG region cells got up to 1.6× slower.

### Reproduce

Prepare the pinned references, then build and run the harness with the
same budget for every decoder:

```bash
scripts/prepare-openjph-reference.sh
scripts/prepare-openhtj2k-reference.sh
export J2K_OPENJPH_SOURCE_DIR="$PWD/target/reference/openjph-0.33.0"
export J2K_OPENJPH_LIB_DIR="$J2K_OPENJPH_SOURCE_DIR/build-reference/src/core"
export J2K_OPENHTJ2K_SOURCE_DIR="$PWD/target/t803/openhtj2k-v0.23.4"
export J2K_OPENHTJ2K_LIB_DIR="$J2K_OPENHTJ2K_SOURCE_DIR/build-reference"
# Grok: set J2K_GROK_SOURCE and J2K_GROK_ROOT to a Release build.
cargo build --profile release-bench -p j2k-compare --bin jp2k_batch_compare

J2K_BATCH_COMPARE_THREADS=8 RAYON_NUM_THREADS=8 J2K_BATCH_COMPARE_REPEATS=11 \
J2K_BATCH_COMPARE_MAX_ABS_DIFF=0 J2K_BATCH_COMPARE_KEEP_GOING=1 \
  target/release-bench/jp2k_batch_compare <set-dir> <image-count>
# Region and zoom: one image, plus J2K_BATCH_COMPARE_VIEWPORT=<reduce>:<count>:<size>:<seed>.
```

What a plain repository build does not yet reproduce:
- **OpenJPEG:** the in-repo build still links `openjpeg-sys`'s vendored 2.5.3
  without `NDEBUG`. This comparison linked upstream 2.5.4 instead.
- **OpenHTJ2K:** the prepare script pins 0.23.4. The speed rows used 0.23.3.
- **Run directories:** the first run's orchestration scripts, input
  generators, raw per-process TSVs and full per-cell report are in the local
  run directory `target/fairbench-20261101`. The rerun's driver
  (`run-rerun.sh`) and raw per-process TSVs are in the `fairbench-20261102`
  run directory on each host. Neither is committed.

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

### Sampled classic Tier-1 on the CPU pool - 2026-10-29

The same Apple M4 Pro decoded eight different groups of 32 interior level-0
tiles from each slide. No tile appeared in two groups. Each sample included
`MetalBatchDecoder::prepare`, submission and completion; output stayed
resident and was compared byte for byte with `CpuBatchDecoder` after timing.
The control disables the new automatic route in the same release binary.

| Slide | GPU Tier-1 median | CPU Tier-1 + Metal reconstruction median | CPU batch median |
| --- | ---: | ---: | ---: |
| JP2K-33003-1 | 12.63 ms | 7.35 ms | 5.95 ms |
| JP2K-33003-2 | 14.94 ms | 8.52 ms | 6.63 ms |

The automatic route raises median submission time to 5.50 and 5.70 ms
because Tier-1 finishes before the Metal command is returned; completion wait
falls to 1.61 and 1.67 ms. It applies only to at least eight distinct,
full-image, sampled classic RGB8 plans whose components can use the stacked
executor, and only when called outside a Rayon worker. Smaller, repeated,
HTJ2K, high-bit, nested-pool and single-worker calls keep the all-Metal route;
an explicit multi-worker limit caps the CPU stage.

The characterization harness accepts consecutive groups with
`J2K_SAMPLED_GROUP_COUNT`:

```bash
J2K_REQUIRE_METAL_RUNTIME=1 \
J2K_SAMPLED_CORPUS=<level{L}-tile{i}.j2k dir> \
J2K_SAMPLED_COUNT=32 J2K_SAMPLED_LEVEL_COUNT=1 \
J2K_SAMPLED_GROUP_COUNT=8 cargo test --release -p j2k-metal --lib \
  local_sampled_color_batch_characterization -- --ignored --nocapture
```

## CUDA batches, nvJPEG2000, and Burn handoff - 2026-10-20

Host: AMD Ryzen 7 5800X3D (8 cores, 16 threads) with an NVIDIA GeForce RTX 4070
SUPER (12 GiB), driver 610.88, Ubuntu 24.04.4 under WSL2. j2k 0.12.0 working
tree, Rust 1.99.0, `release-bench`, CUDA Oxide kernels built for sm_89.

Inputs: the 32 largest level-zero tissue frames of one WSI DICOM export, each
512 x 512 RGB8, HTJ2K lossless, RPCL, three decomposition levels, 64 x 64 code
blocks. Batches larger than 32 cycle the same 32 tiles; they are not larger
sets of unique inputs.

### CUDA surface batches above 32 images

The surface API decodes colour batches in successive 32-image groups on one
session. It keeps the caller's output order, and each group's outputs share
one device allocation, so `Surface::download_batch_tight` copies each adjacent
run of surfaces with one transfer. Before grouping (`HEAD` in the table), all
images decoded in one group: a session keeps at most 2,048 batch scratch
buffers (2 GiB) and one 32-tile batch uses about 1,253, so a larger group
allocated and freed uncached scratch on every decode.

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

The owned API, which `j2k-ml` uses, decodes RGB and RGBA groups larger than 32
images in successive 32-image chunks into one dense allocation, or into
consecutive ranges of a caller's destination. Each chunk except the last
completes before the next starts, so its scratch returns to the cache, and
`submit_batch` waits for all but the final chunk. Before chunking, one warm
65-image group made 1,274 device allocations, against 2 for a warm 32-image
group.

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
path whose exact output is checked by the owned API runs above. Each
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
`cargo xtask promotion-codegen`. Stale tables are caught twice: an xtask unit
test compares the checked-in tables with the manifest on every test run, and
full validation runs `cargo xtask promotion-codegen --check`. The generator checks the manifest schema,
backend, SHA-256 values, that all six operations were measured, workload
identity, limits, and duplicates. A route with no accepted result stays on the
CPU.

The routing runs below date from August and September 2026 and predate the
CPU decode speedups released in 0.13.0 and 0.14.0, so some routed cells may
now be slower than the CPU. On 2026-10-09, `Auto` sent a 640 x 480 `Rgb8`
image to Metal at 1.8 to 2.0 ms while the CPU took 1.4 to 1.5 ms. Rerun the
routing suite before relying on the tables.

Two earlier Metal runs are still declared as sources in
`xtask/routing-promotion-evidence.json` but own no cells, because the
2026-09-24 run below re-measured everything they covered:

| Source | Run | Verified artifact SHA-256 |
| --- | --- | --- |
| `metal_part1` | External Part 1 matrix, 2026-08-04 | `162a47f7a96b2be88abebc100aab672513af04895532863fa1a293660546f879` |
| `metal_part15` | Official Part 15 matrix, 2026-08-07 | `cfa66686d053bb3e2d4c8756abaf84aab65d8505a635795cefd38de53573c1f5` |

### Metal routing run - 2026-09-24

Source of every Metal decode cell and of the lossy RGB8 encode threshold
(`metal_local_combined`).

The corpus is local, not a publication corpus: 512x512 tiles from GDC
whole-slide DICOM files, decoded with `opj_decompress` and stitched into
256x256, 640x480, 1024x1024, and 2048x2048 RGB images and 640x480 and
2048x2048 gray images. Each was re-encoded as Part 1 lossless and lossy
(OpenJPEG), HTJ2K lossless and lossy codestreams (OpenJPH), and HTJ2K lossless
JPH files (Grok). There are 34 cases (26 decode, 8 encode); manifest SHA-256
`dd6363374dccf8d6b34656f90b9557b5a01f8582b94f5014e3165dc61f1248ec`.

`tests/auto_routing_parity.rs` found no mismatch in any full, ROI, or
half-scale decode, and the bench's own parity check passed for every CPU,
Metal, and Auto route. The run used `release-bench`, candidate `c1cc3ff1` with
a dirty tree, and an Apple M4 Pro (16 GPU cores) on macOS 27.0. The verifier
accepted all 124 cells and routed 49 to the GPU. The verified artifact SHA-256
is `66f9d83f932efb7df6cab2de0048849f09343da00915e22794b0056b29f2aa5f`.

Cells routed to Metal:

- **Lossless, repeated batches of 16:** HTJ2K RGB8 from 256x256 (JPH from
  640x480), HTJ2K Gray8 from 640x480, Part 1 RGB8 from 640x480, and Part 1
  Gray8 from 2048x2048.
- **Lossless, single full image:** HTJ2K RGB8 codestreams and JPH files and
  HTJ2K Gray8 codestreams, from 640x480.
- **Lossy, repeated batches of 16:** HTJ2K RGB8 from 256x256 (46% faster at
  256x256), Part 1 RGB8 from 1024x1024 (56% faster; 640x480 measured 10%
  slower), and HTJ2K and Part 1 Gray8 from 2048x2048 (62% and 18% faster;
  HTJ2K Gray8 at 640x480 was 8% faster, below the 10% bar).
- **Lossy, single full image:** HTJ2K RGB8 and Gray8 from 640x480 (44% and 37%
  faster at 640x480, 77% and 78% at 2048x2048).
- **Lossy RGB8 encode:** from 2048x2048 pixels, where Part 1 and HTJ2K encode
  measured 36% and 57% faster.

Staying on the CPU:

- Part 1 single-image decodes: Metal measured 105% to 1400% slower on lossless
  inputs and 66% to 2400% slower on lossy ones.
- Half-scale decodes: none qualified. HTJ2K lossy RGB8 half-scale measured 3%
  to 176% slower.
- ROI and half-scale qualified only for HTJ2K Gray8 at 2048x2048, which Metal
  `Auto` does not route for gray sources.

### External CUDA routing run - 2026-08-05

The uninterrupted CUDA matrix used the same 12 external cases from
`uclouvain/openjpeg-data` commit
`39524bd3a601d90ed8e0177559400d23945f96a9` and manifest SHA-256
`f07072f5d0313c0249e2df5df2310cd5c6c5a4b3414a933537fabf2362d2065c`.
It ran all 36 cells with Cargo `release-bench`, Criterion 0.95 confidence
intervals, ten samples, a one-second warm-up, and a three-second target on an
AMD Ryzen 7 5800X3D with an NVIDIA GeForce RTX 4070 SUPER, driver 596.49,
Linux x86-64, and CUDA 13.2.

The verifier accepted every cell. The routing table has 14 decode cells from
this run (`cuda_part1`), with the measured output sizes as thresholds. RGB8 reversible uses CUDA for full
output from 256 x 149, ROI output from 128 x 74, and half-scale output from
1296 x 972. RGB8 irreversible uses CUDA for full output from 640 x 480 and ROI
or half-scale output from 320 x 240. Gray8 reversible uses CUDA only for full
output from 640 x 480. Gray8 irreversible uses CUDA for full output from
3323 x 891, ROI output from 1661 x 445, and half-scale output from
1662 x 446. Repeated-input batches of 16 use the full-image thresholds, except
RGB8 irreversible, which uses 256 x 149.

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

### Official Part 15 CUDA routing run - 2026-08-07

Source of the eight `cuda_part15` cells.

The run used three external T.803 workloads: raw HTJ2K `p0_04` BSET 12, JPH
file 1 BSET 12, and JPH file 10 component 0 as the encode source. The
schema-2 manifest SHA-256 is
`422f40e4086b53e43f2338f97b468c869257cc23ec4899432cf019585782d48e`.
It ran ten end-to-end cells covering full, ROI, scaled, and repeated batch
decode plus lossless and lossy encode, on an RTX 4070 SUPER with driver 596.49
under WSL2 Linux 5.15.153.1. Every CPU/hybrid pair produced an identical
output hash. There is no GPU-only route, so these measure the hybrid routes.

`Auto` routes raw HT and JPH full, ROI, half-scale, and repeated batch decode
to CUDA (8 of 10 cells), which measured 70.39% to 91.41% faster than the CPU.
CUDA lossless and lossy encode were 528.23% and 305.10% slower than the CPU;
those two cells stay on the CPU. The thresholds apply only to the measured
codec, container, format, dimensions, operation, and repeat count.

The verified artifact SHA-256 is
`77370c83710ebf578139ad0bfa2608ffad989d83faec8d5eee213691290c0088`. The raw
and serialized-report SHA-256 values are
`0b4ac7a08e4adba0e9ed9a7983a4fb3cd9ae0cc81a2ab1162e604f13a785065a` and
`faacdfa190d0c18acdf9c24a584e285606c37af8dcd15cc649dd6eac2eaa8a10`.

The report records base commit `f92646d0e6f0d0ef6c1e60b60beaad29da1afd3b` plus
uncommitted changes. The matrix must be rerun on the release commit before
these numbers are published as release results.

### Metal HTJ2K host-output encode matrix - 2026-08-08

The host-output encode route was measured separately because the T.803
Part 15 inputs are too small for GPU coefficient preparation and HT Tier-1 to
pay for Metal setup. The
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

The SHA-256 of the decision artifact is
`c8defb820b55a99e94acdd5849b4597bce0a1718fd7e0d2bc0aa926bc0e130d4`.
After enabling only the qualifying cells, the full 26-cell matrix reran with
no `Auto` output mismatch; its verified artifact SHA-256, which the generated
routing table records as `METAL_HOST_OUTPUT`, is
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
