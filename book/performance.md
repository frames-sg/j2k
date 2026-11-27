# Performance

This page collects measured results: how J2K's CPU decoder compares with
other JPEG 2000 and JPEG libraries, what the CUDA and Metal backends gain over
the CPU, and where J2K is behind. Each section says when and on which machine
it was measured. To run the benchmarks yourself, see [Reproducing these
results](#reproducing-these-results).

Ratios are **competitor time ÷ J2K time**: above 1 means J2K is faster, below
1 means the competitor is.

## Summary

- **Apple Silicon:** J2K is the fastest decoder we measured in every
  full-image workload except JPEG at high thread counts. On batches of HTJ2K
  images, OpenJPH and Grok take 1.7 to 2.6 times as long, OpenHTJ2K 1.2 to
  1.4 times, and OpenJPEG 3.7 to 5.5 times. On classic JPEG 2000, the others
  take 1.2 to 1.8 times as long. JPEG decodes 14 to 21% faster than
  libjpeg-turbo on one thread and ties it on many.
- **x86:** most of that lead is gone. On 8-bit HTJ2K, OpenJPH and Grok are
  faster with all 16 threads, and on 16-bit HTJ2K, OpenJPH, Grok, and
  OpenHTJ2K are faster at every thread count. libjpeg-turbo is up to 20%
  faster on JPEG. J2K still leads on classic JPEG 2000.
- **Regions of a large image:** J2K is 1.7 to 3.2 times as fast as Grok on
  HTJ2K and 1.05 to 1.4 times on classic JPEG 2000. OpenJPEG returns wrong
  pixels for partial HTJ2K regions, OpenHTJ2K for windows of tiled images, and
  OpenJPH has no region API.
- **Kakadu:** the `j2k` command line is faster than Kakadu's in all 16 cases
  we ran.
- **GPUs:** on Metal, a batch of slide tiles decodes about ten times faster
  than one call per tile, and regions of a large image up to three times
  faster than on the CPU. On CUDA, J2K has 1.3 to 1.8 times nvJPEG2000's
  throughput on lossless HTJ2K tiles, in a comparison with caveats.
  `BackendRequest::Auto` uses a GPU only where it measured faster; see [How
  `Auto` routing thresholds are chosen](#how-auto-routing-thresholds-are-chosen).

## Machines

| Name | CPU | GPU | OS |
| --- | --- | --- | --- |
| mini | Apple M4, 10 cores (4 performance, 6 efficiency) | | macOS |
| studio | Apple M5 Max, 18 cores | | macOS |
| cuda | AMD Ryzen 7 5800X3D, 8 cores, 16 threads (Zen 3, AVX2, no AVX-512) | NVIDIA GeForce RTX 4070 SUPER, 12 GiB | Ubuntu 24.04 under WSL2 |
| pro | Apple M4 Pro, 12 cores (8 performance, 4 efficiency), 48 GB | 16-core GPU | macOS |

## CPU decoding compared with other libraries

Measured 2026-11-02 on mini, studio, and cuda.

| Library | Version | Build |
| --- | --- | --- |
| J2K | `3d253077` (mini, studio), `84b93cd9` (cuda) | `release` profile (fat LTO, one codegen unit), default target CPU |
| OpenJPEG | 2.5.4 | Upstream CMake `Release`, static |
| Grok | 20.4.24 | Upstream `Release` |
| OpenJPH | 0.33.0 | Upstream `Release`. It has AVX2 code for x86 but none for NEON, so the Macs run its generic C++. |
| OpenHTJ2K | 0.23.3 | Upstream default, `-march=native` |
| libjpeg-turbo | 3.2.1 | Upstream `Release` with SIMD. Accurate integer IDCT, fancy upsampling, RGB output. |
| Kakadu | 8.4.1 | Vendor binary (native arm64), command line only, studio only |

Every library except Kakadu is linked into the benchmark process. Each
decoder gets the same codestream bytes, loaded before timing, and writes its
output once into a buffer allocated the same way inside the timed region.
Before timing, its output is compared with J2K's: exact for lossless, within
3 for 9/7. All decoders get the same thread budget: a one-image batch gives
all of it to each decoder's own threading, and a larger batch splits the
images across that many workers, each decoding one image at a time. Each
cell is three fresh processes of 5 to 11 timed repeats, with the order of
sets and decoders rotated, and reports the median of the process medians.
The three ratios of a cell typically agreed within 1% on the Macs and 2.5% on
cuda.

No input was encoded by J2K. All use 5 decomposition levels (6 for the region
image), 64 x 64 code blocks, RPCL order, and one quality layer.

| Set | Images | Codings |
| --- | --- | --- |
| Main | Kodak (24 at 768 x 512), 32 H&E tiles (512 x 512), one 4096 x 2048 tissue mosaic | HTJ2K lossless and Qfactor 90, classic lossless and 2 bpp, each from two encoders; JPEG q90 at 4:2:0 and 4:4:4 |
| Slide tiles | 64 tissue tiles (512 x 512) from each of Camelyon17 patients 000, 040, and 080, and from Aperio CMU-1 | Scanner JPEG, Kakadu classic (CMU-1-JP2K), and the same tiles re-encoded as HTJ2K and classic |
| 16-bit medical | 58 CT slices (512 x 512, 12-bit), 2 radiographs (3733 x 3439), 4 mammograms (2394 x 3062, 12-bit), 64 MR slices (256 x 256, 16-bit), from IDC | Lossless HTJ2K and classic, Gray16 output |
| Region | One 16384 x 16384 tissue image (Camelyon17 patient 000) | HTJ2K lossless and q90, classic lossless and 2 bpp; each tiled (1024 x 1024 tiles, TLM), as one tile with 256 x 256 precincts (PLT), and untiled |

### Batches of images

Geometric mean of competitor time ÷ J2K time, at 1 thread (1T) and at all of
the host's threads (allT). "Classic" is JPEG 2000 Part 1.

Main set:

| Host | Format | OpenJPH 1T | OpenJPH allT | Grok 1T | Grok allT | OpenJPEG 1T | OpenJPEG allT | OpenHTJ2K 1T | OpenHTJ2K allT |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mini (10T) | HTJ2K | 2.54× | 1.91× | 2.58× | 2.07× | 4.80× | 3.73× | 1.34× | 1.27× |
| mini (10T) | Classic | – | – | 1.44× | 1.36× | 1.60× | 1.53× | 1.39× | 1.29× |
| studio (18T) | HTJ2K | 2.51× | 1.80× | 2.57× | 2.12× | 4.99× | 4.02× | 1.36× | 1.21× |
| studio (18T) | Classic | – | – | 1.36× | 1.26× | 1.60× | 1.48× | 1.36× | 1.23× |
| cuda (16T) | HTJ2K | 1.05× | 0.82× | 1.17× | 0.96× | 3.14× | 2.38× | 1.18× | 1.21× |
| cuda (16T) | Classic | – | – | 1.19× | 1.06× | 1.27× | 1.26× | 1.24× | 1.15× |

Slide tiles:

| Host | Format | OpenJPH 1T | OpenJPH allT | Grok 1T | Grok allT | OpenJPEG 1T | OpenJPEG allT | OpenHTJ2K 1T | OpenHTJ2K allT |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mini (10T) | HTJ2K | 2.02× | 1.68× | 2.06× | 1.78× | 5.17× | 3.81× | 1.36× | 1.25× |
| mini (10T) | Classic | – | – | 1.49× | 1.45× | 1.78× | 1.55× | 1.32× | 1.26× |
| studio (18T) | HTJ2K | 2.00× | 1.68× | 2.07× | 2.05× | 5.48× | 4.21× | 1.36× | 1.27× |
| studio (18T) | Classic | – | – | 1.30× | 1.29× | 1.81× | 1.52× | 1.28× | 1.24× |
| cuda (16T) | HTJ2K | 1.05× | 0.83× | 1.26× | 0.91× | 2.79× | 2.02× | 1.21× | 1.09× |
| cuda (16T) | Classic | – | – | 1.20× | 1.12× | 1.32× | 1.26× | 1.18× | 1.21× |

16-bit medical:

| Host | Format | OpenJPH 1T | OpenJPH allT | Grok 1T | Grok allT | OpenJPEG 1T | OpenJPEG allT | OpenHTJ2K 1T | OpenHTJ2K allT |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mini (10T) | HTJ2K | 2.39× | 1.99× | 2.62× | 2.38× | 4.19× | 3.55× | 1.26× | 1.25× |
| mini (10T) | Classic | – | – | 1.47× | 1.41× | 1.52× | 1.46× | 1.44× | 1.38× |
| studio (18T) | HTJ2K | 2.44× | 1.93× | 2.71× | 2.49× | 4.43× | 3.81× | 1.29× | 1.22× |
| studio (18T) | Classic | – | – | 1.44× | 1.38× | 1.52× | 1.46× | 1.42× | 1.36× |
| cuda (16T) | HTJ2K | 0.69× | 0.74× | 0.81× | 0.86× | 2.27× | 2.05× | 0.66× | 0.73× |
| cuda (16T) | Classic | – | – | 1.20× | 1.11× | 1.27× | 1.21× | 1.32× | 1.24× |

On studio with 18 threads, OpenHTJ2K is slightly faster on the CT slices
(0.95 to 0.97).

JPEG against libjpeg-turbo, on batches from the main set and the slide tiles,
and on the 4096 x 2048 mosaic alone (neither library threads a single image):

| Host | Main 1T | Main allT | Slide tiles 1T | Slide tiles allT | Mosaic 1T |
|---|---:|---:|---:|---:|---:|
| mini (10T) | 1.15× | 1.06× | 1.14× | 1.05× | 1.09× |
| studio (18T) | 1.17× | 0.97× | 1.21× | 1.05× | 1.13× |
| cuda (16T) | 0.92× | 0.83× | 1.00× | 0.91× | 0.90× |

On the scanner JPEG tiles alone, the 1-thread ratios range from 1.00 to 1.30
on the Macs and from 0.78 to 1.13 on cuda.

### One large image

The 4096 x 2048 mosaic, decoded with each library's own threading:

| Host | Format | OpenJPH 1T | OpenJPH allT | Grok 1T | Grok allT | OpenJPEG 1T | OpenJPEG allT | OpenHTJ2K 1T | OpenHTJ2K allT |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mini (10T) | HTJ2K | 2.53× | 10.16× | 2.81× | 2.47× | 6.98× | 8.02× | 1.22× | 1.95× |
| mini (10T) | Classic | – | – | 1.40× | 1.60× | 1.83× | 2.19× | 1.37× | 1.53× |
| studio (18T) | HTJ2K | 2.56× | 15.14× | 2.92× | 2.44× | 7.35× | 9.63× | 1.26× | 2.90× |
| studio (18T) | Classic | – | – | 1.38× | 1.35× | 1.82× | 2.49× | 1.33× | 2.15× |
| cuda (16T) | HTJ2K | 1.09× | 4.20× | 1.17× | 1.58× | 3.88× | 4.07× | 1.10× | 2.02× |
| cuda (16T) | Classic | – | – | 1.19× | 1.15× | 1.38× | 1.72× | 1.23× | 1.67× |

OpenJPH can't use more than one thread on one image, which explains most of
its all-thread ratios.

### Regions and zoom

Requests on the 16384 x 16384 image, at seeded positions: a 1024 x 1024
window at full resolution and at 1/4 resolution, a 512 x 512 thumbnail of the
whole image, and 16 full-resolution windows decoded as a batch. Geometric
mean of Grok time ÷ J2K time:

| Host | HTJ2K, 1T | HTJ2K, allT | Classic, 1T | Classic, allT |
|---|---:|---:|---:|---:|
| mini | 3.20 | 2.19 | 1.38 | 1.32 |
| studio | 3.16 | 1.93 | 1.36 | 1.05 |
| cuda | 1.71 | 2.60 | 1.22 | 1.33 |

Grok is faster on classic thumbnails and on 1/4-resolution windows of the
tiled layout at full thread count (0.20 to 0.78; see [Where J2K is
behind](#where-j2k-is-behind)), and on full-resolution windows of the 2 bpp
classic precinct layout, mostly at 1 thread (0.80 to 0.97). Where OpenJPEG's
HTJ2K output is correct (whole-image thumbnails), it is 4 to 86 times slower
than J2K. Where OpenHTJ2K's output is correct, it is 3 to 42 times slower on
HTJ2K and 2 to 22 times slower on classic.

### Kakadu

Command line against command line on studio, with J2K at `70eaacaf`. Each run
reads a 4096 x 2048 RGB codestream, decodes it, and writes a 25 MB PPM,
timed by wall clock over 9 alternating rounds, process start-up included.
Times in ms.

| Set | `j2k` 1T | Kakadu 1T | `j2k` 18T | Kakadu 18T | Max difference from source, J2K / Kakadu |
|---|---:|---:|---:|---:|---|
| classic_2bpp_kdu | 159.3 | 209.9 | 21.4 | 36.4 | 47 / 46 |
| classic_2bpp_opj | 163.2 | 208.4 | 22.7 | 36.1 | 25 / 25 |
| classic_lossless_kdu | 684.0 | 977.4 | 64.7 | 68.9 | 0 / 0 |
| classic_lossless_opj | 685.4 | 978.0 | 63.8 | 68.9 | 0 / 0 |
| ht_lossless_kdu | 45.1 | 64.3 | 20.2 | 34.6 | 0 / 0 |
| ht_lossless_ojph | 44.5 | 65.1 | 19.1 | 36.3 | 0 / 0 |
| ht_q90_kdu | 38.3 | 47.2 | 17.8 | 32.8 | 38 / 38 |
| ht_q90_ojph | 39.6 | 48.8 | 17.6 | 33.5 | 37 / 37 |

### Correctness

| Decoder | Result |
|---|---|
| All, full image | Lossless 8- and 16-bit output matches J2K exactly. 9/7 output is within 2 (Grok) or 1 (others). JPEG output matches libjpeg-turbo exactly. |
| OpenJPEG 2.5.4 | Wrong pixels on every partial HTJ2K region (off by 127 to 128), in all layouts; reproduced with `opj_decompress -d`. Classic regions and full HTJ2K images are correct. |
| OpenHTJ2K 0.23.3 and 0.23.4 | Wrong pixels on windows of tiled images (off by 255); reproduced with its own `region_decode_bench -verify`. Untiled and precinct layouts are correct. |
| OpenJPH 0.33.0 | No region API |
| J2K | The batch region API rejects reduced-resolution requests that span tiles; `J2kDecoder`, used here, handles them. |

## GPU decoding

### Metal: regions of large images

Measured 2026-10-28 on pro. Each sample decodes four 1024 px output tiles at
one reduction of an 18000 x 12000 RGB image, the requests a viewer makes for
one screen. "Metal batch" is `MetalBatchDecoder`, with output left in GPU
memory; "CPU tiles" is one `decode_region_scaled_pow2_into` call per tile,
the four in parallel. Median of seven samples, in ms. Output is identical.

| Input | Level 0 Metal | Level 0 CPU | Level 1 Metal | Level 1 CPU | Level 2 Metal | Level 2 CPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| large-ht.j2c (HTJ2K) | 18 | 26 | 10 | 25 | 9 | 22 |
| large.jp2 (classic) | 136 | 148 | 130 | 228 | 151 | 311 |
| large-plt.jp2 (classic, PLT) | 113 | 139 | 116 | 327 | 143 | 440 |

The same requests through `CpuBatchDecoder`, measured the next day, take 1
to 9% longer than the CPU tiles.

### Metal: subsampled slide tiles

Measured 2026-10-28 on pro, under heavy load from other work. Each sample
decodes 32 tiles (256 x 256 classic, YCbCr with half-width chroma) from the
openslide test slides `JP2K-33003-1.svs` and `JP2K-33003-2.svs` to RGB8 in GPU
memory. "Batch" is `MetalBatchDecoder`, one command buffer per batch; "per
image" is one `decode_request_to_device_with_session` call per tile. Median
of eight samples, in ms.

| Slide | Level 0 batch | Level 0 per image | Level 1 batch | Level 1 per image | Level 2 batch | Level 2 per image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| JP2K-33003-1 | 12.9 | 173.7 | 19.2 | 288.3 | 29.5 | 391.6 |
| JP2K-33003-2 | 12.1 | 142.6 | 14.5 | 132.7 | 22.9 | 187.3 |

Since then, `MetalBatchDecoder` runs the block decoding on the CPU and the
rest on Metal for batches of at least eight distinct, full-image, subsampled
classic RGB8 images, because that measured faster. The CPU batch decoder is
faster still (2026-10-29, eight groups of 32 level-0 tiles, median ms):

| Slide | All on Metal | CPU block decoding + Metal | CPU batch |
| --- | ---: | ---: | ---: |
| JP2K-33003-1 | 12.63 | 7.35 | 5.95 |
| JP2K-33003-2 | 14.94 | 8.52 | 6.63 |

### CUDA: batches of slide tiles

Measured 2026-10-20 on cuda (driver 610.88). Inputs are the 32 largest
level-0 tissue frames of one whole-slide DICOM export: 512 x 512 RGB8,
lossless HTJ2K, RPCL, three decomposition levels, 64 x 64 code blocks. Larger
batches repeat the same 32 tiles. Each time covers parsing, upload, decode,
synchronization, and allocating and freeing the output; "+ readback" adds the
copy to pageable host memory. Median of three process medians of 31 samples,
in ms. Every output was checked.

| Images | Surface API | `CudaBatchDecoder` | nvJPEG2000 | Surface + readback | `CudaBatchDecoder` + readback | nvJPEG2000 + readback |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 5.21 | 6.98 | 9.80 | 8.29 | 10.54 | 13.32 |
| 64 | 10.54 | 13.09 | 18.01 | 20.08 | 24.62 | 28.77 |
| 128 | 20.41 | 25.92 | 35.11 | 41.07 | 46.54 | 55.67 |
| 256 | 41.05 | 53.02 | 63.93 | 81.59 | 92.94 | 106.70 |
| 1024 | 168.49 | 219.10 | 260.07 | 330.80 | 385.00 | 427.55 |

Both J2K APIs cost 0.16 to 0.22 ms per image from 32 to 1024 images.
`CudaBatchDecoder` is slower than the surface API because it also groups
images and writes dense NHWC output.

Treat the nvJPEG2000 column (version 0.10.0.49, ten decode states on ten
streams, interleaved RGB8 output) as a first look, not a fair comparison: the
inputs are all lossless, nvJPEG2000 may be faster with its native planar
output, and it was fed from one host thread while J2K planned on 16.

### CUDA to Burn

Same host and inputs, through `j2k-ml` with NCHW output, measured with a
local variant of the `j2k-ml` CUDA benchmark. The two Burn columns also
convert the `U8` tensor to `f32`, divide by 255, and synchronize the Burn
device in each iteration. Median of three process medians, in ms.

| Tiles | `CudaBatchDecoder` alone | `CudaUploadBurnDecoder` | `CudaDirectBurnDecoder` | Upload ÷ direct |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 6.84 | 18.34 | 8.08 | 2.28× |
| 128 | 25.48 | 85.28 | 28.51 | 2.99× |
| 256 | 53.72 | 162.05 | 55.87 | 2.91× |

Decoding straight into Burn's memory adds 1 to 3 ms over the codec alone. No
speed result for `j2k-mpsgraph` has been published yet.

## How `Auto` routing thresholds are chosen

`BackendRequest::Auto` sends work to a GPU using routing tables built into
J2K; it doesn't measure your machine. A workload goes to the GPU only if the
CPU and GPU routes produced identical bytes on the same input, the GPU route
was at least 10% faster, and the 95% confidence intervals of the two times
didn't overlap. Everything else stays on the CPU. The exact cells and size
thresholds are in
[`xtask/routing-promotion-evidence.json`](https://github.com/frames-sg/j2k/blob/main/xtask/routing-promotion-evidence.json).

- Metal decode (2026-09-24 on pro; slide tiles stitched into images from
  256 x 256 to 2048 x 2048 and re-encoded by OpenJPEG, OpenJPH, and Grok): 49
  of 124 cells go to Metal. Single HTJ2K images go to Metal from 640 x 480,
  and batches of 16 from 256 x 256 to 2048 x 2048 depending on coding and
  format. The lossy decodes that qualified were 18% to 78% faster. Classic
  single images took 1.7 to 25 times as long on Metal, and no
  half-resolution decode qualified.
- Metal encode (same run, and generated images on 2026-08-08): lossy RGB8
  goes to Metal from 2048 x 2048 (36% faster for classic, 57% for HTJ2K).
  Lossless HTJ2K goes to Metal for RGB8 from 1024 x 1024 and Gray8 from
  2048 x 2048: 25.7 ms on the CPU against 16.2 ms on Metal for one RGB8
  1024 x 1024 image, and 120.8 ms against 20.6 ms at 2048 x 2048. The other
  three shapes took 1.7 to 4.3 times as long on Metal.
- CUDA decode (2026-08-05 and 2026-08-07 on cuda; 12 images from
  OpenJPEG's test data, and the T.803 HTJ2K and JPH files): classic 8-bit
  gray and RGB decodes go to CUDA above a size that depends on the request,
  from 128 x 74 for RGB8 lossless regions to 3323 x 891 for full Gray8 lossy
  images. These cover raw codestreams without JP2 color handling, and batches
  of one repeated image only. HTJ2K and JPH decodes of the T.803 inputs
  measured 70% to 91% faster on CUDA.
- CUDA encode was slower than the CPU in every case measured, 4 to 6 times
  on the T.803 inputs, so `Auto` encodes on the CPU.

These runs predate later CPU speedups, so some cells routed to a GPU may now
be slower than the CPU. On 2026-10-09, `Auto` sent a 640 x 480 RGB8 image to
Metal at 1.8 to 2.0 ms while the CPU took 1.4 to 1.5 ms. If GPU speed matters
to you, measure your workload with `BackendRequest::Cpu` and with an explicit
GPU backend.

## Where J2K is behind

- **x86 throughput.** Going from the M4 to the Ryzen at one thread, J2K gets
  2.7 times slower on HTJ2K and 3.9 times slower on 16-bit HTJ2K, while
  OpenJPH and Grok get only 1.1 to 1.7 times slower. Part of the gap is the
  default x86-64 target, which lacks FMA and other instructions our decode
  loops use. Building with `-C target-cpu=native` helps:

  | Image set | Default ÷ native time, 1 thread |
  |---|---:|
  | CT | 1.31× |
  | Radiographs | 1.46× |
  | Mammograms | 1.53× |
  | MR | 1.04× |
  | Camelyon17 tiles; CT classic | 1.02–1.04× |

  The rest of the 16-bit HTJ2K gap isn't understood yet.
- **Reduced-resolution regions of tiled images.** Thumbnails of the tiled
  layout take as long with all threads as with one, and HTJ2K 1/4-resolution
  windows get slower with more threads on studio and cuda (lossless on cuda:
  23 ms rising to 38 ms).
- **JPEG at high thread counts and on x86.** libjpeg-turbo ties J2K at 8 to
  18 threads on the Macs. On cuda the ratios are 0.83 to 1.00, and 0.73 for
  one set; 4:2:0 is the weaker case. Later x86 work (16-bit AVX2 kernels and
  streaming row decode) took Kodak 4:2:0 on cuda at one thread from 0.87 to
  about 1.09 times libjpeg-turbo's speed.

The CPU comparison has limits of its own. x86 is one Zen 3 CPU under WSL2; no
Intel, native Linux, or AVX-512 host was tested. Only one coding setup was
tested (RPCL, one layer, 64 x 64 code blocks), and the only natural images
are Kodak. Kakadu was compared only as a command line.

## Reproducing these results

Build the comparison harness against the pinned OpenJPH and OpenHTJ2K builds
and your own release build of Grok, then run it with the same thread budget
for every decoder:

```bash
scripts/prepare-openjph-reference.sh
scripts/prepare-openhtj2k-reference.sh
export J2K_OPENJPH_SOURCE_DIR="$PWD/target/reference/openjph-0.33.0"
export J2K_OPENJPH_LIB_DIR="$J2K_OPENJPH_SOURCE_DIR/build-reference/src/core"
export J2K_OPENHTJ2K_SOURCE_DIR="$PWD/target/t803/openhtj2k-v0.23.4"
export J2K_OPENHTJ2K_LIB_DIR="$J2K_OPENHTJ2K_SOURCE_DIR/build-reference"
export J2K_GROK_SOURCE=/path/to/grok J2K_GROK_ROOT=/path/to/grok/build/bin
cargo build --profile release-bench -p j2k-compare --bin jp2k_batch_compare

J2K_BATCH_COMPARE_THREADS=8 RAYON_NUM_THREADS=8 J2K_BATCH_COMPARE_REPEATS=11 \
  target/release-bench/jp2k_batch_compare <image-dir> <batch-size>
```

Add `J2K_BATCH_COMPARE_MAX_ABS_DIFF=0` for lossless sets, and
`J2K_BATCH_COMPARE_VIEWPORT=<reduce>:<count>:<size>:<seed>` for region
requests on one image. Two things differ from the runs above: the repository
links OpenJPEG 2.5.3 through `openjpeg-sys` (we linked upstream 2.5.4), and
the prepare script pins OpenHTJ2K 0.23.4 (speed was measured with 0.23.3).

A full benchmark bundle, with external corpora and every comparator, becomes
a report with:

```bash
cargo run -p xtask --features adoption -- adoption-report --run-dir target/j2k-adoption-benchmark/full
```

The other benchmarks, corpora, and their environment variables are described
in the contributor book's [Benchmarks](../contributor-book/benchmarks.md)
chapter.
