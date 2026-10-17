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
decode microbenchmarks by 0.4% and 1.8%; no whole-image CPU decode improvement
is claimed.

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

## HTJ2K DICOM decode development run - 2026-10-09

This run measured the CPU decode, tile-batch, owned-batch, and Metal changes
listed under Unreleased in the changelog, against an export of `8b608120`
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
