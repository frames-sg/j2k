# Benchmark Evidence

This document records published benchmark commands, measurements, and
environment details. JSON and CSV artifacts remain the source of truth when
they are produced by a benchmark harness.

## Publication Status

The codec support boundary is tracked separately in
[`docs/public-support.md`](public-support.md): JPEG 2000 Part 1 codestreams,
JP2 still-image files, HTJ2K Part 15 codestreams, and JPH still-image files.
That repo-local support gate is separate from performance reporting.

This page is the current public benchmark evidence note. Broader adoption-facing
speed reports require an external adoption benchmark bundle:

```bash
cargo xtask adoption-report --run-dir target/j2k-adoption-benchmark/full
```

For adoption reports, `cargo xtask adoption-report` must require a completed
external bundle and identify any missing evidence. Generated repo-local
fixtures and passing codec self-checks remain implementation evidence; use
manifest-backed external rows for adoption-facing speed reports.

## Local Regression Guard - 2026-07-07

Host:

- Machine: local Apple Silicon development host
- CPU: Apple M4 Pro
- Architecture: `arm64`
- Memory: 48 GB
- OS: macOS 26.5 build `25F71`
- Baseline ref: `HEAD` (`29143c8e`)

Commands:

```bash
cargo xtask bench-build
cargo xtask j2k-perf-guard --baseline-ref HEAD --quick
```

Result: both commands passed. The quick guard compared the current remediation
tree against the local `HEAD` baseline with a +10% median regression threshold.
The macOS host cannot build or run the Linux-only cuda-oxide kernels, so
CUDA-labeled rows in this run are CPU fallback rows; strict CUDA runtime
validation remains a separate Linux/NVIDIA hardware gate.

Rows that previously regressed during the remediation and passed in the final
quick guard:

| Row | Baseline median | Current median | Delta |
| --- | ---: | ---: | ---: |
| `htj2k_cleanup_encode_distribution/rho_eq_uq_64x64/2459041792` | 8.592 us | 7.768 us | -9.59% |
| `htj2k_cleanup_encode_distribution/rho_eq_uq_64x64/2459041793` | 8.548 us | 7.846 us | -8.22% |
| `jpeg_cpu_encode_runtime/rgb8_512_420_restart_64` | 2.028 ms | 2.062 ms | +1.66% |

## CUDA JPEG-to-HTJ2K Transcode

Host:

- Machine: self-hosted NVIDIA benchmark host
- GPU: NVIDIA RTX 4070 SUPER
- CUDA target: `sm_89`
- Corpus: pancreas JPEG tile corpus
- Corpus size: 109 JPEG tiles, 7.14 MP total
- Profile: J2K CUDA transform + CUDA HT block encode + CPU packetization
- Options: one decomposition level, quantization scale `1.90`, one warmup,
  three timed iterations, minimum 100 tiles

Historical pre-retirement measurements were captured before the CUDA C/PTX path
was removed. The original CUDA C command and the temporary comparison manifest
are no longer active product workflows. Current CUDA benchmark runs use the
`cuda-runtime` feature with `J2K_REQUIRE_CUDA_RUNTIME=1` and
`J2K_REQUIRE_CUDA_OXIDE_BUILD=1`.

Measured results:

| Historical path | MP/s | Wall | GPU | IDCT + row |
| --- | ---: | ---: | ---: | ---: |
| Original CUDA C | 396.300 | 18.025 ms | 8.049 ms | 0.855 ms |
| cuda-oxide before IDCT basis fix | 40.813 | 175.029 ms | 165.041 ms | 158.390 ms |
| cuda-oxide after IDCT basis fix | 380.411 | 18.778 ms | 9.981 ms | 2.903 ms |

PTXAS evidence for the fix:

- Before: `transcode_dwt97_idct_i16_batch` used a 4096-byte stack frame because
  the Rust IDCT basis array was materialized into per-thread local memory.
- After: `transcode_dwt97_idct_i16_batch` reports `0 bytes stack frame`,
  `0 bytes spill stores`, and `0 bytes spill loads`.

The remaining historical gap was IDCT register pressure. The cuda-oxide IDCT
kernel reported 94 registers after removing the stack frame. CUDA C/PTX is no
longer a supported product backend.

## CUDA Oxide Migration Status

`J2K_REQUIRE_CUDA_OXIDE_BUILD=1` is the shared strict build gate for
Linux/NVIDIA validation hosts.

The temporary NVIDIA comparator harness was removed after the final strict CUDA
Oxide comparison was captured for decode, JPEG decode, and JPEG-to-HTJ2K
transcode. The final comparison required a Linux/NVIDIA host with CUDA, NVIDIA
JPEG/JPEG 2000 comparator libraries, and cuda-oxide available; it could not be
produced by hosted CI or by macOS development machines.

June 29, 2026 CUDA Oxide validation run:

- Host: self-hosted Linux/NVIDIA validation runner
- GPU: NVIDIA GeForce RTX 4070 SUPER, driver `596.49`
- CUDA compiler: `cuda_13.2.r13.2/compiler.37668154_0`
- Rust: `cargo 1.96.0`
- cuda-oxide: `cargo-oxide 0.2.1`
- Oxide arch: `sm_89`
- Historical NVIDIA comparator build before harness removal: strict baseline
  build mode with host-local CUDA and nvJPEG/nvJPEG2000 headers/libraries
- CUDA Oxide build: strict `J2K_REQUIRE_CUDA_OXIDE_BUILD=1`,
  `J2K_REQUIRE_CUDA_RUNTIME=1`
- Artifact directory: `target/bench-logs`

Implemented CUDA Oxide-family evidence captured with strict build requirements:

| Harness | Corpus | CUDA Oxide families | Result |
| --- | --- | --- | --- |
| `transcode_compare --profile-j2k-cuda-only` | 109 pancreas JPEG tiles, 7.143 MP, corpus hash `c1060319d3236928` | transcode + J2K encode | 391.430 MP/s, 18.250 ms wall, 9.001 ms GPU, 0 CPU fallback jobs |
| `decode_compare --fixture-dim 512` | generated Gray8 and RGB8 HTJ2K fixtures | J2K decode store, dequantize, IDWT | Gray8: 1.789 ms wall / 1.477 ms GPU, PSNR `inf`; RGB8: 1.676 ms wall / 1.042 ms GPU, PSNR `inf` |
| `cargo test -p j2k-cuda --all-targets --features cuda-runtime htj2k -- --nocapture` | generated HTJ2K smoke fixtures | HTJ2K cleanup/refinement decode, dequantize, IDWT, decode store | Full, ROI, scaled, ROI-scaled Gray8/RGB8/RGBA8 CUDA rows passed; batch RGB8/RGBA8 rows passed |
| `cargo test -p j2k-cuda --all-targets --features cuda-runtime htj2k_encode -- --nocapture` | generated HTJ2K encode fixtures | HTJ2K codeblock encode + J2K encode compaction/packetization | Scalar codeblock parity passed; two-pass sigprop and three-pass sigprop/magref round-trip cases passed |
| `cargo test -p j2k-jpeg-cuda --test encode --features cuda-runtime -- --nocapture` | generated resident RGB8 fixtures | JPEG baseline encode | Single resident CUDA buffer encode and same-buffer batch encode passed; CPU-backend request rejected without fallback |
| `cargo test -p j2k-jpeg-cuda --all-targets --features cuda-runtime -- --nocapture` | generated JPEG fixtures | JPEG decode + JPEG baseline encode | 4 encode tests, 28 host-surface/decode tests, and CUDA JPEG bench harness targets passed |

Final NVIDIA comparator capture before harness retirement:

| Harness | Corpus | Artifacts | Result |
| --- | --- | --- | --- |
| `transcode_compare` | 109 pancreas JPEG tiles, 7.143 MP, corpus hash `c1060319d3236928` | `final_transcode_cuda_oxide_vs_nvjpeg2000_20260629.{json,csv,log}` | CUDA Oxide transform + CUDA HT block encode: 274.0 MP/s, 26.073 ms wall, 17.171 ms GPU, 0 CPU fallback jobs. NVIDIA reused-session serial: 52.2 MP/s, 136.851 ms wall, 128.608 ms GPU. Byte delta vs NVIDIA: -0.5368%; aggregate PSNR: CUDA Oxide 47.63 dB, NVIDIA 49.19 dB. |
| `decode_compare` | 32 HTJ2K codestreams generated from pancreas JPEG tiles | `final_decode_cuda_oxide_vs_nvjpeg2000_20260629.{json,csv,log}` | CUDA Oxide decode mean: 1.328 ms wall, 1.022 ms GPU. NVIDIA nvJPEG2000 mean: 0.861 ms wall, 0.748 ms GPU. Mean PSNR vs CPU: CUDA Oxide 95.38 dB, NVIDIA 51.46 dB. |
| `jpeg_decode_compare` | generated 1024 x 1024 stress JPEGs, subsampling 4:2:0 / 4:2:2 / 4:4:4 | `final_jpeg_decode_cuda_oxide_vs_nvjpeg_20260629.log` | CUDA Oxide JPEG decode vs nvJPEG wall mean: 3.461 ms vs 6.634 ms for 4:2:0, 4.086 ms vs 8.546 ms for 4:2:2, and 5.079 ms vs 11.921 ms for 4:4:4. Max delta vs CPU matched nvJPEG for 4:2:0 and 4:2:2; 4:4:4 was 0 for CUDA Oxide and 3 for nvJPEG. |

The exact shell commands used for the final comparator capture are preserved in
the corresponding `target/bench-logs/*20260629*.log` files. The
`tests/nvidia-baseline` harness was removed after this capture, so those commands
are historical evidence, not active workflow instructions.

This final capture satisfies the NVIDIA comparator retirement gate. The
remaining publication caveat is that these are repo-local retirement
benchmarks, not a replacement for external adoption-report evidence.

Additional strict CUDA Oxide validation used the host's cuda-oxide, libclang,
CUDA, and linker paths plus:

```bash
export J2K_CUDA_OXIDE_ARCH=sm_89
export J2K_REQUIRE_CUDA_OXIDE_BUILD=1

cargo check -p j2k-cuda-runtime \
  --features cuda-oxide-htj2k-decode,cuda-oxide-j2k-decode-store,cuda-oxide-j2k-dequantize,cuda-oxide-j2k-idwt

cargo test -p j2k-cuda-runtime --lib \
  --features cuda-oxide-htj2k-decode,cuda-oxide-j2k-decode-store,cuda-oxide-j2k-dequantize,cuda-oxide-j2k-idwt \
  cuda_oxide_htj2k -- --nocapture

cargo check -p j2k-cuda-runtime \
  --features cuda-oxide-htj2k-encode,cuda-oxide-j2k-encode

cargo test -p j2k-cuda-runtime --lib \
  --features cuda-oxide-htj2k-encode,cuda-oxide-j2k-encode \
  cuda_oxide_htj2k_encode -- --nocapture

cargo check -p j2k-cuda-runtime \
  --features cuda-oxide-jpeg-encode

cargo test -p j2k-cuda-runtime --lib \
  --features cuda-oxide-jpeg-encode \
  cuda_oxide_jpeg_encode -- --nocapture
```

The validation host also needed its linker search path to include the local
LLVM/libffi runtime used by cuda-oxide.

## Metal Status

Metal acceleration is selective. Public claims should say Metal-accelerated
stages, not complete end-to-end Metal coverage for every encode, decode, or
transcode route.

Existing Apple Silicon encode-stage routing evidence was collected with:

```bash
cargo test -p j2k-metal --test encode_auto_routing_benchmark -- --ignored --nocapture
```

Decode routing evidence can be collected with:

```bash
cargo test -p j2k-metal --release --test metal_decode_benchmark \
  metal_decode_benchmark -- --ignored --nocapture
```

On June 28, 2026, a local benchmark checkout was tested on a MacBook Pro
`Mac16,8`, Apple M4 Pro, 48 GB memory, macOS 26.5. Generated fixture rows
showed no strict Metal decode win over CPU, so `Auto` decode should remain CPU
for these shapes:

| Generated row | CPU | Metal resident | Metal readback |
| --- | ---: | ---: | ---: |
| classic Gray8 512 full | 3.671 ms | 25.682 ms | 25.762 ms |
| HTJ2K Gray8 512 full | 0.701 ms | 2.566 ms | 2.603 ms |
| classic RGB8 512 full | 13.392 ms | 75.281 ms | 74.794 ms |
| classic Gray8 1024 full | 7.923 ms | 28.216 ms | 28.626 ms |
| HTJ2K Gray8 1024 full | 2.081 ms | 2.874 ms | 2.857 ms |
| classic RGB8 1024 full | 28.058 ms | 83.760 ms | 83.090 ms |
| classic RGB8 1024 region+scaled | 13.905 ms | 63.151 ms | 65.418 ms |

Current evidence supports conservative Auto routing for large encode-stage
inputs, not blanket Metal encode coverage. On an Apple M4 Pro, prior routing
evidence showed coefficient-prep stage wins at 512 x 512 and larger stage
inputs, with selected Auto rows such as lossless classic RGB8 improving from
115.098 ms CPU to 65.294 ms Auto at 512 x 512, and lossless HTJ2K RGB8
improving from 284.845 ms CPU to 10.012 ms Auto at 1024 x 1024. Treat these as
routing evidence for the recorded host, not cross-machine performance claims.

Metal JPEG-to-HTJ2K same-geometry batch transcode evidence was collected on
June 28, 2026, from the same local benchmark checkout on a MacBook Pro
`Mac16,8`, Apple M4 Pro, 48 GB memory, macOS 26.5:

```bash
J2K_TRANSCODE_METAL_PROFILE_STAGES=1 \
cargo bench -p j2k-transcode-metal --bench dct97 -- \
  jpeg_to_htj2k_wsi_integer_53_tile_batch/srgb_ybr420_224_batch_128
```

Criterion row, lossless 5/3 HTJ2K, generated 224 x 224 sRGB/YBR 4:2:0 JPEG
fixture, batch size 128:

| Route | Time interval | Throughput interval |
| --- | ---: | ---: |
| Rayon CPU batch | 86.581-89.594 ms | 38.439-39.776 MiB/s |
| Auto Metal batch | 57.334-58.776 ms | 58.593-60.066 MiB/s |
| Strict Metal batch | 58.053-65.069 ms | 52.926-59.323 MiB/s |

The same run emitted profile rows with one stable workload context,
`srgb_ybr420_224_batch_128`: 93 CPU rows with
`request=cpu path=cpu transform_processor=cpu`, 118 Auto rows with
`request=metal_auto path=auto transform_processor=metal`, and 118 strict rows
with `request=metal_explicit path=metal transform_processor=metal`. No CPU
profile row was labeled as Metal.

This row supports the existing same-geometry batch Auto Metal route for the
measured 224 x 224 batch-128 shape on this host. It does not justify broad
single-image transcode Auto expansion or distinct-geometry batch routing.

Release validation should rerun the hardware tests before publishing:

```bash
cargo test -p j2k-jpeg-metal --all-targets
cargo test -p j2k-metal --all-targets
cargo test -p j2k-metal --release --test metal_decode_benchmark \
  metal_decode_benchmark -- --ignored --nocapture
cargo test -p j2k-metal --test encode_auto_routing_benchmark -- --ignored --nocapture
```

If Metal hardware is unavailable, do not expand Metal performance claims for
the release.
