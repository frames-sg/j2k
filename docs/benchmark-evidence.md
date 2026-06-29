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

Original CUDA C baseline command:

```bash
export J2K_REQUIRE_CUDA_RUNTIME=1
export J2K_REQUIRE_CUDA_KERNEL_BUILD=1
export J2K_BENCH_JPEG_DIR=<pancreas-jpeg-tile-corpus>
export CUDA_BENCH_MANIFEST=<test-only-cuda-benchmark-manifest>
export ORIGINAL_CUDA_C_FEATURE=<original-cuda-c-baseline-feature>
cargo run --release --manifest-path "$CUDA_BENCH_MANIFEST" \
  --features "$ORIGINAL_CUDA_C_FEATURE" --bin transcode_compare -- \
  --profile-j2k-cuda-only \
  --decomposition-levels 1 \
  --quant-scales 1.90 \
  --warmup 1 \
  --iterations 3 \
  --min-tiles 100 \
  --json target/bench-logs/transcode_original_v062.json \
  --csv target/bench-logs/transcode_original_v062.csv
```

cuda-oxide command:

```bash
export J2K_REQUIRE_CUDA_RUNTIME=1
export J2K_REQUIRE_CUDA_KERNEL_BUILD=1
export J2K_BENCH_JPEG_DIR=<pancreas-jpeg-tile-corpus>
export J2K_CUDA_OXIDE_ARCH=sm_89
export J2K_REQUIRE_CUDA_OXIDE_TRANSCODE=1
export J2K_REQUIRE_CUDA_OXIDE_J2K_ENCODE=1
export J2K_CUDA_USE_OXIDE_TRANSCODE=1
export J2K_CUDA_USE_OXIDE_J2K_ENCODE=1
export CUDA_BENCH_MANIFEST=<test-only-cuda-benchmark-manifest>
cargo run --release --manifest-path "$CUDA_BENCH_MANIFEST" \
  --features cuda-oxide-transcode,cuda-oxide-j2k-encode \
  --bin transcode_compare -- \
  --profile-j2k-cuda-only \
  --decomposition-levels 1 \
  --quant-scales 1.90 \
  --warmup 1 \
  --iterations 3 \
  --min-tiles 100 \
  --json target/bench-logs/transcode_oxide_v062.json \
  --csv target/bench-logs/transcode_oxide_v062.csv
```

Measured results:

| Path | MP/s | Wall | GPU | IDCT + row |
| --- | ---: | ---: | ---: | ---: |
| Original CUDA C | 396.300 | 18.025 ms | 8.049 ms | 0.855 ms |
| cuda-oxide before IDCT basis fix | 40.813 | 175.029 ms | 165.041 ms | 158.390 ms |
| cuda-oxide after IDCT basis fix | 380.411 | 18.778 ms | 9.981 ms | 2.903 ms |

PTXAS evidence for the fix:

- Before: `transcode_dwt97_idct_i16_batch` used a 4096-byte stack frame because
  the Rust IDCT basis array was materialized into per-thread local memory.
- After: `transcode_dwt97_idct_i16_batch` reports `0 bytes stack frame`,
  `0 bytes spill stores`, and `0 bytes spill loads`.

The remaining gap is IDCT register pressure. The cuda-oxide IDCT kernel reports
94 registers after removing the stack frame, while the CUDA C path remains the
compatibility/performance baseline.

## Metal Status

Metal acceleration is selective. Public claims should say Metal-accelerated
stages, not complete end-to-end Metal coverage for every encode, decode, or
transcode route.

The consolidated codec-only Metal plan lives in
`docs/roadmap/metal-codec-acceleration-plan.md`. Existing Apple Silicon
encode-stage routing evidence was collected with:

```bash
cargo test -p j2k-metal --test encode_auto_routing_benchmark -- --ignored --nocapture
```

Decode routing evidence can be collected with:

```bash
cargo test -p j2k-metal --release --test metal_decode_benchmark \
  metal_decode_benchmark -- --ignored --nocapture
```

On June 28, 2026, commit `931d5815` plus local working-tree changes was tested
on a MacBook Pro `Mac16,8`, Apple M4 Pro, 48 GB memory, macOS 26.5. Generated
fixture rows showed no strict Metal decode win over CPU, so `Auto` decode
should remain CPU for these shapes:

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
June 28, 2026, commit `931d5815` plus local working-tree changes, on a MacBook
Pro `Mac16,8`, Apple M4 Pro, 48 GB memory, macOS 26.5:

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
