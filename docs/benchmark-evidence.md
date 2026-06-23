# Benchmark Evidence

This document records benchmark commands and environment details used for public
performance claims. JSON and CSV artifacts remain the source of truth when they
are produced by a benchmark harness.

## CUDA JPEG-to-HTJ2K Transcode

Host:

- Machine: `self-hosted NVIDIA benchmark host`
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

Existing Apple Silicon encode-stage routing evidence is recorded in
`docs/roadmap/metal-encode-stage-coverage.md`. That evidence was collected with:

```bash
cargo test -p j2k-metal --test encode_auto_routing_benchmark -- --ignored --nocapture
```

Release validation should rerun the hardware tests before publishing:

```bash
cargo test -p j2k-jpeg-metal --all-targets
cargo test -p j2k-metal --all-targets
cargo test -p j2k-metal --test encode_auto_routing_benchmark -- --ignored --nocapture
```

If Metal hardware is unavailable, do not expand Metal performance claims for
the release.
