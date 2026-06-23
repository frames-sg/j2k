# CUDA Benchmark Evidence

This file keeps exact command lines for the test-only NVIDIA comparator harness.
Public docs summarize the evidence without exposing this harness as a supported
public interface.

## CUDA JPEG-to-HTJ2K Transcode

Original CUDA C baseline command:

```bash
export J2K_REQUIRE_CUDA_RUNTIME=1
export J2K_REQUIRE_CUDA_KERNEL_BUILD=1
export J2K_BENCH_JPEG_DIR=tests/nvidia-baseline/benchtiles/pancreas
cargo run --release --manifest-path tests/nvidia-baseline/Cargo.toml \
  --features nvjpeg2000 --bin transcode_compare -- \
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
export J2K_BENCH_JPEG_DIR=tests/nvidia-baseline/benchtiles/pancreas
export J2K_CUDA_OXIDE_ARCH=sm_89
export J2K_REQUIRE_CUDA_OXIDE_TRANSCODE=1
export J2K_REQUIRE_CUDA_OXIDE_J2K_ENCODE=1
export J2K_CUDA_USE_OXIDE_TRANSCODE=1
export J2K_CUDA_USE_OXIDE_J2K_ENCODE=1
cargo run --release --manifest-path tests/nvidia-baseline/Cargo.toml \
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
