# CUDA HTJ2K Decode Batch Baseline

Date: 2026-06-04

Machine:

- Host: `jcwal@cuda-wsl`
- GPU: NVIDIA GeForce RTX 4070 SUPER
- Driver: 596.49
- Rust: `rustc 1.88.0`

Comparison scope:

- Signinum row: real CUDA HTJ2K batch decode through `J2kDecoder::decode_batch_to_device_with_session`.
- NVIDIA row: nvJPEG2000 direct JPEG 2000 decode of the same HTJ2K codestreams.
- Inputs: first 64 JPEG tiles under `tests/nvidia-baseline/benchtiles/pancreas`, converted to HTJ2K with NVIDIA before timed decode.
- Total pixels: 4.194304 MP.
- CSV/JSON artifacts in this directory are the source of truth.

Commands:

```bash
cd /home/jcwal/signinum-codex-decode-clean

SIGNINUM_REQUIRE_NV_BASELINE_BUILD=1 cargo run --release \
  --manifest-path tests/nvidia-baseline/Cargo.toml \
  --features nvjpeg2000 \
  --bin decode_compare -- \
  --profile-signinum-cuda-batch \
  --collect-signinum-stage-timings \
  --skip-signinum-download \
  --jpeg-dir tests/nvidia-baseline/benchtiles/pancreas \
  --max-inputs 64 \
  --min-inputs 64 \
  --warmup 1 \
  --iterations 5 \
  --json /tmp/signinum_decode_batch_no_download.json \
  --csv /tmp/signinum_decode_batch_no_download.csv

SIGNINUM_REQUIRE_NV_BASELINE_BUILD=1 cargo run --release \
  --manifest-path tests/nvidia-baseline/Cargo.toml \
  --features nvjpeg2000 \
  --bin decode_compare -- \
  --profile-signinum-cuda-batch \
  --collect-signinum-stage-timings \
  --jpeg-dir tests/nvidia-baseline/benchtiles/pancreas \
  --max-inputs 64 \
  --min-inputs 64 \
  --warmup 1 \
  --iterations 5 \
  --json /tmp/signinum_decode_batch_download.json \
  --csv /tmp/signinum_decode_batch_download.csv

SIGNINUM_REQUIRE_NV_BASELINE_BUILD=1 cargo run --release \
  --manifest-path tests/nvidia-baseline/Cargo.toml \
  --features nvjpeg2000 \
  --bin decode_compare -- \
  --jpeg-dir tests/nvidia-baseline/benchtiles/pancreas \
  --max-inputs 64 \
  --min-inputs 64 \
  --warmup 1 \
  --iterations 5 \
  --json /tmp/signinum_decode_normal_comparison.json \
  --csv /tmp/signinum_decode_normal_comparison.csv
```

Results:

| Row | MP/s | Wall ms | GPU ms | Stage ms | Download ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Signinum CUDA batch no-download | 487.127 | 8.610 | 4.121 | 6.587 | 0.000 |
| Signinum CUDA batch download | 442.449 | 9.480 | 4.001 | 6.351 | 1.070 |
| Signinum CUDA serial correctness rows | 58.784 | 71.351 | 53.909 | 61.459 | 3.577 |
| NVIDIA nvJPEG2000 direct decode rows | 85.623 | 48.986 | 42.615 | n/a | n/a |

Average NVIDIA PSNR vs CPU over the 64 normal comparison rows: 51.424 dB.
