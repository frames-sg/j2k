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

SIGNINUM_REQUIRE_NV_BASELINE_BUILD=1 cargo run \
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

SIGNINUM_REQUIRE_NV_BASELINE_BUILD=1 cargo run \
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

SIGNINUM_REQUIRE_NV_BASELINE_BUILD=1 cargo run \
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
| Signinum CUDA batch no-download | 124.054 | 33.810 | 8.126 | 26.434 | 0.000 |
| Signinum CUDA batch download | 123.750 | 33.893 | 8.170 | 25.531 | 1.147 |
| Signinum CUDA serial correctness rows | 38.371 | 109.308 | 61.624 | 90.209 | 4.189 |
| NVIDIA nvJPEG2000 direct decode rows | 79.025 | 53.075 | 44.522 | n/a | n/a |

Average NVIDIA PSNR vs CPU over the 64 normal comparison rows: 51.424 dB.
