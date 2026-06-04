# CUDA HTJ2K Batch Correctness Comparison

Date: 2026-06-04

Machine:

- Host: `jcwal@cuda-wsl`
- GPU: NVIDIA GeForce RTX 4070 SUPER
- Driver: 596.49
- Rust: `rustc 1.88.0`

Comparison scope:

- Signinum CUDA is decoded once as a real aggregate batch and downloaded once for per-input PSNR checks.
- Per-input Signinum rows are correctness rows only; Signinum timing lives in the aggregate row.
- NVIDIA nvJPEG2000 is decoded per input for direct JPEG 2000 comparison.
- Inputs: first 64 JPEG tiles under `tests/nvidia-baseline/benchtiles/pancreas`, converted to HTJ2K with NVIDIA before timed decode.
- Total pixels: 4.194304 MP.
- CSV/JSON artifacts in this directory are the source of truth.

Command:

```bash
cd /home/jcwal/signinum-codex-decode-clean

SIGNINUM_REQUIRE_NV_BASELINE_BUILD=1 cargo run \
  --manifest-path tests/nvidia-baseline/Cargo.toml \
  --features nvjpeg2000 \
  --bin decode_compare -- \
  --compare-signinum-cuda-batch \
  --collect-signinum-stage-timings \
  --jpeg-dir tests/nvidia-baseline/benchtiles/pancreas \
  --max-inputs 64 \
  --min-inputs 64 \
  --warmup 1 \
  --iterations 5 \
  --json /tmp/signinum_decode_batch_correctness.json \
  --csv /tmp/signinum_decode_batch_correctness.csv
```

Results:

| Row | MP/s | Wall ms | GPU ms | Stage ms | Download ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Signinum CUDA batch correctness aggregate | 126.556 | 33.142 | 7.744 | 24.765 | 1.261 |

Per-input rows include CPU timing, NVIDIA timing, Signinum-vs-CPU PSNR, and NVIDIA-vs-CPU PSNR. They intentionally do not contain per-input Signinum timing because the Signinum decode was one aggregate batch.
