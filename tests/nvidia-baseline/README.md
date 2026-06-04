# signinum-nvidia-baseline

NVIDIA GPU codec baseline for benchmarking signinum's **coefficient-domain**
JPEG → HTJ2K transcode against the conventional **decode-to-pixels-then-encode**
pipeline.

| Path | Pipeline |
| --- | --- |
| signinum | JPEG → (entropy decode, CPU) → DCT-grid 9/7 + quantize (GPU) → optional CUDA HT encode → HTJ2K |
| NVIDIA | JPEG → pixels (nvJPEG, GPU) → HTJ2K (nvJPEG2000 HT encode, GPU) |

There is no NVIDIA library that transcodes in the coefficient domain, so the fair
comparison is **end-to-end JPEG → HTJ2K throughput**. The two paths produce
different codestreams (signinum skips the pixel round-trip), so size and PSNR are
reported alongside throughput.

## Status

The NVIDIA side (`cuda/nv_baseline.cu`) is compiled and linked **only** with the
`nvjpeg2000` feature, on a host that has `nvcc`, `libnvjpeg` (ships with the CUDA
toolkit), and a **separately-installed `libnvjpeg2k`** (nvJPEG2000 ≥ 0.9.0 for HT
encode). Without it the crate still builds and the harness runs the signinum side,
showing `n/a (not built)` for the NVIDIA columns. The Rust side is validated on
non-CUDA hosts; the FFI itself is validated on the CUDA runner.

## Running the comparison

After installing nvJPEG2000 on the runner:

```bash
# Optional: point at a directory of representative WSI JPEG tiles.
export SIGNINUM_BENCH_JPEG_DIR=/path/to/jpeg/tiles

# Require the C++ baseline to actually build (no silent stub).
export SIGNINUM_REQUIRE_NV_BASELINE_BUILD=1

# If nvJPEG2000 is not under /usr/local/cuda, point at it:
# export NVJPEG2K_LIB_DIR=/opt/nvjpeg2000/lib
# export NVJPEG2K_INCLUDE_DIR=/opt/nvjpeg2000/include

cargo run --release --manifest-path tests/nvidia-baseline/Cargo.toml \
  --features nvjpeg2000 --bin transcode_compare -- tile0.jpg tile1.jpg ...
```

Direct nvJPEG2000 decode comparison uses HTJ2K/J2K codestream inputs. On the
CUDA runner, pass the pathology JPEG tile directory so the comparator first
uses NVIDIA's encoder to produce HTJ2K inputs, then measures signinum CPU,
signinum CUDA, and NVIDIA direct decode on the same codestreams:

```bash
cargo run --release --manifest-path tests/nvidia-baseline/Cargo.toml \
  --features nvjpeg2000 --bin decode_compare -- \
  --jpeg-dir tests/nvidia-baseline/benchtiles/pancreas \
  --json target/decode_compare.json \
  --csv target/decode_compare.csv
```

With neither `--jpeg-dir` nor file arguments, it generates 512-pixel Gray8 and
RGB8 HTJ2K fixtures.

With no file arguments and no `SIGNINUM_BENCH_JPEG_DIR`, `transcode_compare`
falls back to a tiny bundled 16×16 fixture (a build/link smoke test, not a
representative benchmark).

## Reported metrics

- **End-to-end throughput** (MP/s), signinum vs NVIDIA.
- **Wall-clock vs GPU-stage time** — signinum is a CPU/GPU hybrid; NVIDIA uses
  GPU codec stages plus host-side API orchestration.
- **Per-stage breakdown** — signinum's pack/upload, IDCT+row-lift, column-lift,
  quantize, readback, and CUDA HT encode dispatches when enabled; NVIDIA's
  nvJPEG decode and nvJPEG2000 HT encode.
- **Output size + PSNR** — codestream bytes and reconstruction quality vs the
  nvJPEG-decoded source RGB. PSNR is reported as not rate-matched.
- **Direct decode comparator** — signinum CPU decode, signinum strict CUDA
  decode with host download, and NVIDIA nvJPEG2000 decode wall/GPU time for the
  same HTJ2K/J2K codestreams.

Throughput uses the best of `ITERATIONS` runs after warmup. signinum runs through
the batch transform path and, on CUDA builds, the CUDA HT encode accelerator.
The NVIDIA path reuses CUDA stream/events, nvJPEG handles/state, nvJPEG2000
encoder/state/params, and device RGB planes across tiles, but encodes tiles
serially. The supported nvJPEG2000 encode API exposes `nvjpeg2kEncode` and
`nvjpeg2kEncodeRetrieveBitstream`, not an encode-batch entry point, so report it
as **reused-session serial** unless the runner headers prove otherwise.

## CUDA JPEG -> HTJ2K transcode speed gate

The CUDA transcode gate compares two different implementations of the same
end-to-end operation:

| Row | Operation |
| --- | --- |
| Signinum CUDA HT | JPEG entropy/DCT extraction, coefficient-domain DCT -> 9/7 HTJ2K transform, CUDA HT code-block encode, CPU packet assembly |
| NVIDIA reused-session serial | nvJPEG decode to pixels, then nvJPEG2000 HT encode from those pixels |

This is not a claim that NVIDIA exposes a matching coefficient-domain
transcoder. It does not. The fair scope is JPEG -> HTJ2K end-to-end throughput
at similar output bytes, with PSNR and per-channel PSNR reported beside the
speed rows.

The checked-in WSL evidence bundle is:

```text
tests/nvidia-baseline/artifacts/2026-06-03-cuda-transcode-rate-match/
```

The JSON files in that directory are the source of truth; CSV is included only
for review convenience. The CI workflow regenerates equivalent JSON/CSV reports
under `target/` and then runs `scripts/assert_transcode_perf.py` against the
JSON.

Level 1 rate-matched command:

```bash
cd /home/jcwal/signinum-codex-batch
unset RUSTFLAGS
export SIGNINUM_BENCH_JPEG_DIR=tests/nvidia-baseline/benchtiles/pancreas
cargo run --release --manifest-path tests/nvidia-baseline/Cargo.toml \
  --features nvjpeg2000 --bin transcode_compare -- \
  --decomposition-levels 1 \
  --quant-scales 1.90 \
  --match-nvidia-bytes \
  --match-tolerance 0.20 \
  --min-tiles 64 \
  --warmup 3 \
  --iterations 8 \
  --json /tmp/signinum_level1_ratematch_confirm_after_fixes.json \
  --csv /tmp/signinum_level1_ratematch_confirm_after_fixes.csv
```

Observed on 2026-06-03:

- Signinum CUDA HT: 441.0 MP/s, 16.197 ms, bytes -0.54% vs NVIDIA.
- NVIDIA reused-session serial: 49.4 MP/s, 144.690 ms.
- Aggregate PSNR: Signinum 47.63 dB, NVIDIA 49.19 dB.

Level 2 rate-matched command:

```bash
cd /home/jcwal/signinum-codex-batch
unset RUSTFLAGS
export SIGNINUM_BENCH_JPEG_DIR=tests/nvidia-baseline/benchtiles/pancreas
cargo run --release --manifest-path tests/nvidia-baseline/Cargo.toml \
  --features nvjpeg2000 --bin transcode_compare -- \
  --decomposition-levels 2 \
  --quant-scales 1.00 \
  --subband-scales 1.00,1.00,1.00,1.00 \
  --match-nvidia-bytes \
  --match-tolerance 0.20 \
  --min-tiles 64 \
  --warmup 3 \
  --iterations 8 \
  --json /tmp/signinum_level2_confirm_after_fixes.json \
  --csv /tmp/signinum_level2_confirm_after_fixes.csv
```

Observed on 2026-06-03:

- Signinum CUDA HT: 67.7 MP/s, 105.463 ms, bytes +0.13% vs NVIDIA.
- Signinum CUDA transform + CPU HT: 74.0 MP/s, 96.527 ms.
- NVIDIA reused-session serial: 51.3 MP/s, 139.243 ms.
- Aggregate PSNR: Signinum 48.69 dB, NVIDIA 49.19 dB.

CI thresholds are intentionally below the best observed numbers to avoid
turning normal runner noise into false failures:

| Gate | Default threshold |
| --- | --- |
| `SIGNINUM_LEVEL1_CUDA_HT_MIN_MPS` | 350 MP/s |
| `SIGNINUM_LEVEL1_CUDA_HT_MIN_SPEEDUP_VS_NVIDIA` | 4.0x |
| `SIGNINUM_LEVEL2_CUDA_HT_MIN_MPS` | 60 MP/s |
| `SIGNINUM_LEVEL2_CUDA_HT_MIN_SPEEDUP_VS_NVIDIA` | 1.10x |

The gate also requires at least one CUDA HT code-block dispatch and fails if
the JSON byte delta exceeds the configured rate-match tolerance.
