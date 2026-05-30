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

cargo run --release -p signinum-nvidia-baseline \
  --features nvjpeg2000 --bin transcode_compare -- tile0.jpg tile1.jpg ...
```

With no file arguments and no `SIGNINUM_BENCH_JPEG_DIR`, it falls back to a tiny
bundled 16×16 fixture (a build/link smoke test, not a representative benchmark).

## Reported metrics

- **End-to-end throughput** (MP/s), signinum vs NVIDIA.
- **Wall-clock vs GPU-stage time** — signinum is a CPU/GPU hybrid; NVIDIA uses
  GPU codec stages plus host-side API orchestration.
- **Per-stage breakdown** — signinum's pack/upload, IDCT+row-lift, column-lift,
  quantize, readback, and CUDA HT encode dispatches when enabled; NVIDIA's
  nvJPEG decode and nvJPEG2000 HT encode.
- **Output size + PSNR** — codestream bytes and reconstruction quality vs the
  nvJPEG-decoded source RGB. PSNR is reported as not rate-matched.

Throughput uses the best of `ITERATIONS` runs after warmup. signinum runs through
the batch transform path and, on CUDA builds, the CUDA HT encode accelerator.
The NVIDIA path reuses CUDA stream/events, nvJPEG handles/state, nvJPEG2000
encoder/state/params, and device RGB planes across tiles, but encodes tiles
serially. The supported nvJPEG2000 encode API exposes `nvjpeg2kEncode` and
`nvjpeg2kEncodeRetrieveBitstream`, not an encode-batch entry point, so report it
as **reused-session serial** unless the runner headers prove otherwise.
