# signinum-jpeg-cuda

CUDA-facing device-output adapter for `signinum-jpeg`.

Install this crate when a pipeline needs JPEG output in CUDA device memory:

```sh
cargo add signinum-jpeg-cuda --features cuda-runtime
```

`BackendRequest::Cpu` and `BackendRequest::Auto` return host-backed CPU
surfaces. `BackendRequest::Cuda` requires the `cuda-runtime` feature and an
available CUDA driver. For supported full-frame RGB8 4:2:0, 4:2:2, and 4:4:4
YCbCr JPEG decode, strict CUDA requests use Signinum-owned CUDA kernels and
return CUDA-backed `DeviceSurface`s without first decoding to a host RGB buffer.
Region, scaled, and non-RGB8 strict CUDA requests return clear unsupported
errors.

For hot loops, reuse a `CudaSession`. The session caches owned-kernel JPEG
packet state and can provide reusable caller-managed CUDA output buffers for
direct decode via `Codec::decode_tile_rgb8_into_cuda_buffer_with_session`.

Use `cargo bench -p signinum-jpeg-cuda --bench device_decode --features
cuda-runtime` on an NVIDIA host to compare CPU decode, CUDA surface production
through a reused `CudaSession`, and decode-plus-download timing.
Set `SIGNINUM_GPU_BENCH_DIM=4096` for the generated large-tile benchmark, or
set `SIGNINUM_CUDA_BENCH_JPEG` to a large WSI-shaped JPEG tile. The same bench
also compares a CPU batch loop and the public `TileBatchDecodeManyDevice`
adapter path; tune it with `SIGNINUM_GPU_BENCH_BATCH` and
`SIGNINUM_GPU_BENCH_BATCH_DIM`. Generated JPEGs default to 4:2:0; set
`SIGNINUM_CUDA_BENCH_SUBSAMPLING=422` or `444` to benchmark the other owned
CUDA RGB8 kernels.

The stable CPU decode API lives in `signinum-jpeg`.
