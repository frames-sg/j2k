# j2k-cuda

CUDA adapter for JPEG 2000 / HTJ2K decode, GPU-resident encode, and the shared
encode stages.

CPU surface requests return host-memory surfaces. `Auto` can return CUDA-resident
surfaces for promoted workloads when CUDA is available; other `Auto` requests
stay on the CPU. Explicit CUDA decode supports both classic JPEG 2000 and HTJ2K in the
adapter's supported image and code-block profiles, including reversible 5/3 and
irreversible 9/7 transforms. Unsupported explicit CUDA requests return an error.

`Auto` sends a decode to CUDA only for the measured cases listed in
[`xtask/routing-promotion-evidence.json`](https://github.com/frames-sg/j2k/blob/main/xtask/routing-promotion-evidence.json):
`Gray8` or `Rgb8` output of classic codestreams, lossy HTJ2K codestreams, and
lossless HTJ2K JPH files, as a full image, a region, a half-scale image, or a
batch of 16 copies of one input, each above a minimum size. Components wider
than 16 bits are not decoded on CUDA.

Device decodes follow the CPU's sample conversion for signed grayscale and
color sources, including prepared and referenced batches. Grayscale and color
region decodes plan only the code blocks, packets, and wavelet levels the region
needs, and run each inverse DWT level over the region's window rather than whole
bands.

Encode from CUDA buffers is separate: it requires HTJ2K block coding and external
validation, and rejects classic JPEG 2000 block coding. Strict host-input encode
also requires supported CUDA stages throughout. The shared encode-stage adapter
can accelerate individual stages without providing a complete classic encoder.

The host-memory paths and the session types build by default. Enable the
`cuda-runtime` feature for CUDA Driver API dispatch, CUDA-resident surface and
buffer types, and the CUDA-buffer encode APIs. Without it, explicit CUDA
requests return `CudaUnavailable` or an unsupported-request error.

## Host-input lossless encode routing

`CudaLosslessEncoder::encode` follows the `EncodeBackendPreference` in each
job's options:

- `CpuOnly` never touches CUDA.
- `Auto` uses CUDA only for a shape whose CUDA route was measured faster than
  the CPU, as decode routing does. No encode shape is listed in the routing
  evidence today, so `Auto` encodes on the CPU. That is the chosen route, not
  a fallback, so the result carries no fallback reason.
- `RequireDevice` fails unless CUDA runs the whole encode.

A CUDA error is never hidden by retrying on the CPU. The result reports the
requested preference, the backend that completed the encode, the reason when
an `Auto` encode offered to CUDA fell back to the CPU, and the number of CUDA
dispatches per stage.

```rust
use j2k::{
    EncodeBackendPreference, J2kBlockCodingMode, J2kLosslessEncodeOptions,
    J2kLosslessSamples,
};
use j2k_cuda::CudaLosslessEncoder;

let pixels = [0_u8; 16 * 16];
let samples = J2kLosslessSamples::new(&pixels, 16, 16, 1, 8, false)?;
let options = J2kLosslessEncodeOptions::default()
    .with_backend(EncodeBackendPreference::Auto)
    .with_block_coding_mode(J2kBlockCodingMode::HighThroughput);
let mut encoder = CudaLosslessEncoder::new();
let result = encoder.encode(samples, &options)?;

assert_eq!(result.requested_backend(), EncodeBackendPreference::Auto);
# Ok::<(), Box<dyn std::error::Error>>(())
```

`CudaLosslessEncoder::encode_strict_cuda` and the older
`encode_j2k_lossless_with_cuda` function always use `RequireDevice`, whatever
the options say. The encoder can be moved between threads but not shared: each
call takes `&mut self`. Input, routing, and runtime errors clear the cached GPU
state before the next job.

Enabling `cuda-runtime` does not guarantee that the CUDA Oxide kernels were
built. PTX is only generated on Linux hosts with the cuda-oxide toolchain;
other builds may embed placeholder PTX. Set `J2K_REQUIRE_CUDA_OXIDE_BUILD=1` on
CUDA test and benchmark hosts to make missing PTX a build error. Calling a
placeholder kernel returns an error saying the PTX was not built.

NVIDIA performance numbers come from self-hosted benchmark runs.

## Links

- API docs: <https://docs.rs/j2k-cuda>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
