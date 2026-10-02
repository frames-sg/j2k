# j2k-cuda

CUDA adapter for JPEG 2000 / HTJ2K decode, GPU-resident encode, and the shared
encode stages.

CPU and `Auto` surface requests may return host-memory surfaces. Explicit CUDA
decode to GPU memory and encode from CUDA buffers use J2K's own kernels and
currently support HTJ2K only; classic J2K code blocks are rejected on those
paths. The shared encode-stage adapter can still accelerate individual stages
for other inputs. Unsupported explicit CUDA requests return an error.

The host-memory paths and the session types build by default. Enable the
`cuda-runtime` feature for CUDA Driver API dispatch, CUDA-resident surface and
buffer types, and the CUDA-buffer encode APIs. Without it, explicit CUDA
requests return `CudaUnavailable` or an unsupported-request error.

## Host-input lossless encode routing

`CudaLosslessEncoder::encode` follows the `EncodeBackendPreference` in each
job's options. `CpuOnly` never touches CUDA. `Auto` uses CUDA for the stages it
supports and returns a CPU result when no device is available or CUDA cannot
run every stage. `RequireDevice` fails unless CUDA runs the whole encode. A
CUDA error is never hidden by retrying on the CPU.

The result reports the requested preference, the backend that completed the
encode, the reason for any `Auto` fallback, and the number of CUDA dispatches
per stage. A CPU backend with nonzero CUDA dispatches means CUDA ran some
stages and the CPU finished the encode.

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
