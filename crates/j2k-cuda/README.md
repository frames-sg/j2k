# j2k-cuda

CUDA adapter for JPEG 2000 / HTJ2K decode, resident encode, and shared
encode-stage paths.

CPU and Auto surface requests may return host-backed surfaces. Strict
CUDA-resident decode and CUDA-buffer encode use J2K-owned kernels and currently
support HTJ2K codestreams: classic J2K subband plans and classic block coding
are rejected by those resident paths. Separately, the shared encode-stage
adapter can accelerate supported stages without widening the strict resident
codec contract. Unsupported explicit CUDA requests return structured errors.

Host-backed fallbacks and the shared adapter/session types compile in default
builds. Enable `cuda-runtime` for CUDA Driver API dispatch, constructible
CUDA-resident surface and buffer types, and the CUDA-buffer encode APIs. Without
that feature, strict CUDA requests cannot dispatch and return `CudaUnavailable`
or the corresponding structured unsupported-request error.

## Host-input lossless encode routing

`CudaLosslessEncoder::encode` honors the `EncodeBackendPreference` stored in
each job's options. `CpuOnly` never probes CUDA. `Auto` may use supported CUDA
stages and returns a CPU result when the runtime/device is unavailable or CUDA
does not cover every required stage. `RequireDevice` fails unless CUDA satisfies
the complete route. A CUDA execution error is never hidden by retrying the job
on the CPU.

The opaque result reports the requested preference, the backend that satisfied
the complete encode contract, any Auto fallback reason, and per-stage CUDA
dispatches. A CPU backend with nonzero CUDA dispatches means CUDA completed some
stages but the CPU satisfied the overall contract.

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

`CudaLosslessEncoder::encode_strict_cuda` and the compatibility
`encode_j2k_lossless_with_cuda` free function deliberately override the stored
preference with `RequireDevice`. The encoder can be moved between threads, but
is not shareable by reference: each call takes exclusive mutable access.
Input/route/runtime errors clear cached accelerator state before the next job.

`cuda-runtime` is not proof that every CUDA Oxide kernel was built on the local
host. Product PTX is generated only on supported Linux cuda-oxide build hosts;
other builds may embed placeholder PTX. Set `J2K_REQUIRE_CUDA_OXIDE_BUILD=1`
on CUDA validation and benchmark hosts to fail the build when PTX is missing.
Runtime errors for placeholder kernels state that CUDA Oxide PTX was not built.

NVIDIA performance claims require self-hosted benchmark evidence.

## Links

- API docs: <https://docs.rs/j2k-cuda>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
