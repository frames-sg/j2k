# Burn tensor decoding with `j2k-ml`

`j2k-ml` is an experimental, independently maintained integration for Burn
0.21. It is not an official Tracel or Burn crate and remains `publish = false`
through the 0.7 release cycle.

## Contract

The crate accepts borrowed JP2, JPH, raw J2K, and raw HTJ2K bytes together
with the repository's full, ROI, scaled, or ROI-scaled decode request. A
single decode returns a rank-3 tensor; a batch returns one rank-4 tensor while
preserving input order.

Defaults are:

- `ChannelsFirst`, producing `CHW` or `NCHW`.
- `ChannelSelection::Auto`, preserving grayscale as one channel and producing
  RGB for color inputs. Alpha is included only when `Rgba` is explicit.
- `FloatNormalization::Unit`, dividing canonical 8-bit samples by 255 or
  canonical 16-bit samples by 65535.

`decode_u8` and `decode_u16` preserve Burn's native unsigned integer dtype.
`decode_float` produces F32 storage in v1. `Raw` only casts. `MeanStd` applies
unit scaling and then `(x - mean) / std` per channel; channel counts, finite
values, and nonzero standard deviations are validated before pixel decoding.

The result also reports the actual decoded rectangle, codec warnings, and the
route used. Empty batches, corrupt items, allocation overflow, unsupported
dtypes, and mismatched final shapes are errors. The indexed batch error keeps
the original codec or route failure.

## Routes

The `cpu` feature decodes one compact interleaved integer buffer, creates one
Burn `TensorData` for the image or entire batch, and then uses Burn operations
for permutation, F32 conversion, and normalization. It works with any Burn
`Backend` that supports the required dtype.

The `cuda` feature targets Burn's default fused CUDA backend. It retains the
same CUDA primary context used by CubeCL, allocates through Burn/CubeCL, holds
the managed allocation guard while J2K writes, and registers the completed
allocation through public Burn fusion APIs. A Rust `cuda-oxide` kernel fuses
layout conversion, integer-to-F32 conversion, and normalization. There is no
decoded-pixel device-to-host readback, tensor host upload, CUDA C, NVCC, or C
wrapper. Unsupported formats, devices, and direct paths fail with
`CudaDirect`; they never fall back to CPU.

The `metal` feature requests strict resident J2K Metal decode. Native Metal
buffer sharing is not claimed: the route performs one packed readback for the
whole batch, uploads that compact integer representation once to Burn Metal,
and uses Burn operations for layout, conversion, and normalization. It uses no
private wgpu internals and reports `MetalStaged`.

## Burn data loading

The fallible single and batch functions are the primary API. Dataset storage,
labels, sampling, resizing, padding, augmentation, and worker policy remain
application-owned. When Burn's infallible `Batcher` trait is required,
`PanicOnDecodeError` is an explicitly named float-batcher adapter; its panic
includes the failed batch index and codec error.

## Deferred scope

V1 is limited to the existing interleaved Gray/RGB/RGBA 8/16-bit contract.
Arbitrary component tensors, signed native components, precision above 16
bits, F16/BF16 output, resizing, DLPack, dataset formats, and native Metal
buffer sharing are deferred.

## Validation and benchmarks

Portable tests run with `cargo test -p j2k-ml --features cpu`. Metal hardware
validation is part of `cargo xtask release-metal`; CUDA hardware validation is
part of `cargo xtask release-cuda`. Both release lanes set the repository's
fail-closed runtime gates, so missing hardware or skipped accelerator work is a
failure.

Linux AArch64 test and benchmark builds use Burn's NdArray backend. In this
repository's Burn 0.21 all-feature build, Flex selects `gemm-f16` 0.19 and its
debug AArch64 assembly is compiled without the FP16 target-feature gating that
the selected runner requires ([issue
#31](https://github.com/sarah-quinones/gemm/issues/31)). Upstream has a gating
fix under review ([pull request
#43](https://github.com/sarah-quinones/gemm/pull/43)). This substitution is
test-only; the generic `j2k-ml` library API and its release dependencies do not
select either CPU backend. Criterion group names include `flex` or
`ndarray_arm_linux`; results from those groups are not cross-backend
comparisons.

Run `cargo bench -p j2k-ml --bench tensor_decode --features cpu` for the
portable staged cases. Add `metal` on macOS or `cuda` on a Linux CUDA runner to
include the accelerator routes. Benchmark IDs and throughput record compact
upload bytes, packed Metal readback-plus-upload bytes, or zero decoded-pixel
transfer bytes for CUDA direct. CUDA should be described as the fast path only
after direct results beat its staged baseline on the target hardware.
