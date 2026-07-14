# j2k-ml

Experimental JPEG 2000 and HTJ2K tensor decoding for Burn 0.21.

`j2k-ml` is an independent integration maintained by the `j2k` project. It is
not an official Tracel or Burn crate. It remains unpublished during the 0.7
release cycle.

It decodes JP2, JPH, raw J2K, and raw HTJ2K inputs into rank-3 or rank-4
Burn tensors. Defaults are channels-first layout, automatic Gray/RGB channel
selection, and unit-scaled `f32` output.

Enable one or more explicit routes:

- `cpu`: portable host decode into any Burn backend.
- `metal`: strict resident J2K Metal decode, one packed batch readback, and one
  compact upload to Burn Metal.
- `cuda`: strict direct decode and conversion into Burn's default fused CUDA
  allocation. Device conversion kernels are Rust `cuda-oxide`; CUDA C, NVCC,
  and C wrappers are not used.

Every real decode API is fallible. Accelerator routes fail instead of silently
falling back, and batches reject corrupt items and shape mismatches with their
input index. See the [workspace integration guide](../../docs/j2k-ml.md).
