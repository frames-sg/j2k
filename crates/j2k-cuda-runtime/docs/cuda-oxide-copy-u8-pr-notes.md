# cuda-oxide CopyU8 Spike Notes

## Summary

- Adds an opt-in `cuda-oxide-copy-u8` feature for a Rust-authored `j2k_copy_u8`
  CUDA kernel.
- Keeps the existing checked-in PTX `CudaKernel::CopyU8` path as the default.
- Loads the cuda-oxide PTX through the existing `CudaContext` Driver API module
  boundary with a separate module-cache key, so parity can compare both paths in
  one context.

## How To Enable

```bash
cargo test -p j2k-cuda-runtime --features cuda-oxide-copy-u8 --all-targets
```

The build script stages the cuda-oxide device crate into `OUT_DIR`, runs
`cargo oxide build --arch ${J2K_CUDA_OXIDE_ARCH:-sm_80}`, then includes the
generated NUL-terminated PTX when cuda-oxide is available. Ordinary
`--all-features` builds on unsupported hosts write a placeholder PTX and do not
set the generated-PTX cfg; runtime dispatch returns a typed error before loading
that placeholder. Set `J2K_REQUIRE_CUDA_OXIDE_COPY_U8=1` on a Linux cuda-oxide
host to make a missing generated PTX a build failure.

## Build Friction

- cuda-oxide is documented as an early alpha with expected bugs and API
  breakage.
- cuda-oxide is currently Linux-only. On this macOS host, ordinary
  `--all-features` builds skip generation with a warning; strict validation uses
  `J2K_REQUIRE_CUDA_OXIDE_COPY_U8=1`.
- The documented toolchain is heavier than the existing PTX fallback: pinned
  Rust nightly, CUDA Toolkit, LLVM 21+, Clang 21+, and `cargo-oxide`.
- The nested device crate follows cuda-oxide's standalone project template with
  git dependencies pinned to `NVlabs/cuda-oxide` commit
  `a9f964a956f397dd0b3c8db88a3ca5824186c261`. Broader migration should move
  that pin through the normal dependency review process once the spike
  graduates.

## Migration Viability

CopyU8 is viable as an isolated opt-in spike because it has a simple raw-pointer
ABI, no shared memory, and an existing CPU/CUDA parity surface. Broader kernel
migration should remain gated until a Linux CUDA runner validates generated PTX,
records build time, and confirms the pinned cuda-oxide toolchain remains
destabilizing default builds.

## Guidance Applied

- cuda-oxide docs: used `#[kernel]` plus a `#[cuda_module]` device crate; kept
  the function name `j2k_copy_u8` because cuda-oxide preserves the original
  function name as the PTX entry point; defaulted the basic build target to
  `sm_80` with `J2K_CUDA_OXIDE_ARCH` override. Sources:
  [book quick start](https://nvlabs.github.io/cuda-oxide/index.html),
  [installation](https://nvlabs.github.io/cuda-oxide/getting-started/installation.html),
  [launch config](https://nvlabs.github.io/cuda-oxide/gpu-programming/launching-kernels.html).
- Cargo features: kept `cuda-oxide-copy-u8` additive and out of `default`; the
  existing PTX path is not disabled. Source:
  [Cargo feature unification](https://doc.rust-lang.org/cargo/reference/features.html#feature-unification).
- Rust API Guidelines: used a direct, meaningful feature name without `use-` or
  `with-`, and made unsupported explicit builds fail at the boundary. Sources:
  [C-FEATURE](https://rust-lang.github.io/api-guidelines/naming.html#c-feature),
  [C-VALIDATE](https://rust-lang.github.io/api-guidelines/dependability.html#c-validate).
- Clippy docs: did not add blanket lint allowances; the code stays within the
  crate's existing pedantic lint setup and would only use targeted allowances if
  a specific lint needed justification. Source:
  [Clippy lint groups](https://doc.rust-lang.org/clippy/lints.html).
- Unsafe Code Guidelines: kept raw pointer unsafety inside the Rust-authored GPU
  kernel and existing Driver API boundary, while exposing the same safe
  `CudaKernelOutput` API to callers. Source:
  [UCG glossary: soundness and unsafe burden](https://rust-lang.github.io/unsafe-code-guidelines/glossary.html#soundness-of-code--of-a-library).
