# Unsafe Audit

This audit lists Rust sources that contain `unsafe` and records the invariants
expected for broad adoption. Keep this file in sync with:

```sh
cargo xtask unsafe-audit
```

## Safety invariants

- CPU feature detection must gate every architecture-specific SIMD dispatch
  before an intrinsic implementation is called.
- SIMD load/store bounds must be proven by slice lengths, loop trip counts, or
  fixed block geometry before pointer arithmetic executes.
- FFI calls must validate library availability, pointer lifetimes, buffer
  lengths, and error returns before exposing results to safe Rust callers.
- Device-runtime wrappers must surface allocation, copy, kernel, and library
  errors instead of silently falling back for explicit device requests.
- New unsafe blocks should include a local `SAFETY:` comment unless the function
  body is already documented as an unsafe intrinsic implementation.

## Source inventory

- `crates/signinum-core/src/backend.rs` - CPU feature detection for x86/x86_64
  CPUID and XCR0.
- `crates/signinum-core/tests/repo_integrity.rs` - textual audit guard for
  unsafe source inventory.
- `crates/signinum-cuda-runtime/src/lib.rs` - CUDA Driver API and kernel-launch
  wrappers.
- `crates/signinum-cuda-runtime/src/nvjpeg.rs` - dynamically loaded nvJPEG FFI.
- `crates/signinum-j2k-compare/src/grok.rs` - Grok comparator FFI boundary.
- `crates/signinum-j2k-compare/src/openjpeg.rs` - OpenJPEG comparator FFI
  boundary.
- `crates/signinum-j2k-metal/src/compute.rs` - Metal buffer, command, and
  mapped-memory access for J2K decode.
- `crates/signinum-j2k-metal/src/encode.rs` - Metal encode-stage dispatch and
  mapped output handling.
- `crates/signinum-j2k-metal/src/lib.rs` - Metal device/session boundary.
- `crates/signinum-j2k-metal/src/mct.rs` - Metal color-transform buffer access.
- `crates/signinum-jpeg-metal/src/compute.rs` - Metal JPEG device path buffer
  access.
- `crates/signinum-jpeg-metal/src/lib.rs` - Metal device/session boundary.
- `crates/signinum-jpeg/benches/common/libjpeg_turbo.rs` - libjpeg-turbo
  benchmark comparator FFI.
- `crates/signinum-jpeg/src/backend/mod.rs` - SIMD backend dispatch.
- `crates/signinum-jpeg/src/backend/neon.rs` - NEON color conversion and
  upsample kernels.
- `crates/signinum-jpeg/src/backend/x86.rs` - x86 SIMD color conversion and
  upsample kernels.
- `crates/signinum-jpeg/src/bench_support.rs` - benchmark-only SIMD calls.
- `crates/signinum-jpeg/src/decoder.rs` - CPU feature and platform dispatch.
- `crates/signinum-jpeg/src/entropy/sequential.rs` - bounded plane writes in
  sequential decode hot paths.
- `crates/signinum-jpeg/src/idct/avx2.rs` - AVX2 IDCT intrinsic implementation.
- `crates/signinum-jpeg/src/idct/neon.rs` - NEON IDCT intrinsic implementation.
- `crates/signinum-transcode-metal/src/metal.rs` - Metal accelerator dispatch
  and buffer access.

## Verification

Use `cargo xtask test` for behavior, `cargo xtask fuzz-build` and scheduled
fuzzing for parser coverage, and `cargo xtask unsafe-audit` for inventory
drift. Miri is useful for safe interpreter-compatible units, but it cannot run
the SIMD, CUDA, Metal, or external FFI paths directly. sanitizer runs should
target native parser/decode tests and fuzz reproducers on hosts where the
required C/driver libraries are available.

