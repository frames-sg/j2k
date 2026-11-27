# j2k-c module layout

`j2k-c` is a thin adapter over the `j2k` crate in the same workspace. It
doesn't fork or reimplement any codec code.

| Path | Responsibility |
| --- | --- |
| `src/abi.rs` | Fixed-width C values and versioned structs |
| `src/ffi.rs` | Raw pointers, handle ownership, struct-header checks, catching panics |
| `src/decoder.rs` | The owned decoder: one cached prepared request, limits, native sample copies, the decoder's own thread pool |
| `src/operations.rs` | Exact region planning and decoding, sRGB output, operation limits and cancellation |
| `src/rows.rs` | Row streaming through the caller's callback |
| `src/encode.rs` | Lossless and lossy encoding, JP2/JPH wrapping |
| `src/error.rs` | Mapping codec and batch errors to status codes, bounded messages |
| `include/j2k.h` | The C API, mirroring the structs in `src/abi.rs`; compiled by a separate consumer test |
| `include/j2k.hpp` | Header-only C++ ownership and exceptions |
| `native/` | C and C++ examples, and the consumer tests run against an installed package |
| `cmake/`, `pkgconfig/` | CMake package and pkg-config templates |
| `scripts/package.py` | Builds the shared library and the relocatable package, and collects dependency license notices |

`j2k` does the parsing, support checks, geometry, preparation, entropy
decoding, inverse transforms, color interpretation and encoding. The adapter
keeps the input and workspaces in `EncodedImage`, `CpuBatchDecoder` and
`PreparedBatch`, and uses `J2kDecoder` for row streaming and sRGB regions. It
copies finished samples into the caller's buffer, and returns encoded output
in a buffer the caller frees with `j2k_buffer_free`. A decoder opened with
`workers` from 1 to 64 owns a Rayon pool of that size; otherwise work runs on
the codec's shared pool. The adapter adds no codec algorithms and no color or
sample transforms of its own.

All raw-pointer code stays in `src/ffi.rs`, and every unsafe block says what
it assumes about storage and ownership. The ABI uses opaque handles,
fixed-width status values, size and version headers, and per-call error
structs owned by the caller. No Rust enum, string, vector or panic crosses
it.

`include/j2k.h` is maintained by hand alongside the Rust declarations. A
layout change needs layout tests and a C and C++ consumer build. Don't add an
ABI symbol without documenting its ownership and errors in `docs/api.md` and
adding a test that calls it the way a C caller would.

Codec behavior belongs in `j2k` and `j2k-native`. This crate owns the C and
C++ bindings, packaging, and the support statement in `docs/supported.md`. It
has `publish = false`: it ships only as native packages built by
`scripts/package.py`, which never uploads anything.
