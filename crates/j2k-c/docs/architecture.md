# j2k-c boundaries

One adapter crate over the `j2k` crate in the same workspace. It has no fork
of the codec.

| File | Responsibility |
| --- | --- |
| `crates/j2k-c/src/abi.rs` | Fixed-width C values and versioned structures |
| `crates/j2k-c/src/ffi.rs` | Raw pointers, handle ownership, prefix validation, unwind containment |
| `crates/j2k-c/src/decoder.rs` | Safe owned J2K session, one cached prepared request, limits, native sample copy, the decoder's own thread pool |
| `crates/j2k-c/src/operations.rs` | Exact region planning and decode, sRGB output, operation limits and cancellation |
| `crates/j2k-c/src/rows.rs` | Row streaming through the caller's callback |
| `crates/j2k-c/src/encode.rs` | Lossless and lossy encode, JP2/JPH wrapping |
| `crates/j2k-c/src/error.rs` | Typed codec/batch failure mapping and bounded diagnostics |
| `crates/j2k-c/include/j2k.h` | C contract mirrored by the ABI structs and exercised by a separately compiled consumer |
| `crates/j2k-c/include/j2k.hpp` | Header-only C++ ownership and error conversion |
| `crates/j2k-c/native/` | C/C++ examples and installed-package consumer tests |
| `crates/j2k-c/cmake/`, `crates/j2k-c/pkgconfig/` | CMake package and pkg-config templates |
| `crates/j2k-c/scripts/package.py` | Native shared-library build, relocatable package, dependency license notices |

`j2k` owns image parsing, support classification, geometry, preparation,
entropy decode, inverse transforms, color interpretation, and encoding. The
library uses `EncodedImage`, `CpuBatchDecoder`, and `PreparedBatch` to keep
input and workspaces, and `J2kDecoder` for row streaming and sRGB regions. It
copies completed samples into a caller-owned byte buffer and returns encoded
output in a buffer the caller frees with `j2k_buffer_free`. A decoder opened
with `workers` 1-64 owns a Rayon pool of that size; otherwise work runs on the
codec's shared pool. The library does not implement codec algorithms or
additional color or sample transformations.

All raw pointer operations stay in `ffi.rs`. Every unsafe block documents its
storage/ownership assumption. The boundary uses opaque handles, fixed-width
status values, size/version prefixes, and per-call caller-owned diagnostics.
No Rust enum, string, vector, or exception crosses the ABI.

The header is maintained with the Rust declarations. Layout changes require
contract tests and a C/C++ consumer build. Do not add an ABI symbol without
ownership/error documentation and a representative caller-level test.

Codec behavior belongs in `j2k` and `j2k-native`; this crate owns
foreign-language integration, packaging, and its documented support contract.
The crate is `publish = false`: it ships as native packages built by
`scripts/package.py`, which never publishes a release or uploads to a registry.
