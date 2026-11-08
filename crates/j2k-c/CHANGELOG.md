# Changelog

## Unreleased

- Add exact region planning/decoding at native reductions 0–31 without changing
  existing ABI-1 layouts or the legacy 0–3 API. Native U8/U16/I16 values remain
  unchanged; explicit sRGB output provides bounded normalized U8 regions.
- Add optional explicit source preparation, operation working/output controls,
  owned cancellation tokens with `J2K_CANCELLED`, and a derived-plan cache limit.
  A controlled operation on an unprepared decoder prepares the source under its
  own working limit and cancellation, keeping only a successful preparation; raw
  sRGB calls parse their source under the same controls. ICC conversion under a
  working limit requires a matrix/TRC or gray-TRC profile with an XYZ
  connection space. C++ wrappers preserve token and decoder ownership, and
  exact decode can return actual warning metadata.

- Stream the full image row by row: `j2k_decoder_row_info` reports the layout
  and `j2k_decoder_decode_rows` hands each row to a `j2k_row_callback`, in order
  and once each, holding at most `max_rows_per_stripe` rows. Rows are the
  bytes `j2k_decoder_decode_into` writes for 8-bit unsigned and 9- to 16-bit
  samples, and need no whole-image buffer, so `max_output_bytes` does not limit
  them. A nonzero callback return stops the decode with the new `J2K_STOPPED`.
  The C++ `decoder::decode_rows` takes a callable and rethrows its exceptions.
- Encode losslessly with `j2k_encode_lossless` and a `j2k_encode_options`:
  classic or HTJ2K, 1- to 16-bit unsigned or signed samples, as a raw
  codestream, JP2, or JPH. The library allocates the result;
  `j2k_buffer_free` releases it. Encoder failures return the new
  `J2K_ENCODE_FAILED`. C++ callers use `j2k::encode_lossless`.
- `workers` bounds threads: 1–64 gives a decoder its own pool of exactly that
  many threads, on which preparation, decode, and row streaming run; 0 keeps
  the codec's shared pool.
- `J2K_OPEN_LENIENT` (in `j2k_open_options.flags`, formerly `reserved`) accepts
  the codec's JP2/JPH metadata recoveries; `j2k_output_info.warnings` reports
  `J2K_WARNING_LENIENT_RECOVERY` when a request applies one.
- `j2k_image_info` reports the image origin and tile grid on the reference grid
  (`image_x0`, `image_y0`, `tile_x0`, `tile_y0`, `tile_width`, `tile_height`).
- Move the C ABI and C++ wrapper from the private `j2k-sdk` repository into the
  public `j2k` workspace as `crates/j2k-c`, licensed MIT OR Apache-2.0 like the
  rest of the workspace.
  It now builds against the workspace `j2k` crate and shares its version
  instead of depending on published `j2k` 0.13.1. Packages carry the workspace
  license files and `NOTICES.md`, and collect third-party notices only for the
  packages the library links or builds with.
- Rename every public name from `j2k_sdk` to `j2k`. Functions are `j2k_*`
  (`j2k_sdk_decoder_open` is now `j2k_decoder_open`), types `j2k_*`, constants
  and macros `J2K_*`, headers `j2k.h` and `j2k.hpp`, the C++ namespace `j2k`,
  the CMake package `J2K` with target `J2K::j2k`, and the shared library
  `libj2k` (`libj2k.so.1`, `libj2k.1.dylib`). Layouts, values, and behavior are
  unchanged; programs built against `libj2k_sdk` must be rebuilt. Entries
  below use the new names.
- Reduced-resolution output follows the image's reference-grid origin. A 23x17
  image at origin (3, 5) now reports and writes 11x8, 6x4, and 3x2 at reductions
  1–3, matching OpenJPEG, instead of sizes computed as if the origin were zero.
- Full-frame decodes are no longer limited by `j2k` 0.13.1's fixed 512 MiB
  per-decode working memory; 8192x8192 RGB decodes (`docs/supported.md`).
- Reject a request with `J2K_RESOURCE_LIMIT` before codec planning when its
  area at the requested resolution exceeds `max_output_bytes` even at one byte
  per pixel. A 2 KiB codestream with forged image and tile dimensions previously
  cost about 30 s of CPU per call.
- Return `J2K_INVALID_INPUT`, not `J2K_UNSUPPORTED`, for input that is
  neither a JP2/JPH file nor a raw JPEG 2000 codestream.
- `j2k_decoder_open` now sets `*out` to `NULL` on every failure, including
  a rejected misaligned `error` pointer.
- Copy encoded input once instead of twice, halving its peak memory; allocator
  exhaustion during that copy aborts, as documented in `docs/api.md`.
- Document that `workers` does not bound threads with `j2k` 0.13.1, and that
  `RAYON_NUM_THREADS` does; the SDK no longer allocates idle batch workers.
- Document the codec's fixed per-decode working-memory ceiling, which large
  full-frame requests can exceed after `output_info` succeeds, and the region
  and reduction workaround (`docs/supported.md`).
- Shared libraries are named by C ABI revision: GNU/Linux `libj2k.so.1`
  (SONAME `libj2k.so.1`) and macOS `libj2k.1.dylib` (install name
  `@rpath/libj2k.1.dylib`, compatibility version 1.0.0), with unversioned
  development symlinks. Applications must deploy the versioned file.
- Packaged libraries are stripped to their exported symbols, embed no
  build-machine paths, and build reproducibly regardless of source location.
  The GNU/Linux glibc floor is documented.
- CI builds a relocated package on GNU/Linux and macOS and runs the C and C++
  consumers against it.
- Tests check the C header's constants and struct layouts against the Rust ABI,
  pin exact statuses for malformed input, and verify stride padding for every
  sample type.

## 0.1.0 — initial implementation

- Add an opaque-handle CPU C ABI over published `j2k` 0.13.1.
- Inspect source/component metadata and decode full, region, and reduced native
  integer images into caller-owned buffers.
- Add input/output limits, checked ABI prefixes, per-call diagnostics, and panic
  containment with poisoned-handle rejection.
- Add a move-only C++17 wrapper, C/C++ examples, and relocatable native packaging.

This is an evaluation SDK. GPU, encoding, slide containers, and regulated-use
qualification are not included. See `docs/supported.md` for the supported subset.
