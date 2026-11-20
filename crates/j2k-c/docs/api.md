# C API reference

`include/j2k.h` declares the C API. It decodes full images, regions and
reduced resolutions into buffers you own, streams full images row by row,
reports the color specification that decoding applies, and encodes
interleaved samples losslessly or lossy to rate, size or quality targets. One
decoder can move to a new codestream with `j2k_decoder_feed`, and
`j2k_decoder_transfer_syntax` reports how the codestream it holds is coded.
Full images decode through the batch route or the core whole-image route.
`include/j2k.hpp` wraps it for C++17 with move-only ownership and turns error
statuses into `j2k::error` exceptions. The library is the CPU codec of the
`j2k` Rust crate it is built and versioned with.

## Functions

| Function | Purpose |
| --- | --- |
| `j2k_abi_version`, `j2k_version` | The ABI revision (`J2K_ABI_VERSION`) and the library version string. The string lives as long as the library; never free it. |
| `j2k_decoder_open`, `j2k_decoder_close` | Create and destroy a decoder for one encoded image. |
| `j2k_decoder_feed`, `j2k_decoder_transfer_syntax` | Move a decoder to a new codestream, keeping its threads, settings and limits, and report how the codestream it holds is coded. |
| `j2k_decoder_info`, `j2k_decoder_component_info` | Image size, component count, resolution levels, reference grid, and per-component precision, signedness and sampling. |
| `j2k_decoder_color_info`, `j2k_decoder_icc_profile` | The color specification decoding applies, and its ICC profile. |
| `j2k_decoder_output_info`, `j2k_decoder_decode_into` | Plan and decode a full image, region or reduction (0 to 3) in native samples. |
| `j2k_decoder_output_info_native`, `j2k_decoder_decode_full_into` | Plan and decode a full image through the core whole-image route. |
| `j2k_decoder_row_info`, `j2k_decoder_decode_rows` | Stream the full image row by row through a callback. |
| `j2k_decoder_prepare_operations`, `j2k_decoder_plan_region`, `j2k_decoder_decode_region_into`, `j2k_decoder_set_operation_cache_limit` | Exact region operations: reductions 0 to 31, 8-bit sRGB output, memory limits and cancellation. |
| `j2k_cancellation_create`, `j2k_cancellation_cancel`, `j2k_cancellation_free` | Cancellation tokens for exact operations. |
| `j2k_encode_lossless`, `j2k_encode`, `j2k_buffer_free` | Encode interleaved samples, and free the encoded buffer. |

In C++, `j2k::decoder` has `info()`, `component(index)`, `color_info()`,
`icc_profile()`, `output_info()`, `decode_into()`, `row_info()`,
`decode_rows()`, `prepare_operations()`, `plan_region()`,
`decode_region_into()` and `set_operation_cache_limit()`.
`j2k::cancellation_token` has `cancel()` and `native_handle()`, and
`j2k::encode_lossless()` and `j2k::encode()` return a `std::vector<uint8_t>`.
`j2k::error::status()` gives the failing `j2k_status`.

## Persistent feed and transfer syntax

`j2k_decoder_feed` replaces the codestream a live decoder holds and keeps its
threads, settings, and input and output limits. It applies the same signature
and input-limit checks as `j2k_decoder_open` and inspects the new headers
completely before replacing any state, so a failed feed leaves the previous
codestream inspectable and decodable. The decoder keeps its own copy of the
input, so you can release yours after the call. Feed is for successive
independent images through one decoder configuration; it doesn't append or
combine codestreams into one image.

`j2k_decoder_transfer_syntax` reports how the held codestream is coded:
`J2K_SYNTAX_UNKNOWN` (0), `J2K_SYNTAX_JPEG2000_LOSSLESS` (1),
`J2K_SYNTAX_JPEG2000_LOSSY` (2), `J2K_SYNTAX_HT_JPEG2000_LOSSLESS` (3),
`J2K_SYNTAX_HT_JPEG2000_LOSSY` (4), or `J2K_SYNTAX_OTHER` (5) when this ABI
revision doesn't name the class. Check the returned status before reading the
value. It describes the compressed coding, not the decoded sample layout; use
`j2k_decoder_output_info` for that. C++ callers use `decoder::feed` and
`decoder::transfer_syntax`.

## Decoding

1. Initialize a `j2k_open_options` with `J2K_OPEN_OPTIONS_INIT`, or pass
   `NULL` for the defaults. `flags` can set `J2K_OPEN_LENIENT` (see
   [Lenient decoding](#lenient-decoding)); `workers` picks the threads (see
   [Limits and threading](#limits-and-threading)).
2. Call `j2k_decoder_open` with a complete raw codestream, JP2 file or JPH
   file. Input that starts with neither the JP2/JPH signature box nor the SOC
   and SIZ markers returns `J2K_INVALID_INPUT`. On success the decoder owns a
   copy of the input, so you can release yours. Opening reads the headers
   only: it doesn't prove the pixel data is valid or that a given output can
   be produced.
3. Read the image and component metadata you need.
4. Call `j2k_decoder_output_info` to prepare a request and learn its sample
   type, dimensions, channel count, significant bits and alpha.
5. Allocate a buffer and call `j2k_decoder_decode_into` with the same
   options. It checks the layout again, so step 4 is optional. To avoid
   holding the whole image, use `j2k_decoder_decode_rows` instead (see
   [Row streaming](#row-streaming)).
6. Call `j2k_decoder_close(&decoder, ...)`. It sets the handle to `NULL`, and
   closing a `NULL` handle succeeds. Never `free` a decoder handle.

The decoder keeps the plan for the latest request. Repeating that request
reuses the plan and the codec's CPU workspaces; a different request replaces
it.

## Output layout

Output is interleaved by pixel: gray, RGB or RGBA, with `J2K_U8`,
`J2K_U16_LE` (little-endian) or `J2K_I16_LE` (little-endian two's complement)
samples. Signed sources of 8 bits or fewer come back as `J2K_I16_LE`.
`precision` gives the significant bits: a 12-bit unsigned image keeps its
0-4095 range.

J2K applies the codestream's inverse transforms and the color interpretation
it supports. Native output adds no ICC conversion, display windowing,
normalization, orientation, resizing or alpha compositing, so an RGB layout
is not necessarily sRGB. `alpha` says whether alpha is straight or
premultiplied. Representations the decoder can't produce return
`J2K_UNSUPPORTED`.

`stride` is in bytes, also for 16-bit samples, and must be at least
`row_bytes`. The buffer must hold

```text
stride * (height - 1) + row_bytes
```

bytes; the last row needs no padding. Padding between rows and bytes past the
required length are left untouched. Mind alignment and byte order before
reading 16-bit samples through a wider pointer type.

## Core whole-image route

`j2k_decoder_output_info_native` works out the output layout of a full-image,
full-resolution request from the headers alone, without the batch route's
per-image planning. It covers raw codestreams with one or three unsigned
components of equal 8- to 16-bit precision. JP2/JPH files, regions,
reductions, signed components, mixed precision, samples under 8 or over 16
bits, uniform component subsampling, and other channel counts return
`J2K_UNSUPPORTED`; `j2k_decoder_output_info` stays the general route for
those. Mixed component sampling is supported and upsampled onto the output
grid. A successful result has the same layout fields as
`j2k_decoder_output_info` for the same request.

`j2k_decoder_decode_full_into` decodes such a request through the core
whole-image decoder, with the same stride contract as
`j2k_decoder_decode_into` and the same output bytes. It writes straight into
your buffer, so a failure partway through may leave some rows written; check
the status before using the result. C++ callers use
`decoder::output_info_native` and `decoder::decode_full_into`.

## Regions and reductions

An all-zero rectangle selects the full image. Any other rectangle is in
full-resolution image coordinates, must have a positive width and height,
and must lie inside the image.

`reduction` 0 to 3 drops that many wavelet resolution levels. A request
deeper than the image's levels fails; the library never resizes instead. A
reduced region covers the requested source rectangle, so odd coordinates can
add a pixel at the edge. The `x`, `y`, `width` and `height` that
`j2k_decoder_output_info` returns describe the covering rectangle at the
reduced resolution; allocate from them.

Coverage follows the image's position on the JPEG 2000 reference grid: a
reduction by `2^r` spans `ceil(x0 / 2^r)` to `ceil((x0 + width) / 2^r)`,
where `x0` is the image origin. `j2k_image_info` reports the origin
(`image_x0`, `image_y0`) and the tile grid (`tile_x0`, `tile_y0`,
`tile_width`, `tile_height`), so you can align regions with tiles.

## Exact region operations

`j2k_decoder_plan_region` takes a `j2k_operation_options`
(`J2K_OPERATION_OPTIONS_INIT`) and fills a `j2k_region_plan`. `reduction` is
an exact level count from 0 to 31, bounded by the levels every component and
tile has; the plan reports that common `max_reduction`. It also reports the
source rectangle, the covered output rectangle in `output`, the
interpretation, the sample metadata and the exact byte lengths. A region that
has no samples at the requested reduction returns `J2K_UNSUPPORTED`. Geometry
follows the reference-grid rules above.

`interpretation` selects the output:

- `J2K_INTERPRETATION_NATIVE` keeps the native `U8`/`U16`/`I16` samples and
  their alpha association. The plan's `route` is `J2K_ROUTE_NATIVE_CPU`: the
  CPU picks how to decode when the operation runs, so a plan doesn't check
  the entropy-coded data or promise that scratch memory will fit.
- `J2K_INTERPRETATION_SRGB8` returns 8-bit gray, RGB or RGBA in sRGB with
  straight alpha, converting the supported ICC profiles (`route` is
  `J2K_ROUTE_REGION_PLANES`). Premultiplied color is un-premultiplied at the
  source precision before sYCC conversion and quantization. Premultiplied CIE
  Lab, and premultiplied subsampled or signed input other than 8-bit mapped
  samples without color conversion, return `J2K_UNSUPPORTED`.

Allocate from `plan.output` and call `j2k_decoder_decode_region_into` with
the same options and a byte stride. An error leaves the buffer unchanged. Its
optional `j2k_output_info` result reports the actual metadata and warnings;
sRGB warnings may only be known once the decode runs.

`j2k_decoder_prepare_operations` parses the image for exact operations
outside any operation's limits. It is optional: the first native operation
on an unprepared decoder prepares the image itself, under that operation's
working limit and cancellation, and keeps the result only if preparation
succeeds, so a cancelled call can be retried. Once prepared, the parsed image
and the compressed input don't count against later working limits, which
cover derived plans, the codec's current output and scratch. Plans kept by
`j2k_decoder_output_info` are released before a controlled operation runs.
The limit is not a process memory (RSS) limit, and it can't raise the
codec's own ceiling.

Operation limits:

- `max_output_bytes`, when nonzero, bounds the caller's buffer span,
  including gaps between rows. It can't raise the decoder's `max_output_bytes`
  from `j2k_decoder_open`.
- `max_working_memory_bytes`, when nonzero, bounds the codec's working
  memory. sRGB operations also parse their source under the operation's
  limits and cancellation. Under a working limit, ICC conversion counts a
  bound on its color-management state and returns `J2K_UNSUPPORTED` for any
  profile that isn't a matrix/TRC or gray-TRC profile with an XYZ connection
  space.
- `cancellation` takes a token from `j2k_cancellation_create`. Cancel it with
  `j2k_cancellation_cancel`; the operation returns `J2K_CANCELLED`.
  Cancellation is permanent, so use a new token for later work. The
  operation clones the token before codec work. Only
  `j2k_cancellation_cancel` may run at the same time as calls that use the
  token; keep it alive throughout them. `j2k_cancellation_free` releases it
  and sets the handle to `NULL`. In C++, use the move-only
  `j2k::cancellation_token` and pass `native_handle()`. Calls on the decoder
  itself are still serialized.

`j2k_decoder_set_operation_cache_limit` bounds the derived plans kept between
operations; 0 disables that cache. The parsed image stays until the decoder
is closed. Controlled operations drop unused cached plans and run with
bounded memory, so an earlier request's admission decision never carries over
to a different operation.

## Row streaming

`j2k_decoder_decode_rows` decodes the full image at full resolution and calls
your `j2k_row_callback` once per row, in order. Rows have the layout
`j2k_decoder_row_info` reports, which matches what
`j2k_decoder_output_info` reports for the full image: the same channels,
sample type, precision, alpha and row bytes. `options` must be `NULL` or
select the full image at reduction 0; anything else returns
`J2K_UNSUPPORTED`.

The library holds at most `max_rows_per_stripe` decoded rows at a time (0
means 64). No whole-image buffer is allocated and `max_output_bytes` doesn't
apply, so an image too large for `j2k_decoder_output_info` can still be
streamed. The decoder's working memory still depends on the image:

- An image whose tiles don't span its width decodes each stripe as a region.
- Coding the header can't describe in advance, such as HTJ2K code blocks
  mixed with classic ones, makes the image decode whole before its rows are
  delivered. The `j2k` crate's `J2kDecoder::rows_decode_plan` reports which
  plan an image gets.

Rows keep native values for 8-bit unsigned samples and for 9- to 16-bit
samples, unsigned or signed. Other precisions (unsigned below 8 bits, signed
8 bits or fewer) return `J2K_UNSUPPORTED`; decode them with
`j2k_decoder_decode_into`.

A row pointer is valid only during its callback. Return 0 to continue or
nonzero to stop: the call then returns `J2K_STOPPED`, rows already delivered
stay delivered, and the decoder stays usable. The callback runs on the
calling thread or, for a decoder with its own threads, on one of those while
the call blocks. It is never called concurrently, must not unwind, and must
not call back into the same decoder. On success, the optional `out` receives
the row layout and its warnings. The C++ `decoder::decode_rows` takes a
callable that returns `true` to continue, and returns the row layout, or
`std::nullopt` when the callable stopped the decode. If the callable throws,
the decode stops and the exception is rethrown after the call.

## Color specification

`j2k_decoder_color_info` describes the JP2/JPH color specification (COLR) box
that decoding applies: the first one in the file, as ISO/IEC 15444-1 I.5.3.3
directs. `specification` is one of:

- `J2K_COLOR_ENUMERATED`, with the `EnumCS` value in `value` (16 sRGB,
  17 greyscale, 18 sYCC);
- `J2K_COLOR_ICC`, with the profile length in `icc_profile_bytes`;
- `J2K_COLOR_UNKNOWN`, with the uninterpreted COLR method in `value`;
- `J2K_COLOR_NONE` for a raw codestream.

`specifications` counts every COLR box. Later boxes are kept in the file but
not applied; `has_icc_profile` in `j2k_image_info` covers all of them.
`j2k_decoder_icc_profile` copies the applied profile into a buffer of at
least `icc_profile_bytes` bytes. A shorter buffer returns
`J2K_BUFFER_TOO_SMALL` and is left untouched; an image without an applied
profile returns `J2K_INVALID_ARGUMENT`. Native output applies no ICC
conversion; sRGB operations convert the supported profiles.

## Encoding

`j2k_encode_lossless` encodes interleaved samples described by a
`j2k_encode_options` (`J2K_ENCODE_OPTIONS_INIT`): `width`, `height`,
`components`, and a `bit_depth` of 1 to 16. Samples of up to 8 bits take one
byte; deeper samples take two, little-endian. `flags` combines:

- `J2K_ENCODE_SIGNED`: two's-complement samples.
- `J2K_ENCODE_HTJ2K`: HTJ2K block coding instead of classic JPEG 2000.
- `J2K_ENCODE_FILE`: a JP2 file (JPH with HTJ2K) instead of a raw codestream.

The encoder decodes its result and compares it with the samples before
returning it. On success the library allocates the output; release it with
`j2k_buffer_free(data, length)` exactly once. On failure `*out` is `NULL` and
`*out_len` is 0. Encoding runs on the codec's shared CPU thread pool.

`j2k_encode` takes the same samples and options, a `j2k_encode_params`
(`J2K_ENCODE_PARAMS_INIT`; `NULL` encodes exactly like `j2k_encode_lossless`)
and an optional `j2k_encode_report`. Zero fields select defaults, except
`levels`, whose default `J2K_LEVELS_AUTO` the init macro sets. The fields map
onto the codec's encode options, and the codec checks every combination:

- `mode`: `J2K_MODE_LOSSLESS` (reversible 5/3) or `J2K_MODE_LOSSY`
  (irreversible 9/7). Three lossy components use the irreversible color
  transform. Three lossless components use the reversible one unless `flags`
  has `J2K_PARAMS_NO_COLOR_TRANSFORM`.
- `levels`: the maximum number of decompositions, or `J2K_LEVELS_AUTO`. The
  image and tile size can lower it.
- `progression`: `J2K_PROGRESSION_LRCP`, `_RLCP`, `_RPCL`, `_PCRL` or
  `_CPRL`.
- `tile_width`, `tile_height`: both 0 for a single tile.
  `tile_part_packets` caps the complete packets per tile-part (1 to 65535;
  0 means no limit).
- `markers`: any of `J2K_MARKER_SOP`, `_EPH`, `_TLM`, `_PLT`, `_PLM`, `_PPM`
  and `_PPT`.
- `quality_layers`: lossless layers, 1 to 32 (0 writes one).
- Lossy targets: `rate` gives the unit (`J2K_RATE_BITS_PER_PIXEL`,
  `J2K_RATE_BYTES` or `J2K_RATE_PSNR_DB`), and `targets` holds `target_count`
  ascending values: one target for the codestream, or 2 to 32 cumulative
  layer targets. Byte targets must be whole numbers. `qfactor` (1 to 100)
  selects HTJ2K's quality factor instead of a target.
- `icc_profile`, `icc_profile_len`: with `J2K_ENCODE_FILE`, the ICC profile
  for the COLR box. Without one, the file declares the enumerated color space
  the samples imply.

Invalid field values, and combinations the library can check itself, return
`J2K_INVALID_ARGUMENT`. Combinations the codec refuses, such as a Qfactor with
classic block coding or with a rate target, return `J2K_UNSUPPORTED`. A
target the encoder can't reach returns `J2K_ENCODE_FAILED`. On success the
report holds the codestream length (without the file wrapper), its bits per
image pixel, the PSNR when the encoder measured it (otherwise 0), and the
number of quality layers. Lossless output must reproduce the samples exactly.

## Lenient decoding

`J2K_OPEN_LENIENT` accepts the JP2/JPH metadata recoveries of the codec's
lenient mode: malformed trailing or optional boxes, and an undeclared alpha
channel. Codestream validation stays strict. A request that relies on a
recovery sets `J2K_WARNING_LENIENT_RECOVERY` in `j2k_output_info.warnings`,
as returned by `j2k_decoder_output_info`, `j2k_decoder_decode_rows` and
`j2k_decoder_decode_region_into`. A strict decoder rejects such input.

## Limits and threading

`max_input_bytes` and `max_output_bytes` in `j2k_open_options` default to
512 MiB each; nonzero values replace them. The input size is checked before
the input is read or copied. Before planning, a request is rejected with
`J2K_RESOURCE_LIMIT` if its area at the requested resolution exceeds the
output limit even at one byte per pixel, so forged image or tile dimensions
can't make planning expensive. The exact output length is checked before
decoding and before anything is written to your buffer.

These limits don't cap the memory of the process or the codec. J2K enforces
its own allocation and structure budgets, which a request can exceed after
`j2k_decoder_output_info` succeeds (see
[supported configurations](supported.md)). Decoding writes into memory the
codec owns and then copies into your buffer, so peak memory includes a second
copy of the output. The input copy is one allocation. If the allocator runs
out of memory where Rust or a dependency allocates infallibly, including for
that copy, the process aborts.

`workers` chooses where a decoder runs. 0 uses the codec's shared,
process-wide thread pool, sized to the host; set `RAYON_NUM_THREADS` before
the first decode to change its size. 1 to 64 gives the decoder its own pool
of exactly that many threads, started by `j2k_decoder_open` and stopped by
`j2k_decoder_close`; preparation, decoding and row streaming run there while
the call blocks. Higher values return `J2K_INVALID_ARGUMENT`.

Serialize every call on one decoder, including metadata queries and
`j2k_decoder_close`. A decoder may move between threads, and different
decoders can run at the same time. There is no global error state,
process-wide lock or GPU routing.

## Errors

Every function that returns a `j2k_status` takes an optional `j2k_error`
that you own. On failure its `code` matches the returned status and
`message` holds NUL-terminated UTF-8; a successful call clears it. There is
no thread-local "last error".

| Status | Meaning |
| --- | --- |
| `J2K_OK` | Success. |
| `J2K_INVALID_ARGUMENT` | A null or out-of-range argument, unknown flag bits, a nonzero reserved field, or a combination the library rejects. |
| `J2K_ABI_MISMATCH` | A struct's header names another ABI revision or a smaller size than this library needs. |
| `J2K_INVALID_INPUT` | The input isn't JPEG 2000, or is malformed or truncated. |
| `J2K_UNSUPPORTED` | A valid request the library can't serve. |
| `J2K_BUFFER_TOO_SMALL` | The output buffer is shorter than required. |
| `J2K_RESOURCE_LIMIT` | A size or memory limit was exceeded. Exact byte counts aren't always available. |
| `J2K_ALLOCATION_FAILED` | An allocation or worker thread couldn't be created. |
| `J2K_INTERNAL_ERROR` | An internal check failed. The decoder is poisoned. |
| `J2K_PANIC` | A Rust panic was caught, or the decoder is poisoned. |
| `J2K_DECODE_FAILED` | The codec failed without a narrower classification; the message has its diagnostic. This alone doesn't prove the input is corrupt. |
| `J2K_STOPPED` | A row callback asked to stop. |
| `J2K_ENCODE_FAILED` | The encoder failed, for example on an unreachable target. |
| `J2K_CANCELLED` | A cancellation token stopped the operation. |

Result structures are written only on success. Ordinary errors leave pixel
buffers unchanged, because J2K finishes decoding before it copies samples.
`j2k_decoder_decode_full_into` is the exception: it decodes straight into your
buffer, so a failure partway through may leave some rows written.
After `J2K_PANIC` or `J2K_INTERNAL_ERROR`, discard any output and close the
decoder: every later call on it returns `J2K_PANIC`. A panic hook may still
write to stderr. Process aborts and invalid pointer accesses can't be caught.

Pointers must point to live, suitably aligned allocations of the stated size.
Output arguments must not overlap each other, inputs or decoder storage. The
library checks for null pointers, alignment and lengths, but that doesn't
make dangling or forged pointers safe.

## ABI compatibility

Initialize every versioned struct with its `*_INIT` macro. ABI 1 structs
start with a `j2k_header` holding their size and ABI revision. A struct
smaller than the library expects, or from another ABI revision, returns
`J2K_ABI_MISMATCH`. A larger ABI 1 struct is accepted: only the known prefix
is read or written, the rest is left untouched, and an output's header
reports the size written. Fields added later must be optional for older
libraries; behavior a caller depends on needs explicit version or capability
negotiation.

ABI 1 is the first C ABI revision. Its struct layouts, status values,
function signatures and ownership rules never change in place. A breaking
change gets a new ABI revision and migration notes.

## Static linking

To link the C ABI statically, define `J2K_STATIC` and link `lib/libj2k.a`
explicitly. `pkg-config --static --libs j2k` reports the archive's private
dependencies but also includes `-lj2k`; remove that entry so the linker
doesn't pick the shared library:

```sh
cc -DJ2K_STATIC app.c /path/to/package/lib/libj2k.a \
    $(pkg-config --cflags j2k) \
    $(pkg-config --static --libs-only-l j2k | sed 's/-lj2k//')
```

CMake consumers link the exported `J2K::j2k_static` target, which defines
`J2K_STATIC` and carries the archive's private link libraries.
