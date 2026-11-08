# C ABI contract

The C header is `include/j2k.h`. It decodes full images, regions, and reduced
resolutions into caller-owned buffers, streams full images row by row, and
encodes interleaved samples losslessly. The C++17 wrapper in `j2k.hpp` adds
move-only ownership and converts error statuses to `j2k::error` exceptions.
The library is built from the `j2k` CPU codec in the same workspace and is
versioned with it.

## Lifecycle and ownership

1. Initialize options with `J2K_OPEN_OPTIONS_INIT`, or pass `NULL` for defaults.
   `flags` may set `J2K_OPEN_LENIENT` (see [Lenient decoding](#lenient-decoding));
   `workers` chooses the decode threads (see [Limits and threading](#limits-and-threading)).
2. Call `j2k_decoder_open` with a complete raw codestream, JP2, or JPH image.
   Input that starts with neither the JP2/JPH signature box nor the codestream
   SOC and SIZ markers returns `J2K_INVALID_INPUT`.
   A successful open owns a copy of the input; the caller can then release it.
   Open inspects headers. It does not guarantee that the pixel data is valid or
   that the requested output is representable.
3. Inspect source metadata and individual component precision/sampling as needed.
4. Call `j2k_decoder_output_info` to prepare a request and discover its native
   sample type, dimensions, channel count, significant bits, and alpha meaning.
5. Allocate an output buffer and call `j2k_decoder_decode_into` with the same
   options. The library checks the layout again; a prior query is not mandatory.
   To avoid holding the whole image, stream it with `j2k_decoder_decode_rows`
   instead (see [Row streaming](#row-streaming)).
6. Call `j2k_decoder_close(&decoder, ...)`. The slot becomes `NULL`; repeating
   close on that null slot succeeds. Never call `free` on a decoder handle.

The latest prepared request is cached. Repeating it reuses its plan and the
codec's retained CPU worker workspaces. A different request replaces the plan.
Input storage is owned through J2K's `EncodedImage`; no borrowed Rust decoder
is given an artificial lifetime.

## Output interpretation

Output is interleaved by pixel (HWC for this single-image API): gray, RGB, or
RGBA, with U8, little-endian U16, or little-endian two's-complement I16 samples.
Signed sources of eight bits or less are sign-extended to I16. Significant bits
remain explicit; a 12-bit unsigned image keeps its 0–4095 sample range.

J2K performs the codestream's inverse transforms and supported color
interpretation. Native output adds no ICC conversion, display windowing, normalization,
orientation, arbitrary resizing, or alpha compositing. An RGB layout is not a
promise of color-managed sRGB. Use the alpha field to distinguish straight and
premultiplied output. Unsupported representations return `J2K_UNSUPPORTED`.

`stride` is in **bytes**, including for 16-bit samples. It must be at least
`row_bytes`. Required buffer length is:

```text
stride * (height - 1) + row_bytes
```

Trailing padding after the last row is unnecessary. Inter-row padding and bytes
beyond the required span remain untouched. Do not cast 16-bit output to a host
integer pointer without considering alignment and little-endian byte order.

## Regions and reduction

A completely zero rectangle requests the full image. Other rectangles use
full-resolution image-relative coordinates and must have positive dimensions
and stay inside the source. The library delegates geometry validation to J2K.

Reduction 0, 1, 2, or 3 discards that many wavelet resolution levels. Requests
deeper than the source ladder fail; they never fall back to resizing. A region
at reduced resolution covers the requested source rectangle, so odd coordinates
can add a boundary pixel. The returned x/y/width/height describe the actual
covering rectangle at the reduced resolution. Allocate from that result.

Coverage follows the image's position on the JPEG 2000 reference grid: a
reduction by `2^r` spans `ceil(x0 / 2^r)` to `ceil((x0 + width) / 2^r)`, where
`x0` is the image origin. `j2k_image_info` reports that origin
(`image_x0`, `image_y0`) and the tile grid (`tile_x0`, `tile_y0`,
`tile_width`, `tile_height`), so callers can align regions with tiles.

## Exact region operations

`j2k_decoder_plan_region` takes `j2k_operation_options` and fills a
`j2k_region_plan`. Its reduction is an exact count from 0 through 31, bounded
by the effective component and tile ladders. The plan reports their common
maximum, the source rectangle, covered output rectangle, interpretation,
sample metadata, and checked packed lengths. A region with no samples at that
reduction returns `J2K_UNSUPPORTED`. Geometry uses the same reference-grid
rules as the legacy API; no scale is inferred from rounded dimensions.

`J2K_INTERPRETATION_NATIVE` retains the native U8/U16/I16 values and alpha
association. `J2K_ROUTE_NATIVE_CPU` means the CPU selects its prepared,
streamed, or component route during execution; the query does not guarantee
entropy validity or that execution scratch will fit.
`J2K_INTERPRETATION_SRGB8` requests bounded region reconstruction and explicit
colour normalization, including supported restricted ICC profiles. Its output
is gray/RGB/RGBA U8 with straight alpha. Premultiplied colour is unassociated at
the source precision before sYCC conversion and quantization; premultiplied CIE
Lab input, and premultiplied subsampled or signed input other than eight-bit
mapped samples without colour conversion, return `J2K_UNSUPPORTED`. Ordinary
native output retains its existing capabilities.

Allocate from `plan.output`, then call `j2k_decoder_decode_region_into` with
the same options and a byte stride. As with legacy decoding, errors leave the
destination unchanged. Its optional `j2k_output_info` result reports actual
metadata and warnings; normalized warnings need not be known during planning.
The C++ wrapper exposes `prepare_operations`, `plan_region`, and
`decode_region_into` on the existing `decoder`.

`j2k_decoder_prepare_operations` prepares immutable input facts explicitly,
outside any operation's controls. It is optional: the first native operation on
an unprepared decoder prepares the source itself, under that operation's working
limit and cancellation when it has them, and a failed or cancelled preparation
is not kept, so the call can be retried. Once prepared, the source preparation
and compressed input are input owners; later working limits cover derived
operation plans, current codec-owned output, and execution scratch. Optional old
legacy plans and workspaces are released before controlled work. This is not an
RSS limit and cannot raise the native engine's ceiling.

Nonzero `max_output_bytes` additionally bounds the final caller span including
row gaps, and cannot raise the decoder's open-time output cap. Nonzero
`max_working_memory_bytes` sets the native working bound. Raw sRGB operations
parse their source under the operation's working limit and cancellation, as well
as its output limit. Under a working limit, ICC conversion charges a bound on its
colour-management state and returns `J2K_UNSUPPORTED` for profiles outside the
bounded matrix/TRC or gray-TRC shape with an XYZ connection space.

Create a token with `j2k_cancellation_create`, attach its handle to the options,
and cancel it with `j2k_cancellation_cancel`. Cancellation is permanent and
returns `J2K_CANCELLED`; use a new token for independent later work. The token
is cloned before codec work. Only cancellation may run concurrently with calls
using that handle; keep it live throughout those calls and never free it during
use. `j2k_cancellation_free` releases and nulls the slot. C++ callers use the
move-only `cancellation_token`. Decoder access itself remains serialized.

`j2k_decoder_set_operation_cache_limit` bounds only derived native operation
plans retained between calls. Zero disables that cache; original warmed source
preparation remains until decoder close. Controlled operations discard unused
offset caches and use bounded native execution, so a previous request cannot
silently retain a different operation's admission decision.

## Row streaming

`j2k_decoder_decode_rows` decodes the full image at full resolution and calls
the caller's `j2k_row_callback` with each row, in order and exactly once. Rows
have the layout `j2k_decoder_row_info` reports, which is the layout
`j2k_decoder_output_info` reports for the same full-image request: the same
channels, sample type, precision, alpha, and bytes. `options` must be `NULL` or
select the full image at reduction 0; other requests return
`J2K_UNSUPPORTED`.

The library holds at most `max_rows_per_stripe` decoded rows at a time (`0`
selects 64), so no whole-image output buffer or `max_output_bytes` limit
applies: an image too large for `j2k_decoder_output_info` can still stream.
Decoder working memory still depends on the image; a codestream whose tiles or
coding keep its rows from streaming natively decodes whole before its rows are
handed over (the `j2k` crate's `rows_decode_plan` describes when).

Rows keep native values for 8-bit unsigned and 9- to 16-bit samples, unsigned
or signed. Other precisions (unsigned below 8 bits, signed of 8 bits or less)
return `J2K_UNSUPPORTED`; decode them with `j2k_decoder_decode_into`.

A row pointer is valid only during its callback. Return `0` to continue or
nonzero to stop: the call then returns `J2K_STOPPED`, rows already delivered
stay delivered, and the decoder stays usable. The callback runs on the calling
thread or, for a decoder with its own threads, on one of them while the call
blocks. It is never called concurrently, must not unwind, and must not call
back into the same decoder. The C++ wrapper's `decoder::decode_rows` catches an
exception from its callable, stops the decode, and rethrows it after the call.

## Lossless encoding

`j2k_encode_lossless` encodes interleaved samples described by a
`j2k_encode_options` (`J2K_ENCODE_OPTIONS_INIT`): width, height, components,
and a bit depth of 1–16. Samples of up to 8 bits take one byte; deeper samples
take two, little-endian, and `J2K_ENCODE_SIGNED` marks them two's-complement.
`J2K_ENCODE_HTJ2K` selects HTJ2K block coding, and `J2K_ENCODE_FILE` wraps the
codestream in a JP2 file (JPH with HTJ2K); otherwise the result is a raw
codestream. The encoder decodes its result and compares it with the samples
before returning it. On success the library allocates the result; release it
with `j2k_buffer_free(data, length)` exactly once. On failure `*out` is `NULL`
and `*out_len` is 0. Encoding runs on the codec's shared CPU pool.

## Lenient decoding

`J2K_OPEN_LENIENT` accepts the JP2/JPH metadata recoveries of the codec's
lenient decoding: malformed trailing or optional boxes and an undeclared alpha
channel. Codestream validation stays strict. A request that applies a recovery
reports `J2K_WARNING_LENIENT_RECOVERY` in `j2k_output_info.warnings`, from
`j2k_decoder_output_info`, `j2k_decoder_decode_rows`, and the optional exact
decode result. A strict decoder
rejects such input instead.

## Limits and threading

The default encoded-input and output-span limits are each 512 MiB. Nonzero
options replace the respective limits. Input size is checked before it is
read or copied. Before codec planning, a request is rejected with
`J2K_RESOURCE_LIMIT` when its area at the requested resolution exceeds the
output limit even at one byte per pixel, so forged image or tile dimensions
cannot make planning expensive. The exact output span is checked before pixel
decode and before any destination writes.

These limits are **not a total-process or total-codec memory cap**. J2K
independently enforces its own allocation and structural budgets, which a
request can still exceed after `j2k_decoder_output_info` succeeds; see
[supported configurations](supported.md). Preparation and native output use
codec-owned memory; results are copied to the caller's buffer, so a decode's
peak memory includes a codec-owned copy of the whole output. The input copy
is one allocation. Ordinary allocator exhaustion, including for that copy, may
abort the process where Rust or a dependency uses an infallible allocation.

`workers` chooses where a decoder's work runs. `0` uses the codec's shared,
process-wide CPU pool, sized to the host (`RAYON_NUM_THREADS` sets its size
before the first decode). `1`–`64` gives the decoder its own pool of exactly
that many threads, started by `j2k_decoder_open` and stopped by
`j2k_decoder_close`; preparation, decode, and row streaming run there while the
call blocks. Serialize **all** access to a given handle, including metadata
queries and destruction. Separate handles can run concurrently. No global error
buffer, process-wide library lock, or GPU routing is involved.

## Errors and ABI compatibility

Every status-returning function accepts an optional caller-owned error object.
`J2K_STOPPED` reports a row callback's stop and `J2K_ENCODE_FAILED` an encoder
failure; both carry a message.
Its code matches the returned status; its message is NUL-terminated UTF-8. A
successful call clears it. No thread-local "last error" state needs retrieval.
`J2K_DECODE_FAILED` retains a codec diagnostic when the public J2K error does
not provide a narrower classification; it does not by itself prove corrupt input.
Resource-budget refusals return `J2K_RESOURCE_LIMIT`; exact byte counts are not
always available from the codec. `J2K_CANCELLED` identifies cooperative
cancellation separately from a row callback's `J2K_STOPPED`.

Result structures are written only on success. Ordinary errors leave pixel
buffers unchanged because J2K completes native decode before the library copies
samples. A contained Rust panic or internal invariant failure poisons the
handle; discard any output and close/recreate it. Panic hooks may still write to
stderr. Process aborts and invalid C pointer accesses cannot be contained.

Pointers must reference live, appropriately aligned allocations of the declared
size. Output arguments cannot overlap other arguments or handle storage. Null,
alignment, and length checks do not make dangling or forged pointers valid.

Initialize all versioned structs using their `*_INIT` macros. ABI 1 uses fixed
integer fields and a size/version prefix. Too-small prefixes and different ABI
revisions fail. Larger ABI-1 structures are accepted; only the current prefix is
read/written, and trailing storage is left untouched. An output's header reports
the written prefix size. Future extension fields must be optional for older
libraries; required new behavior needs explicit version/capability negotiation.

ABI 1 is the first C ABI revision. ABI-1 field layouts, status values,
function signatures, and ownership rules must not be changed in place. A
breaking ABI change requires a new revision and corresponding migration notes.
