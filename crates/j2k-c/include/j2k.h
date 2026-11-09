/* SPDX-License-Identifier: MIT OR Apache-2.0 */
/* ABI declarations: crates/j2k-c/src/abi.rs. */
#ifndef J2K_H
#define J2K_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define J2K_API __declspec(dllimport)
#else
#define J2K_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define J2K_ABI_VERSION UINT32_C(1)
typedef uint32_t j2k_status;
#define J2K_OK UINT32_C(0)
#define J2K_INVALID_ARGUMENT UINT32_C(1)
#define J2K_ABI_MISMATCH UINT32_C(2)
#define J2K_INVALID_INPUT UINT32_C(3)
#define J2K_UNSUPPORTED UINT32_C(4)
#define J2K_BUFFER_TOO_SMALL UINT32_C(5)
#define J2K_RESOURCE_LIMIT UINT32_C(6)
#define J2K_ALLOCATION_FAILED UINT32_C(7)
#define J2K_INTERNAL_ERROR UINT32_C(8)
#define J2K_PANIC UINT32_C(9)
#define J2K_DECODE_FAILED UINT32_C(10)
/* A row callback returned nonzero; rows already delivered stay delivered. */
#define J2K_STOPPED UINT32_C(11)
#define J2K_ENCODE_FAILED UINT32_C(12)
#define J2K_CANCELLED UINT32_C(13)

#define J2K_INTERPRETATION_NATIVE UINT32_C(0)
#define J2K_INTERPRETATION_SRGB8 UINT32_C(1)
#define J2K_ROUTE_NATIVE_CPU UINT32_C(1)
#define J2K_ROUTE_REGION_PLANES UINT32_C(2)

/* j2k_open_options.flags: accept the JP2/JPH metadata recoveries of lenient
 * decoding (malformed trailing or optional boxes, an undeclared alpha). */
#define J2K_OPEN_LENIENT UINT32_C(1)
/* j2k_output_info.warnings: a lenient metadata recovery applies. */
#define J2K_WARNING_LENIENT_RECOVERY UINT32_C(1)
/* j2k_encode_options.flags: HTJ2K (Part 15) instead of classic Part 1;
 * two's-complement signed samples; a JP2 file (JPH with HTJ2K). */
#define J2K_ENCODE_HTJ2K UINT32_C(1)
#define J2K_ENCODE_SIGNED UINT32_C(2)
#define J2K_ENCODE_FILE UINT32_C(4)
/* j2k_encode_params.mode: reversible lossless coding (the default), or
 * irreversible lossy coding, optionally to rate or quality targets. */
#define J2K_MODE_LOSSLESS UINT32_C(0)
#define J2K_MODE_LOSSY UINT32_C(1)
/* j2k_encode_params.flags: code three lossless components without the
 * reversible colour transform. */
#define J2K_PARAMS_NO_COLOR_TRANSFORM UINT32_C(1)
/* j2k_encode_params.levels: the codec's default for the image geometry. */
#define J2K_LEVELS_AUTO UINT32_C(4294967295)
/* j2k_encode_params.progression: packet progression order. */
#define J2K_PROGRESSION_LRCP UINT32_C(0)
#define J2K_PROGRESSION_RLCP UINT32_C(1)
#define J2K_PROGRESSION_RPCL UINT32_C(2)
#define J2K_PROGRESSION_PCRL UINT32_C(3)
#define J2K_PROGRESSION_CPRL UINT32_C(4)
/* j2k_encode_params.rate: the unit of lossy targets. */
#define J2K_RATE_NONE UINT32_C(0)
#define J2K_RATE_BITS_PER_PIXEL UINT32_C(1)
#define J2K_RATE_BYTES UINT32_C(2)
#define J2K_RATE_PSNR_DB UINT32_C(3)
/* j2k_encode_params.markers: optional marker segments to write. */
#define J2K_MARKER_SOP UINT32_C(1)
#define J2K_MARKER_EPH UINT32_C(2)
#define J2K_MARKER_TLM UINT32_C(4)
#define J2K_MARKER_PLT UINT32_C(8)
#define J2K_MARKER_PLM UINT32_C(16)
#define J2K_MARKER_PPM UINT32_C(32)
#define J2K_MARKER_PPT UINT32_C(64)
/* j2k_color_info.specification: the first COLR box, which decoding applies. */
#define J2K_COLOR_NONE UINT32_C(0)
#define J2K_COLOR_ENUMERATED UINT32_C(1)
#define J2K_COLOR_ICC UINT32_C(2)
#define J2K_COLOR_UNKNOWN UINT32_C(3)

#define J2K_U8 UINT32_C(1)
#define J2K_U16_LE UINT32_C(2)
#define J2K_I16_LE UINT32_C(3)
#define J2K_ALPHA_NONE UINT32_C(0)
#define J2K_ALPHA_STRAIGHT UINT32_C(1)
#define J2K_ALPHA_PREMULTIPLIED UINT32_C(2)

typedef struct j2k_decoder j2k_decoder;
typedef struct j2k_cancellation j2k_cancellation;

typedef struct j2k_header {
    uint32_t struct_size;
    uint32_t abi_version;
} j2k_header;

/* Optional, per-call, caller-owned error. Always cleared on success.
 * message is NUL-terminated UTF-8. The layout is fixed for ABI 1. */
typedef struct j2k_error {
    uint32_t code;
    char message[256];
} j2k_error;

typedef struct j2k_open_options {
    j2k_header header;
    uint64_t max_input_bytes;  /* 0 = 512 MiB; checked before input is copied. */
    uint64_t max_output_bytes; /* 0 = 512 MiB; includes inter-row padding. */
    uint32_t workers;          /* 0 = shared codec pool; 1-64 = own pool of that many threads. */
    uint32_t flags;            /* J2K_OPEN_*; other bits must be zero. */
} j2k_open_options;

typedef struct j2k_decode_options {
    j2k_header header;
    uint32_t x;         /* Full-resolution image-relative coordinates. */
    uint32_t y;
    uint32_t width;     /* An entirely zero rectangle selects the full image. */
    uint32_t height;    /* Other rectangles require positive width and height. */
    uint32_t reduction; /* 0-3, bounded by the source resolution ladder. */
    uint32_t reserved;  /* Must be zero. */
} j2k_decode_options;

/* Additive exact-operation options. Input and source facts prepared earlier are
 * outside the working bound. Zero byte limits retain the default/decoder cap. */
typedef struct j2k_operation_options {
    j2k_header header;
    uint32_t x;                 /* Full-resolution source ROI; all zero = full. */
    uint32_t y;
    uint32_t width;
    uint32_t height;
    uint32_t reduction;          /* Exact wavelet levels, 0-31. */
    uint32_t interpretation;     /* J2K_INTERPRETATION_*. */
    uint64_t max_working_memory_bytes;
    uint64_t max_output_bytes;   /* Includes caller row gaps, excludes final padding. */
    const j2k_cancellation *cancellation; /* Optional; keep live through the call. */
} j2k_operation_options;

typedef struct j2k_image_info {
    j2k_header header;
    uint32_t width;
    uint32_t height;
    uint32_t components;
    uint32_t resolution_levels;
    uint32_t has_signed_components; /* 0 or 1. */
    uint32_t has_icc_profile;       /* 0 or 1. */
    uint32_t image_x0;    /* Image origin on the reference grid (XOsiz, YOsiz). */
    uint32_t image_y0;
    uint32_t tile_x0;     /* Tile-grid origin (XTOsiz, YTOsiz). */
    uint32_t tile_y0;
    uint32_t tile_width;  /* Nominal tile size (XTsiz, YTsiz). */
    uint32_t tile_height;
} j2k_image_info;

typedef struct j2k_component_info {
    j2k_header header;
    uint32_t precision;
    uint32_t is_signed; /* 0 or 1. */
    uint32_t sampling_x;
    uint32_t sampling_y;
} j2k_component_info;

typedef struct j2k_output_info {
    j2k_header header;
    uint32_t x; /* Covering rectangle in reduced-resolution coordinates. */
    uint32_t y;
    uint32_t width;
    uint32_t height;
    uint32_t channels;    /* Interleaved: 1 = gray, 3 = RGB, 4 = RGBA. */
    uint32_t sample_type; /* J2K_U8, J2K_U16_LE, or J2K_I16_LE. */
    uint32_t precision;   /* Significant output bits; native output keeps its range. */
    uint32_t alpha;       /* J2K_ALPHA_*. */
    uint64_t row_bytes;   /* Minimum stride in bytes. */
    uint64_t buffer_bytes;/* Tightly packed byte length. */
    uint32_t warnings;    /* J2K_WARNING_* recoveries the request applies. */
    uint32_t reserved;    /* Written as zero. */
} j2k_output_info;

/* Geometry and output sizes are exact. Native CPU execution chooses its
 * prepared/streamed/component route later; planning does not validate entropy. */
typedef struct j2k_region_plan {
    j2k_header header;
    j2k_output_info output;
    uint32_t source_x;
    uint32_t source_y;
    uint32_t source_width;
    uint32_t source_height;
    uint32_t reduction;
    uint32_t max_reduction;  /* Common to every component and tile. */
    uint32_t interpretation;
    uint32_t route;          /* J2K_ROUTE_*. */
} j2k_region_plan;

typedef struct j2k_encode_options {
    j2k_header header;
    uint32_t width;
    uint32_t height;
    uint32_t components; /* Interleaved samples per pixel. */
    uint32_t bit_depth;  /* 1-16; up to 8 bits take one byte, more take two (LE). */
    uint32_t flags;      /* J2K_ENCODE_*; other bits must be zero. */
    uint32_t reserved;   /* Must be zero. */
} j2k_encode_options;

/* Coding parameters for j2k_encode. Zero fields select defaults, except
 * levels, whose default J2K_LEVELS_AUTO the init macro sets. The codec
 * validates every combination; see docs/api.md. */
typedef struct j2k_encode_params {
    j2k_header header;
    uint32_t mode;              /* J2K_MODE_*. */
    uint32_t flags;             /* J2K_PARAMS_*; other bits must be zero. */
    uint32_t levels;            /* Maximum wavelet decompositions, or J2K_LEVELS_AUTO. */
    uint32_t progression;       /* J2K_PROGRESSION_*. */
    uint32_t tile_width;        /* Both zero = one tile. */
    uint32_t tile_height;
    uint32_t markers;           /* J2K_MARKER_* bits; other bits must be zero. */
    uint32_t tile_part_packets; /* Complete packets per tile-part, 1-65535; 0 = no limit. */
    uint32_t quality_layers;    /* Lossless only: 1-32; 0 = one. */
    uint32_t qfactor;           /* Lossy HTJ2K only: 1-100; 0 = none. */
    uint32_t rate;              /* Lossy: J2K_RATE_*, the unit of targets. */
    uint32_t target_count;      /* Lossy: 0, 1 codestream target, or 2-32 cumulative layers. */
    const double *targets;      /* Ascending targets in rate units; bytes are whole numbers. */
    const uint8_t *icc_profile; /* With J2K_ENCODE_FILE: the COLR ICC profile; NULL = enumerated. */
    size_t icc_profile_len;
} j2k_encode_params;

/* What an encode produced, measured on its codestream. */
typedef struct j2k_encode_report {
    j2k_header header;
    uint64_t codestream_bytes; /* Excluding any JP2/JPH wrapper. */
    double bits_per_pixel;     /* Codestream bits per image pixel. */
    double psnr_db;            /* When the encode measured it; otherwise 0. */
    uint32_t quality_layers;
    uint32_t reserved;         /* Written as zero. */
} j2k_encode_report;

/* The colour specification decoding applies: the file's first COLR box. */
typedef struct j2k_color_info {
    j2k_header header;
    uint32_t specification;     /* J2K_COLOR_*. */
    uint32_t value;             /* EnumCS when ENUMERATED (16 sRGB, 17 greyscale, 18 sYCC),
                                 * the COLR method when UNKNOWN, otherwise 0. */
    uint32_t specifications;    /* COLR boxes; later boxes are preserved, not applied. */
    uint32_t reserved;          /* Written as zero. */
    uint64_t icc_profile_bytes; /* Applied ICC profile length; 0 without one. */
} j2k_color_info;

/* Receives one row of row_bytes bytes, valid only during the call. Return 0 to
 * continue or nonzero to stop the decode with J2K_STOPPED. */
typedef int32_t (*j2k_row_callback)(
    void *user_data, uint32_t y, const uint8_t *row, size_t row_bytes);

#define J2K_HEADER_INIT(type) {(uint32_t)sizeof(type), J2K_ABI_VERSION}
#define J2K_OPEN_OPTIONS_INIT {J2K_HEADER_INIT(j2k_open_options), 0, 0, 0, 0}
#define J2K_DECODE_OPTIONS_INIT {J2K_HEADER_INIT(j2k_decode_options), 0, 0, 0, 0, 0, 0}
#define J2K_OPERATION_OPTIONS_INIT {J2K_HEADER_INIT(j2k_operation_options), 0, 0, 0, 0, 0, J2K_INTERPRETATION_NATIVE, 0, 0, NULL}
#define J2K_IMAGE_INFO_INIT {J2K_HEADER_INIT(j2k_image_info), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
#define J2K_COMPONENT_INFO_INIT {J2K_HEADER_INIT(j2k_component_info), 0, 0, 0, 0}
#define J2K_OUTPUT_INFO_INIT {J2K_HEADER_INIT(j2k_output_info), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
#define J2K_REGION_PLAN_INIT {J2K_HEADER_INIT(j2k_region_plan), J2K_OUTPUT_INFO_INIT, 0, 0, 0, 0, 0, 0, 0, 0}
#define J2K_ENCODE_OPTIONS_INIT {J2K_HEADER_INIT(j2k_encode_options), 0, 0, 0, 0, 0, 0}
#define J2K_ENCODE_PARAMS_INIT {J2K_HEADER_INIT(j2k_encode_params), J2K_MODE_LOSSLESS, 0, J2K_LEVELS_AUTO, J2K_PROGRESSION_LRCP, 0, 0, 0, 0, 0, 0, J2K_RATE_NONE, 0, NULL, NULL, 0}
#define J2K_ENCODE_REPORT_INIT {J2K_HEADER_INIT(j2k_encode_report), 0, 0, 0, 0, 0}
#define J2K_COLOR_INFO_INIT {J2K_HEADER_INIT(j2k_color_info), J2K_COLOR_NONE, 0, 0, 0, 0}

/* Pointer contract for every call:
 * - Pointers must reference valid, aligned storage for their declared lengths.
 * - Output arguments must not overlap each other, inputs, or decoder storage.
 * - Serialize all calls on the same decoder, including inspection and close.
 *   Different decoders may be used concurrently; a handle may move between threads.
 * - Options may be NULL for defaults. Error may be NULL to discard diagnostics.
 * - Initialize options and result structures with the macros above. Larger
 *   structures with ABI 1 are accepted; only the known prefix is read/written.
 * - Null, size, and alignment checks do not validate dangling/forged pointers.
 * See docs/api.md for memory, sample, failure, and compatibility contracts. */

J2K_API uint32_t j2k_abi_version(void);
/* Immutable, NUL-terminated string with library lifetime; never free it. */
J2K_API const char *j2k_version(void);

/* Cancellation is permanent. Only cancel may run concurrently with calls that
 * use the same token; never free during use. Operations clone the owned token
 * before codec work. Create requires an unowned output slot and nulls it on
 * failure; free nulls the slot and accepts a null handle. */
J2K_API j2k_status j2k_cancellation_create(j2k_cancellation **out, j2k_error *error);
J2K_API j2k_status j2k_cancellation_cancel(const j2k_cancellation *token, j2k_error *error);
J2K_API j2k_status j2k_cancellation_free(j2k_cancellation **token, j2k_error *error);

/* Copies input on success. Caller may then release or modify input.
 * out must not already own a decoder; it is set to NULL on failure.
 * Header inspection succeeds independently of pixel-decode support. */
J2K_API j2k_status j2k_decoder_open(
    const uint8_t *input, size_t input_len, const j2k_open_options *options,
    j2k_decoder **out, j2k_error *error);

/* Releases a handle and sets *decoder to NULL; closing a NULL handle succeeds. */
J2K_API j2k_status j2k_decoder_close(
    j2k_decoder **decoder, j2k_error *error);

J2K_API j2k_status j2k_decoder_info(
    j2k_decoder *decoder, j2k_image_info *out, j2k_error *error);

J2K_API j2k_status j2k_decoder_component_info(
    j2k_decoder *decoder, uint32_t index,
    j2k_component_info *out, j2k_error *error);

/* The first COLR box, which decoding applies; a raw codestream reports
 * J2K_COLOR_NONE. has_icc_profile in j2k_image_info covers every COLR box. */
J2K_API j2k_status j2k_decoder_color_info(
    j2k_decoder *decoder, j2k_color_info *out, j2k_error *error);

/* Copies the applied ICC profile (j2k_color_info.icc_profile_bytes long).
 * J2K_INVALID_ARGUMENT without one; J2K_BUFFER_TOO_SMALL leaves output untouched. */
J2K_API j2k_status j2k_decoder_icc_profile(
    j2k_decoder *decoder, uint8_t *output, size_t output_len, j2k_error *error);

/* Prepare immutable native input facts explicitly, outside per-operation
 * limits/cancellation. Optional: a native exact operation on an unprepared
 * decoder prepares it, under that operation's working limit and cancellation
 * when it has them; only a successful preparation is kept. */
J2K_API j2k_status j2k_decoder_prepare_operations(j2k_decoder *decoder, j2k_error *error);

/* Native output preserves U8/U16/I16 values and association. Explicit sRGB
 * output is eight-bit with straight alpha. Raw sRGB operations construct their
 * source under the operation's working limit and cancellation as well as its
 * output limit. ICC conversion under a working limit requires a matrix/TRC or
 * gray-TRC profile with an XYZ connection space. Native
 * working limits include live derived plans, current owned output and scratch,
 * not input facts already prepared. */
J2K_API j2k_status j2k_decoder_plan_region(
    j2k_decoder *decoder, const j2k_operation_options *options,
    j2k_region_plan *out, j2k_error *error);
/* No writes on failure. Optional out receives actual metadata and warnings;
 * normalized warnings need not be known during planning. */
J2K_API j2k_status j2k_decoder_decode_region_into(
    j2k_decoder *decoder, const j2k_operation_options *options,
    uint8_t *output, size_t output_len, size_t stride,
    j2k_output_info *out, j2k_error *error);
/* Bounds only derived native operation-plan retention; zero disables caching.
 * Original source preparation remains with the decoder until close. */
J2K_API j2k_status j2k_decoder_set_operation_cache_limit(
    j2k_decoder *decoder, size_t bytes, j2k_error *error);

/* Prepares and validates an output layout. Does not guarantee valid entropy data
 * or that the codec's working-memory budget suffices; see docs/supported.md. */
J2K_API j2k_status j2k_decoder_output_info(
    j2k_decoder *decoder, const j2k_decode_options *options,
    j2k_output_info *out, j2k_error *error);

/* Exact native integer samples, interleaved by pixel; 16-bit values are LE.
 * stride >= row_bytes. Required span = stride*(height-1)+row_bytes; last-row
 * padding is not required. Padding and bytes beyond that span are untouched.
 * On ordinary errors, output is unchanged. After PANIC/INTERNAL_ERROR, discard
 * output and close the decoder. No exception or unwinding crosses this ABI. */
J2K_API j2k_status j2k_decoder_decode_into(
    j2k_decoder *decoder, const j2k_decode_options *options,
    uint8_t *output, size_t output_len, size_t stride, j2k_error *error);

/* The layout j2k_decoder_decode_rows delivers: the full image at full
 * resolution, in the native samples j2k_decoder_output_info would report for
 * it. options must be NULL or select the full image at reduction 0. No
 * whole-image output limit applies. 8-bit unsigned and 9- to 16-bit samples
 * only; other precisions return J2K_UNSUPPORTED (use decode_into). */
J2K_API j2k_status j2k_decoder_row_info(
    j2k_decoder *decoder, const j2k_decode_options *options,
    j2k_output_info *out, j2k_error *error);

/* Decodes the full image and calls callback with each row, in order and once
 * each, holding at most max_rows_per_stripe decoded rows (0 = 64). The callback
 * runs on the calling thread or, for a decoder with its own threads, on one of
 * them while this call blocks; it is never called concurrently and must not
 * call back into this decoder. A nonzero return stops the decode with
 * J2K_STOPPED; rows already delivered stay delivered. On success out (may be
 * NULL) receives the j2k_decoder_row_info layout with warnings. */
J2K_API j2k_status j2k_decoder_decode_rows(
    j2k_decoder *decoder, const j2k_decode_options *options,
    uint32_t max_rows_per_stripe, j2k_row_callback callback, void *user_data,
    j2k_output_info *out, j2k_error *error);

/* Encodes interleaved samples losslessly into a buffer this library allocates.
 * The encoder decodes its result and compares it with the samples before
 * returning it. On success *out and *out_len describe the encoded bytes;
 * release them with j2k_buffer_free. On failure *out is NULL and *out_len 0.
 * Encoding runs on the codec's shared CPU pool. */
J2K_API j2k_status j2k_encode_lossless(
    const uint8_t *samples, size_t samples_len, const j2k_encode_options *options,
    uint8_t **out, size_t *out_len, j2k_error *error);

/* Encodes interleaved samples with coding parameters (NULL = lossless
 * defaults, as j2k_encode_lossless) into a buffer this library allocates. The
 * encoder decodes its result before returning it; lossless output must match
 * the samples. On success *out and *out_len describe the encoded bytes and
 * report (may be NULL) what the codec measured; release the bytes with
 * j2k_buffer_free. On failure *out is NULL, *out_len 0, and report unwritten. */
J2K_API j2k_status j2k_encode(
    const uint8_t *samples, size_t samples_len, const j2k_encode_options *options,
    const j2k_encode_params *params, uint8_t **out, size_t *out_len,
    j2k_encode_report *report, j2k_error *error);

/* Releases a buffer from j2k_encode or j2k_encode_lossless with its length.
 * NULL is ignored. */
J2K_API void j2k_buffer_free(uint8_t *data, size_t length);

#ifdef __cplusplus
}
#endif
#endif
