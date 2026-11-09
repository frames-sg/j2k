// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fixed-width values shared with `include/j2k.h`.

#![forbid(unsafe_code)]

use std::ffi::c_char;

/// C ABI revision, independent of the library's package version.
pub const J2K_ABI_VERSION: u32 = 1;
/// Success.
pub const J2K_OK: u32 = 0;
/// Invalid argument, pointer, index, region, or stride.
pub const J2K_INVALID_ARGUMENT: u32 = 1;
/// Unsupported ABI revision or structure prefix size.
pub const J2K_ABI_MISMATCH: u32 = 2;
/// Invalid, truncated, or non-JPEG 2000 encoded input.
pub const J2K_INVALID_INPUT: u32 = 3;
/// Valid input or request outside this library's supported subset.
pub const J2K_UNSUPPORTED: u32 = 4;
/// Destination does not cover the required output bytes.
pub const J2K_BUFFER_TOO_SMALL: u32 = 5;
/// A resource limit or representable byte count was exceeded.
pub const J2K_RESOURCE_LIMIT: u32 = 6;
/// A fallible host allocation failed.
pub const J2K_ALLOCATION_FAILED: u32 = 7;
/// An internal codec or library invariant failed; close and recreate the affected decoder.
pub const J2K_INTERNAL_ERROR: u32 = 8;
/// A panic was contained; close and recreate the affected decoder.
pub const J2K_PANIC: u32 = 9;
/// Codec failure whose public error does not expose a narrower classification.
pub const J2K_DECODE_FAILED: u32 = 10;
/// A row callback returned nonzero; the rows it already received stay delivered.
pub const J2K_STOPPED: u32 = 11;
/// The encoder failed; the message says why.
pub const J2K_ENCODE_FAILED: u32 = 12;
/// The operation's cancellation token was cancelled.
pub const J2K_CANCELLED: u32 = 13;

/// Preserve native integer sample values and alpha association.
pub const J2K_INTERPRETATION_NATIVE: u32 = 0;
/// Normalize to eight-bit sRGB with straight alpha.
pub const J2K_INTERPRETATION_SRGB8: u32 = 1;
/// The CPU selects a native prepared, streamed, or component route during execution.
pub const J2K_ROUTE_NATIVE_CPU: u32 = 1;
/// Bounded native region reconstruction followed by colour normalization.
pub const J2K_ROUTE_REGION_PLANES: u32 = 2;

/// Open flag: accept the JP2/JPH metadata recoveries of lenient decoding.
pub const J2K_OPEN_LENIENT: u32 = 1;
/// Every open flag this ABI revision defines.
pub(crate) const J2K_OPEN_FLAGS: u32 = J2K_OPEN_LENIENT;
/// Output warning: a lenient JP2/JPH metadata recovery was applied.
pub const J2K_WARNING_LENIENT_RECOVERY: u32 = 1;

/// Encode flag: HTJ2K (Part 15) block coding instead of classic Part 1.
pub const J2K_ENCODE_HTJ2K: u32 = 1;
/// Encode flag: the samples are signed two's-complement values.
pub const J2K_ENCODE_SIGNED: u32 = 2;
/// Encode flag: wrap the codestream in a JP2 file, or JPH with HTJ2K.
pub const J2K_ENCODE_FILE: u32 = 4;
/// Every encode flag this ABI revision defines.
pub(crate) const J2K_ENCODE_FLAGS: u32 = J2K_ENCODE_HTJ2K | J2K_ENCODE_SIGNED | J2K_ENCODE_FILE;

/// Encode mode: reversible 5/3 coding that reproduces the samples exactly.
pub const J2K_MODE_LOSSLESS: u32 = 0;
/// Encode mode: irreversible 9/7 coding, optionally to rate or quality targets.
pub const J2K_MODE_LOSSY: u32 = 1;
/// Parameter flag: code three lossless components without the reversible colour transform.
pub const J2K_PARAMS_NO_COLOR_TRANSFORM: u32 = 1;
/// Every parameter flag this ABI revision defines.
pub(crate) const J2K_PARAMS_FLAGS: u32 = J2K_PARAMS_NO_COLOR_TRANSFORM;
/// Decomposition levels: the codec's default for the image geometry.
pub const J2K_LEVELS_AUTO: u32 = u32::MAX;
/// Layer-resolution-component-position packet progression (the default).
pub const J2K_PROGRESSION_LRCP: u32 = 0;
/// Resolution-layer-component-position packet progression.
pub const J2K_PROGRESSION_RLCP: u32 = 1;
/// Resolution-position-component-layer packet progression.
pub const J2K_PROGRESSION_RPCL: u32 = 2;
/// Position-component-resolution-layer packet progression.
pub const J2K_PROGRESSION_PCRL: u32 = 3;
/// Component-position-resolution-layer packet progression.
pub const J2K_PROGRESSION_CPRL: u32 = 4;
/// Lossy targets: none; the codec's default quantization.
pub const J2K_RATE_NONE: u32 = 0;
/// Lossy targets in codestream bits per image pixel.
pub const J2K_RATE_BITS_PER_PIXEL: u32 = 1;
/// Lossy targets in codestream bytes.
pub const J2K_RATE_BYTES: u32 = 2;
/// Lossy targets in PSNR decibels against the samples.
pub const J2K_RATE_PSNR_DB: u32 = 3;
/// Marker segment: SOP before each packet.
pub const J2K_MARKER_SOP: u32 = 1;
/// Marker segment: EPH after each packet header.
pub const J2K_MARKER_EPH: u32 = 2;
/// Marker segment: TLM tile-part lengths.
pub const J2K_MARKER_TLM: u32 = 4;
/// Marker segment: PLT packet lengths in each tile-part header.
pub const J2K_MARKER_PLT: u32 = 8;
/// Marker segment: PLM packet lengths in the main header.
pub const J2K_MARKER_PLM: u32 = 16;
/// Marker segment: PPM packed packet headers in the main header.
pub const J2K_MARKER_PPM: u32 = 32;
/// Marker segment: PPT packed packet headers in each tile-part header.
pub const J2K_MARKER_PPT: u32 = 64;
/// Every marker segment bit this ABI revision defines.
pub(crate) const J2K_MARKERS: u32 = 127;

/// Colour specification: none; a raw codestream carries no COLR box.
pub const J2K_COLOR_NONE: u32 = 0;
/// Colour specification: an enumerated colour space (`value` holds `EnumCS`).
pub const J2K_COLOR_ENUMERATED: u32 = 1;
/// Colour specification: an embedded ICC profile.
pub const J2K_COLOR_ICC: u32 = 2;
/// Colour specification: a COLR method this codec does not interpret (`value` holds it).
pub const J2K_COLOR_UNKNOWN: u32 = 3;

/// Unsigned eight-bit samples.
pub const J2K_U8: u32 = 1;
/// Unsigned sixteen-bit samples, stored little-endian.
pub const J2K_U16_LE: u32 = 2;
/// Signed sixteen-bit samples, stored little-endian.
pub const J2K_I16_LE: u32 = 3;
/// Output has no alpha channel.
pub const J2K_ALPHA_NONE: u32 = 0;
/// Output has straight alpha.
pub const J2K_ALPHA_STRAIGHT: u32 = 1;
/// Output has premultiplied alpha.
pub const J2K_ALPHA_PREMULTIPLIED: u32 = 2;

/// Prefix of every extensible options or result structure.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct J2kCHeader {
    /// Size of the caller's complete structure, in bytes.
    pub struct_size: u32,
    /// Required ABI revision.
    pub abi_version: u32,
}

impl J2kCHeader {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "const fn cannot use u32::try_from; the assertion bounds the size first"
    )]
    pub(crate) const fn of<T>() -> Self {
        assert!(std::mem::size_of::<T>() <= u32::MAX as usize);
        Self {
            struct_size: std::mem::size_of::<T>() as u32,
            abi_version: J2K_ABI_VERSION,
        }
    }
}

/// Optional caller-owned diagnostic. Its fixed layout is part of ABI 1.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct J2kCError {
    /// Same status as the function return value.
    pub code: u32,
    /// NUL-terminated UTF-8, truncated at a character boundary if necessary.
    pub message: [c_char; 256],
}

impl Default for J2kCError {
    fn default() -> Self {
        Self {
            code: J2K_OK,
            message: [0; 256],
        }
    }
}

/// Per-decoder configuration. Zero byte limits select 512 MiB defaults.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct J2kCOpenOptions {
    /// Initialize to this structure's size and ABI 1.
    pub header: J2kCHeader,
    /// Maximum encoded input length. Not a total-memory limit.
    pub max_input_bytes: u64,
    /// Maximum output span, including inter-row padding.
    pub max_output_bytes: u64,
    /// Decode threads: 0 uses the codec's shared, host-sized CPU pool; 1–64
    /// gives the decoder its own pool of exactly that many threads.
    pub workers: u32,
    /// `J2K_OPEN_*` flags; other bits must be zero.
    pub flags: u32,
}

impl Default for J2kCOpenOptions {
    fn default() -> Self {
        Self {
            header: J2kCHeader::of::<Self>(),
            max_input_bytes: 0,
            max_output_bytes: 0,
            workers: 0,
            flags: 0,
        }
    }
}

/// Decode geometry. A completely zero region means the entire image.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct J2kCDecodeOptions {
    /// Initialize to this structure's size and ABI 1.
    pub header: J2kCHeader,
    /// Left edge in full-resolution image coordinates.
    pub x: u32,
    /// Top edge in full-resolution image coordinates.
    pub y: u32,
    /// Region width; must be positive for a non-default region.
    pub width: u32,
    /// Region height; must be positive for a non-default region.
    pub height: u32,
    /// Number of wavelet resolution levels to discard: 0–3.
    pub reduction: u32,
    /// Must be zero.
    pub reserved: u32,
}

impl Default for J2kCDecodeOptions {
    fn default() -> Self {
        Self {
            header: J2kCHeader::of::<Self>(),
            x: 0,
            y: 0,
            width: 0,
            height: 0,
            reduction: 0,
            reserved: 0,
        }
    }
}

/// Exact region operation. The all-zero source rectangle selects the full image.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct J2kCOperationOptions {
    /// Initialize to this structure's size and ABI 1.
    pub header: J2kCHeader,
    /// Left edge in full-resolution image coordinates.
    pub x: u32,
    /// Top edge in full-resolution image coordinates.
    pub y: u32,
    /// Source region width.
    pub width: u32,
    /// Source region height.
    pub height: u32,
    /// Exact wavelet resolution levels to discard, 0–31.
    pub reduction: u32,
    /// One of the `J2K_INTERPRETATION_*` constants.
    pub interpretation: u32,
    /// Native operation working bound; zero keeps the engine's default.
    /// Original source preparation and input are separate from this bound.
    pub max_working_memory_bytes: u64,
    /// Final output span bound; zero uses the decoder's configured output cap.
    pub max_output_bytes: u64,
    /// Optional live cancellation handle. Only cancellation may run concurrently.
    pub cancellation: *const crate::ffi::J2kCCancellation,
}

impl Default for J2kCOperationOptions {
    fn default() -> Self {
        Self {
            header: J2kCHeader::of::<Self>(),
            x: 0,
            y: 0,
            width: 0,
            height: 0,
            reduction: 0,
            interpretation: J2K_INTERPRETATION_NATIVE,
            max_working_memory_bytes: 0,
            max_output_bytes: 0,
            cancellation: std::ptr::null(),
        }
    }
}

/// Source header metadata. Inspection does not establish decodability.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct J2kCImageInfo {
    /// Initialize to this structure's size and ABI 1.
    pub header: J2kCHeader,
    /// Source width.
    pub width: u32,
    /// Source height.
    pub height: u32,
    /// Number of codestream components.
    pub components: u32,
    /// Header-reported resolution count; preparation validates each request.
    pub resolution_levels: u32,
    /// One if any source component is signed, otherwise zero.
    pub has_signed_components: u32,
    /// One if the file declares an ICC profile, otherwise zero.
    pub has_icc_profile: u32,
    /// Horizontal image origin on the reference grid (`XOsiz`).
    pub image_x0: u32,
    /// Vertical image origin on the reference grid (`YOsiz`).
    pub image_y0: u32,
    /// Horizontal tile-grid origin (`XTOsiz`).
    pub tile_x0: u32,
    /// Vertical tile-grid origin (`YTOsiz`).
    pub tile_y0: u32,
    /// Nominal tile width (`XTsiz`).
    pub tile_width: u32,
    /// Nominal tile height (`YTsiz`).
    pub tile_height: u32,
}

impl Default for J2kCImageInfo {
    fn default() -> Self {
        Self {
            header: J2kCHeader::of::<Self>(),
            width: 0,
            height: 0,
            components: 0,
            resolution_levels: 0,
            has_signed_components: 0,
            has_icc_profile: 0,
            image_x0: 0,
            image_y0: 0,
            tile_x0: 0,
            tile_y0: 0,
            tile_width: 0,
            tile_height: 0,
        }
    }
}

/// Source precision and sampling for one codestream component.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct J2kCComponentInfo {
    /// Initialize to this structure's size and ABI 1.
    pub header: J2kCHeader,
    /// Significant source bits.
    pub precision: u32,
    /// One for signed samples, zero for unsigned samples.
    pub is_signed: u32,
    /// Horizontal source sampling factor.
    pub sampling_x: u32,
    /// Vertical source sampling factor.
    pub sampling_y: u32,
}

impl Default for J2kCComponentInfo {
    fn default() -> Self {
        Self {
            header: J2kCHeader::of::<Self>(),
            precision: 0,
            is_signed: 0,
            sampling_x: 0,
            sampling_y: 0,
        }
    }
}

/// Output geometry and sample interpretation.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct J2kCOutputInfo {
    /// Initialize to this structure's size and ABI 1.
    pub header: J2kCHeader,
    /// Left edge of the covering output rectangle at the requested resolution.
    pub x: u32,
    /// Top edge of the covering output rectangle at the requested resolution.
    pub y: u32,
    /// Output width.
    pub width: u32,
    /// Output height.
    pub height: u32,
    /// Interleaved channels: 1 (gray), 3 (RGB), or 4 (RGBA).
    pub channels: u32,
    /// One of `J2K_U8`, `J2K_U16_LE`, or `J2K_I16_LE`.
    pub sample_type: u32,
    /// Significant output bits; native operations retain the source integer range.
    pub precision: u32,
    /// One of the `J2K_ALPHA_*` constants.
    pub alpha: u32,
    /// Minimum byte stride for one tightly packed row.
    pub row_bytes: u64,
    /// Byte length for tightly packed output.
    pub buffer_bytes: u64,
    /// `J2K_WARNING_*` bits: recoveries the request applies or applied.
    pub warnings: u32,
    /// Written as zero.
    pub reserved: u32,
}

impl Default for J2kCOutputInfo {
    fn default() -> Self {
        Self {
            header: J2kCHeader::of::<Self>(),
            x: 0,
            y: 0,
            width: 0,
            height: 0,
            channels: 0,
            sample_type: 0,
            precision: 0,
            alpha: 0,
            row_bytes: 0,
            buffer_bytes: 0,
            warnings: 0,
            reserved: 0,
        }
    }
}

/// Exact operation geometry and output interpretation, without pixel allocation.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct J2kCRegionPlan {
    /// Initialize to this structure's size and ABI 1.
    pub header: J2kCHeader,
    /// Covered output rectangle, packed lengths, and sample metadata.
    pub output: J2kCOutputInfo,
    /// Requested full-resolution source left edge.
    pub source_x: u32,
    /// Requested full-resolution source top edge.
    pub source_y: u32,
    /// Requested source width.
    pub source_width: u32,
    /// Requested source height.
    pub source_height: u32,
    /// Exact requested reduction.
    pub reduction: u32,
    /// Largest reduction common to every source component and tile.
    pub max_reduction: u32,
    /// One of the `J2K_INTERPRETATION_*` constants.
    pub interpretation: u32,
    /// One of the `J2K_ROUTE_*` constants. Later validation can still fail.
    pub route: u32,
}

impl Default for J2kCRegionPlan {
    fn default() -> Self {
        Self {
            header: J2kCHeader::of::<Self>(),
            output: J2kCOutputInfo::default(),
            source_x: 0,
            source_y: 0,
            source_width: 0,
            source_height: 0,
            reduction: 0,
            max_reduction: 0,
            interpretation: 0,
            route: 0,
        }
    }
}

/// Geometry and coding of interleaved samples to encode losslessly.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct J2kCEncodeOptions {
    /// Initialize to this structure's size and ABI 1.
    pub header: J2kCHeader,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Interleaved components per pixel.
    pub components: u32,
    /// Significant bits per sample, 1–16. Samples of up to 8 bits take one
    /// byte; deeper samples take two, little-endian.
    pub bit_depth: u32,
    /// `J2K_ENCODE_*` flags; other bits must be zero.
    pub flags: u32,
    /// Must be zero.
    pub reserved: u32,
}

impl Default for J2kCEncodeOptions {
    fn default() -> Self {
        Self {
            header: J2kCHeader::of::<Self>(),
            width: 0,
            height: 0,
            components: 0,
            bit_depth: 0,
            flags: 0,
            reserved: 0,
        }
    }
}

/// Coding parameters for [`crate::j2k_encode`]. Zero fields select defaults,
/// except `levels`, whose default is [`J2K_LEVELS_AUTO`].
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct J2kCEncodeParams {
    /// Initialize to this structure's size and ABI 1.
    pub header: J2kCHeader,
    /// `J2K_MODE_LOSSLESS` or `J2K_MODE_LOSSY`.
    pub mode: u32,
    /// `J2K_PARAMS_*` flags; other bits must be zero.
    pub flags: u32,
    /// Maximum wavelet decompositions, or `J2K_LEVELS_AUTO`; the image and
    /// tile geometry can lower it.
    pub levels: u32,
    /// One of the `J2K_PROGRESSION_*` constants.
    pub progression: u32,
    /// Tile width; zero with a zero tile height codes one tile.
    pub tile_width: u32,
    /// Tile height; zero with a zero tile width codes one tile.
    pub tile_height: u32,
    /// `J2K_MARKER_*` segments to write; other bits must be zero.
    pub markers: u32,
    /// Complete packets per tile-part, 1–65535; zero places no limit.
    pub tile_part_packets: u32,
    /// Lossless quality layers, 1–32; zero writes one. Lossy layers come from `targets`.
    pub quality_layers: u32,
    /// Lossy HTJ2K visual quality factor, 1–100; zero uses none.
    pub qfactor: u32,
    /// Lossy: one of the `J2K_RATE_*` constants, the unit of `targets`.
    pub rate: u32,
    /// Lossy: zero, one codestream target, or 2–32 cumulative layer targets.
    pub target_count: u32,
    /// `target_count` ascending targets in `rate` units; byte targets are whole numbers.
    pub targets: *const f64,
    /// With `J2K_ENCODE_FILE`: the ICC profile for the COLR box. Null writes
    /// the enumerated colour space the samples imply.
    pub icc_profile: *const u8,
    /// Byte length of `icc_profile`.
    pub icc_profile_len: usize,
}

impl Default for J2kCEncodeParams {
    fn default() -> Self {
        Self {
            header: J2kCHeader::of::<Self>(),
            mode: J2K_MODE_LOSSLESS,
            flags: 0,
            levels: J2K_LEVELS_AUTO,
            progression: J2K_PROGRESSION_LRCP,
            tile_width: 0,
            tile_height: 0,
            markers: 0,
            tile_part_packets: 0,
            quality_layers: 0,
            qfactor: 0,
            rate: J2K_RATE_NONE,
            target_count: 0,
            targets: std::ptr::null(),
            icc_profile: std::ptr::null(),
            icc_profile_len: 0,
        }
    }
}

/// What an encode produced, measured on its codestream.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct J2kCEncodeReport {
    /// Initialize to this structure's size and ABI 1.
    pub header: J2kCHeader,
    /// Codestream length, excluding any JP2/JPH wrapper.
    pub codestream_bytes: u64,
    /// Codestream bits per image pixel.
    pub bits_per_pixel: f64,
    /// PSNR against the samples when the encode measured it, otherwise zero.
    pub psnr_db: f64,
    /// Quality layers in the codestream.
    pub quality_layers: u32,
    /// Written as zero.
    pub reserved: u32,
}

impl Default for J2kCEncodeReport {
    fn default() -> Self {
        Self {
            header: J2kCHeader::of::<Self>(),
            codestream_bytes: 0,
            bits_per_pixel: 0.0,
            psnr_db: 0.0,
            quality_layers: 0,
            reserved: 0,
        }
    }
}

/// The colour specification decoding applies: the file's first COLR box.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct J2kCColorInfo {
    /// Initialize to this structure's size and ABI 1.
    pub header: J2kCHeader,
    /// One of the `J2K_COLOR_*` constants.
    pub specification: u32,
    /// `EnumCS` for an enumerated space, the COLR method for an unknown one, else zero.
    pub value: u32,
    /// COLR boxes in the file; boxes after the first are preserved, not applied.
    pub specifications: u32,
    /// Written as zero.
    pub reserved: u32,
    /// Byte length of the applied ICC profile; zero without one.
    pub icc_profile_bytes: u64,
}

impl Default for J2kCColorInfo {
    fn default() -> Self {
        Self {
            header: J2kCHeader::of::<Self>(),
            specification: J2K_COLOR_NONE,
            value: 0,
            specifications: 0,
            reserved: 0,
            icc_profile_bytes: 0,
        }
    }
}
