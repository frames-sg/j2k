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

/// Prepared output geometry and native sample interpretation.
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
    /// Significant sample bits; samples retain the native integer range.
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
