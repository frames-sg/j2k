// SPDX-License-Identifier: MIT OR Apache-2.0

//! Inverse discrete cosine transform. The scalar ISLOW path is the parity
//! oracle — every SIMD variant is proptested for bit-exact match against
//! `scalar::idct_islow`.

use j2k_codec_math::jpeg::idct;

/// Fraction bits of the ISLOW gains, as `i32`: the type of shift amounts in
/// the scalar transforms and of the SIMD shift-count generics.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "shared IDCT shift constants are small positive compile-time values"
)]
pub(crate) const CONST_BITS: i32 = idct::CONST_BITS as i32;
/// Fraction bits the 8-bit transform keeps between its passes.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "shared IDCT shift constants are small positive compile-time values"
)]
pub(crate) const PASS1_BITS: i32 = idct::PASS1_BITS as i32;
/// Right shift ending the 8-bit column pass.
pub(crate) const PASS1_SHIFT: i32 = CONST_BITS - PASS1_BITS;
/// Right shift ending the 8-bit row pass, including the 1/8 normalisation.
pub(crate) const PASS2_SHIFT: i32 = CONST_BITS + PASS1_BITS + 3;
/// Round-half-up addends of the two 8-bit passes.
pub(crate) const PASS1_ROUNDING: i32 = 1 << (PASS1_SHIFT - 1);
pub(crate) const PASS2_ROUNDING: i32 = 1 << (PASS2_SHIFT - 1);
/// The row pass's DC-only shortcut is `(p0 + DC_ROUNDING) >> DC_SHIFT`; it
/// equals the full row transform while `p0 << CONST_BITS` fits in `i32`.
pub(crate) const DC_SHIFT: i32 = PASS1_BITS + 3;
pub(crate) const DC_ROUNDING: i32 = 1 << (DC_SHIFT - 1);

/// Fraction bits the 12-bit transform keeps between its passes:
/// libjpeg-turbo's `jidctint.c` keeps one instead of two to leave headroom.
pub(crate) const PASS1_BITS_12: i32 = 1;
/// Right shifts ending the 12-bit column and row passes.
pub(crate) const PASS1_SHIFT_12: i32 = CONST_BITS - PASS1_BITS_12;
pub(crate) const PASS2_SHIFT_12: i32 = CONST_BITS + PASS1_BITS_12 + 3;

pub(crate) mod downscale;
pub(crate) mod scalar;

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2;

pub(crate) use scalar::idct_islow;
pub(crate) use scalar::idct_islow_12bit_dc_only_sample;
pub(crate) use scalar::idct_islow_dc_only;
pub(crate) use scalar::idct_islow_dc_only_pixel;
