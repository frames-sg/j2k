// SPDX-License-Identifier: MIT OR Apache-2.0

//! Inverse discrete cosine transform. The ISLOW maths lives in
//! `j2k_codec_math::jpeg::idct`; the scalar path here calls it and is the
//! parity oracle — every SIMD variant is proptested for bit-exact match
//! against `scalar::idct_islow`.

/// [`j2k_codec_math::jpeg::idct::CONST_BITS`] as `i32`, the type of the SIMD
/// shift-count generics.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "shared IDCT shift constants are small positive compile-time values"
)]
pub(crate) const CONST_BITS: i32 = j2k_codec_math::jpeg::idct::CONST_BITS as i32;

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
