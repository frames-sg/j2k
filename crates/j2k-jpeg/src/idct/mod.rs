// SPDX-License-Identifier: MIT OR Apache-2.0

//! Inverse discrete cosine transform. The scalar ISLOW path is the parity
//! oracle — every SIMD variant is proptested for bit-exact match against
//! `scalar::idct_islow`.

pub(crate) mod downscale;
pub(crate) mod scalar;

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2;

pub(crate) use scalar::idct_islow;
pub(crate) use scalar::idct_islow_12bit;
pub(crate) use scalar::idct_islow_12bit_dc_only_sample;
pub(crate) use scalar::idct_islow_dc_only;
pub(crate) use scalar::idct_islow_dc_only_pixel;
