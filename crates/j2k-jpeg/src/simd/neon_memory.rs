// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fixed-size `AArch64` NEON memory operations.

use core::arch::aarch64::{
    int16x8_t, uint8x16_t, uint8x8_t, uint8x8x3_t, vld1_u8, vld1q_s16, vst1_u8, vst1q_u8, vst3_u8,
};

#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn load_u8x8(src: &[u8; 8]) -> uint8x8_t {
    // SAFETY:
    // - Feature availability: callers run inside a `Neon` token kernel.
    // - Bounds: the array reference proves eight readable bytes.
    // - Alignment: AArch64 `vld1_u8` supports unaligned byte addresses.
    // - Aliasing: the shared reference permits reads and no writes occur.
    // - Initialization: all bytes behind a Rust reference are initialized.
    unsafe { vld1_u8(src.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn load_u8x8_triplet(src: &[u8; 10]) -> (uint8x8_t, uint8x8_t, uint8x8_t) {
    // SAFETY:
    // - Feature availability: callers run inside a `Neon` token kernel.
    // - Bounds: the fixed-size array proves that eight-byte loads beginning at
    //   offsets zero, one, and two are all readable.
    // - Alignment: AArch64 `vld1_u8` supports unaligned byte addresses.
    // - Aliasing: the shared reference permits reads and no writes occur.
    // - Initialization: all ten bytes behind the Rust reference are initialized.
    unsafe {
        let ptr = src.as_ptr();
        (vld1_u8(ptr), vld1_u8(ptr.add(1)), vld1_u8(ptr.add(2)))
    }
}

#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn load_i16x8(src: &[i16; 8]) -> int16x8_t {
    // SAFETY:
    // - Feature availability: callers run inside a `Neon` token kernel.
    // - Bounds: the array reference proves eight readable i16 coefficients.
    // - Alignment: AArch64 `vld1q_s16` supports unaligned i16 addresses.
    // - Aliasing: the shared reference permits reads and no writes occur.
    // - Initialization: all coefficients behind the reference are initialized.
    unsafe { vld1q_s16(src.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn store_u8x8(dst: &mut [u8; 8], values: uint8x8_t) {
    // SAFETY:
    // - Feature availability: callers run inside a `Neon` token kernel.
    // - Bounds: the array reference proves eight writable bytes.
    // - Alignment: AArch64 `vst1_u8` supports unaligned byte addresses.
    // - Aliasing: the exclusive reference prevents overlapping live access.
    // - Initialization: the store initializes every output byte.
    unsafe { vst1_u8(dst.as_mut_ptr(), values) };
}

#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn store_u8x16(dst: &mut [u8; 16], values: uint8x16_t) {
    // SAFETY:
    // - Feature availability: callers run inside a `Neon` token kernel.
    // - Bounds: the array reference proves sixteen writable bytes.
    // - Alignment: AArch64 `vst1q_u8` supports unaligned byte addresses.
    // - Aliasing: the exclusive reference prevents overlapping live access.
    // - Initialization: the store initializes every output byte.
    unsafe { vst1q_u8(dst.as_mut_ptr(), values) };
}

#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn store_rgb8(dst: &mut [u8; 24], red: uint8x8_t, green: uint8x8_t, blue: uint8x8_t) {
    // SAFETY:
    // - Feature availability: callers run inside a `Neon` token kernel.
    // - Bounds: the array reference proves space for eight three-byte pixels.
    // - Alignment: AArch64 `vst3_u8` supports unaligned byte addresses.
    // - Aliasing: the exclusive reference prevents overlapping live access.
    // - Initialization: the store initializes all twenty-four output bytes.
    unsafe { vst3_u8(dst.as_mut_ptr(), uint8x8x3_t(red, green, blue)) };
}
