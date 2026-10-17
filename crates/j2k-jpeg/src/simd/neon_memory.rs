// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fixed-size `AArch64` NEON memory operations.

use core::arch::aarch64::{
    int16x8_t, uint16x8_t, uint8x16_t, uint8x16x3_t, uint8x8_t, uint8x8x3_t, vaddq_u8, vdupq_n_u8,
    vld1_u8, vld1q_s16, vld1q_u8, vminq_u8, vqtbl1q_u8, vst1_u8, vst1q_u16, vst1q_u8, vst3_u8,
    vst3q_u8,
};

const WINDOW_LANE_OFFSETS: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
const HEAD_WINDOW_LANE_OFFSETS: [u8; 16] = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];

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
pub(crate) fn store_u16x8(dst: &mut [u16; 8], values: uint16x8_t) {
    // SAFETY:
    // - Feature availability: callers run inside a `Neon` token kernel.
    // - Bounds: the array reference proves eight writable u16 samples.
    // - Alignment: AArch64 `vst1q_u16` supports unaligned u16 addresses.
    // - Aliasing: the exclusive reference prevents overlapping live access.
    // - Initialization: the store initializes every output sample.
    unsafe { vst1q_u16(dst.as_mut_ptr(), values) };
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

#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn load_u8x16(src: &[u8; 16]) -> uint8x16_t {
    // SAFETY:
    // - Feature availability: callers run inside a `Neon` token kernel.
    // - Bounds: the array reference proves sixteen readable bytes.
    // - Alignment: AArch64 `vld1q_u8` supports unaligned byte addresses.
    // - Aliasing: the shared reference permits reads and no writes occur.
    // - Initialization: all bytes behind a Rust reference are initialized.
    unsafe { vld1q_u8(src.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn store_rgb8x16(
    dst: &mut [u8; 48],
    red: uint8x16_t,
    green: uint8x16_t,
    blue: uint8x16_t,
) {
    // SAFETY:
    // - Feature availability: callers run inside a `Neon` token kernel.
    // - Bounds: the array reference proves space for sixteen three-byte pixels.
    // - Alignment: AArch64 `vst3q_u8` supports unaligned byte addresses.
    // - Aliasing: the exclusive reference prevents overlapping live access.
    // - Initialization: the store initializes all forty-eight output bytes.
    unsafe { vst3q_u8(dst.as_mut_ptr(), uint8x16x3_t(red, green, blue)) };
}

#[target_feature(enable = "neon")]
pub(crate) fn load_tail_window(src: &[u8], start: usize, len: usize) -> [u8; 16] {
    debug_assert!(start < src.len());
    debug_assert!(len > 0);
    debug_assert!(len <= 16);
    if src.len() >= 16 && start < src.len() && (1..=16).contains(&len) {
        // Load the nearest in-bounds vector, then clamp every table index to
        // the final source byte that belongs to this replicated window.
        let base = start.min(src.len() - 16);
        let window = src[base..]
            .first_chunk::<16>()
            .expect("tail window load stays within the source row");
        let offset = start - base;
        let copy_len = (src.len() - start).min(len);
        let last =
            u8::try_from(offset + copy_len - 1).expect("tail window last index fits one lane");
        let offset = u8::try_from(offset).expect("tail window offset fits one lane");
        let indices = vminq_u8(
            vaddq_u8(load_u8x16(&WINDOW_LANE_OFFSETS), vdupq_n_u8(offset)),
            vdupq_n_u8(last),
        );
        let mut out = [0u8; 16];
        store_u8x16(&mut out, vqtbl1q_u8(load_u8x16(window), indices));
        return out;
    }
    let mut out = [0u8; 16];
    let available = src.len() - start;
    let copy_len = available.min(len);
    out[..copy_len].copy_from_slice(&src[start..start + copy_len]);
    if copy_len < len {
        let pad = out[copy_len - 1];
        for value in &mut out[copy_len..len] {
            *value = pad;
        }
    }
    if len < 16 {
        let pad = out[len - 1];
        for value in &mut out[len..16] {
            *value = pad;
        }
    }
    out
}

#[target_feature(enable = "neon")]
pub(crate) fn load_head_window(src: &[u8], len: usize) -> [u8; 16] {
    debug_assert!(
        !src.is_empty(),
        "head window needs at least one source byte"
    );
    debug_assert!(len > 0);
    debug_assert!(len <= 16);
    if src.len() >= 16 && (1..16).contains(&len) {
        // Lane zero is the replicated left edge. Later lanes advance through
        // the requested prefix and clamp to its final source byte.
        let window = src
            .first_chunk::<16>()
            .expect("head window load stays within the source row");
        let copy_len = src.len().min(len);
        let last = u8::try_from(copy_len - 1).expect("head window last index fits one lane");
        let indices = vminq_u8(load_u8x16(&HEAD_WINDOW_LANE_OFFSETS), vdupq_n_u8(last));
        let mut out = [0u8; 16];
        store_u8x16(&mut out, vqtbl1q_u8(load_u8x16(window), indices));
        return out;
    }
    let mut out = [0u8; 16];
    let copy_len = src.len().min(len);
    out[0] = src[0];
    out[1..=copy_len].copy_from_slice(&src[..copy_len]);
    if copy_len < len {
        let pad = out[copy_len];
        for value in &mut out[copy_len + 1..=len] {
            *value = pad;
        }
    }
    if len + 1 < 16 {
        let pad = out[len];
        for value in &mut out[len + 1..16] {
            *value = pad;
        }
    }
    out
}
