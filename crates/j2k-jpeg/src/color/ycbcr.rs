// SPDX-License-Identifier: MIT OR Apache-2.0

//! YCbCr → RGB conversion. Scalar implementation uses libjpeg-turbo's 16-bit
//! fixed-point coefficients (`jdcolor.c`), so outputs match their ISLOW path
//! byte-for-byte.
//!
//! Coefficients (all × 2^16, rounded):
//!     R = Y                        + 1.40200 * (Cr - 128)
//!     G = Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)
//!     B = Y + 1.77200 * (Cb - 128)

pub(crate) const FIX_1_40200: i32 = 91_881; // (int)(1.40200 * 65536 + 0.5)
pub(crate) const FIX_0_34414: i32 = 22_554; // (int)(0.34414 * 65536 + 0.5)
pub(crate) const FIX_0_71414: i32 = 46_802; // (int)(0.71414 * 65536 + 0.5)
pub(crate) const FIX_1_77200: i32 = 116_130; // (int)(1.77200 * 65536 + 0.5)
pub(crate) const ROUND: i32 = 1 << 15; // 0.5 in 16-bit fixed point

const fn clamp_to_u8(v: i32) -> u8 {
    if v < 0 {
        0
    } else if v > 255 {
        255
    } else {
        v as u8
    }
}

const fn clamp_to_12bit(v: i32) -> u16 {
    if v < 0 {
        0
    } else if v > 4095 {
        4095
    } else {
        v as u16
    }
}

const fn clamp_to_u16(v: i64) -> u16 {
    if v < 0 {
        0
    } else if v > u16::MAX as i64 {
        u16::MAX
    } else {
        v as u16
    }
}

/// Convert one YCbCr pixel to RGB. `y`, `cb`, `cr` are the 8-bit component
/// values as read from the decoded block after IDCT and upsample.
///
/// Returns `(R, G, B)` clamped to `[0, 255]`.
pub(crate) fn ycbcr_to_rgb(y: u8, cb: u8, cr: u8) -> (u8, u8, u8) {
    let y = y as i32;
    let cb_centered = cb as i32 - 128;
    let cr_centered = cr as i32 - 128;
    let r = y + ((FIX_1_40200 * cr_centered + ROUND) >> 16);
    let g = y - ((FIX_0_34414 * cb_centered + FIX_0_71414 * cr_centered + ROUND) >> 16);
    let b = y + ((FIX_1_77200 * cb_centered + ROUND) >> 16);

    (clamp_to_u8(r), clamp_to_u8(g), clamp_to_u8(b))
}

/// Convert one 12-bit YCbCr pixel to RGB samples stored in `Rgb16` output.
///
/// Returned values are clamped to the native 12-bit range `[0, 4095]`, not
/// scaled to the full `u16` range.
pub(crate) fn ycbcr12_to_rgb16(y: u16, cb: u16, cr: u16) -> (u16, u16, u16) {
    let y = i32::from(y);
    let cb_centered = i32::from(cb) - 2048;
    let cr_centered = i32::from(cr) - 2048;
    let r = y + ((FIX_1_40200 * cr_centered + ROUND) >> 16);
    let g = y - ((FIX_0_34414 * cb_centered + FIX_0_71414 * cr_centered + ROUND) >> 16);
    let b = y + ((FIX_1_77200 * cb_centered + ROUND) >> 16);

    (clamp_to_12bit(r), clamp_to_12bit(g), clamp_to_12bit(b))
}

/// Convert one 16-bit lossless YCbCr pixel to RGB samples stored in `Rgb16`
/// output.
///
/// Returned values are clamped to the native 16-bit range `[0, 65535]`.
pub(crate) fn ycbcr16_to_rgb16(y: u16, cb: u16, cr: u16) -> (u16, u16, u16) {
    let y = i64::from(y);
    let cb_centered = i64::from(cb) - 32768;
    let cr_centered = i64::from(cr) - 32768;
    let r = y + ((i64::from(FIX_1_40200) * cr_centered + i64::from(ROUND)) >> 16);
    let g = y
        - ((i64::from(FIX_0_34414) * cb_centered
            + i64::from(FIX_0_71414) * cr_centered
            + i64::from(ROUND))
            >> 16);
    let b = y + ((i64::from(FIX_1_77200) * cb_centered + i64::from(ROUND)) >> 16);

    (clamp_to_u16(r), clamp_to_u16(g), clamp_to_u16(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutral_gray_roundtrips_to_equal_rgb_channels() {
        let (r, g, b) = ycbcr_to_rgb(128, 128, 128);
        assert_eq!((r, g, b), (128, 128, 128));
    }

    #[test]
    fn bright_red_maps_to_high_r_low_gb() {
        // libjpeg-turbo: Y=76 Cb=85 Cr=255 ≈ pure red (255, 0, 0).
        let (r, g, b) = ycbcr_to_rgb(76, 85, 255);
        assert!(r > 240 && g < 15 && b < 15, "got ({r}, {g}, {b})");
    }

    #[test]
    fn clamps_out_of_range_arithmetic_to_0_255() {
        // Y=255, large Cr pushes R arithmetic above 255 → saturate high.
        let (r, _, _) = ycbcr_to_rgb(255, 128, 255);
        assert_eq!(r, 255, "R must saturate at 255");
        // Y=255, large Cb pushes B arithmetic above 255 → saturate high.
        let (_, _, b) = ycbcr_to_rgb(255, 255, 128);
        assert_eq!(b, 255, "B must saturate at 255");
        // Y=0, small Cr pushes R arithmetic below 0 → saturate low.
        let (r, _, _) = ycbcr_to_rgb(0, 128, 0);
        assert_eq!(r, 0, "R must saturate at 0");
    }

    #[test]
    fn matches_libjpeg_turbo_fixed_point_expectations() {
        // Sampled checks against libjpeg-turbo jdcolor.c computed values.
        // Y=100 Cb=150 Cr=200 → R=201, G=41, B=139 (16-bit fixed point).
        let (r, g, b) = ycbcr_to_rgb(100, 150, 200);
        assert!((r as i32 - 201).abs() <= 1, "R={r}, expected ≈201");
        assert!((g as i32 - 41).abs() <= 1, "G={g}, expected ≈41");
        assert!((b as i32 - 139).abs() <= 1, "B={b}, expected ≈139");
    }

    #[test]
    fn ycbcr12_to_rgb16_uses_native_12_bit_range() {
        assert_eq!(ycbcr12_to_rgb16(2048, 2048, 2048), (2048, 2048, 2048));
        assert_eq!(ycbcr12_to_rgb16(2064, 2072, 2032), (2042, 2067, 2107));
        assert_eq!(ycbcr12_to_rgb16(4095, 2048, 4095).0, 4095);
        assert_eq!(ycbcr12_to_rgb16(0, 2048, 0).0, 0);
    }

    #[test]
    fn ycbcr16_to_rgb16_uses_native_16_bit_range() {
        assert_eq!(ycbcr16_to_rgb16(32768, 32768, 32768), (32768, 32768, 32768));
        assert_eq!(ycbcr16_to_rgb16(33000, 35000, 40000), (43139, 27067, 36955));
        assert_eq!(ycbcr16_to_rgb16(u16::MAX, 32768, u16::MAX).0, u16::MAX);
        assert_eq!(ycbcr16_to_rgb16(0, 32768, 0).0, 0);
    }
}
