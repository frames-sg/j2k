// SPDX-License-Identifier: MIT OR Apache-2.0

//! Adobe-style CMYK/YCCK to RGB conversion.

use crate::color::ycbcr::{ycbcr12_to_rgb16, ycbcr_to_rgb};

#[expect(
    clippy::cast_possible_truncation,
    reason = "the normalized product of two u8 samples is bounded to the u8 range"
)]
fn multiply_u8(a: u8, b: u8) -> u8 {
    ((u16::from(a) * u16::from(b) + 127) / 255) as u8
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the normalized product of two 12-bit samples is bounded to the u16 range"
)]
fn multiply_u12(a: u16, b: u16) -> u16 {
    ((u32::from(a) * u32::from(b) + 2047) / 4095) as u16
}

/// Convert complemented Adobe CMYK samples to RGB.
///
/// Adobe CMYK JPEG stores samples in the complemented domain, so each channel
/// acts like the amount of display light remaining after ink coverage.
pub(crate) fn inverted_cmyk_to_rgb(c: u8, m: u8, y: u8, k: u8) -> (u8, u8, u8) {
    (multiply_u8(c, k), multiply_u8(m, k), multiply_u8(y, k))
}

/// Convert one row of Adobe CMYK (`ycck == false`) or YCCK samples to RGB in
/// place: the first three rows receive red, green, and blue.
pub(crate) fn four_component_row_to_rgb(
    ycck: bool,
    first: &mut [u8],
    second: &mut [u8],
    third: &mut [u8],
    black: &[u8],
) {
    for (((first, second), third), &black) in first
        .iter_mut()
        .zip(second.iter_mut())
        .zip(third.iter_mut())
        .zip(black)
    {
        let (red, green, blue) = if ycck {
            ycck_to_rgb(*first, *second, *third, black)
        } else {
            inverted_cmyk_to_rgb(*first, *second, *third, black)
        };
        *first = red;
        *second = green;
        *third = blue;
    }
}

/// Convert 12-bit complemented Adobe CMYK samples to native-range RGB16.
pub(crate) fn inverted_cmyk12_to_rgb16(c: u16, m: u16, y: u16, k: u16) -> (u16, u16, u16) {
    (multiply_u12(c, k), multiply_u12(m, k), multiply_u12(y, k))
}

/// Convert Adobe YCCK samples directly to RGB.
///
/// Adobe encoders complement CMYK before the YCC transform, so the YCC
/// triple decodes to complemented CMY: libjpeg-turbo's `ycck_cmyk_convert`
/// yields `C = 255 - R(Y, Cb, Cr)` and passes K through unchanged. It uses the
/// C tables, whose green rounding equals `ycbcr_to_rgb`'s SIMD form for every
/// 8-bit chroma pair (the forms differ only on exact ties, which 8-bit
/// chroma never produces).
pub(crate) fn ycck_to_rgb(y: u8, cb: u8, cr: u8, k: u8) -> (u8, u8, u8) {
    let (red, green, blue) = ycbcr_to_rgb(y, cb, cr);
    inverted_cmyk_to_rgb(u8::MAX - red, u8::MAX - green, u8::MAX - blue, k)
}

/// Convert 12-bit Adobe YCCK samples directly to native-range RGB16.
pub(crate) fn ycck12_to_rgb16(y: u16, cb: u16, cr: u16, k: u16) -> (u16, u16, u16) {
    const MAX_12BIT: u16 = 4095;
    let (red, green, blue) = ycbcr12_to_rgb16(y, cb, cr);
    inverted_cmyk12_to_rgb16(MAX_12BIT - red, MAX_12BIT - green, MAX_12BIT - blue, k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverted_cmyk_multiplies_color_channels_by_black_channel() {
        assert_eq!(inverted_cmyk_to_rgb(255, 255, 255, 255), (255, 255, 255));
        assert_eq!(inverted_cmyk_to_rgb(255, 0, 0, 255), (255, 0, 0));
        assert_eq!(inverted_cmyk_to_rgb(128, 128, 128, 128), (64, 64, 64));
    }

    #[test]
    fn ycck_neutral_samples_match_inverted_cmyk_neutral() {
        assert_eq!(ycck_to_rgb(128, 128, 128, 128), (64, 64, 64));
    }

    #[test]
    fn ycck_complements_the_ycc_triple_before_applying_black() {
        // White YCC is no ink: (255, 255, 255) complemented CMY is 0.
        assert_eq!(ycck_to_rgb(255, 128, 128, 255), (0, 0, 0));
        assert_eq!(ycck_to_rgb(0, 128, 128, 255), (255, 255, 255));
    }

    #[test]
    fn eight_bit_green_c_tables_never_tie_so_simd_form_matches_libjpeg_turbo() {
        use crate::color::ycbcr::{FIX_0_34414, FIX_0_71414, ROUND};
        for cb in -128..128 {
            for cr in -128..128 {
                let c_tables = (-FIX_0_34414 * cb - FIX_0_71414 * cr + ROUND) >> 16;
                let simd_form = -((FIX_0_34414 * cb + FIX_0_71414 * cr + ROUND) >> 16);
                assert_eq!(c_tables, simd_form, "cb {cb} cr {cr}");
            }
        }
    }

    #[test]
    fn cmyk12_and_ycck12_neutral_samples_use_native_12_bit_range() {
        assert_eq!(
            inverted_cmyk12_to_rgb16(2048, 2048, 2048, 2048),
            (1024, 1024, 1024)
        );
        assert_eq!(ycck12_to_rgb16(2048, 2048, 2048, 2048), (1024, 1024, 1024));
    }
}
