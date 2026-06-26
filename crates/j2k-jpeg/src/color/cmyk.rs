// SPDX-License-Identifier: MIT OR Apache-2.0

//! Adobe-style CMYK/YCCK to RGB conversion.

use crate::color::ycbcr::ycbcr12_to_rgb16;
use crate::color::ycbcr::ycbcr_to_rgb;

fn multiply_u8(a: u8, b: u8) -> u8 {
    ((u16::from(a) * u16::from(b) + 127) / 255) as u8
}

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

/// Convert 12-bit complemented Adobe CMYK samples to native-range RGB16.
pub(crate) fn inverted_cmyk12_to_rgb16(c: u16, m: u16, y: u16, k: u16) -> (u16, u16, u16) {
    (multiply_u12(c, k), multiply_u12(m, k), multiply_u12(y, k))
}

/// Convert Adobe YCCK samples directly to RGB.
pub(crate) fn ycck_to_rgb(y: u8, cb: u8, cr: u8, k: u8) -> (u8, u8, u8) {
    let (c, m, y) = ycbcr_to_rgb(y, cb, cr);
    inverted_cmyk_to_rgb(c, m, y, k)
}

/// Convert 12-bit Adobe YCCK samples directly to native-range RGB16.
pub(crate) fn ycck12_to_rgb16(y: u16, cb: u16, cr: u16, k: u16) -> (u16, u16, u16) {
    let (c, m, y) = ycbcr12_to_rgb16(y, cb, cr);
    inverted_cmyk12_to_rgb16(c, m, y, k)
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
    fn cmyk12_and_ycck12_neutral_samples_use_native_12_bit_range() {
        assert_eq!(
            inverted_cmyk12_to_rgb16(2048, 2048, 2048, 2048),
            (1024, 1024, 1024)
        );
        assert_eq!(ycck12_to_rgb16(2048, 2048, 2048, 2048), (1024, 1024, 1024));
    }
}
