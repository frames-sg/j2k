// SPDX-License-Identifier: Apache-2.0

//! Adobe-style CMYK/YCCK to RGB conversion.

use crate::color::ycbcr::ycbcr_to_rgb;

fn multiply_u8(a: u8, b: u8) -> u8 {
    ((u16::from(a) * u16::from(b) + 127) / 255) as u8
}

/// Convert complemented Adobe CMYK samples to RGB.
///
/// Adobe CMYK JPEG stores samples in the complemented domain, so each channel
/// acts like the amount of display light remaining after ink coverage.
pub(crate) fn inverted_cmyk_to_rgb(c: u8, m: u8, y: u8, k: u8) -> (u8, u8, u8) {
    (multiply_u8(c, k), multiply_u8(m, k), multiply_u8(y, k))
}

/// Convert Adobe YCCK samples directly to RGB.
pub(crate) fn ycck_to_rgb(y: u8, cb: u8, cr: u8, k: u8) -> (u8, u8, u8) {
    let (c, m, y) = ycbcr_to_rgb(y, cb, cr);
    inverted_cmyk_to_rgb(c, m, y, k)
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
}
