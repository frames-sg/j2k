// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reduced-size ISLOW IDCTs derived from libjpeg-turbo's `jidctred.c`. The
//! maths is `j2k_codec_math::jpeg::idct`; these are the CPU entry points.

use j2k_codec_math::jpeg::idct;

use super::scalar::{idct_islow_12bit_dc_only_sample, idct_islow_dc_only_pixel as dc_only_pixel};

pub(crate) fn idct_islow_4x4(input: &[i16; 64], output: &mut [u8; 16]) {
    idct::islow_block_4x4(input, output);
}

pub(crate) fn idct_islow_4x4_dc_only(dc_coeff: i16, output: &mut [u8; 16]) {
    output.fill(dc_only_pixel(dc_coeff));
}

pub(crate) fn idct_islow_2x2(input: &[i16; 64], output: &mut [u8; 4]) {
    idct_islow_2x2_scalar(input, output);
}

pub(crate) fn idct_islow_2x2_dc_only(dc_coeff: i16, output: &mut [u8; 4]) {
    output.fill(dc_only_pixel(dc_coeff));
}

pub(crate) fn idct_islow_2x2_scalar(input: &[i16; 64], output: &mut [u8; 4]) {
    idct::islow_block_2x2(input, output);
}

pub(crate) fn idct_islow_1x1(input: &[i16; 64]) -> u8 {
    dc_only_pixel(input[0])
}

/// Reduced 4x4 IDCT of a 12-bit block (`jpeg_idct_4x4`).
///
/// 12-bit `jidctred.c`: `BITS_IN_JSAMPLE == 12` keeps `PASS1_BITS = 1` for
/// headroom and computes in 64-bit `JLONG`. The C code's zero-AC shortcuts give
/// the same results as the full computation and are omitted, as in
/// [`super::scalar::idct_islow_12bit`].
pub(crate) fn idct_islow_12bit_4x4(input: &[i16; 64], output: &mut [u16; 16]) {
    idct::islow_block_12bit_4x4(input, output);
}

/// Reduced 2x2 IDCT of a 12-bit block (`jpeg_idct_2x2`).
pub(crate) fn idct_islow_12bit_2x2(input: &[i16; 64], output: &mut [u16; 4]) {
    idct::islow_block_12bit_2x2(input, output);
}

/// Reduced 1x1 IDCT of a 12-bit block (`jpeg_idct_1x1`).
pub(crate) fn idct_islow_12bit_1x1(input: &[i16; 64]) -> u16 {
    idct_islow_12bit_dc_only_sample(input[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduced_idcts_preserve_zero_block_level_shift() {
        let input = [0i16; 64];
        let mut out4 = [0u8; 16];
        let mut out2 = [0u8; 4];
        idct_islow_4x4(&input, &mut out4);
        idct_islow_2x2(&input, &mut out2);
        assert!(out4.iter().all(|&px| px == 128));
        assert!(out2.iter().all(|&px| px == 128));
        assert_eq!(idct_islow_1x1(&input), 128);
    }

    #[test]
    fn reduced_dc_only_helpers_match_full_reduced_idcts() {
        for dc in [-300, -37, 0, 37, 300] {
            let mut input = [0i16; 64];
            input[0] = dc;
            let mut expected4 = [0u8; 16];
            let mut actual4 = [0u8; 16];
            let mut expected2 = [0u8; 4];
            let mut actual2 = [0u8; 4];

            idct_islow_4x4(&input, &mut expected4);
            idct_islow_4x4_dc_only(input[0], &mut actual4);
            idct_islow_2x2(&input, &mut expected2);
            idct_islow_2x2_dc_only(input[0], &mut actual2);

            assert_eq!(actual4, expected4);
            assert_eq!(actual2, expected2);
        }
    }

    #[test]
    fn twelve_bit_reduced_idcts_of_dc_blocks_are_uniform_rounded_samples() {
        // DESCALE(dc, 3) + 2048: the DC-only result every reduced size shares.
        for (dc, expected) in [
            (4i16, 2049u16),
            (-4, 2048),
            (12, 2050),
            (-5, 2047),
            (32767, 4095),
        ] {
            let mut input = [0i16; 64];
            input[0] = dc;
            let mut out4 = [0u16; 16];
            let mut out2 = [0u16; 4];
            idct_islow_12bit_4x4(&input, &mut out4);
            idct_islow_12bit_2x2(&input, &mut out2);
            assert_eq!(idct_islow_12bit_1x1(&input), expected, "1x1 dc={dc}");
            assert!(out4.iter().all(|&px| px == expected), "4x4 dc={dc}");
            assert!(out2.iter().all(|&px| px == expected), "2x2 dc={dc}");
        }
    }

    #[test]
    fn reduced_idcts_round_dc_half_up_like_libjpeg_descale() {
        // jpeg_idct_1x1 and the DC-only paths of jpeg_idct_{4x4,2x2} compute
        // DESCALE(dc, 3) = (dc + 4) >> 3. A plain shift gives 128, 127, 129 for
        // the first three inputs.
        for (dc, expected) in [(4i16, 129u8), (-4, 128), (12, 130), (-5, 127), (3, 128)] {
            let mut input = [0i16; 64];
            input[0] = dc;
            let mut out4 = [0u8; 16];
            let mut out2 = [0u8; 4];
            idct_islow_4x4(&input, &mut out4);
            idct_islow_2x2(&input, &mut out2);

            assert_eq!(idct_islow_1x1(&input), expected, "1x1 dc={dc}");
            assert!(out4.iter().all(|&px| px == expected), "4x4 dc={dc}");
            assert!(out2.iter().all(|&px| px == expected), "2x2 dc={dc}");
        }
    }
}
