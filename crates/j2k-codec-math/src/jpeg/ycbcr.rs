//!
//! The YCbCr to RGB constants are libjpeg-turbo's, which every decoder
//! backend applies. The full-range RGB to YCbCr transform below is the one
//! every baseline encoder backend applies: the CPU encoder calls these
//! functions, the CUDA-Oxide encode kernel calls them on the device, and the
//! Metal encode shaders read the constants from a fragment that
//! `j2k-jpeg-metal`'s build script renders.
//!
//! The forward weights are the T.871 coefficients rounded to `2^-16`. Each row
//! is exact on the neutral axis: the Y weights sum to `2^16` and each chroma
//! row sums to zero, so a grey `(v, v, v)` converts to `(v, center, center)`.

/// Fraction bits of every fixed-point constant in this module.
pub const FRACTION_BITS: i32 = 16;

/// Fixed-point integer approximation of 1.40200 * 2^16.
pub const FIX_1_40200: i32 = 91_881;
/// Fixed-point integer approximation of 0.34414 * 2^16.
pub const FIX_0_34414: i32 = 22_554;
/// Fixed-point integer approximation of 0.71414 * 2^16.
pub const FIX_0_71414: i32 = 46_802;
/// Fixed-point integer approximation of 1.77200 * 2^16.
pub const FIX_1_77200: i32 = 116_130;
/// Rounding addend for 16-bit fixed-point color conversion.
pub const ROUND: i32 = 1 << 15;

/// 0.29900 * 2^16, the red weight of Y.
pub const FIX_0_29900: i32 = 19_595;
/// 0.58700 * 2^16, the green weight of Y.
pub const FIX_0_58700: i32 = 38_470;
/// 0.11400 * 2^16, the blue weight of Y.
pub const FIX_0_11400: i32 = 7_471;
/// 0.16874 * 2^16, the negated red weight of Cb.
pub const FIX_0_16874: i32 = 11_059;
/// 0.33126 * 2^16, the negated green weight of Cb.
pub const FIX_0_33126: i32 = 21_709;
/// 0.50000 * 2^16, the blue weight of Cb and the red weight of Cr.
pub const FIX_0_50000: i32 = 32_768;
/// 0.41869 * 2^16, the negated green weight of Cr.
pub const FIX_0_41869: i32 = 27_439;
/// 0.08131 * 2^16, the negated blue weight of Cr.
pub const FIX_0_08131: i32 = 5_329;

/// Y of an RGB sample, rounded to nearest and not clamped.
///
/// Inputs are unsigned samples of at most 12 bits, so every product and sum
/// fits `i32`.
#[inline]
pub const fn rgb_to_y(red: i32, green: i32, blue: i32) -> i32 {
    (FIX_0_29900 * red + FIX_0_58700 * green + FIX_0_11400 * blue + ROUND) >> FRACTION_BITS
}

/// Cb of an RGB sample, offset by `center` (half the sample range: 128 for
/// 8-bit samples), rounded to nearest and not clamped.
///
/// Inputs are unsigned samples of at most 12 bits, so every product and sum
/// fits `i32`.
#[inline]
pub const fn rgb_to_cb(red: i32, green: i32, blue: i32, center: i32) -> i32 {
    (-FIX_0_16874 * red - FIX_0_33126 * green
        + FIX_0_50000 * blue
        + (center << FRACTION_BITS)
        + ROUND)
        >> FRACTION_BITS
}

/// Cr of an RGB sample, offset by `center` (half the sample range: 128 for
/// 8-bit samples), rounded to nearest and not clamped.
///
/// Inputs are unsigned samples of at most 12 bits, so every product and sum
/// fits `i32`.
#[inline]
pub const fn rgb_to_cr(red: i32, green: i32, blue: i32, center: i32) -> i32 {
    (FIX_0_50000 * red - FIX_0_41869 * green - FIX_0_08131 * blue
        + (center << FRACTION_BITS)
        + ROUND)
        >> FRACTION_BITS
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `coefficient / 100_000` in `2^-16` units, rounded to nearest.
    const fn fixed(coefficient: i64) -> i64 {
        (coefficient * 65_536 + 50_000) / 100_000
    }

    #[test]
    fn forward_weights_round_the_t871_coefficients() {
        let weights = [
            (FIX_0_29900, 29_900),
            (FIX_0_58700, 58_700),
            (FIX_0_11400, 11_400),
            (FIX_0_16874, 16_874),
            (FIX_0_33126, 33_126),
            (FIX_0_50000, 50_000),
            (FIX_0_41869, 41_869),
            (FIX_0_08131, 8_131),
        ];
        for (weight, coefficient) in weights {
            assert_eq!(i64::from(weight), fixed(coefficient), "{coefficient}");
        }
        assert_eq!(FIX_0_50000, ROUND);
    }

    #[test]
    fn forward_rows_are_exact_on_the_neutral_axis() {
        assert_eq!(FIX_0_29900 + FIX_0_58700 + FIX_0_11400, 1 << FRACTION_BITS);
        assert_eq!(FIX_0_16874 + FIX_0_33126, FIX_0_50000);
        assert_eq!(FIX_0_41869 + FIX_0_08131, FIX_0_50000);
        for (center, max) in [(128, 255), (2_048, 4_095)] {
            for value in 0..=max {
                assert_eq!(rgb_to_y(value, value, value), value);
                assert_eq!(rgb_to_cb(value, value, value, center), center);
                assert_eq!(rgb_to_cr(value, value, value, center), center);
            }
        }
    }

    #[test]
    fn primaries_convert_to_the_rounded_t871_values() {
        assert_eq!(
            (
                rgb_to_y(255, 0, 0),
                rgb_to_cb(255, 0, 0, 128),
                rgb_to_cr(255, 0, 0, 128)
            ),
            (76, 85, 256)
        );
        assert_eq!(
            (
                rgb_to_y(0, 255, 0),
                rgb_to_cb(0, 255, 0, 128),
                rgb_to_cr(0, 255, 0, 128)
            ),
            (150, 44, 21)
        );
        assert_eq!(
            (
                rgb_to_y(0, 0, 255),
                rgb_to_cb(0, 0, 255, 128),
                rgb_to_cr(0, 0, 255, 128)
            ),
            (29, 256, 107)
        );
    }
}
