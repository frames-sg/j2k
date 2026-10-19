//! Lossless (SOF3) JPEG prediction and difference coding (T.81 Annex H).
//!
//! Every encoder backend predicts, differences, and categorizes samples with
//! these helpers, so CPU, CUDA-Oxide, and Metal output stays identical. The
//! arithmetic matches libjpeg-turbo: predictions and differences are taken
//! modulo 2^16, and predictors 5 and 6 halve with an arithmetic shift.
//!
//! Every function is `#[inline]`, which exports the MIR the cuda-oxide device
//! compiler needs; rustc exports it unprompted only for leaf functions.

/// Largest predictor selection value `Ss`.
pub const MAX_PREDICTOR: u8 = 7;

/// Largest difference category `SSSS`; category 16 codes the difference
/// 32768 with no additional bits.
pub const MAX_DIFFERENCE_CATEGORY: u8 = 16;

/// Number of Huffman symbols a lossless table can code (categories 0..=16).
pub const DIFFERENCE_SYMBOLS: usize = MAX_DIFFERENCE_CATEGORY as usize + 1;

/// Longest bit string one difference can take: a 16-bit code plus 15
/// additional bits.
pub const MAX_DIFFERENCE_BITS: u32 = 31;

/// The prediction `Px` with selection value `predictor` (1..=7) from the
/// reconstructed samples to the left (`ra`), above (`rb`), and above-left
/// (`rc`), modulo 2^16. Values above 7 predict as 7.
#[inline]
#[must_use]
pub const fn predict(predictor: u8, ra: u16, rb: u16, rc: u16) -> u16 {
    let (a, b, c) = (ra as i32, rb as i32, rc as i32);
    let prediction = match predictor {
        1 => a,
        2 => b,
        3 => c,
        4 => a + b - c,
        5 => a + ((b - c) >> 1),
        6 => b + ((a - c) >> 1),
        _ => (a + b) >> 1,
    };
    // Truncation is the modulo 2^16 reduction T.81 H.1.2.1 specifies.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "predictions are reduced modulo 2^16"
    )]
    let wrapped = prediction as u16;
    wrapped
}

/// The prediction for the sample at column `x` of row `row` of a component,
/// where `row` counts from the start of the scan or of the current restart
/// interval. The first row predicts from the left, seeded with
/// `initial_predictor` (`2^(P - Pt - 1)`); later rows predict their first
/// sample from above and the rest with `predictor`.
#[inline]
#[must_use]
pub const fn predict_at(
    predictor: u8,
    initial_predictor: u16,
    x: usize,
    row: usize,
    [ra, rb, rc]: [u16; 3],
) -> u16 {
    match (row, x) {
        (0, 0) => initial_predictor,
        (0, _) => ra,
        (_, 0) => rb,
        _ => predict(predictor, ra, rb, rc),
    }
}

/// The 1-D predictor seed `2^(P - Pt - 1)` for precision `precision` and point
/// transform `point_transform`, or `None` unless `point_transform <
/// precision <= 16`.
#[inline]
#[must_use]
pub const fn initial_predictor(precision: u8, point_transform: u8) -> Option<u16> {
    if precision > 16 || point_transform >= precision {
        return None;
    }
    Some(1 << (precision - point_transform - 1))
}

/// The category `SSSS` of `difference` (taken modulo 2^16) and its
/// additional bits, low `SSSS` bits significant (T.81 H.1.2.2). The
/// difference 32768 has category 16 and no additional bits.
#[inline]
#[must_use]
pub const fn difference_category(difference: u16) -> (u8, u16) {
    #[expect(
        clippy::cast_possible_wrap,
        reason = "differences are read as two's-complement values modulo 2^16"
    )]
    let signed = difference as i16;
    if signed == i16::MIN {
        return (MAX_DIFFERENCE_CATEGORY, 0);
    }
    let magnitude = signed.unsigned_abs();
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a u16 bit length is at most 16"
    )]
    let category = (u16::BITS - magnitude.leading_zeros()) as u8;
    let bits = if signed >= 0 {
        magnitude
    } else {
        // Negative differences send the low bits of `difference - 1`.
        difference.wrapping_sub(1) & ((1 << category) - 1)
    };
    (category, bits)
}

/// Additional bits that follow the Huffman code of category `category`.
#[inline]
#[must_use]
pub const fn additional_bits(category: u8) -> u8 {
    if category >= MAX_DIFFERENCE_CATEGORY {
        0
    } else {
        category
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The decoder's inverse: extend `bits` of `category` back to a difference.
    fn decode_difference(category: u8, bits: u16) -> u16 {
        match category {
            0 => 0,
            MAX_DIFFERENCE_CATEGORY => 32768,
            _ if bits < 1 << (category - 1) => bits.wrapping_sub((1 << category) - 1),
            _ => bits,
        }
    }

    #[test]
    fn every_difference_round_trips_through_its_category() {
        for difference in 0..=u16::MAX {
            let (category, bits) = difference_category(difference);
            assert!(category <= MAX_DIFFERENCE_CATEGORY);
            assert!(u32::from(bits) < 1 << additional_bits(category));
            assert_eq!(
                decode_difference(category, bits),
                difference,
                "{difference}"
            );
        }
    }

    #[test]
    fn categories_follow_the_magnitude() {
        assert_eq!(difference_category(0), (0, 0));
        assert_eq!(difference_category(1), (1, 1));
        assert_eq!(difference_category(u16::MAX), (1, 0));
        assert_eq!(difference_category(255), (8, 255));
        assert_eq!(difference_category(0u16.wrapping_sub(255)), (8, 0));
        assert_eq!(difference_category(32767), (15, 32767));
        assert_eq!(difference_category(32768), (16, 0));
        assert_eq!(difference_category(32769), (15, 0));
    }

    #[test]
    fn predictors_match_t81_table_h1_modulo_2_16() {
        let (ra, rb, rc) = (1000_u16, 3000_u16, 500_u16);
        assert_eq!(predict(1, ra, rb, rc), 1000);
        assert_eq!(predict(2, ra, rb, rc), 3000);
        assert_eq!(predict(3, ra, rb, rc), 500);
        assert_eq!(predict(4, ra, rb, rc), 3500);
        assert_eq!(predict(5, ra, rb, rc), 2250);
        assert_eq!(predict(6, ra, rb, rc), 3250);
        assert_eq!(predict(7, ra, rb, rc), 2000);
        // Negative halves floor, and every result wraps modulo 2^16.
        assert_eq!(predict(5, 0, 0, 3), 0u16.wrapping_sub(2));
        assert_eq!(predict(6, 0, 0, 3), 0u16.wrapping_sub(2));
        assert_eq!(predict(4, u16::MAX, u16::MAX, 0), u16::MAX - 1);
        assert_eq!(predict(7, u16::MAX, u16::MAX, 0), u16::MAX);
    }

    #[test]
    fn edges_use_the_one_dimensional_predictors() {
        let samples = [10, 20, 30];
        assert_eq!(predict_at(7, 512, 0, 0, samples), 512);
        assert_eq!(predict_at(7, 512, 3, 0, samples), 10);
        assert_eq!(predict_at(7, 512, 0, 4, samples), 20);
        assert_eq!(predict_at(7, 512, 3, 4, samples), 15);
    }

    #[test]
    fn the_seed_is_half_the_transformed_range() {
        assert_eq!(initial_predictor(16, 0), Some(32768));
        assert_eq!(initial_predictor(12, 2), Some(512));
        assert_eq!(initial_predictor(2, 1), Some(1));
        assert_eq!(initial_predictor(8, 8), None);
        assert_eq!(initial_predictor(17, 0), None);
    }
}
