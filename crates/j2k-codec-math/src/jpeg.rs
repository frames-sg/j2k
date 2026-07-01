//! Baseline JPEG constants shared by CPU and device backends.

/// T.81 zigzag order from stream coefficient index to natural 8x8 index.
#[rustfmt::skip]
pub const ZIGZAG: [u8; 64] = [
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
];

/// Fixed-point constants used by the baseline JPEG integer IDCT.
pub mod idct {
    /// Number of fractional bits in the integer IDCT constants.
    pub const CONST_BITS: usize = 13;
    /// First-pass post-IDCT scaling bits.
    pub const PASS1_BITS: usize = 2;

    /// Fixed-point integer approximation of 0.298631336.
    pub const FIX_0_298631336: i32 = 2_446;
    /// Fixed-point integer approximation of 0.390180644.
    pub const FIX_0_390180644: i32 = 3_196;
    /// Fixed-point integer approximation of 0.541196100.
    pub const FIX_0_541196100: i32 = 4_433;
    /// Fixed-point integer approximation of 0.765366865.
    pub const FIX_0_765366865: i32 = 6_270;
    /// Fixed-point integer approximation of 0.899976223.
    pub const FIX_0_899976223: i32 = 7_373;
    /// Fixed-point integer approximation of 1.175875602.
    pub const FIX_1_175875602: i32 = 9_633;
    /// Fixed-point integer approximation of 1.501321110.
    pub const FIX_1_501321110: i32 = 12_299;
    /// Fixed-point integer approximation of 1.847759065.
    pub const FIX_1_847759065: i32 = 15_137;
    /// Fixed-point integer approximation of 1.961570560.
    pub const FIX_1_961570560: i32 = 16_069;
    /// Fixed-point integer approximation of 2.053119869.
    pub const FIX_2_053119869: i32 = 16_819;
    /// Fixed-point integer approximation of 2.562915447.
    pub const FIX_2_562915447: i32 = 20_995;
    /// Fixed-point integer approximation of 3.072711026.
    pub const FIX_3_072711026: i32 = 25_172;
}

/// Fixed-point constants used by JPEG YCbCr to RGB conversion.
pub mod ycbcr {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zigzag_is_a_permutation_of_one_block() {
        let mut seen = [false; 64];
        for &idx in &ZIGZAG {
            assert!(idx < 64);
            assert!(!seen[idx as usize], "duplicate zigzag index {idx}");
            seen[idx as usize] = true;
        }
        assert!(seen.into_iter().all(|entry| entry));
    }

    #[test]
    fn idct_constants_match_existing_integer_backend() {
        assert_eq!(idct::CONST_BITS, 13);
        assert_eq!(idct::PASS1_BITS, 2);
        assert_eq!(idct::FIX_0_298631336, 2_446);
        assert_eq!(idct::FIX_0_390180644, 3_196);
        assert_eq!(idct::FIX_0_541196100, 4_433);
        assert_eq!(idct::FIX_0_765366865, 6_270);
        assert_eq!(idct::FIX_0_899976223, 7_373);
        assert_eq!(idct::FIX_1_175875602, 9_633);
        assert_eq!(idct::FIX_1_501321110, 12_299);
        assert_eq!(idct::FIX_1_847759065, 15_137);
        assert_eq!(idct::FIX_1_961570560, 16_069);
        assert_eq!(idct::FIX_2_053119869, 16_819);
        assert_eq!(idct::FIX_2_562915447, 20_995);
        assert_eq!(idct::FIX_3_072711026, 25_172);
    }
}
