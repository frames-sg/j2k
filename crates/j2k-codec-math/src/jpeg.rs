//! Baseline JPEG constants shared by CPU and device backends.

use core::fmt;

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

/// Canonical JPEG Huffman table derivation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanonicalHuffmanDerivation {
    /// Smallest code for each code length; `i32::MAX` when absent.
    pub min_code: [i32; 17],
    /// Largest code for each code length; `-1` when absent.
    pub max_code: [i32; 17],
    /// Value-index offset for each code length.
    pub val_offset: [i32; 17],
    /// Canonical Huffman code for each value index.
    pub huffcode: [u16; 256],
    /// Canonical Huffman code length for each value index.
    pub huffsize: [u8; 256],
    /// Number of valid entries in `huffcode` and `huffsize`.
    pub huffsize_len: usize,
}

/// Error returned by [`derive_canonical_huffman`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanonicalHuffmanError {
    /// BITS entries describe more than 256 symbols.
    BitsExceedTableCapacity,
    /// BITS counts do not match the supplied HUFFVAL length.
    BitsValuesLenMismatch,
    /// Canonical code assignment overflowed the JPEG 16-bit code space.
    CodeOverflow,
}

impl fmt::Display for CanonicalHuffmanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BitsExceedTableCapacity => f.write_str("BITS exceed table capacity"),
            Self::BitsValuesLenMismatch => f.write_str("BITS do not match HUFFVAL length"),
            Self::CodeOverflow => f.write_str("canonical code overflow"),
        }
    }
}

/// Derive canonical JPEG Huffman codes and slow-path decode bounds from the
/// T.81 DHT `BITS` vector and HUFFVAL length.
///
/// The caller owns the HUFFVAL bytes; this helper derives only the canonical
/// code metadata shared by CPU and device backends.
pub fn derive_canonical_huffman(
    bits: &[u8; 16],
    values_len: usize,
) -> Result<CanonicalHuffmanDerivation, CanonicalHuffmanError> {
    if values_len > 256 {
        return Err(CanonicalHuffmanError::BitsExceedTableCapacity);
    }

    let mut huffsize = [0u8; 256];
    let mut huffsize_len = 0usize;
    for (len_minus_1, &count) in bits.iter().enumerate() {
        let len = u8::try_from(len_minus_1 + 1).map_err(|_| CanonicalHuffmanError::CodeOverflow)?;
        for _ in 0..count {
            if huffsize_len >= values_len || huffsize_len >= huffsize.len() {
                return Err(CanonicalHuffmanError::BitsExceedTableCapacity);
            }
            huffsize[huffsize_len] = len;
            huffsize_len += 1;
        }
    }
    if huffsize_len != values_len {
        return Err(CanonicalHuffmanError::BitsValuesLenMismatch);
    }

    let mut huffcode = [0u16; 256];
    let mut code = 0u32;
    let mut si = huffsize.first().copied().unwrap_or(0);
    for (idx, &size) in huffsize[..huffsize_len].iter().enumerate() {
        while size != si {
            code <<= 1;
            si = si.saturating_add(1);
        }
        if si > 16 || code >= (1u32 << si) {
            return Err(CanonicalHuffmanError::CodeOverflow);
        }
        huffcode[idx] = u16::try_from(code).map_err(|_| CanonicalHuffmanError::CodeOverflow)?;
        code = code
            .checked_add(1)
            .ok_or(CanonicalHuffmanError::CodeOverflow)?;
    }

    let mut min_code = [i32::MAX; 17];
    let mut max_code = [-1i32; 17];
    let mut val_offset = [0i32; 17];
    let mut cursor = 0usize;
    for (len_minus_1, &count) in bits.iter().enumerate() {
        let len = len_minus_1 + 1;
        let count = usize::from(count);
        if count == 0 {
            continue;
        }
        min_code[len] = i32::from(huffcode[cursor]);
        max_code[len] = i32::from(huffcode[cursor + count - 1]);
        val_offset[len] =
            i32::try_from(cursor).map_err(|_| CanonicalHuffmanError::CodeOverflow)? - min_code[len];
        cursor += count;
    }

    Ok(CanonicalHuffmanDerivation {
        min_code,
        max_code,
        val_offset,
        huffcode,
        huffsize,
        huffsize_len,
    })
}

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
    fn canonical_huffman_derivation_matches_t81_luma_dc_table() {
        let bits = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
        let Ok(canonical) = derive_canonical_huffman(&bits, 12) else {
            panic!("canonical table should derive");
        };

        assert_eq!(canonical.huffsize_len, 12);
        assert_eq!(
            &canonical.huffsize[..12],
            &[2, 3, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9]
        );
        assert_eq!(
            &canonical.huffcode[..12],
            &[0, 2, 3, 4, 5, 6, 14, 30, 62, 126, 254, 510]
        );
        assert_eq!(canonical.min_code[2], 0);
        assert_eq!(canonical.max_code[3], 6);
        assert_eq!(canonical.val_offset[3], -1);
    }

    #[test]
    fn canonical_huffman_derivation_rejects_mismatched_value_count() {
        let bits = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        assert_eq!(
            derive_canonical_huffman(&bits, 0),
            Err(CanonicalHuffmanError::BitsExceedTableCapacity)
        );
        assert_eq!(
            derive_canonical_huffman(&bits, 2),
            Err(CanonicalHuffmanError::BitsValuesLenMismatch)
        );
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
