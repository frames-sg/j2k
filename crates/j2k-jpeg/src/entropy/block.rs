// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-block entropy decode: one 8×8 DCT coefficient block.
//!
//! Steps per T.81 §F.2.1:
//! 1. Decode DC category (`T`) via the DC Huffman table; read `T` bits to get
//!    the DC difference; add to `prev_dc` to recover absolute DC.
//! 2. Loop up to 63 AC coefficients: decode a byte `rs` via the AC Huffman
//!    table; `rrrr = rs >> 4` is a run of zeros; `ssss = rs & 0x0F` is the
//!    next value's category. `rs == 0x00` means EOB (all remaining AC = 0);
//!    `rs == 0xF0` means ZRL (16 zeros, continue).
//! 3. Dequantize by multiplying each surviving coefficient with its quant
//!    table entry; write to the output block in zigzag-inverted position.
//!
//! Produces a 64-entry array in row-major (natural) order, suitable for
//! direct consumption by the IDCT.

#[cfg(test)]
use crate::entropy::huffman::HuffmanTable;
use crate::entropy::huffman::{
    ac_decoded_run, ac_decoded_value, AcHuffmanTable, DcHuffmanTable, AC_FAST_EOB,
    AC_FAST_KIND_MASK, AC_FAST_VALUE, AC_FAST_ZRL,
};
use crate::entropy::ZIGZAG;
use crate::error::{HuffmanFailure, JpegError};
use crate::internal::bit_reader::BitReader;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlockActivity {
    DcOnly,
    BottomHalfZero,
    General,
}

#[derive(Debug, Clone)]
pub(crate) struct CoefficientBlock {
    coeffs: [i16; 64],
    /// Set by any AC store. Clearing a block that holds AC coefficients
    /// zeroes all 128 bytes (a handful of vector stores); a DC-only block
    /// clears just the DC. This keeps per-coefficient bookkeeping to a single
    /// flag store instead of an index list maintained in memory.
    ac_written: bool,
}

impl Default for CoefficientBlock {
    fn default() -> Self {
        Self {
            coeffs: [0; 64],
            ac_written: false,
        }
    }
}

impl CoefficientBlock {
    #[expect(
        clippy::inline_always,
        reason = "measured entropy-block hot path requires cross-helper inlining"
    )]
    #[inline(always)]
    fn clear_touched(&mut self) {
        if self.ac_written {
            self.coeffs = [0; 64];
            self.ac_written = false;
        } else {
            self.coeffs[0] = 0;
        }
    }

    #[expect(
        clippy::inline_always,
        reason = "measured entropy-block hot path requires cross-helper inlining"
    )]
    #[inline(always)]
    fn store_dc(&mut self, value: i16) {
        self.coeffs[0] = value;
    }

    #[expect(
        clippy::inline_always,
        reason = "measured entropy-block hot path requires cross-helper inlining"
    )]
    #[inline(always)]
    fn store(&mut self, idx: usize, value: i16) {
        self.coeffs[idx] = value;
        self.ac_written = true;
    }

    #[expect(
        clippy::inline_always,
        reason = "measured entropy-block hot path requires cross-helper inlining"
    )]
    #[inline(always)]
    pub(crate) fn coefficients(&self) -> &[i16; 64] {
        &self.coeffs
    }

    #[expect(
        clippy::inline_always,
        reason = "measured entropy-block hot path requires cross-helper inlining"
    )]
    #[inline(always)]
    pub(crate) fn dc_coeff(&self) -> i16 {
        self.coeffs[0]
    }
}

/// Classify a block from the OR of every stored AC coefficient's natural
/// index. AC indices are nonzero, so any AC store makes the OR nonzero, and
/// bit 5 is set exactly when some index is at least 32 (rows 4-7). Keeping
/// one OR per coefficient replaces a per-coefficient state-machine update.
#[expect(
    clippy::inline_always,
    reason = "measured entropy-block hot path requires cross-helper inlining"
)]
#[inline(always)]
fn activity_from_ac_indices(ac_index_or: usize) -> BlockActivity {
    if ac_index_or == 0 {
        BlockActivity::DcOnly
    } else if ac_index_or & 32 == 0 {
        BlockActivity::BottomHalfZero
    } else {
        BlockActivity::General
    }
}

/// Decode one 8×8 DCT block from the scan.
///
/// - `prev_dc` is read and updated in place so the caller threads DC prediction
///   across blocks of the same component.
/// - `quant` is the 64-entry quant table (natural / zigzag-natural order matches
///   how DQT stored it: linear). Multiplication is a straight elementwise scale.
/// - `out` is cleared and filled with the dequantized coefficients in row-major
///   order (natural 8×8 layout), ready for the IDCT.
#[cfg(test)]
pub(crate) fn decode_block(
    br: &mut BitReader<'_>,
    dc_table: DcHuffmanTable<'_>,
    ac_table: AcHuffmanTable<'_>,
    prev_dc: &mut i32,
    quant: &[u16; 64],
    block: &mut CoefficientBlock,
) -> Result<(), JpegError> {
    decode_block_with_activity(br, dc_table, ac_table, prev_dc, quant, block).map(|_| ())
}

#[expect(
    clippy::inline_always,
    reason = "measured entropy-block hot path requires cross-helper inlining"
)]
#[inline(always)]
pub(crate) fn decode_block_with_activity(
    br: &mut BitReader<'_>,
    dc_table: DcHuffmanTable<'_>,
    ac_table: AcHuffmanTable<'_>,
    prev_dc: &mut i32,
    quant: &[u16; 64],
    block: &mut CoefficientBlock,
) -> Result<BlockActivity, JpegError> {
    block.clear_touched();

    // DC.
    let diff = dc_table.decode_fast_dc(br)?;
    *prev_dc = prev_dc.wrapping_add(diff);
    // Dequant the DC in natural-order position 0 (zigzag index 0 → natural 0).
    let dc_dequant = (*prev_dc).wrapping_mul(i32::from(quant[0]));
    block.store_dc(clamp_i16(dc_dequant));

    let mut ac_index_or = 0usize;
    drive_ac_fast::<false, _>(br, ac_table, |k, ac| {
        let natural_idx = ZIGZAG[k] as usize;
        // Quant table entries are stored in zigzag order per T.81 §B.2.4.1,
        // so `quant[k]` is the matching coefficient (not `quant[natural_idx]`).
        let value = ac_decoded_value(ac);
        let dequant = value.wrapping_mul(i32::from(quant[k]));
        block.store(natural_idx, clamp_i16(dequant));
        ac_index_or |= natural_idx;
        Ok(())
    })?;
    Ok(activity_from_ac_indices(ac_index_or))
}

/// Decode one dequantized block directly into a freshly zeroed output block.
///
/// The caller must pass an output block that is all zeroes; this function only
/// writes non-zero decoded coefficients.
#[expect(
    clippy::inline_always,
    reason = "measured entropy-block hot path requires cross-helper inlining"
)]
#[inline(always)]
pub(crate) fn decode_block_dequantized_into(
    br: &mut BitReader<'_>,
    dc_table: DcHuffmanTable<'_>,
    ac_table: AcHuffmanTable<'_>,
    prev_dc: &mut i32,
    quant: &[u16; 64],
    out: &mut [i16; 64],
) -> Result<(), JpegError> {
    // DC.
    let diff = dc_table.decode_fast_dc(br)?;
    *prev_dc = prev_dc.wrapping_add(diff);
    let dc_dequant = (*prev_dc).wrapping_mul(i32::from(quant[0]));
    out[0] = clamp_i16(dc_dequant);

    drive_ac_fast::<false, _>(br, ac_table, |k, ac| {
        let natural_idx = ZIGZAG[k] as usize;
        let value = ac_decoded_value(ac);
        let dequant = value.wrapping_mul(i32::from(quant[k]));
        out[natural_idx] = clamp_i16(dequant);
        Ok(())
    })?;
    Ok(())
}

#[expect(
    clippy::inline_always,
    reason = "measured entropy-block hot path requires cross-helper inlining"
)]
#[inline(always)]
pub(crate) fn decode_block_quantized_and_dequantized_with_activity(
    br: &mut BitReader<'_>,
    dc_table: DcHuffmanTable<'_>,
    ac_table: AcHuffmanTable<'_>,
    prev_dc: &mut i32,
    quant: &[u16; 64],
    quantized_block: &mut CoefficientBlock,
    dequantized_block: &mut CoefficientBlock,
) -> Result<BlockActivity, JpegError> {
    quantized_block.clear_touched();
    dequantized_block.clear_touched();

    // DC.
    let diff = dc_table.decode_fast_dc(br)?;
    *prev_dc = prev_dc.wrapping_add(diff);
    quantized_block.store_dc(clamp_i16(*prev_dc));
    let dc_dequant = (*prev_dc).wrapping_mul(i32::from(quant[0]));
    dequantized_block.store_dc(clamp_i16(dc_dequant));

    let mut ac_index_or = 0usize;
    drive_ac_fast::<false, _>(br, ac_table, |k, ac| {
        let natural_idx = ZIGZAG[k] as usize;
        let value = ac_decoded_value(ac);
        quantized_block.store(natural_idx, clamp_i16(value));
        let dequant = value.wrapping_mul(i32::from(quant[k]));
        dequantized_block.store(natural_idx, clamp_i16(dequant));
        ac_index_or |= natural_idx;
        Ok(())
    })?;
    Ok(activity_from_ac_indices(ac_index_or))
}

#[expect(
    clippy::inline_always,
    reason = "measured entropy-block hot path requires cross-helper inlining"
)]
#[inline(always)]
#[cfg(test)]
pub(crate) fn decode_block_with_dc_status(
    br: &mut BitReader<'_>,
    dc_table: DcHuffmanTable<'_>,
    ac_table: AcHuffmanTable<'_>,
    prev_dc: &mut i32,
    quant: &[u16; 64],
    block: &mut CoefficientBlock,
) -> Result<bool, JpegError> {
    block.clear_touched();

    let diff = dc_table.decode_fast_dc(br)?;
    *prev_dc = prev_dc.wrapping_add(diff);
    let dc_dequant = (*prev_dc).wrapping_mul(i32::from(quant[0]));
    block.store_dc(clamp_i16(dc_dequant));

    let mut dc_only = true;
    drive_ac_fast::<false, _>(br, ac_table, |k, ac| {
        let natural_idx = ZIGZAG[k] as usize;
        let value = ac_decoded_value(ac);
        let dequant = value.wrapping_mul(i32::from(quant[k]));
        block.store(natural_idx, clamp_i16(dequant));
        dc_only = false;
        Ok(())
    })?;
    Ok(dc_only)
}

#[expect(
    clippy::inline_always,
    reason = "measured entropy-block hot path requires cross-helper inlining"
)]
#[inline(always)]
pub(crate) fn skip_block(
    br: &mut BitReader<'_>,
    dc_table: DcHuffmanTable<'_>,
    ac_table: AcHuffmanTable<'_>,
    prev_dc: &mut i32,
) -> Result<(), JpegError> {
    let diff = dc_table.decode_fast_dc(br)?;
    *prev_dc = prev_dc.wrapping_add(diff);

    drive_ac_fast::<true, _>(br, ac_table, |_, _| Ok(()))?;
    Ok(())
}

#[expect(
    clippy::inline_always,
    reason = "measured entropy-block hot path requires cross-helper inlining"
)]
#[inline(always)]
fn drive_ac_fast<const SKIP_VALUES: bool, F>(
    br: &mut BitReader<'_>,
    ac_table: AcHuffmanTable<'_>,
    mut on_value: F,
) -> Result<(), JpegError>
where
    F: FnMut(usize, u32) -> Result<(), JpegError>,
{
    let mut k: usize = 1;
    while k < 64 {
        let ac = if SKIP_VALUES {
            ac_table.skip_fast_ac(br)?
        } else {
            ac_table.decode_fast_ac(br)?
        };
        match ac & AC_FAST_KIND_MASK {
            AC_FAST_EOB => break,
            AC_FAST_ZRL => k += 16,
            AC_FAST_VALUE => {
                k += ac_decoded_run(ac);
                if k >= 64 {
                    return Err(invalid_huffman_symbol());
                }
                on_value(k, ac)?;
                k += 1;
            }
            _ => unreachable!("invalid AC fast-table tag"),
        }
    }
    Ok(())
}

#[expect(
    clippy::inline_always,
    reason = "measured entropy-block hot path requires cross-helper inlining"
)]
#[inline(always)]
fn invalid_huffman_symbol() -> JpegError {
    JpegError::HuffmanDecode {
        mcu: 0,
        reason: HuffmanFailure::InvalidSymbol,
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "dequantized coefficients are explicitly clamped to i16 before storage"
)]
pub(crate) fn clamp_i16(v: i32) -> i16 {
    if v > i32::from(i16::MAX) {
        i16::MAX
    } else if v < i32::from(i16::MIN) {
        i16::MIN
    } else {
        v as i16
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::tables::{HuffmanTableRole, HuffmanValues, RawHuffmanTable};

    /// DC table that decodes bit `0` → symbol `0` (DC category 0 = no diff).
    /// Single code of length 1 → symbol 0.
    fn trivial_dc_table() -> HuffmanTable {
        let raw = RawHuffmanTable {
            bits: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: HuffmanValues::from_slice(&[0]),
        };
        HuffmanTable::from_raw(&raw, HuffmanTableRole::Dc).unwrap()
    }

    /// AC table that decodes bit `0` → symbol `0x00` (EOB).
    fn eob_ac_table() -> HuffmanTable {
        let raw = RawHuffmanTable {
            bits: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: HuffmanValues::from_slice(&[0x00]),
        };
        HuffmanTable::from_raw(&raw, HuffmanTableRole::Ac).unwrap()
    }

    #[test]
    fn decodes_all_zero_block() {
        // DC code `0` (→ category 0, no diff bits) then AC code `0` (EOB).
        // Pad with zeros so the Huffman decoder's 8-bit peek never runs dry.
        let bytes = [0u8; 4];
        let mut br = BitReader::new(&bytes);
        let dc = trivial_dc_table();
        let ac = eob_ac_table();
        let quant = [1u16; 64];
        let mut prev_dc = 0i32;
        let mut out = CoefficientBlock::default();
        let activity = decode_block_with_activity(
            &mut br,
            dc.dc().unwrap(),
            ac.ac().unwrap(),
            &mut prev_dc,
            &quant,
            &mut out,
        )
        .unwrap();
        assert_eq!(prev_dc, 0);
        assert_eq!(activity, BlockActivity::DcOnly);
        assert!(out.coefficients().iter().all(|&c| c == 0));
    }

    #[test]
    fn dequantizes_dc_coefficient() {
        let raw = RawHuffmanTable {
            bits: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: HuffmanValues::from_slice(&[2]),
        };
        let dc = HuffmanTable::from_raw(&raw, HuffmanTableRole::Dc).unwrap();
        let ac = eob_ac_table();
        // Bits: 00 (DC code → ssss=2) 11 (extend → diff=3) 0 (EOB).
        // Trailing zero bytes satisfy the decoder's 8-bit peek requirement.
        let bytes = [0b0011_0000u8, 0, 0, 0];
        let mut br = BitReader::new(&bytes);
        let quant = [7u16; 64];
        let mut prev_dc = 0i32;
        let mut out = CoefficientBlock::default();
        let activity = decode_block_with_activity(
            &mut br,
            dc.dc().unwrap(),
            ac.ac().unwrap(),
            &mut prev_dc,
            &quant,
            &mut out,
        )
        .unwrap();
        assert_eq!(prev_dc, 3);
        assert_eq!(activity, BlockActivity::DcOnly);
        assert_eq!(out.coefficients()[0], 21, "DC = 3 * quant 7 = 21");
        assert!(out.coefficients()[1..].iter().all(|&c| c == 0));
    }

    #[test]
    fn dc_prediction_accumulates_across_blocks() {
        let raw = RawHuffmanTable {
            bits: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: HuffmanValues::from_slice(&[2]),
        };
        let dc = HuffmanTable::from_raw(&raw, HuffmanTableRole::Dc).unwrap();
        let ac = eob_ac_table();
        // Block 1: 00 11 0 (diff=+3). Block 2: 00 11 0 (diff=+3). Pad for peek.
        let bytes = [0b0011_0001u8, 0b1000_0000u8, 0, 0];
        let mut br = BitReader::new(&bytes);
        let quant = [1u16; 64];
        let mut prev_dc = 10i32;
        let mut out = CoefficientBlock::default();
        decode_block(
            &mut br,
            dc.dc().unwrap(),
            ac.ac().unwrap(),
            &mut prev_dc,
            &quant,
            &mut out,
        )
        .unwrap();
        assert_eq!(prev_dc, 13);
        decode_block(
            &mut br,
            dc.dc().unwrap(),
            ac.ac().unwrap(),
            &mut prev_dc,
            &quant,
            &mut out,
        )
        .unwrap();
        assert_eq!(prev_dc, 16);
    }

    #[test]
    fn reports_general_activity_when_ac_coefficient_is_present() {
        let dc = trivial_dc_table();
        let raw = RawHuffmanTable {
            bits: [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: HuffmanValues::from_slice(&[0x01, 0x00]),
        };
        let ac = HuffmanTable::from_raw(&raw, HuffmanTableRole::Ac).unwrap();
        // AC symbols: `00` => 0x01 (run 0, size 1), then `010` => EOB.
        // Payload bit `1` gives AC value +1 at zigzag slot 1.
        let bytes = [0b0001_0100u8, 0, 0, 0];
        let mut br = BitReader::new(&bytes);
        let quant = [1u16; 64];
        let mut prev_dc = 0i32;
        let mut out = CoefficientBlock::default();
        let activity = decode_block_with_activity(
            &mut br,
            dc.dc().unwrap(),
            ac.ac().unwrap(),
            &mut prev_dc,
            &quant,
            &mut out,
        )
        .unwrap();
        assert_eq!(activity, BlockActivity::BottomHalfZero);
        assert_eq!(out.coefficients()[crate::entropy::ZIGZAG[1] as usize], 1);
    }

    #[test]
    fn dc_status_decoder_matches_block_coefficients_without_activity_classification() {
        let dc = trivial_dc_table();
        let raw = RawHuffmanTable {
            bits: [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: HuffmanValues::from_slice(&[0x01, 0x00]),
        };
        let ac = HuffmanTable::from_raw(&raw, HuffmanTableRole::Ac).unwrap();
        let bytes = [0b0001_0100u8, 0, 0, 0];
        let quant = [1u16; 64];
        let mut activity_reader = BitReader::new(&bytes);
        let mut dc_status_reader = BitReader::new(&bytes);
        let mut activity_prev_dc = 0i32;
        let mut dc_status_prev_dc = 0i32;
        let mut activity_block = CoefficientBlock::default();
        let mut dc_status_block = CoefficientBlock::default();

        let activity = decode_block_with_activity(
            &mut activity_reader,
            dc.dc().unwrap(),
            ac.ac().unwrap(),
            &mut activity_prev_dc,
            &quant,
            &mut activity_block,
        )
        .unwrap();
        let dc_only = decode_block_with_dc_status(
            &mut dc_status_reader,
            dc.dc().unwrap(),
            ac.ac().unwrap(),
            &mut dc_status_prev_dc,
            &quant,
            &mut dc_status_block,
        )
        .unwrap();

        assert_eq!(activity, BlockActivity::BottomHalfZero);
        assert!(!dc_only);
        assert_eq!(dc_status_prev_dc, activity_prev_dc);
        assert_eq!(
            dc_status_block.coefficients(),
            activity_block.coefficients()
        );
        assert_eq!(dc_status_reader.snapshot(), activity_reader.snapshot());
    }

    #[test]
    fn skip_block_consumes_stream_and_updates_dc_like_decode() {
        let dc_raw = RawHuffmanTable {
            bits: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: HuffmanValues::from_slice(&[2]),
        };
        let dc = HuffmanTable::from_raw(&dc_raw, HuffmanTableRole::Dc).unwrap();
        let ac_raw = RawHuffmanTable {
            bits: [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: HuffmanValues::from_slice(&[0x01, 0x00]),
        };
        let ac = HuffmanTable::from_raw(&ac_raw, HuffmanTableRole::Ac).unwrap();
        let bytes = [0b0011_0010u8, 0b1000_0000, 0, 0];
        let quant = [1u16; 64];

        let mut decoded_reader = BitReader::new(&bytes);
        let mut skipped_reader = BitReader::new(&bytes);
        let mut decoded_prev_dc = 5i32;
        let mut skipped_prev_dc = 5i32;
        let mut out = CoefficientBlock::default();

        decode_block_with_activity(
            &mut decoded_reader,
            dc.dc().unwrap(),
            ac.ac().unwrap(),
            &mut decoded_prev_dc,
            &quant,
            &mut out,
        )
        .unwrap();
        skip_block(
            &mut skipped_reader,
            dc.dc().unwrap(),
            ac.ac().unwrap(),
            &mut skipped_prev_dc,
        )
        .unwrap();

        assert_eq!(skipped_prev_dc, decoded_prev_dc);
        assert_eq!(skipped_reader.snapshot(), decoded_reader.snapshot());
    }

    #[test]
    fn top_half_ac_indices_classify_as_bottom_half_zero() {
        for indices in [&[31usize][..], &[7], &[1, 8, 31], &[3, 17, 26]] {
            let or = indices.iter().fold(0, |acc, &idx| acc | idx);
            assert_eq!(
                activity_from_ac_indices(or),
                BlockActivity::BottomHalfZero,
                "{indices:?}"
            );
        }
        assert_eq!(activity_from_ac_indices(0), BlockActivity::DcOnly);
    }

    #[test]
    fn any_bottom_half_ac_index_classifies_as_general() {
        for indices in [&[32usize][..], &[40], &[1, 63], &[7, 31, 32]] {
            let or = indices.iter().fold(0, |acc, &idx| acc | idx);
            assert_eq!(
                activity_from_ac_indices(or),
                BlockActivity::General,
                "{indices:?}"
            );
        }
    }

    #[test]
    fn clear_zeroes_every_coefficient_after_ac_stores() {
        let mut block = CoefficientBlock::default();
        block.store_dc(9);
        for (i, idx) in [1usize, 8, 16, 24, 63].into_iter().enumerate() {
            block.store(
                idx,
                i16::try_from(i + 1).expect("fixture coefficient fits in i16"),
            );
        }

        block.clear_touched();
        assert!(block.coefficients().iter().all(|&c| c == 0));

        // A DC-only block that follows must also clear back to all zeroes.
        block.store_dc(-4);
        block.clear_touched();
        assert!(block.coefficients().iter().all(|&c| c == 0));
    }
}
