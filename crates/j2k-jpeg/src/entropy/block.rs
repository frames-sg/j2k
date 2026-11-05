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
    ac_decoded_run, ac_decoded_value, ac_pair_distance, ac_pair_ends_block, ac_pair_guard,
    ac_pair_len, ac_pair_value, AcHuffmanTable, DcHuffmanTable, AC_FAST_EOB, AC_FAST_KIND_MASK,
    AC_FAST_VALUE, AC_FAST_ZRL,
};
use crate::entropy::ZIGZAG;
use crate::error::{HuffmanFailure, JpegError};
use crate::internal::bit_reader::BitReader;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlockActivity {
    DcOnly,
    TopLeft2x2,
    TopLeftQuadrant,
    BottomHalfZero,
    RightHalfZero,
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
/// bit 5 marks rows 4-7 and bit 2 marks columns 4-7. One OR per coefficient
/// therefore identifies both half-block shortcuts without rescanning the block.
#[expect(
    clippy::inline_always,
    reason = "measured entropy-block hot path requires cross-helper inlining"
)]
#[inline(always)]
fn activity_from_ac_indices(ac_index_or: usize) -> BlockActivity {
    if ac_index_or == 0 {
        BlockActivity::DcOnly
    } else {
        match ac_index_or & (32 | 4) {
            0 if ac_index_or & (16 | 2) == 0 => BlockActivity::TopLeft2x2,
            0 => BlockActivity::TopLeftQuadrant,
            4 => BlockActivity::BottomHalfZero,
            32 => BlockActivity::RightHalfZero,
            _ => BlockActivity::General,
        }
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
pub(crate) fn decode_block<const N: usize>(
    br: &mut BitReader<'_>,
    dc_table: DcHuffmanTable<'_, N>,
    ac_table: AcHuffmanTable<'_, N>,
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
pub(crate) fn decode_block_with_activity<const N: usize>(
    br: &mut BitReader<'_>,
    dc_table: DcHuffmanTable<'_, N>,
    ac_table: AcHuffmanTable<'_, N>,
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
    drive_ac_fast::<N, false, _>(br, ac_table, |k, value| {
        let natural_idx = ZIGZAG[k] as usize;
        // Quant table entries are stored in zigzag order per T.81 §B.2.4.1,
        // so `quant[k]` is the matching coefficient (not `quant[natural_idx]`).
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
pub(crate) fn decode_block_dequantized_into<const N: usize>(
    br: &mut BitReader<'_>,
    dc_table: DcHuffmanTable<'_, N>,
    ac_table: AcHuffmanTable<'_, N>,
    prev_dc: &mut i32,
    quant: &[u16; 64],
    out: &mut [i16; 64],
) -> Result<(), JpegError> {
    // DC.
    let diff = dc_table.decode_fast_dc(br)?;
    *prev_dc = prev_dc.wrapping_add(diff);
    let dc_dequant = (*prev_dc).wrapping_mul(i32::from(quant[0]));
    out[0] = clamp_i16(dc_dequant);

    drive_ac_fast::<N, false, _>(br, ac_table, |k, value| {
        let natural_idx = ZIGZAG[k] as usize;
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
pub(crate) fn decode_block_quantized_and_dequantized_with_activity<const N: usize>(
    br: &mut BitReader<'_>,
    dc_table: DcHuffmanTable<'_, N>,
    ac_table: AcHuffmanTable<'_, N>,
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
    drive_ac_fast::<N, false, _>(br, ac_table, |k, value| {
        let natural_idx = ZIGZAG[k] as usize;
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
pub(crate) fn decode_block_with_dc_status<const N: usize>(
    br: &mut BitReader<'_>,
    dc_table: DcHuffmanTable<'_, N>,
    ac_table: AcHuffmanTable<'_, N>,
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
    drive_ac_fast::<N, false, _>(br, ac_table, |k, value| {
        let natural_idx = ZIGZAG[k] as usize;
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
pub(crate) fn skip_block<const N: usize>(
    br: &mut BitReader<'_>,
    dc_table: DcHuffmanTable<'_, N>,
    ac_table: AcHuffmanTable<'_, N>,
    prev_dc: &mut i32,
) -> Result<(), JpegError> {
    let diff = dc_table.decode_fast_dc(br)?;
    *prev_dc = prev_dc.wrapping_add(diff);

    drive_ac_fast::<N, true, _>(br, ac_table, |_, _| Ok(()))?;
    Ok(())
}

#[expect(
    clippy::inline_always,
    reason = "measured entropy-block hot path requires cross-helper inlining"
)]
#[inline(always)]
fn drive_ac_fast<const N: usize, const SKIP_VALUES: bool, F>(
    br: &mut BitReader<'_>,
    ac_table: AcHuffmanTable<'_, N>,
    on_value: F,
) -> Result<(), JpegError>
where
    F: FnMut(usize, i32) -> Result<(), JpegError>,
{
    if ac_table.uses_wide_lookup() {
        drive_ac_fast_with_width::<N, SKIP_VALUES, true, F>(br, ac_table, on_value)
    } else {
        drive_ac_fast_with_width::<N, SKIP_VALUES, false, F>(br, ac_table, on_value)
    }
}

/// AC coefficient loop: one primary lookup per iteration. A non-wide value
/// entry also carries the next symbol when the window holds it (a value or
/// EOB). That symbol is taken from this lookup only when the one-symbol loop
/// would decode it from the same window, that is when the buffered bits after
/// the first symbol cover its next lookup and it falls inside the block;
/// otherwise the first symbol is decoded alone. Refill points, coefficients
/// and errors therefore match the one-symbol loop exactly.
#[expect(
    clippy::inline_always,
    reason = "the lookup width is selected once before the AC coefficient loop"
)]
#[inline(always)]
fn drive_ac_fast_with_width<const N: usize, const SKIP_VALUES: bool, const WIDE: bool, F>(
    br: &mut BitReader<'_>,
    ac_table: AcHuffmanTable<'_, N>,
    mut on_value: F,
) -> Result<(), JpegError>
where
    F: FnMut(usize, i32) -> Result<(), JpegError>,
{
    let mut k: usize = 1;
    while k < 64 {
        let lookup = ac_table.lookup_ac::<WIDE>(br)?;
        if !WIDE && lookup.packed & AC_FAST_KIND_MASK == AC_FAST_VALUE {
            // A single-value entry repeats its value at distance zero, so the
            // second store below rewrites the first.
            let first = k + ac_decoded_run(lookup.packed);
            let second = first + ac_pair_distance(lookup.pair);
            let ends_block = ac_pair_ends_block(lookup.pair);
            if second + usize::from(ends_block) < 64 && br.has_bits(ac_pair_guard(lookup.pair)) {
                br.consume_bits(ac_pair_len(lookup.pair));
                on_value(first, ac_decoded_value(lookup.packed))?;
                on_value(second, ac_pair_value(lookup.pair))?;
                if ends_block {
                    break;
                }
                k = second + 1;
                continue;
            }
        }
        let ac = ac_table.finish_ac::<WIDE, SKIP_VALUES>(br, lookup)?;
        match ac & AC_FAST_KIND_MASK {
            AC_FAST_EOB => break,
            AC_FAST_ZRL => k += 16,
            AC_FAST_VALUE => {
                k += ac_decoded_run(ac);
                if k >= 64 {
                    return Err(invalid_huffman_symbol());
                }
                on_value(k, ac_decoded_value(ac))?;
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
    use crate::adapter::JpegBaselineHuffmanTable;
    use crate::baseline_entropy::{encode_block, BitWriter};
    use crate::entropy::huffman::{
        HuffmanLookupRequirement, FULL_PRIMARY_ENTRIES, NARROW_PRIMARY_ENTRIES,
    };
    use crate::internal::bit_reader::tests::next_random;
    use crate::optimal_huffman::{optimal_huffman_spec, HuffmanFrequencies};
    use crate::parse::tables::{raw_table, HuffmanTableRole, HuffmanValues, RawHuffmanTable};
    use crate::standard_huffman::{
        STD_LUMA_AC_BITS, STD_LUMA_AC_CANONICAL, STD_LUMA_AC_VALUES, STD_LUMA_DC_BITS,
        STD_LUMA_DC_CANONICAL, STD_LUMA_DC_VALUES,
    };

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
        assert_eq!(activity, BlockActivity::TopLeft2x2);
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

        assert_eq!(activity, BlockActivity::TopLeft2x2);
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
    fn activity_tracks_top_and_left_half_occupancy() {
        assert_eq!(activity_from_ac_indices(0), BlockActivity::DcOnly);
        for first in 1usize..64 {
            for second in 1usize..64 {
                let bottom = first / 8 >= 4 || second / 8 >= 4;
                let right = first % 8 >= 4 || second % 8 >= 4;
                let top_left_2x2 = [first, second]
                    .into_iter()
                    .all(|index| index / 8 < 2 && index % 8 < 2);
                let expected = if top_left_2x2 {
                    BlockActivity::TopLeft2x2
                } else {
                    match (bottom, right) {
                        (false, false) => BlockActivity::TopLeftQuadrant,
                        (false, true) => BlockActivity::BottomHalfZero,
                        (true, false) => BlockActivity::RightHalfZero,
                        (true, true) => BlockActivity::General,
                    }
                };
                assert_eq!(
                    activity_from_ac_indices(first | second),
                    expected,
                    "AC positions {first}, {second}"
                );
            }
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

    /// The AC loop before pair extensions: one `decode_fast_ac` (or, when
    /// `skip`, `skip_fast_ac`) per symbol. Returns the dequantized block.
    fn one_symbol_block<const N: usize>(
        br: &mut BitReader<'_>,
        dc_table: DcHuffmanTable<'_, N>,
        ac_table: AcHuffmanTable<'_, N>,
        prev_dc: &mut i32,
        quant: &[u16; 64],
        skip: bool,
    ) -> Result<([i16; 64], BlockActivity), JpegError> {
        let mut out = [0i16; 64];
        *prev_dc = prev_dc.wrapping_add(dc_table.decode_fast_dc(br)?);
        out[0] = clamp_i16(prev_dc.wrapping_mul(i32::from(quant[0])));
        let mut ac_index_or = 0usize;
        let mut k = 1usize;
        while k < 64 {
            let ac = match (skip, ac_table.uses_wide_lookup()) {
                (false, false) => ac_table.decode_fast_ac::<false>(br)?,
                (false, true) => ac_table.decode_fast_ac::<true>(br)?,
                (true, false) => ac_table.skip_fast_ac::<false>(br)?,
                (true, true) => ac_table.skip_fast_ac::<true>(br)?,
            };
            match ac & AC_FAST_KIND_MASK {
                AC_FAST_EOB => break,
                AC_FAST_ZRL => k += 16,
                _ => {
                    k += ac_decoded_run(ac);
                    if k >= 64 {
                        return Err(invalid_huffman_symbol());
                    }
                    let natural = usize::from(ZIGZAG[k]);
                    let value = ac_decoded_value(ac).wrapping_mul(i32::from(quant[k]));
                    out[natural] = clamp_i16(value);
                    ac_index_or |= natural;
                    k += 1;
                }
            }
        }
        Ok((out, activity_from_ac_indices(ac_index_or)))
    }

    fn random_below(state: &mut u64, bound: u64) -> u64 {
        next_random(state) % bound
    }

    /// Entropy-code `blocks` random blocks: dense blocks whose last value sits
    /// at zigzag 63 (no EOB), sparse blocks of short runs and small values that
    /// pair up, long runs that need ZRL, and occasional 10-bit magnitudes.
    fn random_scan(
        state: &mut u64,
        blocks: usize,
        dc: &JpegBaselineHuffmanTable,
        ac: &JpegBaselineHuffmanTable,
    ) -> Vec<u8> {
        let mut writer = BitWriter::try_with_max_bytes(blocks * 256 + 64).unwrap();
        let mut prev_dc = 0;
        for _ in 0..blocks {
            let mut coeffs = [0i32; 64];
            coeffs[0] = i32::try_from(random_below(state, 1024)).unwrap() - 512;
            let style = random_below(state, 4);
            let mut k = 1 + usize::try_from(random_below(state, 2)).unwrap();
            while k < 64 {
                let size = if random_below(state, 12) == 0 {
                    1 + random_below(state, 10)
                } else {
                    1 + random_below(state, 3)
                };
                let low = 1i32 << (size - 1);
                let magnitude = low + i32::try_from(random_below(state, 1 << (size - 1))).unwrap();
                coeffs[usize::from(ZIGZAG[k])] = if next_random(state) & 1 == 0 {
                    magnitude
                } else {
                    -magnitude
                };
                if style != 0 && random_below(state, 8) == 0 {
                    break;
                }
                let run = match style {
                    0 => 0,
                    1 => random_below(state, 3),
                    2 => random_below(state, 24),
                    _ => random_below(state, 6),
                };
                k += 1 + usize::try_from(run).unwrap();
            }
            encode_block(&coeffs, &mut prev_dc, dc, ac, &mut writer).unwrap();
        }
        writer.into_bytes().unwrap()
    }

    /// Clean, truncated, bit-flipped and marker-damaged copies of a scan.
    fn scan_variants(state: &mut u64, scan: &[u8]) -> Vec<Vec<u8>> {
        let terminated = |bytes: &[u8]| [bytes, &[0xff, 0xd9]].concat();
        let mut variants = vec![terminated(scan), scan.to_vec()];
        for _ in 0..4 {
            let cut =
                usize::try_from(random_below(state, u64::try_from(scan.len()).unwrap())).unwrap();
            variants.push(terminated(&scan[..cut]));
            variants.push(scan[..cut].to_vec());
            let mut flipped = scan.to_vec();
            flipped[cut] ^= 1 << random_below(state, 8);
            variants.push(terminated(&flipped));
            let marker = [0xff, 0xd0 | u8::try_from(random_below(state, 8)).unwrap()];
            variants.push(terminated(&[&scan[..cut], &marker, &scan[cut..]].concat()));
        }
        variants
    }

    fn assert_pair_loop_matches_one_symbol_loop<const N: usize>(
        bytes: &[u8],
        dc: &HuffmanTable,
        ac: &HuffmanTable,
        wide: bool,
        quant: &[u16; 64],
        blocks: usize,
    ) {
        let dc_table = dc.dc_with_entries::<N>().unwrap();
        let ac_table = ac.ac_with_entries::<N>().unwrap().with_wide_lookup(wide);
        let mut decoded = BitReader::new(bytes);
        let mut dequantized = BitReader::new(bytes);
        let mut skipped = BitReader::new(bytes);
        let mut expected_decode = BitReader::new(bytes);
        let mut expected_skip = BitReader::new(bytes);
        let mut prev_dc = [0i32; 5];
        let mut block = CoefficientBlock::default();
        for index in 0..blocks {
            let context = format!("N={N} wide={wide} block {index}");
            let expected = one_symbol_block(
                &mut expected_decode,
                dc_table,
                ac_table,
                &mut prev_dc[0],
                quant,
                false,
            );
            let actual = decode_block_with_activity(
                &mut decoded,
                dc_table,
                ac_table,
                &mut prev_dc[1],
                quant,
                &mut block,
            )
            .map(|activity| (*block.coefficients(), activity));
            assert_eq!(actual, expected, "{context}");
            assert_eq!(decoded.snapshot(), expected_decode.snapshot(), "{context}");
            assert_eq!(prev_dc[1], prev_dc[0], "{context}");

            let mut out = [0i16; 64];
            let into = decode_block_dequantized_into(
                &mut dequantized,
                dc_table,
                ac_table,
                &mut prev_dc[2],
                quant,
                &mut out,
            )
            .map(|()| out);
            let expected_into = expected.clone().map(|(coeffs, _)| coeffs);
            assert_eq!(into, expected_into, "{context}");
            assert_eq!(
                dequantized.snapshot(),
                expected_decode.snapshot(),
                "{context}"
            );

            let expected_skipped = one_symbol_block(
                &mut expected_skip,
                dc_table,
                ac_table,
                &mut prev_dc[3],
                quant,
                true,
            )
            .map(|_| ());
            let actual_skipped = skip_block(&mut skipped, dc_table, ac_table, &mut prev_dc[4]);
            assert_eq!(actual_skipped, expected_skipped, "{context} skip");
            assert_eq!(
                skipped.snapshot(),
                expected_skip.snapshot(),
                "{context} skip"
            );
            assert_eq!(prev_dc[4], prev_dc[3], "{context} skip");
            if expected.is_err() {
                break;
            }
        }
    }

    struct PairTestTables {
        dc_raw: RawHuffmanTable,
        dc_codes: JpegBaselineHuffmanTable,
        ac_raw: RawHuffmanTable,
        ac_codes: JpegBaselineHuffmanTable,
    }

    /// Standard luma tables, plus optimal tables for a skewed AC distribution
    /// (one- and two-bit codes, so two symbols often share a narrow window).
    fn pair_test_tables() -> Vec<PairTestTables> {
        let mut dc_frequencies = HuffmanFrequencies::new();
        let mut ac_frequencies = HuffmanFrequencies::new();
        for symbol in 0..=11 {
            dc_frequencies.add_count(symbol, 1);
        }
        for run in 0..16u8 {
            for size in 1..=10u8 {
                ac_frequencies.add_count((run << 4) | size, 1);
            }
        }
        for (symbol, count) in [
            (0x00, 4000),
            (0x01, 3000),
            (0x02, 1000),
            (0x11, 800),
            (0xf0, 50),
        ] {
            ac_frequencies.add_count(symbol, count);
        }
        let dc_spec = optimal_huffman_spec(&dc_frequencies).unwrap();
        let ac_spec = optimal_huffman_spec(&ac_frequencies).unwrap();
        vec![
            PairTestTables {
                dc_raw: raw_table(STD_LUMA_DC_BITS, &STD_LUMA_DC_VALUES),
                dc_codes: STD_LUMA_DC_CANONICAL,
                ac_raw: raw_table(STD_LUMA_AC_BITS, &STD_LUMA_AC_VALUES),
                ac_codes: STD_LUMA_AC_CANONICAL,
            },
            PairTestTables {
                dc_raw: raw_table(dc_spec.bits, dc_spec.values()),
                dc_codes: dc_spec.table(),
                ac_raw: raw_table(ac_spec.bits, ac_spec.values()),
                ac_codes: ac_spec.table(),
            },
        ]
    }

    #[test]
    fn pair_loop_matches_one_symbol_loop_on_encoded_and_damaged_scans() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut quant = [1u16; 64];
        for (index, q) in quant.iter_mut().enumerate() {
            // Small steps, plus large ones that saturate the i16 clamp.
            *q = if index % 9 == 8 {
                4000
            } else {
                1 + u16::try_from(index % 7).unwrap()
            };
        }
        let blocks = 48;
        for tables in pair_test_tables() {
            for requirement in [
                HuffmanLookupRequirement::FusedFull,
                HuffmanLookupRequirement::FusedNarrow,
            ] {
                let build = |raw: &RawHuffmanTable, role| {
                    let mut live_bytes = 0;
                    HuffmanTable::from_raw_with_requirement_and_live_budget(
                        raw,
                        role,
                        requirement,
                        &mut live_bytes,
                        j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                    )
                    .unwrap()
                };
                let dc = build(&tables.dc_raw, HuffmanTableRole::Dc);
                let ac = build(&tables.ac_raw, HuffmanTableRole::Ac);
                for _ in 0..6 {
                    let scan = random_scan(&mut state, blocks, &tables.dc_codes, &tables.ac_codes);
                    for bytes in scan_variants(&mut state, &scan) {
                        let budget = blocks + 2;
                        if requirement == HuffmanLookupRequirement::FusedFull {
                            for wide in [false, true] {
                                assert_pair_loop_matches_one_symbol_loop::<FULL_PRIMARY_ENTRIES>(
                                    &bytes, &dc, &ac, wide, &quant, budget,
                                );
                            }
                        }
                        assert_pair_loop_matches_one_symbol_loop::<NARROW_PRIMARY_ENTRIES>(
                            &bytes, &dc, &ac, false, &quant, budget,
                        );
                    }
                }
            }
        }
    }
}
