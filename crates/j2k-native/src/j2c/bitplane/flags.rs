// SPDX-License-Identifier: MIT OR Apache-2.0

//! Packed per-stripe-column state for the normal-neighbor arithmetic path.
//!
//! One `u32` per stripe column (four rows) carries everything the three
//! passes consult, so a coefficient's zero-coding context is a shift and a
//! table load, and a newly significant coefficient updates three words (six
//! at a stripe edge) instead of eight neighbor bytes plus scan masks.
//!
//! Bit layout, for window rows `-1..=4` (the stripe's four rows plus the row
//! above and below) and columns `-1..=1`:
//! - `3 * (row + 1) + (col + 1)`: significance of that neighbor (bits 0-17);
//! - 18: sign of row -1; `19 + 3 * row` for rows 0-4: sign of the column's
//!   own coefficient in that row (bits 19, 22, 25, 28, 31);
//! - `20 + 3 * row`, rows 0-3: magnitude refinement has run (bits 20-29);
//! - `21 + 3 * row`, rows 0-3: coded in this bitplane's significance
//!   propagation pass (bits 21-30).
//!
//! Shifting a word right by `3 * row` aligns row `row`'s 3x3 neighborhood
//! with bits 0-8 and its own refinement and visit bits with bits 20 and 21.
//! Sign bits are only ever set together with the matching significance bit.

use super::super::build::SubBandType;
use j2k_codec_math::classic::{
    SIGN_CONTEXT_LOOKUP, ZERO_CTX_HH_LOOKUP, ZERO_CTX_HL_LOOKUP, ZERO_CTX_LL_LH_LOOKUP,
};

/// The coefficient's own significance within a row-aligned window.
pub(super) const SIGMA_THIS: u32 = 1 << 4;
/// The eight neighbors within a row-aligned window.
pub(super) const SIGMA_NEIGHBOURS: u32 = 0x1EF;
/// Magnitude refinement has run, row-aligned.
pub(super) const MU_THIS: u32 = 1 << 20;
/// Coded in this bitplane's significance propagation pass, row-aligned.
pub(super) const PI_THIS: u32 = 1 << 21;
/// All four rows' `PI_THIS` bits.
pub(super) const PI_ALL: u32 = PI_THIS | PI_THIS << 3 | PI_THIS << 6 | PI_THIS << 9;

/// Zero-coding context label for a row-aligned 3x3 window (bits 0-8, the
/// centre ignored), derived from the byte-indexed Table D.1 lookups.
const fn zero_coding_window_table(byte_table: &[u8; 256]) -> [u8; 512] {
    const fn bit(window: usize, index: usize) -> usize {
        (window >> index) & 1
    }
    let mut table = [0_u8; 512];
    let mut window = 0;
    while window < 512 {
        // Neighbor byte, MSB to LSB: TL, T, TR, L, BL, R, BR, B.
        let byte = (bit(window, 0) << 7)
            | (bit(window, 1) << 6)
            | (bit(window, 2) << 5)
            | (bit(window, 3) << 4)
            | (bit(window, 6) << 3)
            | (bit(window, 5) << 2)
            | (bit(window, 8) << 1)
            | bit(window, 7);
        table[window] = byte_table[byte];
        window += 1;
    }
    table
}

static ZERO_CODING_LL_LH: [u8; 512] = zero_coding_window_table(&ZERO_CTX_LL_LH_LOOKUP);
static ZERO_CODING_HL: [u8; 512] = zero_coding_window_table(&ZERO_CTX_HL_LOOKUP);
static ZERO_CODING_HH: [u8; 512] = zero_coding_window_table(&ZERO_CTX_HH_LOOKUP);

pub(super) fn zero_coding_table(sub_band_type: SubBandType) -> &'static [u8; 512] {
    match sub_band_type {
        SubBandType::LowLow | SubBandType::LowHigh => &ZERO_CODING_LL_LH,
        SubBandType::HighLow => &ZERO_CODING_HL,
        SubBandType::HighHigh => &ZERO_CODING_HH,
    }
}

/// Zero-coding context label for row `row` of the column word `flags`.
#[expect(
    clippy::inline_always,
    reason = "Tier-1 context lookup is a measured coefficient-loop hot path"
)]
#[inline(always)]
pub(super) fn zero_coding_label(table: &[u8; 512], flags: u32, row: u32) -> u8 {
    table[((flags >> (3 * row)) & SIGMA_NEIGHBOURS) as usize]
}

/// Sign-coding context label and XOR bit (Table D.3) for row `row`, from the
/// column word and its left and right neighbors.
#[expect(
    clippy::inline_always,
    reason = "Tier-1 context lookup is a measured coefficient-loop hot path"
)]
#[inline(always)]
pub(super) fn sign_coding_label(flags: u32, left: u32, right: u32, row: u32) -> (u8, u8) {
    let window = flags >> (3 * row);
    let significances = (((window >> 1) & 1) << 6)
        | (((window >> 3) & 1) << 4)
        | (((window >> 5) & 1) << 2)
        | ((window >> 7) & 1);
    // The sign of row - 1 is bit 18 for row 0 and 19 + 3 * (row - 1) after.
    let north_sign_bit = if row == 0 { 18 } else { 16 + 3 * row };
    let signs = (((flags >> north_sign_bit) & 1) << 6)
        | (((left >> (19 + 3 * row)) & 1) << 4)
        | (((right >> (19 + 3 * row)) & 1) << 2)
        | ((flags >> (22 + 3 * row)) & 1);
    let merged = ((significances & signs) << 1) | (significances & !signs);
    SIGN_CONTEXT_LOOKUP[merged as usize]
}

/// Record that row `row` of the column word at `index` became significant
/// with sign `sign`. The column's own word is the caller's register copy
/// `own`; neighbor words are updated in `flags`, whose row stride is
/// `stride` (one padding column and one padding stripe on every side).
#[expect(clippy::inline_always, reason = "Tier-1 coefficient-loop hot path")]
#[inline(always)]
pub(super) fn set_significant(
    flags: &mut [u32],
    own: &mut u32,
    index: usize,
    stride: usize,
    row: u32,
    sign: u32,
) {
    let shift = 3 * row;
    *own |= (SIGMA_THIS | sign << 19) << shift;
    flags[index - 1] |= (1 << 5) << shift;
    flags[index + 1] |= (1 << 3) << shift;
    if row == 0 {
        // Row 4 of the stripe above: significance bits 15-17, sign bit 31.
        let north = index - stride;
        flags[north - 1] |= 1 << 17;
        flags[north] |= 1 << 16 | sign << 31;
        flags[north + 1] |= 1 << 15;
    } else if row == 3 {
        // Row -1 of the stripe below: significance bits 0-2, sign bit 18.
        let south = index + stride;
        flags[south - 1] |= 1 << 2;
        flags[south] |= 1 << 1 | sign << 18;
        flags[south + 1] |= 1;
    }
}
