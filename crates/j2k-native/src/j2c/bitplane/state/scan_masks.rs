// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-stripe-column bit masks that let the passes test four coefficients at
//! once instead of loading each coefficient's state.

use super::{BitPlaneDecodeContext, COEFFICIENTS_PADDING, HAS_ZERO_CODING_MASK};

impl BitPlaneDecodeContext {
    #[expect(clippy::inline_always, reason = "Tier-1 coefficient-loop hot path")]
    #[inline(always)]
    pub(in crate::j2c::bitplane) fn set_zero_coding_index(
        &mut self,
        idx: usize,
        y: usize,
        padded_width: usize,
    ) {
        self.coefficient_states[idx].0 |= HAS_ZERO_CODING_MASK;
        let (scan_unit, bit) = self.scan_unit_mask_index(idx, y, padded_width);
        self.zero_coding_scan_masks[scan_unit] |= bit;
    }

    #[expect(clippy::inline_always, reason = "Tier-1 coefficient-loop hot path")]
    #[inline(always)]
    pub(in crate::j2c::bitplane) fn set_significant_scan_mask(
        &mut self,
        idx: usize,
        y: usize,
        padded_width: usize,
    ) {
        let (scan_unit, bit) = self.scan_unit_mask_index(idx, y, padded_width);
        self.significant_scan_masks[scan_unit] |= bit;
    }

    #[expect(clippy::inline_always, reason = "Tier-1 coefficient-loop hot path")]
    #[inline(always)]
    pub(in crate::j2c::bitplane) fn scan_unit_mask_index(
        &self,
        idx: usize,
        y: usize,
        padded_width: usize,
    ) -> (usize, u8) {
        // Callers pass the row they are scanning, so recovering the column
        // costs a multiply; deriving the row from `idx` needed a division on
        // every significance and zero-coding update.
        let pad = COEFFICIENTS_PADDING as usize;
        debug_assert!(idx / padded_width == y + pad);
        let x = idx - (y + pad) * padded_width - pad;
        debug_assert!(y < self.height as usize);
        debug_assert!(x < self.width as usize);

        let scan_unit = (y >> 2) * self.width as usize + x;
        (scan_unit, 1u8 << (y & 3))
    }
}
