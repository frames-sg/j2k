// SPDX-License-Identifier: MIT OR Apache-2.0

use core::mem::size_of;

// JPEG 2000 Part 1 permits up to 38 sample bits; keep additional headroom for
// guard bits and ROI-shifted code-block magnitudes while reserving one sign bit.
#[expect(clippy::cast_possible_truncation, reason = "u64 is eight bytes")]
pub(crate) const BITPLANE_BIT_SIZE: u32 = size_of::<u64>() as u32 * 8 - 1;

pub(in crate::j2c::bitplane) const HAS_MAGNITUDE_REFINEMENT_SHIFT: u8 = 6;
pub(in crate::j2c::bitplane) const HAS_ZERO_CODING_SHIFT: u8 = 5;
pub(in crate::j2c::bitplane) const SIGNIFICANCE_MASK: u8 = 1 << 7;
pub(in crate::j2c::bitplane) const HAS_MAGNITUDE_REFINEMENT_MASK: u8 =
    1 << HAS_MAGNITUDE_REFINEMENT_SHIFT;
pub(in crate::j2c::bitplane) const HAS_ZERO_CODING_MASK: u8 = 1 << HAS_ZERO_CODING_SHIFT;

/// Bit-packed coefficient state (only 3 bits used):
/// - Bit 7: significance state (set when first non-zero bit is encountered)
/// - Bit 6: has had magnitude refinement pass
/// - Bit 5: zero coded in current bitplane's significance propagation pass
#[derive(Default, Copy, Clone)]
pub(crate) struct CoefficientState(pub(in crate::j2c::bitplane) u8);

impl CoefficientState {
    #[expect(clippy::inline_always, reason = "Tier-1 coefficient-loop hot path")]
    #[inline(always)]
    pub(in crate::j2c::bitplane) fn set_significant(&mut self) {
        self.0 |= SIGNIFICANCE_MASK;
    }

    #[expect(clippy::inline_always, reason = "Tier-1 coefficient-loop hot path")]
    #[inline(always)]
    pub(in crate::j2c::bitplane) fn is_significant(self) -> bool {
        self.0 & SIGNIFICANCE_MASK != 0
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub(crate) struct Coefficient(u64);

impl Coefficient {
    #[cfg(test)]
    #[expect(clippy::trivially_copy_pass_by_ref, reason = "stable accessor")]
    pub(crate) fn get(&self) -> i32 {
        i32::try_from(
            self.get_i64()
                .clamp(i64::from(i32::MIN), i64::from(i32::MAX)),
        )
        .expect("coefficient is clamped to the i32 range")
    }

    #[expect(clippy::trivially_copy_pass_by_ref, reason = "stable accessor")]
    pub(crate) fn get_i64(&self) -> i64 {
        let mut magnitude = (self.0 & !(1_u64 << 63)).cast_signed();
        // Map sign (0 for positive, 1 for negative) to 1, -1.
        magnitude *= 1 - 2 * i64::from(self.sign() != 0);
        magnitude
    }

    pub(in crate::j2c::bitplane) fn set_sign(&mut self, sign: u8) {
        self.0 |= u64::from(sign) << 63;
    }

    pub(in crate::j2c::bitplane) fn sign(self) -> u64 {
        (self.0 >> 63) & 1
    }

    pub(in crate::j2c::bitplane) fn push_bit_at(&mut self, bit: u32, position: u8) {
        self.0 |= u64::from(bit) << position;
    }
}

pub(in crate::j2c::bitplane) const COEFFICIENTS_PADDING: u32 = 1;

/// Neighbor significance bits ordered as top-left, top, top-right, left,
/// bottom-left, right, bottom-right, bottom from MSB to LSB.
#[derive(Default, Copy, Clone)]
pub(in crate::j2c::bitplane) struct NeighborSignificances(pub(in crate::j2c::bitplane) u8);

impl NeighborSignificances {
    pub(in crate::j2c::bitplane) fn set_top_left(&mut self) {
        self.0 |= 1 << 7;
    }

    pub(in crate::j2c::bitplane) fn set_top(&mut self) {
        self.0 |= 1 << 6;
    }

    pub(in crate::j2c::bitplane) fn set_top_right(&mut self) {
        self.0 |= 1 << 5;
    }

    pub(in crate::j2c::bitplane) fn set_left(&mut self) {
        self.0 |= 1 << 4;
    }

    pub(in crate::j2c::bitplane) fn set_bottom_left(&mut self) {
        self.0 |= 1 << 3;
    }

    pub(in crate::j2c::bitplane) fn set_right(&mut self) {
        self.0 |= 1 << 2;
    }

    pub(in crate::j2c::bitplane) fn set_bottom_right(&mut self) {
        self.0 |= 1 << 1;
    }

    pub(in crate::j2c::bitplane) fn set_bottom(&mut self) {
        self.0 |= 1;
    }

    pub(in crate::j2c::bitplane) fn all(self) -> u8 {
        self.0
    }

    pub(in crate::j2c::bitplane) fn all_without_bottom(self) -> u8 {
        self.0 & 0b1111_0100
    }
}
