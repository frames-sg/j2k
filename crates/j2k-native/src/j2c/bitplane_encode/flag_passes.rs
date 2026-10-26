// SPDX-License-Identifier: MIT OR Apache-2.0

//! The three arithmetic-coded passes over the packed column words that the
//! decoder scans, described in `bitplane::flags`, for code blocks without
//! vertically causal context. Decisions, contexts and their order match the
//! per-coefficient passes in `passes.rs` exactly; only the bookkeeping
//! differs.

use alloc::vec::Vec;

use super::super::arithmetic_encoder::{ArithmeticEncoder, ArithmeticEncoderContext};
use super::super::bitplane::flags::{
    set_significant, sign_coding_label, zero_coding_label, zero_coding_table, MU_THIS, PI_ALL,
    PI_THIS, SIGMA_NEIGHBOURS, SIGMA_THIS,
};
use super::super::build::SubBandType;
use super::super::codestream::CodeBlockStyle;
use super::super::coefficient_view::{CoefficientBlockView, SignedCoefficient};
use super::super::encode::allocation::try_untracked_vec_filled;
use super::{code_passes, CodedBlock, CodingPass};
use crate::EncodeResult;

/// Bit 0 of each of the stripe's four row windows.
const ROW_BITS: u32 = 1 | 1 << 3 | 1 << 6 | 1 << 9;

/// Rows neither significant nor coded in this bitplane's significance pass,
/// as `ROW_BITS`.
#[expect(clippy::inline_always, reason = "Tier-1 coefficient-loop hot path")]
#[inline(always)]
fn uncoded_rows(own: u32) -> u32 {
    !(own >> SIGMA_THIS.trailing_zeros() | own >> PI_THIS.trailing_zeros()) & ROW_BITS
}

/// Uncoded rows with a significant neighbour: the significance propagation
/// candidates.
#[expect(clippy::inline_always, reason = "Tier-1 coefficient-loop hot path")]
#[inline(always)]
fn propagation_rows(own: u32) -> u32 {
    let any_in_row = own | own >> 1 | own >> 2;
    let neighbors = any_in_row | any_in_row >> 6 | own >> 3 | own >> 5;
    neighbors & uncoded_rows(own)
}

/// Sample words keep the magnitude below the sign in bit 31.
pub(super) const MAX_BITPLANES: u8 = 31;

/// One word per sample, the four rows of a stripe column consecutive and rows
/// past the block's end zero: magnitude in bits 0-30, sign in bit 31. The
/// caller checks that every magnitude fits before building these words.
fn try_column_samples<T: SignedCoefficient>(
    coefficients: CoefficientBlockView<'_, T>,
) -> EncodeResult<Vec<u32>> {
    let width = coefficients.width();
    let mut samples = try_untracked_vec_filled(
        coefficients.height().div_ceil(4) * width * 4,
        0_u32,
        "classic Tier-1 column samples",
    )?;
    for (y, row) in coefficients.rows().enumerate() {
        let first = (y / 4) * width * 4 + y % 4;
        for (word, &coefficient) in samples[first..].iter_mut().step_by(4).zip(row) {
            let magnitude = coefficient.unsigned_magnitude();
            #[expect(
                clippy::cast_possible_truncation,
                reason = "blocks with more than 31 coded bitplanes are rejected before coding"
            )]
            let low = magnitude as u32 & !(1 << 31);
            *word = low | u32::from(coefficient.is_negative()) << 31;
        }
    }
    Ok(samples)
}

/// Everything one pass touches.
struct Scan<'a> {
    encoder: &'a mut ArithmeticEncoder,
    contexts: &'a mut [ArithmeticEncoderContext; 19],
    flags: &'a mut [u32],
    samples: &'a [u32],
    zero_coding: &'static [u8; 512],
    width: usize,
    height: usize,
    /// Row stride of `flags`: one padding column on each side.
    stride: usize,
    bitplane: u32,
}

impl Scan<'_> {
    #[expect(clippy::inline_always, reason = "per-symbol MQ encode")]
    #[inline(always)]
    fn code(&mut self, bit: u32, label: u8) {
        self.encoder
            .encode(bit, &mut self.contexts[usize::from(label)]);
    }

    /// Flag index of column 0 in `stripe`, index of its first sample word,
    /// and the `ROW_BITS` of the rows the stripe holds.
    fn stripe_start(&self, stripe: usize) -> (usize, usize, u32) {
        let rows = (self.height - 4 * stripe).min(4);
        (
            (stripe + 1) * self.stride + 1,
            stripe * self.width * 4,
            ROW_BITS & ((1 << (3 * rows)) - 1),
        )
    }

    /// The sample word in the row at window shift `shift` of the column whose
    /// first word is `column`.
    #[expect(clippy::inline_always, reason = "Tier-1 coefficient-loop hot path")]
    #[inline(always)]
    fn sample(&self, column: usize, shift: u32) -> u32 {
        self.samples[column + (shift / 3) as usize]
    }

    /// Code the sign of the coefficient that just became significant and
    /// record its significance.
    #[expect(clippy::inline_always, reason = "Tier-1 coefficient-loop hot path")]
    #[inline(always)]
    fn become_significant(&mut self, own: &mut u32, flag_index: usize, sample: u32, shift: u32) {
        let sign = sample >> 31;
        let (label, xor) = sign_coding_label(
            *own,
            self.flags[flag_index - 1],
            self.flags[flag_index + 1],
            shift / 3,
        );
        self.code(sign ^ u32::from(xor), label);
        set_significant(self.flags, own, flag_index, self.stride, shift / 3, sign);
    }

    /// Zero-code the insignificant coefficient in the row at window shift
    /// `shift`; returns whether it became significant.
    #[expect(clippy::inline_always, reason = "Tier-1 coefficient-loop hot path")]
    #[inline(always)]
    fn zero_code(&mut self, own: &mut u32, flag_index: usize, column: usize, shift: u32) -> bool {
        let sample = self.sample(column, shift);
        let bit = (sample >> self.bitplane) & 1;
        self.code(bit, zero_coding_label(self.zero_coding, *own, shift / 3));
        if bit != 0 {
            self.become_significant(own, flag_index, sample, shift);
        }
        bit != 0
    }

    fn significance_propagation(&mut self) {
        for stripe in 0..self.height.div_ceil(4) {
            let (flag_row, sample_row, rows) = self.stripe_start(stripe);
            for x in 0..self.width {
                let flag_index = flag_row + x;
                let mut own = self.flags[flag_index];
                let mut candidates = propagation_rows(own) & rows;
                if candidates == 0 {
                    continue;
                }
                let column = sample_row + 4 * x;
                while candidates != 0 {
                    let shift = candidates.trailing_zeros();
                    own |= PI_THIS << shift;
                    if self.zero_code(&mut own, flag_index, column, shift) {
                        // The new significant row neighbors the rows below it.
                        candidates = propagation_rows(own) & rows & (u32::MAX << (shift + 1));
                    } else {
                        candidates &= candidates - 1;
                    }
                }
                self.flags[flag_index] = own;
            }
        }
    }

    fn magnitude_refinement(&mut self) {
        for stripe in 0..self.height.div_ceil(4) {
            let (flag_row, sample_row, rows) = self.stripe_start(stripe);
            for x in 0..self.width {
                let flag_index = flag_row + x;
                let mut own = self.flags[flag_index];
                // Significant before this bitplane's significance pass.
                let mut candidates = (own >> SIGMA_THIS.trailing_zeros())
                    & !(own >> PI_THIS.trailing_zeros())
                    & rows;
                if candidates == 0 {
                    continue;
                }
                let column = sample_row + 4 * x;
                while candidates != 0 {
                    let shift = candidates.trailing_zeros();
                    let window = own >> shift;
                    let label = if window & MU_THIS != 0 {
                        16
                    } else if window & SIGMA_NEIGHBOURS != 0 {
                        15
                    } else {
                        14
                    };
                    self.code((self.sample(column, shift) >> self.bitplane) & 1, label);
                    own |= MU_THIS << shift;
                    candidates &= candidates - 1;
                }
                self.flags[flag_index] = own;
            }
        }
    }

    fn cleanup(&mut self) {
        for stripe in 0..self.height.div_ceil(4) {
            let (flag_row, sample_row, rows) = self.stripe_start(stripe);
            for x in 0..self.width {
                let flag_index = flag_row + x;
                let column = sample_row + 4 * x;
                let mut own = self.flags[flag_index];
                // A zero word is a full column of insignificant, unvisited
                // coefficients with zero contexts: Annex D's run-length case.
                if rows == ROW_BITS && own == 0 {
                    let bits = (0..4).fold(0_u32, |bits, row| {
                        bits | ((self.samples[column + row] >> self.bitplane) & 1) << row
                    });
                    if bits == 0 {
                        self.code(0, 17);
                        continue;
                    }
                    let first = bits.trailing_zeros();
                    self.code(1, 17);
                    self.code(first >> 1, 18);
                    self.code(first & 1, 18);
                    let sample = self.sample(column, 3 * first);
                    self.become_significant(&mut own, flag_index, sample, 3 * first);
                    for row in first + 1..4 {
                        self.zero_code(&mut own, flag_index, column, 3 * row);
                    }
                } else {
                    let mut candidates = uncoded_rows(own) & rows;
                    while candidates != 0 {
                        self.zero_code(&mut own, flag_index, column, candidates.trailing_zeros());
                        candidates &= candidates - 1;
                    }
                }
                // Visit marks last one bitplane.
                self.flags[flag_index] = own & !PI_ALL;
            }
        }
    }
}

/// Code every pass of a block's `num_bitplanes`, at most `MAX_BITPLANES`.
#[expect(clippy::trivially_copy_pass_by_ref, reason = "shared style borrow")]
pub(super) fn try_encode_column_passes<T: SignedCoefficient>(
    coefficients: CoefficientBlockView<'_, T>,
    sub_band_type: SubBandType,
    num_bitplanes: u8,
    style: &CodeBlockStyle,
    payload_bytes: usize,
) -> EncodeResult<CodedBlock> {
    debug_assert!(num_bitplanes <= MAX_BITPLANES);
    let samples = try_column_samples(coefficients)?;
    let (width, height) = (coefficients.width(), coefficients.height());
    let stride = width + 2;
    let mut flags = try_untracked_vec_filled(
        (height.div_ceil(4) + 2) * stride,
        0_u32,
        "classic Tier-1 column flags",
    )?;
    let zero_coding = zero_coding_table(sub_band_type);
    code_passes(
        num_bitplanes,
        style,
        payload_bytes,
        |pass, bitplane, encoder, contexts| {
            let mut scan = Scan {
                encoder,
                contexts,
                flags: &mut flags,
                samples: &samples,
                zero_coding,
                width,
                height,
                stride,
                bitplane,
            };
            match pass {
                CodingPass::SignificancePropagation => scan.significance_propagation(),
                CodingPass::MagnitudeRefinement => scan.magnitude_refinement(),
                CodingPass::Cleanup => scan.cleanup(),
            }
        },
    )
}
