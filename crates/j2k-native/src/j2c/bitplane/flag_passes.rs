// SPDX-License-Identifier: MIT OR Apache-2.0

//! The coding passes for code blocks without vertically causal contexts,
//! scanning the packed column words described in [`super::flags`]: the three
//! arithmetic-coded passes, and the raw ones of selective bypass segments.
//!
//! Decisions, contexts, and their order match the per-coefficient passes in
//! `arithmetic.rs` exactly; only the bookkeeping differs.

use super::super::arithmetic_decoder::{ArithmeticDecoder, ArithmeticDecoderContext};
use super::bypass::RawBits;
use super::flags::{
    set_significant, sign_coding_label, zero_coding_label, zero_coding_table, MU_THIS, PI_ALL,
    PI_THIS, SIGMA_NEIGHBOURS, SIGMA_THIS,
};
use super::state::{BitPlaneDecodeContext, Coefficient};

/// Bit `3 * row` for each of a column's four rows, the shift that aligns the
/// row's window with bit 0. Passes visit a column's rows as a set of these
/// bits, lowest first, instead of branching on each row.
const ROW_BITS: u32 = 0x249;

/// Rows neither significant nor coded in this bitplane's significance
/// propagation pass, as `ROW_BITS`.
#[expect(clippy::inline_always, reason = "Tier-1 coefficient-loop hot path")]
#[inline(always)]
fn uncoded_rows(own: u32) -> u32 {
    !(own >> SIGMA_THIS.trailing_zeros() | own >> PI_THIS.trailing_zeros()) & ROW_BITS
}

/// Uncoded rows with a significant neighbor, as `ROW_BITS`.
#[expect(clippy::inline_always, reason = "Tier-1 coefficient-loop hot path")]
#[inline(always)]
fn propagation_rows(own: u32) -> u32 {
    // Bit 3k: any significance in window row k (row k - 1 of the stripe).
    let any_in_row = own | own >> 1 | own >> 2;
    let neighbors = any_in_row | any_in_row >> 6 | own >> 3 | own >> 5;
    neighbors & uncoded_rows(own)
}

/// Register copies of everything a pass touches. The MQ decoder is a local
/// copy and the arrays are plain slices, so neither the MQ registers nor the
/// slice pointers round-trip through memory that the pass's stores alias.
struct Scan<'c, 'd> {
    decoder: ArithmeticDecoder<'d>,
    contexts: &'c mut [ArithmeticDecoderContext; 256],
    flags: &'c mut [u32],
    coefficients: &'c mut [Coefficient],
    zero_coding: &'static [u8; 512],
    width: usize,
    height: usize,
    /// Row stride of both `flags` and `coefficients` (width plus padding).
    stride: usize,
    bit_position: u8,
}

impl<'c, 'd> Scan<'c, 'd> {
    fn new(ctx: &'c mut BitPlaneDecodeContext, decoder: &ArithmeticDecoder<'d>) -> Self {
        Self {
            decoder: decoder.clone(),
            zero_coding: zero_coding_table(ctx.sub_band_type),
            width: ctx.width as usize,
            height: ctx.height as usize,
            stride: ctx.padded_width as usize,
            bit_position: ctx.current_bit_position,
            contexts: &mut ctx.contexts,
            flags: &mut ctx.flags,
            coefficients: &mut ctx.coefficients,
        }
    }

    fn finish(self, decoder: &mut ArithmeticDecoder<'d>) {
        *decoder = self.decoder;
    }

    /// Flag and top-row coefficient indices of column 0 in `stripe`, and the
    /// `ROW_BITS` of the rows the stripe holds.
    fn stripe_start(&self, stripe: usize) -> (usize, usize, u32) {
        let rows = (self.height - 4 * stripe).min(4);
        (
            (stripe + 1) * self.stride + 1,
            (4 * stripe + 1) * self.stride + 1,
            ROW_BITS & ((1 << (3 * rows)) - 1),
        )
    }

    #[expect(clippy::inline_always, reason = "per-symbol MQ decode")]
    #[inline(always)]
    fn read(&mut self, label: u8) -> u32 {
        self.decoder
            .read_bit(&mut self.contexts[usize::from(label)])
    }

    /// The coefficient in the row at window shift `shift` just decoded a one
    /// while insignificant: record the bit, decode its sign, and mark it
    /// significant.
    #[expect(clippy::inline_always, reason = "Tier-1 coefficient-loop hot path")]
    #[inline(always)]
    fn become_significant(&mut self, own: &mut u32, flag_index: usize, top: usize, shift: u32) {
        let index = top + (shift / 3) as usize * self.stride;
        self.coefficients[index].push_bit_at(1, self.bit_position);
        let (label, xor) = sign_coding_label(
            *own,
            self.flags[flag_index - 1],
            self.flags[flag_index + 1],
            shift,
        );
        let sign = self.read(label) ^ u32::from(xor);
        self.coefficients[index].set_sign(u8::from(sign != 0));
        set_significant(self.flags, own, flag_index, self.stride, shift, sign);
    }

    /// Zero-code the insignificant coefficient in the row at window shift
    /// `shift`; returns whether it became significant.
    #[expect(clippy::inline_always, reason = "Tier-1 coefficient-loop hot path")]
    #[inline(always)]
    fn zero_code(&mut self, own: &mut u32, flag_index: usize, top: usize, shift: u32) -> bool {
        let significant = self.read(zero_coding_label(self.zero_coding, *own, shift)) != 0;
        if significant {
            self.become_significant(own, flag_index, top, shift);
        }
        significant
    }
}

pub(super) fn significance_propagation_pass_flags(
    ctx: &mut BitPlaneDecodeContext,
    decoder: &mut ArithmeticDecoder<'_>,
) {
    let mut scan = Scan::new(ctx, decoder);
    for stripe in 0..scan.height.div_ceil(4) {
        let (flag_row, top_row, rows) = scan.stripe_start(stripe);
        for x in 0..scan.width {
            let flag_index = flag_row + x;
            let mut own = scan.flags[flag_index];
            let mut candidates = propagation_rows(own) & rows;
            if candidates == 0 {
                continue;
            }
            while candidates != 0 {
                let shift = candidates.trailing_zeros();
                own |= PI_THIS << shift;
                if scan.zero_code(&mut own, flag_index, top_row + x, shift) {
                    // The new significant row neighbors the rows below it.
                    candidates = propagation_rows(own) & rows & (u32::MAX << (shift + 1));
                } else {
                    candidates &= candidates - 1;
                }
            }
            scan.flags[flag_index] = own;
        }
    }
    scan.finish(decoder);
}

pub(super) fn magnitude_refinement_pass_flags(
    ctx: &mut BitPlaneDecodeContext,
    decoder: &mut ArithmeticDecoder<'_>,
) {
    let mut scan = Scan::new(ctx, decoder);
    for stripe in 0..scan.height.div_ceil(4) {
        let (flag_row, top_row, rows) = scan.stripe_start(stripe);
        for x in 0..scan.width {
            let flag_index = flag_row + x;
            let mut own = scan.flags[flag_index];
            // Significant before this bitplane's significance pass.
            let mut candidates =
                (own >> SIGMA_THIS.trailing_zeros()) & !(own >> PI_THIS.trailing_zeros()) & rows;
            if candidates == 0 {
                continue;
            }
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
                let bit = scan.read(label);
                scan.coefficients[top_row + x + (shift / 3) as usize * scan.stride]
                    .push_bit_at(bit, scan.bit_position);
                own |= MU_THIS << shift;
                candidates &= candidates - 1;
            }
            scan.flags[flag_index] = own;
        }
    }
    scan.finish(decoder);
}

pub(super) fn cleanup_pass_flags(
    ctx: &mut BitPlaneDecodeContext,
    decoder: &mut ArithmeticDecoder<'_>,
) {
    let mut scan = Scan::new(ctx, decoder);
    // The run-length and uniform contexts are only used here; holding them in
    // locals keeps consecutive run-length decodes off a store-to-load chain
    // through the context array.
    let mut run_length = scan.contexts[17];
    let mut uniform = scan.contexts[18];
    for stripe in 0..scan.height.div_ceil(4) {
        let (flag_row, top_row, rows) = scan.stripe_start(stripe);
        for x in 0..scan.width {
            let flag_index = flag_row + x;
            let top = top_row + x;
            let mut own = scan.flags[flag_index];
            // A zero word is a full column of insignificant, unvisited
            // coefficients with zero contexts: Annex D's run-length case.
            if rows == ROW_BITS && own == 0 {
                if scan.decoder.read_bit(&mut run_length) == 0 {
                    continue;
                }
                let first = (scan.decoder.read_bit(&mut uniform) << 1)
                    | scan.decoder.read_bit(&mut uniform);
                scan.become_significant(&mut own, flag_index, top, 3 * first);
                for row in first + 1..4 {
                    scan.zero_code(&mut own, flag_index, top, 3 * row);
                }
            } else {
                let mut candidates = uncoded_rows(own) & rows;
                while candidates != 0 {
                    scan.zero_code(&mut own, flag_index, top, candidates.trailing_zeros());
                    candidates &= candidates - 1;
                }
            }
            // Visit marks last one bitplane.
            scan.flags[flag_index] = own & !PI_ALL;
        }
    }
    scan.contexts[17] = run_length;
    scan.contexts[18] = uniform;
    scan.finish(decoder);
}

/// Register copies for the raw (bypass) passes, as [`Scan`] keeps for the
/// arithmetic ones. Raw bits carry no context, and a raw sign bit is the sign
/// itself.
struct RawScan<'c, 'r, 'd> {
    bits: &'r mut RawBits<'d>,
    flags: &'c mut [u32],
    coefficients: &'c mut [Coefficient],
    width: usize,
    height: usize,
    stride: usize,
    bit_position: u8,
}

impl<'c, 'r, 'd> RawScan<'c, 'r, 'd> {
    fn new(ctx: &'c mut BitPlaneDecodeContext, bits: &'r mut RawBits<'d>) -> Self {
        Self {
            bits,
            width: ctx.width as usize,
            height: ctx.height as usize,
            stride: ctx.padded_width as usize,
            bit_position: ctx.current_bit_position,
            flags: &mut ctx.flags,
            coefficients: &mut ctx.coefficients,
        }
    }

    fn stripe_start(&self, stripe: usize) -> (usize, usize, u32) {
        let rows = (self.height - 4 * stripe).min(4);
        (
            (stripe + 1) * self.stride + 1,
            (4 * stripe + 1) * self.stride + 1,
            ROW_BITS & ((1 << (3 * rows)) - 1),
        )
    }

    #[expect(clippy::inline_always, reason = "Tier-1 coefficient-loop hot path")]
    #[inline(always)]
    fn become_significant(
        &mut self,
        own: &mut u32,
        flag_index: usize,
        top: usize,
        shift: u32,
    ) -> Option<()> {
        let index = top + (shift / 3) as usize * self.stride;
        self.coefficients[index].push_bit_at(1, self.bit_position);
        let sign = self.bits.read()?;
        self.coefficients[index].set_sign(u8::from(sign != 0));
        set_significant(self.flags, own, flag_index, self.stride, shift, sign);
        Some(())
    }

    #[expect(clippy::inline_always, reason = "Tier-1 coefficient-loop hot path")]
    #[inline(always)]
    fn zero_code(
        &mut self,
        own: &mut u32,
        flag_index: usize,
        top: usize,
        shift: u32,
    ) -> Option<bool> {
        let significant = self.bits.read()? != 0;
        if significant {
            self.become_significant(own, flag_index, top, shift)?;
        }
        Some(significant)
    }
}

/// The significance propagation pass of a raw (bypass) segment, with the
/// decisions and order of [`significance_propagation_pass_flags`].
pub(super) fn significance_propagation_pass_raw(
    ctx: &mut BitPlaneDecodeContext,
    bits: &mut RawBits<'_>,
) -> Option<()> {
    let mut scan = RawScan::new(ctx, bits);
    for stripe in 0..scan.height.div_ceil(4) {
        let (flag_row, top_row, rows) = scan.stripe_start(stripe);
        for x in 0..scan.width {
            let flag_index = flag_row + x;
            let mut own = scan.flags[flag_index];
            let mut candidates = propagation_rows(own) & rows;
            if candidates == 0 {
                continue;
            }
            while candidates != 0 {
                let shift = candidates.trailing_zeros();
                own |= PI_THIS << shift;
                if scan.zero_code(&mut own, flag_index, top_row + x, shift)? {
                    candidates = propagation_rows(own) & rows & (u32::MAX << (shift + 1));
                } else {
                    candidates &= candidates - 1;
                }
            }
            scan.flags[flag_index] = own;
        }
    }
    Some(())
}

/// The magnitude refinement pass of a raw (bypass) segment, with the
/// decisions and order of [`magnitude_refinement_pass_flags`].
pub(super) fn magnitude_refinement_pass_raw(
    ctx: &mut BitPlaneDecodeContext,
    bits: &mut RawBits<'_>,
) -> Option<()> {
    let scan = RawScan::new(ctx, bits);
    for stripe in 0..scan.height.div_ceil(4) {
        let (flag_row, top_row, rows) = scan.stripe_start(stripe);
        for x in 0..scan.width {
            let flag_index = flag_row + x;
            let mut own = scan.flags[flag_index];
            let mut candidates =
                (own >> SIGMA_THIS.trailing_zeros()) & !(own >> PI_THIS.trailing_zeros()) & rows;
            if candidates == 0 {
                continue;
            }
            while candidates != 0 {
                let shift = candidates.trailing_zeros();
                let bit = scan.bits.read()?;
                scan.coefficients[top_row + x + (shift / 3) as usize * scan.stride]
                    .push_bit_at(bit, scan.bit_position);
                own |= MU_THIS << shift;
                candidates &= candidates - 1;
            }
            scan.flags[flag_index] = own;
        }
    }
    Some(())
}

/// The cleanup pass of a raw segment, with the decisions and order of
/// [`cleanup_pass_flags`]. Selective bypass codes cleanup passes with the MQ
/// coder; a raw segment only reaches one when its pass counts say so.
pub(super) fn cleanup_pass_raw(
    ctx: &mut BitPlaneDecodeContext,
    bits: &mut RawBits<'_>,
) -> Option<()> {
    let mut scan = RawScan::new(ctx, bits);
    for stripe in 0..scan.height.div_ceil(4) {
        let (flag_row, top_row, rows) = scan.stripe_start(stripe);
        for x in 0..scan.width {
            let flag_index = flag_row + x;
            let top = top_row + x;
            let mut own = scan.flags[flag_index];
            if rows == ROW_BITS && own == 0 {
                if scan.bits.read()? == 0 {
                    continue;
                }
                let first = (scan.bits.read()? << 1) | scan.bits.read()?;
                scan.become_significant(&mut own, flag_index, top, 3 * first)?;
                for row in first + 1..4 {
                    scan.zero_code(&mut own, flag_index, top, 3 * row)?;
                }
            } else {
                let mut candidates = uncoded_rows(own) & rows;
                while candidates != 0 {
                    scan.zero_code(&mut own, flag_index, top, candidates.trailing_zeros())?;
                    candidates &= candidates - 1;
                }
            }
            scan.flags[flag_index] = own & !PI_ALL;
        }
    }
    Some(())
}
