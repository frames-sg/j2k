// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use super::super::ht_block_decode::sigma_stride;
use super::allocation::HtWorkerAllocation;
use crate::j2c::coefficient_view::CoefficientBlockView;
use crate::j2c::encode::allocation::{try_untracked_vec, try_untracked_vec_filled};
use crate::{EncodeError, EncodeResult};

const SIGPROP_SPREAD_MASKS: [u32; 16] = [
    0x33, 0x76, 0xEC, 0xC8, 0x330, 0x760, 0xEC0, 0xC80, 0x3300, 0x7600, 0xEC00, 0xC800, 0x33000,
    0x76000, 0xEC000, 0xC8000,
];

mod writers;
use writers::{ForwardRefinementBitWriter, ReverseRefinementBitWriter};

pub(super) fn try_encode_refinement_segment_view(
    coefficients: CoefficientBlockView<'_, i32>,
    cleanup_significance_threshold: i32,
    num_coding_passes: u8,
    allocation: HtWorkerAllocation,
) -> EncodeResult<Vec<u8>> {
    let width = coefficients.width();
    let height = coefficients.height();
    let width_u32 = u32::try_from(width).map_err(|_| EncodeError::InvalidInput {
        what: "HTJ2K code-block width exceeds u32 range",
    })?;
    let height_u32 = u32::try_from(height).map_err(|_| EncodeError::InvalidInput {
        what: "HTJ2K code-block height exceeds u32 range",
    })?;
    let mstr = sigma_stride(width_u32);
    let sigma_rows = usize::try_from(height_u32.div_ceil(4))
        .map_err(|_| EncodeError::ArithmeticOverflow {
            what: "HTJ2K sigma rows",
        })?
        .checked_add(1)
        .ok_or(EncodeError::ArithmeticOverflow {
            what: "HTJ2K sigma rows",
        })?;
    let sigma_entries = sigma_rows
        .checked_mul(mstr)
        .ok_or(EncodeError::ArithmeticOverflow {
            what: "HTJ2K sigma entries",
        })?;
    if sigma_entries != allocation.sigma_entries {
        return Err(EncodeError::InternalInvariant {
            what: "HTJ2K sigma allocation plan disagrees with geometry",
        });
    }
    let mut sigma = try_untracked_vec_filled(sigma_entries, 0_u16, "HTJ2K sigma significance map")?;
    build_sigma_from_coefficients(
        coefficients,
        cleanup_significance_threshold,
        mstr,
        &mut sigma,
    )
    .map_err(ht_refinement_invariant)?;
    let sigprop = write_sigprop_refinement_bits(
        &sigma,
        coefficients,
        width_u32,
        height_u32,
        mstr,
        allocation,
    )?;
    let magref = if num_coding_passes > 2 {
        write_magref_refinement_bits(
            &sigma,
            coefficients,
            width_u32,
            height_u32,
            mstr,
            allocation,
        )?
    } else {
        Vec::new()
    };
    let combined_len =
        sigprop
            .len()
            .checked_add(magref.len())
            .ok_or(EncodeError::ArithmeticOverflow {
                what: "HTJ2K refinement segment length",
            })?;
    if combined_len > allocation.refinement_bytes {
        return Err(EncodeError::InternalInvariant {
            what: "HTJ2K refinement exceeded its checked bound",
        });
    }
    let mut refinement = try_untracked_vec(combined_len, "HTJ2K refinement segment")?;
    refinement.extend_from_slice(&sigprop);
    refinement.extend_from_slice(&magref);
    Ok(refinement)
}

#[expect(
    clippy::cast_sign_loss,
    reason = "the cleanup significance threshold is constructed as a positive bitplane magnitude"
)]
fn build_sigma_from_coefficients(
    coefficients: CoefficientBlockView<'_, i32>,
    significance_threshold: i32,
    mstr: usize,
    sigma: &mut [u16],
) -> Result<(), &'static str> {
    let width = coefficients.width();
    let height = coefficients.height();

    let group_rows = height.div_ceil(4);
    let group_cols = width.div_ceil(4);
    for group_y in 0..group_rows {
        let sigma_row = group_y
            .checked_mul(mstr)
            .ok_or("HTJ2K sigma row offset overflow")?;
        for group_x in 0..group_cols {
            let mut bits = 0u16;
            for dy in 0..4 {
                let y = group_y * 4 + dy;
                if y >= height {
                    continue;
                }
                for dx in 0..4 {
                    let x = group_x * 4 + dx;
                    if x >= width {
                        continue;
                    }
                    let coefficient = coefficients
                        .get(x, y)
                        .ok_or("HTJ2K sigma coefficient offset out of range")?;
                    if coefficient.unsigned_abs() >= significance_threshold as u32 {
                        bits |= 1u16 << (dx * 4 + dy);
                    }
                }
            }
            sigma[sigma_row + group_x] = bits;
        }
    }

    Ok(())
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "validated block coordinates are bounded and packed significance stores the low 16-bit half"
)]
fn write_sigprop_refinement_bits(
    sigma: &[u16],
    coefficients: CoefficientBlockView<'_, i32>,
    width: u32,
    height: u32,
    mstr: usize,
    allocation: HtWorkerAllocation,
) -> EncodeResult<Vec<u8>> {
    let mut prev_row_sig = try_untracked_vec_filled(
        allocation.previous_sigma_entries,
        0_u16,
        "HTJ2K previous-row significance",
    )?;
    let mut writer = ForwardRefinementBitWriter::try_new(allocation.sigprop_bytes)?;
    for y in (0..height).step_by(4) {
        let mut pattern = 0xFFFFu32;
        if height - y < 4 {
            pattern = 0x7777;
            if height - y < 3 {
                pattern = 0x3333;
                if height - y < 2 {
                    pattern = 0x1111;
                }
            }
        }

        let mut prev = 0u32;
        let cur_row = (y >> 2) as usize * mstr;
        let next_row = cur_row + mstr;

        for x in (0..width).step_by(4) {
            let mut col_pattern = pattern;
            let shift_cols = (x as i32 + 4 - width as i32).max(0) as u32;
            col_pattern >>= shift_cols * 4;

            let idx = (x >> 2) as usize;
            let ps = read_sigma_pair(&prev_row_sig, idx)?;
            let ns = read_sigma_pair(sigma, next_row + idx)?;
            let mut u = (ps & 0x8888_8888) >> 3;
            u |= (ns & 0x1111_1111) << 3;

            let cs = read_sigma_pair(sigma, cur_row + idx)?;
            let mut mbr = cs;
            mbr |= (cs & 0x7777_7777) << 1;
            mbr |= (cs & 0xEEEE_EEEE) >> 1;
            mbr |= u;
            let t = mbr;
            mbr |= t << 4;
            mbr |= t >> 4;
            mbr |= prev >> 12;
            mbr &= col_pattern;
            mbr &= !cs;

            let mut new_sig = 0u32;
            if mbr != 0 {
                let inv_sig = !cs & col_pattern;
                let mut candidates = mbr;
                let mut processed = 0u32;

                while candidates != 0 {
                    let bit = candidates.trailing_zeros();
                    let sample_mask = 1u32 << bit;
                    candidates &= !sample_mask;
                    processed |= sample_mask;

                    let coeff = coefficient_for_sigprop_bit(coefficients, x, y, bit)?;
                    let significant = coeff != 0;
                    writer.push_bit(significant)?;
                    if significant {
                        new_sig |= sample_mask;
                        candidates |= SIGPROP_SPREAD_MASKS[bit as usize] & inv_sig & !processed;
                    }
                }

                let mut sign_bits = new_sig;
                while sign_bits != 0 {
                    let bit = sign_bits.trailing_zeros();
                    sign_bits &= !(1u32 << bit);
                    let coeff = coefficient_for_sigprop_bit(coefficients, x, y, bit)?;
                    writer.push_bit(coeff < 0)?;
                }
            }

            let combined_sig = new_sig | cs;
            prev_row_sig[idx] = combined_sig as u16;
            if idx + 1 < prev_row_sig.len() {
                prev_row_sig[idx + 1] = (combined_sig >> 16) as u16;
            }

            let t = combined_sig;
            let mut next_prev = combined_sig;
            next_prev |= (t & 0x7777) << 1;
            next_prev |= (t & 0xEEEE) >> 1;
            prev = (next_prev | u) & 0xF000;
        }
    }

    writer.finish()
}

fn write_magref_refinement_bits(
    sigma: &[u16],
    coefficients: CoefficientBlockView<'_, i32>,
    width: u32,
    height: u32,
    mstr: usize,
    allocation: HtWorkerAllocation,
) -> EncodeResult<Vec<u8>> {
    let mut writer =
        ReverseRefinementBitWriter::try_new(allocation.magref_bits, allocation.magref_bytes)?;
    for y in (0..height).step_by(4) {
        let mut cur_sig_idx = (y >> 2) as usize * mstr;

        for x in (0..width).step_by(8) {
            let sig = read_sigma_pair(sigma, cur_sig_idx)?;
            cur_sig_idx += 2;
            let mut col_mask = 0xFu32;

            if sig != 0 {
                for _ in 0..8 {
                    if (sig & col_mask) != 0 {
                        let mut sample_mask = 0x1111_1111u32 & col_mask;

                        for _ in 0..4 {
                            if (sig & sample_mask) != 0 {
                                let bit = sample_mask.trailing_zeros();
                                let coeff = coefficient_for_sigprop_bit(coefficients, x, y, bit)?;
                                writer.push_bit((coeff.unsigned_abs() & 1) != 0)?;
                            }
                            sample_mask <<= 1;
                        }
                    }
                    col_mask <<= 4;
                }
            }
        }
    }

    writer.finish()
}

fn coefficient_for_sigprop_bit(
    coefficients: CoefficientBlockView<'_, i32>,
    group_x: u32,
    group_y: u32,
    bit: u32,
) -> EncodeResult<i32> {
    let x = group_x as usize + (bit >> 2) as usize;
    let y = group_y as usize + (bit & 3) as usize;
    coefficients
        .get(x, y)
        .ok_or(EncodeError::InternalInvariant {
            what: "HTJ2K sigprop coefficient offset is out of range",
        })
}

fn read_sigma_pair(values: &[u16], index: usize) -> EncodeResult<u32> {
    let high_index = index
        .checked_add(1)
        .ok_or(EncodeError::ArithmeticOverflow {
            what: "HTJ2K sigma pair offset",
        })?;
    if high_index >= values.len() {
        return Err(EncodeError::InternalInvariant {
            what: "HTJ2K sigma pair is out of range",
        });
    }
    Ok(u32::from(values[index]) | (u32::from(values[high_index]) << 16))
}

fn ht_refinement_invariant(what: &'static str) -> EncodeError {
    EncodeError::InternalInvariant { what }
}
