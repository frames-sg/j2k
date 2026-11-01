// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "simd")]
use fearless_simd::{u32x4, Simd, SimdBase};

use super::readers::{read_u32_pair, ForwardBitReader};
#[cfg(feature = "simd")]
use crate::math::{dispatch, Level};

/// The samples each newly significant sample of a 4x4 group makes a `SigProp`
/// candidate, as bits `4 * column + row`: itself, the sample below it, and
/// the three nearest of the next column. The reference for
/// [`SIGPROP_COLUMN`], and the order the encoder visits.
#[cfg(test)]
pub(super) const SIGPROP_SPREAD_MASKS: [u32; 16] = [
    0x33, 0x76, 0xEC, 0xC8, 0x330, 0x760, 0xEC0, 0xC80, 0x3300, 0x7600, 0xEC00, 0xC800, 0x33000,
    0x76000, 0xEC000, 0xC8000,
];

/// `SigProp` decoding of one four-sample column of a stripe, indexed by
/// `bits | insignificant << 4 | candidates << 8`: the next four refinement
/// bits, the column's samples not yet significant, and its candidates. A
/// candidate takes one bit, top to bottom; a one makes it significant and
/// the insignificant sample below it a candidate. Each entry is
/// `used | ones << 5 | spread << 8 | significant << 12`: the bits used (bits
/// 3-4 stay clear, so the entry is also the shift past them), the new
/// samples' count, the candidates they add to the next column (beside and
/// diagonally next to them), and the new samples. `spread` sits where the
/// next column's index takes its candidates, so one lookup feeds the next
/// through a mask.
const SIGPROP_COLUMN: [u16; 4096] = sigprop_column_table();

const fn sigprop_column_table() -> [u16; 4096] {
    let mut table = [0; 4096];
    let mut index = 0;
    while index < 4096 {
        let bits = index & 0xF;
        let insignificant = (index >> 4) & 0xF;
        let mut candidates = index >> 8;
        let mut used = 0_u16;
        let mut significant = 0_u16;
        let mut row = 0;
        while row < 4 {
            if candidates & (1 << row) != 0 {
                if (bits >> used) & 1 != 0 {
                    significant |= 1 << row;
                    candidates |= (1 << (row + 1)) & insignificant;
                }
                used += 1;
            }
            row += 1;
        }
        let spread = (significant | (significant << 1) | (significant >> 1)) & 0xF;
        let ones = (significant & 1)
            + ((significant >> 1) & 1)
            + ((significant >> 2) & 1)
            + (significant >> 3);
        table[index] = used | (ones << 5) | (spread << 8) | (significant << 12);
        index += 1;
    }
    table
}

/// The significance bits of one 4x4 `SigProp` group: `candidates` are the
/// samples next to a sample significant before this pass, `insignificant`
/// the samples that can still become significant (bits `4 * column + row`),
/// and `cwd` the group's refinement bits. Newly significant samples make
/// their later neighbors candidates; the scan is column by column, top to
/// bottom. Returns the newly significant samples, the significance bits
/// used, and the count of new samples (one sign bit each).
#[expect(clippy::inline_always, reason = "per-group hot path")]
#[inline(always)]
pub(super) fn decode_group_significance(
    candidates: u32,
    insignificant: u32,
    cwd: u32,
) -> (u32, u32, u32) {
    let mut bits = cwd;
    let mut significant = 0;
    // Bits used in bits 0-4 and new samples from bit 5: each column adds at
    // most four of either.
    let mut sums = 0;
    // The previous column's entry; its `spread` field is in index position.
    let mut previous = 0;
    for column in 0..4 {
        let shift = 4 * column;
        let column_insignificant = (insignificant >> shift) & 0xF;
        let column_candidates = (candidates >> shift) & 0xF;
        let index = (previous & (column_insignificant << 8))
            | (column_candidates << 8)
            | (column_insignificant << 4)
            | (bits & 0xF);
        let entry = u32::from(SIGPROP_COLUMN[index as usize]);
        bits = bits.wrapping_shr(entry);
        sums += entry & 0xE7;
        significant |= (entry >> 12) << shift;
        previous = entry;
    }
    (significant, sums & 0x1F, sums >> 5)
}

pub(crate) fn sigma_stride(width: u32) -> usize {
    ((width.div_ceil(4) + 2 + 7) & !7) as usize
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::inline_always,
    reason = "the assembled cleanup significance mask is exactly two packed 16-bit groups in a hot scan"
)]
#[inline(always)]
pub(super) fn build_sigma_from_cleanup_phase(
    cleanup: &[u16],
    sigma: &mut [u16],
    width: u32,
    height: u32,
    sstr: usize,
    mstr: usize,
) -> Option<()> {
    let sigma_rows = height.div_ceil(4) as usize + 1;
    if sigma.len() < sigma_rows * mstr {
        return None;
    }

    let mut y = 0u32;
    while y < height {
        let sp_base = (y >> 1) as usize * sstr;
        let dp_base = (y >> 2) as usize * mstr;
        let mut x = 0u32;
        let mut sp = sp_base;
        let mut dp = dp_base;
        while x < width {
            let mut t0 =
                ((u32::from(cleanup[sp]) & 0x30) >> 4) | ((u32::from(cleanup[sp]) & 0xC0) >> 2);
            t0 |= ((u32::from(cleanup[sp + 2]) & 0x30) << 4)
                | ((u32::from(cleanup[sp + 2]) & 0xC0) << 6);
            let mut t1 = ((u32::from(cleanup[sp + sstr]) & 0x30) >> 2)
                | (u32::from(cleanup[sp + sstr]) & 0xC0);
            t1 |= ((u32::from(cleanup[sp + sstr + 2]) & 0x30) << 6)
                | ((u32::from(cleanup[sp + sstr + 2]) & 0xC0) << 8);
            sigma[dp] = (t0 | t1) as u16;
            x += 4;
            sp += 4;
            dp += 1;
        }
        sigma[dp] = 0;
        y += 4;
    }

    let dp_base = (height.div_ceil(4) as usize) * mstr;
    for x in 0..=width.div_ceil(4) as usize {
        sigma[dp_base + x] = 0;
    }

    Some(())
}

#[expect(
    clippy::inline_always,
    clippy::too_many_arguments,
    reason = "the significance scan is a hot stable phase boundary with explicit geometry"
)]
#[inline(always)]
pub(super) fn apply_significance_propagation_phase(
    refinement_data: &[u8],
    sigma: &[u16],
    decoded_data: &mut [u32],
    width: u32,
    height: u32,
    stride: u32,
    mstr: usize,
    stripe_causal: bool,
    p: u32,
    prev_row_sig: &mut [u16],
) -> Option<()> {
    let geometry = (width, height, stride, mstr);
    #[cfg(feature = "simd")]
    {
        dispatch!(Level::new(), simd => propagate_stripes(
            refinement_data,
            sigma,
            decoded_data,
            geometry,
            stripe_causal,
            p,
            prev_row_sig,
            |data: &mut [u32], base, new_sig, cwd, value| {
                write_group_signs(simd, data, base, stride as usize, new_sig, cwd, value)
            },
        ))
    }
    #[cfg(not(feature = "simd"))]
    {
        propagate_stripes(
            refinement_data,
            sigma,
            decoded_data,
            geometry,
            stripe_causal,
            p,
            prev_row_sig,
            |data: &mut [u32], base, new_sig, cwd, value| {
                write_signs_by_bit(data, base, stride as usize, new_sig, cwd, value);
                Some(())
            },
        )
    }
}

/// The new samples of a `SigProp` group with their sign bits, in bit order
/// (`4 * column + row`), one sample at a time.
#[expect(clippy::inline_always, reason = "per-group hot path")]
#[inline(always)]
fn write_signs_by_bit(
    decoded_data: &mut [u32],
    base: usize,
    stride: usize,
    new_sig: u32,
    cwd: u32,
    value: u32,
) {
    let mut sign_bits = new_sig;
    let mut cwd = cwd;
    while sign_bits != 0 {
        let bit = sign_bits.trailing_zeros();
        sign_bits &= sign_bits - 1;
        decoded_data[base + (bit >> 2) as usize + (bit & 3) as usize * stride] =
            ((cwd & 1) << 31) | value;
        cwd >>= 1;
    }
}

/// [`write_signs_by_bit`] for a group inside the block, one row of four
/// samples per vector: each new sample's sign bit sits at its rank among the
/// group's new samples in bit order, and the other samples keep their value.
/// The per-sample loop ends on a hard-to-predict branch; this has none.
#[cfg(feature = "simd")]
#[expect(clippy::inline_always, reason = "per-group hot path")]
#[inline(always)]
fn write_group_signs<S: Simd>(
    simd: S,
    decoded_data: &mut [u32],
    base: usize,
    stride: usize,
    new_sig: u32,
    cwd: u32,
    value: u32,
) -> Option<()> {
    let zero = u32x4::splat(simd, 0);
    // Column `c`'s four new-sample bits (rows 0-3) in lane `c`.
    let nibbles = (u32x4::splat(simd, new_sig) >> u32x4::from_slice(simd, &[0, 4, 8, 12])) & 0xF;
    let flags = [0, 1, 2, 3].map(|row| (nibbles >> row) & 1);
    // Sign bits taken by the columns before each column.
    let counts = flags[0] + flags[1] + flags[2] + flags[3];
    let pairs = counts + zero.slide_within_blocks::<3>(counts);
    let inclusive = pairs + zero.slide_within_blocks::<2>(pairs);
    let mut rank = inclusive - counts;
    let bits = u32x4::splat(simd, cwd);
    for (row, flag) in flags.into_iter().enumerate() {
        let new_value = (((bits >> rank) & 1) << 31) | value;
        let keep = flag - 1;
        let at = base + row * stride;
        let samples = decoded_data.get_mut(at..at + 4)?;
        let old = u32x4::from_slice(simd, samples);
        ((old & keep) | (new_value & !keep)).store_slice(samples);
        rank += flag;
    }
    Some(())
}

/// The `SigProp` scan over 4x4 groups: `whole_group_signs` writes the new
/// samples of a group that lies entirely inside the block, given the group's
/// new samples, the bits after its significance bits, and the sample value.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::inline_always,
    clippy::too_many_arguments,
    reason = "the hot significance scan uses bounded block coordinates and stores the low packed mask half"
)]
#[inline(always)]
fn propagate_stripes(
    refinement_data: &[u8],
    sigma: &[u16],
    decoded_data: &mut [u32],
    (width, height, stride, mstr): (u32, u32, u32, usize),
    stripe_causal: bool,
    p: u32,
    prev_row_sig: &mut [u16],
    mut whole_group_signs: impl FnMut(&mut [u32], usize, u32, u32, u32) -> Option<()>,
) -> Option<()> {
    if prev_row_sig.len() < width.div_ceil(4) as usize + 8 {
        return None;
    }

    prev_row_sig.fill(0);
    let mut sigprop = ForwardBitReader::<0>::new(refinement_data);
    let stride_us = stride as usize;

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
        let dpp = (y * stride) as usize;

        for x in (0..width).step_by(4) {
            let mut col_pattern = pattern;
            let mut s = x as i32 + 4 - width as i32;
            s = s.max(0);
            col_pattern >>= (s * 4) as u32;

            let idx = (x >> 2) as usize;
            let ps = u32::from(prev_row_sig[idx]) | (u32::from(prev_row_sig[idx + 1]) << 16);
            let ns = read_u32_pair(sigma, next_row + idx);
            let mut u = (ps & 0x8888_8888) >> 3;
            if !stripe_causal {
                u |= (ns & 0x1111_1111) << 3;
            }

            let cs = read_u32_pair(sigma, cur_row + idx);
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
                let mut cwd = sigprop.fetch();
                let (significant, used, ones) =
                    decode_group_significance(mbr, !cs & col_pattern, cwd);
                new_sig = significant;
                cwd >>= used;

                if new_sig != 0 {
                    let value = 3u32 << (p - 2);
                    let block_base = dpp + x as usize;
                    if x + 4 <= width && y + 4 <= height {
                        whole_group_signs(decoded_data, block_base, new_sig, cwd, value)?;
                    } else {
                        write_signs_by_bit(
                            decoded_data,
                            block_base,
                            stride_us,
                            new_sig,
                            cwd,
                            value,
                        );
                    }
                }

                sigprop.advance(used + ones);
            }

            let combined_sig = new_sig | (cs & 0xFFFF);
            prev_row_sig[idx] = combined_sig as u16;

            let t = combined_sig;
            let mut next_prev = combined_sig;
            next_prev |= (t & 0x7777) << 1;
            next_prev |= (t & 0xEEEE) >> 1;
            prev = (next_prev | u) & 0xF000;
        }
    }

    Some(())
}

#[cfg(test)]
mod tests {
    use alloc::{vec, vec::Vec};

    use super::{apply_significance_propagation_phase, sigma_stride};
    use crate::j2c::ht_block_decode::readers::{read_u32_pair, ForwardBitReader};

    /// The `SigProp` phase as first written (candidate scan and per-sample
    /// sign loop), the reference for the column table and vector signs.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::too_many_arguments,
        reason = "the test reference keeps the phase's original scalar arithmetic and signature"
    )]
    fn reference(
        refinement_data: &[u8],
        sigma: &[u16],
        decoded_data: &mut [u32],
        width: u32,
        height: u32,
        stride: u32,
        mstr: usize,
        stripe_causal: bool,
        p: u32,
        prev_row_sig: &mut [u16],
    ) -> Option<()> {
        if prev_row_sig.len() < width.div_ceil(4) as usize + 8 {
            return None;
        }

        prev_row_sig.fill(0);
        let mut sigprop = ForwardBitReader::<0>::new(refinement_data);
        let stride_us = stride as usize;

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
            let dpp = (y * stride) as usize;

            for x in (0..width).step_by(4) {
                let mut col_pattern = pattern;
                let mut s = x as i32 + 4 - width as i32;
                s = s.max(0);
                col_pattern >>= (s * 4) as u32;

                let idx = (x >> 2) as usize;
                let ps = u32::from(prev_row_sig[idx]) | (u32::from(prev_row_sig[idx + 1]) << 16);
                let ns = read_u32_pair(sigma, next_row + idx);
                let mut u = (ps & 0x8888_8888) >> 3;
                if !stripe_causal {
                    u |= (ns & 0x1111_1111) << 3;
                }

                let cs = read_u32_pair(sigma, cur_row + idx);
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
                    let mut cwd = sigprop.fetch();
                    let mut cnt = 0u32;
                    let inv_sig = !cs & col_pattern;
                    let mut candidates = mbr;
                    let mut processed = 0u32;

                    while candidates != 0 {
                        let bit = candidates.trailing_zeros();
                        let sample_mask = 1u32 << bit;
                        candidates &= !sample_mask;
                        processed |= sample_mask;

                        if (cwd & 1) != 0 {
                            new_sig |= sample_mask;
                            candidates |=
                                super::SIGPROP_SPREAD_MASKS[bit as usize] & inv_sig & !processed;
                        }
                        cwd >>= 1;
                        cnt += 1;
                    }

                    if new_sig != 0 {
                        let value = 3u32 << (p - 2);
                        let block_base = dpp + x as usize;
                        let mut sign_bits = new_sig;

                        while sign_bits != 0 {
                            let bit = sign_bits.trailing_zeros();
                            let sample_mask = 1u32 << bit;
                            sign_bits &= !sample_mask;

                            let offset = (bit >> 2) as usize + ((bit & 3) as usize * stride_us);
                            decoded_data[block_base + offset] = ((cwd & 1) << 31) | value;
                            cwd >>= 1;
                            cnt += 1;
                        }
                    }

                    sigprop.advance(cnt);
                }

                let combined_sig = new_sig | (cs & 0xFFFF);
                prev_row_sig[idx] = combined_sig as u16;

                let t = combined_sig;
                let mut next_prev = combined_sig;
                next_prev |= (t & 0x7777) << 1;
                next_prev |= (t & 0xEEEE) >> 1;
                prev = (next_prev | u) & 0xF000;
            }
        }

        Some(())
    }

    #[test]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the test draws its random fields from the low bits of 64-bit generator words"
    )]
    fn significance_propagation_matches_the_candidate_scan() {
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for (width, height, stride) in [
            (64_u32, 64_u32, 64_u32),
            (64, 64, 70),
            (24, 16, 24),
            (13, 7, 16),
            (4, 4, 4),
            (5, 9, 5),
        ] {
            for case in 0..60_u32 {
                let mstr = sigma_stride(width);
                let sigma_rows = height.div_ceil(4) as usize + 1;
                let density = case % 4;
                let mut sigma: Vec<u16> = (0..sigma_rows * mstr)
                    .map(|_| {
                        let word = next();
                        let mut bits = word as u16;
                        for shift in 0..density {
                            bits &= (word >> (16 * (shift + 1))) as u16;
                        }
                        bits
                    })
                    .collect();
                // As the sigma build leaves them: nothing past the block.
                let groups = width.div_ceil(4) as usize;
                for row in 0..sigma_rows {
                    for group in groups..mstr {
                        sigma[row * mstr + group] = 0;
                    }
                }
                for group in 0..mstr {
                    sigma[(sigma_rows - 1) * mstr + group] = 0;
                }
                let refinement: Vec<u8> = (0..4096).map(|_| next() as u8).collect();
                let len = ((height - 1) * stride + width) as usize;
                let original: Vec<u32> = (0..len).map(|_| (next() as u32) & 0x7FFF_FFF0).collect();
                let p = 5 + case % 20;
                let stripe_causal = case % 3 == 0;
                let mut expected = original.clone();
                let mut prev = vec![0u16; groups + 8];
                reference(
                    &refinement,
                    &sigma,
                    &mut expected,
                    width,
                    height,
                    stride,
                    mstr,
                    stripe_causal,
                    p,
                    &mut prev,
                )
                .expect("reference sigprop");
                let mut decoded = original.clone();
                let mut prev = vec![0u16; groups + 8];
                apply_significance_propagation_phase(
                    &refinement,
                    &sigma,
                    &mut decoded,
                    width,
                    height,
                    stride,
                    mstr,
                    stripe_causal,
                    p,
                    &mut prev,
                )
                .expect("sigprop");
                assert_eq!(
                    decoded, expected,
                    "{width}x{height} stride {stride} case {case}"
                );
            }
        }
    }
}
