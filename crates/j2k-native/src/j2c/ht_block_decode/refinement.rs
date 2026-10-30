// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "simd")]
use fearless_simd::{u32x4, u32x8, Simd, SimdBase};

use super::readers::{read_u32_pair, ReverseBitReader};
#[cfg(feature = "simd")]
use crate::math::{dispatch, Level};

#[expect(
    clippy::inline_always,
    clippy::too_many_arguments,
    reason = "the magnitude-refinement scan is a hot stable phase boundary with explicit geometry"
)]
#[inline(always)]
pub(super) fn apply_magnitude_refinement_phase(
    refinement_data: &[u8],
    sigma: &[u16],
    decoded_data: &mut [u32],
    width: u32,
    height: u32,
    stride: u32,
    mstr: usize,
    p: u32,
) -> Option<()> {
    if p < 2 {
        return None;
    }
    let geometry = (width, height, stride as usize, mstr);

    #[cfg(feature = "simd")]
    {
        dispatch!(Level::new(), simd => refine_stripes(
            refinement_data,
            sigma,
            decoded_data,
            geometry,
            p,
            |data: &mut [u32], base, sig, cwd| refine_whole_group(simd, data, base, stride as usize, sig, cwd, p),
        ))
    }
    #[cfg(not(feature = "simd"))]
    {
        refine_stripes(
            refinement_data,
            sigma,
            decoded_data,
            geometry,
            p,
            |data: &mut [u32], base, sig, cwd| {
                refine_group_bits(data, base, stride as usize, sig, u32::MAX, cwd, p);
                Some(())
            },
        )
    }
}

/// The MagRef scan over 8x4 groups of the block: `whole_group` refines a
/// group that lies entirely inside the block, the rest go sample by sample.
#[expect(
    clippy::inline_always,
    reason = "inlined into the dispatched SIMD level"
)]
#[inline(always)]
fn refine_stripes(
    refinement_data: &[u8],
    sigma: &[u16],
    decoded_data: &mut [u32],
    (width, height, stride, mstr): (u32, u32, usize, usize),
    p: u32,
    mut whole_group: impl FnMut(&mut [u32], usize, u32, u32) -> Option<()>,
) -> Option<()> {
    let mut magref = ReverseBitReader::new_mrp(refinement_data);

    for y in (0..height).step_by(4) {
        let mut cur_sig_idx = (y >> 2) as usize * mstr;
        let dpp = y as usize * stride;
        // One bit per row of this stripe that lies inside the code block.
        let rows_in_block = (1u32 << (height - y).min(4)) - 1;

        for i in (0..width).step_by(8) {
            let cwd = magref.fetch();
            let sig = read_u32_pair(sigma, cur_sig_idx);
            cur_sig_idx += 2;
            let base = dpp + i as usize;
            if sig != 0 {
                if rows_in_block == 0xF && i + 8 <= width {
                    whole_group(decoded_data, base, sig, cwd)?;
                } else {
                    // The cleanup pass can mark samples of a quad that lies
                    // across the block edge as significant when the
                    // codestream is malformed. Those samples still consume
                    // their refinement bits, but, as in the cleanup pass,
                    // nothing is written for them.
                    let in_block = (0x1111_1111u32 * rows_in_block)
                        & (u32::MAX >> (32 - 4 * (width - i).min(8)));
                    refine_group_bits(decoded_data, base, stride, sig, in_block, cwd, p);
                }
            }

            magref.advance(sig.count_ones());
        }
    }

    Some(())
}

/// Refine one 8x4 group sample by sample. Significant samples take
/// refinement bits in bit order: column by column, top to bottom (bit
/// `4 * column + row`); only those in `in_block` are written.
#[expect(clippy::inline_always, reason = "per-group hot path")]
#[inline(always)]
fn refine_group_bits(
    decoded_data: &mut [u32],
    base: usize,
    stride: usize,
    sig: u32,
    in_block: u32,
    cwd: u32,
    p: u32,
) {
    let half = 1u32 << (p - 2);
    let mut pending = sig;
    let mut cwd_mut = cwd;
    while pending != 0 {
        let bit = pending.trailing_zeros();
        pending &= pending - 1;
        if in_block & (1 << bit) != 0 {
            let sym = ((1 - (cwd_mut & 1)) << (p - 1)) | half;
            decoded_data[base + (bit >> 2) as usize + (bit & 3) as usize * stride] ^= sym;
        }
        cwd_mut >>= 1;
    }
}

/// [`refine_group_bits`] for a group inside the block, one row of eight
/// samples per vector and no per-sample branches: each sample's refinement
/// bit sits at its rank among the group's significant samples in bit order.
#[cfg(feature = "simd")]
#[expect(clippy::inline_always, reason = "per-group hot path")]
#[inline(always)]
fn refine_whole_group<S: Simd>(
    simd: S,
    decoded_data: &mut [u32],
    base: usize,
    stride: usize,
    sig: u32,
    cwd: u32,
    p: u32,
) -> Option<()> {
    let per_column = |values: [u32; 8]| u32x8::from_slice(simd, &values);
    let zero = u32x8::splat(simd, 0);
    // Column `c`'s four significance bits (rows 0-3) in lane `c`.
    let nibbles = (u32x8::splat(simd, sig) >> per_column([0, 4, 8, 12, 16, 20, 24, 28])) & 0xF;
    let flags = [0, 1, 2, 3].map(|row| (nibbles >> row) & 1);
    // Bits taken by the columns before each column: an exclusive prefix sum
    // of the column counts, carried from the low four lanes to the high four.
    let counts = flags[0] + flags[1] + flags[2] + flags[3];
    let pairs = counts + zero.slide_within_blocks::<3>(counts);
    let inclusive = pairs + zero.slide_within_blocks::<2>(pairs);
    let (low, high) = simd.split_u32x8(inclusive);
    let high = high + u32x4::splat(simd, low.to_array()[3]);
    let mut rank = simd.combine_u32x4(low, high) - counts;

    let bits = u32x8::splat(simd, cwd);
    // The symbol of a zero refinement bit; a one clears bit `p - 1`.
    let zero_bit_symbol = (1u32 << (p - 1)) | (1u32 << (p - 2));
    for (row, flag) in flags.into_iter().enumerate() {
        let refinement = (bits >> rank) & 1;
        let symbol =
            (u32x8::splat(simd, zero_bit_symbol) - (refinement << (p - 1))) & (zero - flag);
        let at = base + row * stride;
        let samples = decoded_data.get_mut(at..at + 8)?;
        (u32x8::from_slice(simd, samples) ^ symbol).store_slice(samples);
        rank += flag;
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::apply_magnitude_refinement_phase;
    use crate::j2c::ht_block_decode::significance::sigma_stride;

    /// The MagRef scan as first written: eight columns by four rows of
    /// branches per group.
    fn reference(
        refinement: &[u8],
        sigma: &[u16],
        decoded: &mut [u32],
        (width, height, stride, mstr): (u32, u32, u32, usize),
        p: u32,
    ) {
        let mut magref = super::ReverseBitReader::new_mrp(refinement);
        let half = 1u32 << (p - 2);
        for y in (0..height).step_by(4) {
            let mut cur = (y >> 2) as usize * mstr;
            let dpp = (y * stride) as usize;
            let rows_in_block = (1u32 << (height - y).min(4)) - 1;
            for i in (0..width).step_by(8) {
                let mut cwd = magref.fetch();
                let sig = super::read_u32_pair(sigma, cur);
                cur += 2;
                let in_block =
                    (0x1111_1111u32 * rows_in_block) & (u32::MAX >> (32 - 4 * (width - i).min(8)));
                for j in 0..8 {
                    for row in 0..4 {
                        let bit = 4 * j + row;
                        if sig & (1 << bit) != 0 {
                            if in_block & (1 << bit) != 0 {
                                let sym = ((1 - (cwd & 1)) << (p - 1)) | half;
                                decoded[dpp + i as usize + j as usize + (row * stride) as usize] ^=
                                    sym;
                            }
                            cwd >>= 1;
                        }
                    }
                }
                magref.advance(sig.count_ones());
            }
        }
    }

    #[test]
    fn magnitude_refinement_matches_the_per_sample_scan() {
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
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
            (8, 4, 8),
            (5, 9, 5),
        ] {
            for case in 0..40 {
                let mstr = sigma_stride(width);
                let sigma_rows = height.div_ceil(4) as usize + 1;
                let density = case % 4;
                let sigma: Vec<u16> = (0..sigma_rows * mstr)
                    .map(|_| {
                        let word = next();
                        let mut bits = word as u16;
                        for shift in 0..density {
                            bits &= (word >> (16 * (shift + 1))) as u16;
                        }
                        bits
                    })
                    .collect();
                let refinement: Vec<u8> = (0..4096).map(|_| next() as u8).collect();
                let len = ((height - 1) * stride + width) as usize;
                let original: Vec<u32> = (0..len).map(|_| (next() as u32) & 0x7FFF_FFF0).collect();
                let p = 5 + (case % 20) as u32;
                let mut expected = original.clone();
                reference(
                    &refinement,
                    &sigma,
                    &mut expected,
                    (width, height, stride, mstr),
                    p,
                );
                let mut decoded = original.clone();
                apply_magnitude_refinement_phase(
                    &refinement,
                    &sigma,
                    &mut decoded,
                    width,
                    height,
                    stride,
                    mstr,
                    p,
                )
                .expect("magref");
                assert_eq!(
                    decoded, expected,
                    "{width}x{height} stride {stride} case {case}"
                );
            }
        }
    }
}
