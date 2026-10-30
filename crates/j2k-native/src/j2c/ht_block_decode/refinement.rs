// SPDX-License-Identifier: MIT OR Apache-2.0

use super::readers::{read_u32_pair, ReverseBitReader};

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

    let mut magref = ReverseBitReader::new_mrp(refinement_data);
    let half = 1u32 << (p - 2);

    for y in (0..height).step_by(4) {
        let mut cur_sig_idx = (y >> 2) as usize * mstr;
        let dpp = (y * stride) as usize;
        // One bit per row of this stripe that lies inside the code block.
        let rows_in_block = (1u32 << (height - y).min(4)) - 1;

        for i in (0..width).step_by(8) {
            let cwd = magref.fetch();
            let sig = read_u32_pair(sigma, cur_sig_idx);
            cur_sig_idx += 2;
            // The cleanup pass can mark samples of a quad that lies across
            // the block edge as significant when the codestream is malformed.
            // Those samples still consume their refinement bits, but, as in
            // the cleanup pass, nothing is written for them.
            let in_block =
                (0x1111_1111u32 * rows_in_block) & (u32::MAX >> (32 - 4 * (width - i).min(8)));
            // Significant samples take refinement bits in bit order: column
            // by column, top to bottom (bit `4 * column + row`).
            let base = dpp + i as usize;
            let mut pending = sig;
            let mut cwd_mut = cwd;
            while pending != 0 {
                let bit = pending.trailing_zeros();
                pending &= pending - 1;
                if in_block & (1 << bit) != 0 {
                    let sym = ((1 - (cwd_mut & 1)) << (p - 1)) | half;
                    decoded_data
                        [base + (bit >> 2) as usize + (bit & 3) as usize * stride as usize] ^= sym;
                }
                cwd_mut >>= 1;
            }

            magref.advance(sig.count_ones());
        }
    }

    Some(())
}
