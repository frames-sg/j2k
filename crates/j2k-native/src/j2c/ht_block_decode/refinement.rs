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

        for i in (0..width).step_by(8) {
            let cwd = magref.fetch();
            let sig = read_u32_pair(sigma, cur_sig_idx);
            cur_sig_idx += 2;
            let mut col_mask = 0xFu32;
            let mut cwd_mut = cwd;

            if sig != 0 {
                for j in 0..8 {
                    if (sig & col_mask) != 0 {
                        let mut dp = dpp + i as usize + j;
                        let mut sample_mask = 0x1111_1111u32 & col_mask;

                        for _ in 0..4 {
                            if (sig & sample_mask) != 0 {
                                let mut sym = cwd_mut & 1;
                                sym = (1 - sym) << (p - 1);
                                sym |= half;
                                decoded_data[dp] ^= sym;
                                cwd_mut >>= 1;
                            }
                            sample_mask <<= 1;
                            dp += stride as usize;
                        }
                    }
                    col_mask <<= 4;
                }
            }

            magref.advance(sig.count_ones());
        }
    }

    Some(())
}
