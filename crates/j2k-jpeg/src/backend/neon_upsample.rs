// SPDX-License-Identifier: MIT OR Apache-2.0

//! NEON fancy 4:2:0 chroma row upsampling.

use core::arch::aarch64::{
    vaddq_u16, vcombine_u8, vdup_n_u8, vdupq_n_u16, vdupq_n_u8, vget_low_u8, vmlal_high_u8,
    vmlal_u8, vmlaq_n_u16, vmovl_high_u8, vmovl_u8, vshrn_n_u16, vzip1q_u8, vzip2q_u8,
};

use crate::simd::neon_memory;

fearless_simd::kernel! {
    /// Fill the leading interior pairs of a fancy 4:2:0 chroma row, sixteen
    /// chroma samples at a time. With column sums `cs(i) = 3 * curr[i] +
    /// near[i]`, pair `i` is `interior[2i] = (3 * cs(i) + cs(i + 1) + 7) >> 4`
    /// and `interior[2i + 1] = (3 * cs(i + 1) + cs(i) + 8) >> 4`, exactly as
    /// libjpeg-turbo's `h2v2_fancy_upsample`. Returns how many pairs were
    /// filled; the caller finishes the rest.
    pub(crate) fn h2v2_fancy_pairs(
        neon: Neon,
        curr: &[u8],
        near: &[u8],
        interior: &mut [u8],
    ) -> usize {
        h2v2_fancy_pairs_neon(curr, near, interior)
    }
}

#[target_feature(enable = "neon")]
fn h2v2_fancy_pairs_neon(curr: &[u8], near: &[u8], interior: &mut [u8]) -> usize {
    let pairs = interior.len() / 2;
    let mut done = 0;
    // Each step reads samples `done..done + 17` and writes 32 outputs.
    while done + 16 <= pairs {
        let (Some(c0), Some(c1), Some(n0), Some(n1)) = (
            curr.get(done..).and_then(<[u8]>::first_chunk::<16>),
            curr.get(done + 1..).and_then(<[u8]>::first_chunk::<16>),
            near.get(done..).and_then(<[u8]>::first_chunk::<16>),
            near.get(done + 1..).and_then(<[u8]>::first_chunk::<16>),
        ) else {
            break;
        };
        let Some((lo, rest)) = interior[2 * done..].split_first_chunk_mut::<16>() else {
            break;
        };
        let Some(hi) = rest.first_chunk_mut::<16>() else {
            break;
        };
        let (c0, c1) = (neon_memory::load_u8x16(c0), neon_memory::load_u8x16(c1));
        let (n0, n1) = (neon_memory::load_u8x16(n0), neon_memory::load_u8x16(n1));
        let (three, three_q) = (vdup_n_u8(3), vdupq_n_u8(3));
        // Column sums of samples `i` (left) and `i + 1` (right), in u16.
        let left_lo = vmlal_u8(vmovl_u8(vget_low_u8(n0)), vget_low_u8(c0), three);
        let left_hi = vmlal_high_u8(vmovl_high_u8(n0), c0, three_q);
        let right_lo = vmlal_u8(vmovl_u8(vget_low_u8(n1)), vget_low_u8(c1), three);
        let right_hi = vmlal_high_u8(vmovl_high_u8(n1), c1, three_q);
        // At most 3 * 1020 + 1020 + 8 = 4088, so the u16 lanes cannot wrap.
        let (seven, eight) = (vdupq_n_u16(7), vdupq_n_u16(8));
        let odd = vcombine_u8(
            vshrn_n_u16::<4>(vaddq_u16(vmlaq_n_u16(right_lo, left_lo, 3), seven)),
            vshrn_n_u16::<4>(vaddq_u16(vmlaq_n_u16(right_hi, left_hi, 3), seven)),
        );
        let even = vcombine_u8(
            vshrn_n_u16::<4>(vaddq_u16(vmlaq_n_u16(left_lo, right_lo, 3), eight)),
            vshrn_n_u16::<4>(vaddq_u16(vmlaq_n_u16(left_hi, right_hi, 3), eight)),
        );
        neon_memory::store_u8x16(lo, vzip1q_u8(odd, even));
        neon_memory::store_u8x16(hi, vzip2q_u8(odd, even));
        done += 16;
    }
    done
}
