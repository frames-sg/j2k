// SPDX-License-Identifier: MIT OR Apache-2.0

use super::readers::{read_u32_pair, ForwardBitReader};

pub(super) const SIGPROP_SPREAD_MASKS: [u32; 16] = [
    0x33, 0x76, 0xEC, 0xC8, 0x330, 0x760, 0xEC0, 0xC80, 0x3300, 0x7600, 0xEC00, 0xC800, 0x33000,
    0x76000, 0xEC000, 0xC8000,
];

pub(crate) fn sigma_stride(width: u32) -> usize {
    ((width.div_ceil(4) + 2 + 7) & !7) as usize
}

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
                        candidates |= SIGPROP_SPREAD_MASKS[bit as usize] & inv_sig & !processed;
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

    Some(())
}
