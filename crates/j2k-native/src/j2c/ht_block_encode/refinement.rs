// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::{vec, vec::Vec};

use super::super::ht_block_decode::sigma_stride;

const SIGPROP_SPREAD_MASKS: [u32; 16] = [
    0x33, 0x76, 0xEC, 0xC8, 0x330, 0x760, 0xEC0, 0xC80, 0x3300, 0x7600, 0xEC00, 0xC800, 0x33000,
    0x76000, 0xEC000, 0xC8000,
];

struct ForwardRefinementBitWriter {
    data: Vec<u8>,
    used_bits: u8,
    max_bits: u8,
    tmp: u8,
}

impl ForwardRefinementBitWriter {
    fn new() -> Self {
        Self {
            data: Vec::new(),
            used_bits: 0,
            max_bits: 8,
            tmp: 0,
        }
    }

    fn push_bit(&mut self, bit: bool) {
        if bit {
            self.tmp |= 1 << self.used_bits;
        }
        self.used_bits += 1;
        if self.used_bits == self.max_bits {
            self.flush_full_byte();
        }
    }

    fn flush_full_byte(&mut self) {
        self.data.push(self.tmp);
        self.max_bits = if self.tmp == 0xFF { 7 } else { 8 };
        self.tmp = 0;
        self.used_bits = 0;
    }

    fn finish(mut self) -> Vec<u8> {
        if self.used_bits > 0 {
            self.data.push(self.tmp);
        }
        if self.data.is_empty() {
            self.data.push(0);
        }
        self.data
    }
}

struct ReverseRefinementBitWriter {
    bits: Vec<bool>,
}

impl ReverseRefinementBitWriter {
    fn new() -> Self {
        Self { bits: Vec::new() }
    }

    fn push_bit(&mut self, bit: bool) {
        self.bits.push(bit);
    }

    fn finish(self) -> Vec<u8> {
        let mut read_order = Vec::new();
        let mut offset = 0usize;
        let mut unstuff = true;

        while offset < self.bits.len() {
            let remaining = self.bits.len() - offset;
            let first_seven_are_ones =
                remaining >= 7 && self.bits[offset..offset + 7].iter().all(|bit| *bit);
            let capacity = if unstuff && first_seven_are_ones {
                7
            } else {
                8
            };
            let take = capacity.min(remaining);
            let mut byte = 0u8;
            for bit_idx in 0..take {
                if self.bits[offset + bit_idx] {
                    byte |= 1 << bit_idx;
                }
            }
            read_order.push(byte);
            offset += take;
            unstuff = byte > 0x8F;
        }

        if read_order.is_empty() {
            read_order.push(0);
        }
        read_order.reverse();
        read_order
    }
}

pub(super) fn encode_refinement_segment(
    coefficients: &[i32],
    width: usize,
    height: usize,
    cleanup_significance_threshold: i32,
    num_coding_passes: u8,
) -> Result<Vec<u8>, &'static str> {
    let width_u32 = u32::try_from(width).map_err(|_| "HTJ2K code-block width exceeds u32 range")?;
    let height_u32 =
        u32::try_from(height).map_err(|_| "HTJ2K code-block height exceeds u32 range")?;
    let mstr = sigma_stride(width_u32);
    let sigma_rows = height_u32.div_ceil(4) as usize + 1;
    let mut sigma = vec![0u16; sigma_rows * mstr];
    build_sigma_from_coefficients(
        coefficients,
        width,
        height,
        cleanup_significance_threshold,
        mstr,
        &mut sigma,
    )?;
    let mut refinement =
        write_sigprop_refinement_bits(&sigma, coefficients, width_u32, height_u32, mstr)?;
    if num_coding_passes > 2 {
        let magref =
            write_magref_refinement_bits(&sigma, coefficients, width_u32, height_u32, mstr)?;
        refinement.extend_from_slice(&magref);
    }
    Ok(refinement)
}

fn build_sigma_from_coefficients(
    coefficients: &[i32],
    width: usize,
    height: usize,
    significance_threshold: i32,
    mstr: usize,
    sigma: &mut [u16],
) -> Result<(), &'static str> {
    if coefficients.len() < width.saturating_mul(height) {
        return Err("HTJ2K coefficient block is shorter than its dimensions");
    }

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
                let row = y
                    .checked_mul(width)
                    .ok_or("HTJ2K coefficient row offset overflow")?;
                for dx in 0..4 {
                    let x = group_x * 4 + dx;
                    if x >= width {
                        continue;
                    }
                    if coefficients[row + x].unsigned_abs() >= significance_threshold as u32 {
                        bits |= 1u16 << (dx * 4 + dy);
                    }
                }
            }
            sigma[sigma_row + group_x] = bits;
        }
    }

    Ok(())
}

fn write_sigprop_refinement_bits(
    sigma: &[u16],
    coefficients: &[i32],
    width: u32,
    height: u32,
    mstr: usize,
) -> Result<Vec<u8>, &'static str> {
    let mut prev_row_sig = vec![0u16; width.div_ceil(4) as usize + 8];
    let mut writer = ForwardRefinementBitWriter::new();
    let width_usize = width as usize;

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

                    let coeff = coefficient_for_sigprop_bit(coefficients, width_usize, x, y, bit)?;
                    let significant = coeff != 0;
                    writer.push_bit(significant);
                    if significant {
                        new_sig |= sample_mask;
                        candidates |= SIGPROP_SPREAD_MASKS[bit as usize] & inv_sig & !processed;
                    }
                }

                let mut sign_bits = new_sig;
                while sign_bits != 0 {
                    let bit = sign_bits.trailing_zeros();
                    sign_bits &= !(1u32 << bit);
                    let coeff = coefficient_for_sigprop_bit(coefficients, width_usize, x, y, bit)?;
                    writer.push_bit(coeff < 0);
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

    Ok(writer.finish())
}

fn write_magref_refinement_bits(
    sigma: &[u16],
    coefficients: &[i32],
    width: u32,
    height: u32,
    mstr: usize,
) -> Result<Vec<u8>, &'static str> {
    let mut writer = ReverseRefinementBitWriter::new();
    let width_usize = width as usize;

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
                                let coeff = coefficient_for_sigprop_bit(
                                    coefficients,
                                    width_usize,
                                    x,
                                    y,
                                    bit,
                                )?;
                                writer.push_bit((coeff.unsigned_abs() & 1) != 0);
                            }
                            sample_mask <<= 1;
                        }
                    }
                    col_mask <<= 4;
                }
            }
        }
    }

    Ok(writer.finish())
}

fn coefficient_for_sigprop_bit(
    coefficients: &[i32],
    width: usize,
    group_x: u32,
    group_y: u32,
    bit: u32,
) -> Result<i32, &'static str> {
    let x = group_x as usize + (bit >> 2) as usize;
    let y = group_y as usize + (bit & 3) as usize;
    let idx = y
        .checked_mul(width)
        .and_then(|row| row.checked_add(x))
        .ok_or("HTJ2K sigprop coefficient offset overflow")?;
    coefficients
        .get(idx)
        .copied()
        .ok_or("HTJ2K sigprop coefficient offset out of range")
}

fn read_sigma_pair(values: &[u16], index: usize) -> Result<u32, &'static str> {
    let high_index = index
        .checked_add(1)
        .ok_or("HTJ2K sigma pair offset overflow")?;
    if high_index >= values.len() {
        return Err("HTJ2K sigma pair is out of range");
    }
    Ok(u32::from(values[index]) | (u32::from(values[high_index]) << 16))
}

#[cfg(test)]
mod tests {
    use super::{ForwardRefinementBitWriter, ReverseRefinementBitWriter};

    #[test]
    fn refinement_writer_stuffing_matches_pre_split_goldens() {
        let mut forward = ForwardRefinementBitWriter::new();
        for bit in [
            true, true, true, true, true, true, true, true, true, false, true, false, true, false,
            true, true, false,
        ] {
            forward.push_bit(bit);
        }
        assert_eq!(forward.finish(), [0xFF, 0x55, 0x01]);

        let mut reverse = ReverseRefinementBitWriter::new();
        for bit in [
            true, true, true, true, true, true, true, true, false, true, false, true, false, true,
            true, false,
        ] {
            reverse.push_bit(bit);
        }
        assert_eq!(reverse.finish(), [0x00, 0xD5, 0x7F]);
    }
}
