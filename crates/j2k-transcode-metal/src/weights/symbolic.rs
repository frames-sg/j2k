// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    shared::{WaveletBand, WaveletKind, ALPHA, BETA, DELTA, GAMMA, INV_KAPPA, KAPPA},
    SparseWeightRow, SparseWeightRowsError, SparseWeightTap,
};
use j2k_codec_math::dwt::{linearized_dwt53_row, Dwt53Band};

const MAX_SYMBOLIC_TAPS: usize = 16;

#[derive(Clone, Copy)]
struct LinearWeightTap {
    sample_idx: usize,
    weight: f64,
}

impl LinearWeightTap {
    const ZERO: Self = Self {
        sample_idx: 0,
        weight: 0.0,
    };
}

#[derive(Clone, Copy)]
struct LinearWeightRow {
    taps: [LinearWeightTap; MAX_SYMBOLIC_TAPS],
    len: usize,
}

impl LinearWeightRow {
    fn unit(sample_idx: usize, weight: f64) -> Self {
        let mut row = Self {
            taps: [LinearWeightTap::ZERO; MAX_SYMBOLIC_TAPS],
            len: 1,
        };
        row.taps[0] = LinearWeightTap { sample_idx, weight };
        row
    }

    fn add(&mut self, sample_idx: usize, weight: f64) -> Result<(), SparseWeightRowsError> {
        if weight == 0.0 {
            return Ok(());
        }
        if let Some(position) = self.taps[..self.len]
            .iter()
            .position(|tap| tap.sample_idx == sample_idx)
        {
            self.taps[position].weight += weight;
            if self.taps[position].weight == 0.0 {
                self.remove(position);
            }
            return Ok(());
        }
        if self.len == MAX_SYMBOLIC_TAPS {
            return Err(SparseWeightRowsError::SizeOverflow);
        }
        let position = self.taps[..self.len]
            .iter()
            .position(|tap| tap.sample_idx > sample_idx)
            .unwrap_or(self.len);
        for index in (position..self.len).rev() {
            self.taps[index + 1] = self.taps[index];
        }
        self.taps[position] = LinearWeightTap { sample_idx, weight };
        self.len += 1;
        Ok(())
    }

    fn remove(&mut self, position: usize) {
        for index in position + 1..self.len {
            self.taps[index - 1] = self.taps[index];
        }
        self.len -= 1;
        self.taps[self.len] = LinearWeightTap::ZERO;
    }

    fn targets(&self, odd: bool) -> ([LinearWeightTap; MAX_SYMBOLIC_TAPS], usize) {
        let mut targets = [LinearWeightTap::ZERO; MAX_SYMBOLIC_TAPS];
        let mut count = 0usize;
        for &tap in &self.taps[..self.len] {
            if (tap.sample_idx % 2 == 1) == odd {
                targets[count] = tap;
                count += 1;
            }
        }
        (targets, count)
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "Metal projection tables intentionally store scalar f64 weights in the f32 shader ABI"
)]
pub(super) fn write_symbolic_row(
    output: &mut SparseWeightRow,
    sample_len: usize,
    output_index: usize,
    band: WaveletBand,
    wavelet: WaveletKind,
) -> Result<(), SparseWeightRowsError> {
    if matches!(wavelet, WaveletKind::Reversible53) {
        return write_dwt53_row(output, sample_len, output_index, band);
    }
    let output_sample_idx = output_index
        .checked_mul(2)
        .and_then(|index| index.checked_add(usize::from(matches!(band, WaveletBand::High))))
        .ok_or(SparseWeightRowsError::SizeOverflow)?;
    let symbolic = symbolic_row_97(sample_len, output_sample_idx)?;
    for tap in &symbolic.taps[..symbolic.len] {
        push_tap(output, tap.sample_idx, tap.weight as f32)?;
    }
    Ok(())
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "Metal projection tables intentionally store scalar f64 weights in the f32 shader ABI"
)]
fn write_dwt53_row(
    output: &mut SparseWeightRow,
    sample_len: usize,
    output_index: usize,
    band: WaveletBand,
) -> Result<(), SparseWeightRowsError> {
    let band = match band {
        WaveletBand::Low => Dwt53Band::Low,
        WaveletBand::High => Dwt53Band::High,
    };
    let row = linearized_dwt53_row(sample_len, band, output_index)
        .ok_or(SparseWeightRowsError::SizeOverflow)?;
    for tap in row.taps() {
        push_tap(output, tap.sample_index(), tap.weight() as f32)?;
    }
    Ok(())
}

fn push_tap(
    output: &mut SparseWeightRow,
    sample_idx: usize,
    weight: f32,
) -> Result<(), SparseWeightRowsError> {
    if weight.to_bits() == 0 {
        return Ok(());
    }
    if output.taps.len() == output.taps.capacity() {
        return Err(SparseWeightRowsError::SizeOverflow);
    }
    output.taps.push(SparseWeightTap { sample_idx, weight });
    Ok(())
}

fn symbolic_row_97(
    sample_len: usize,
    output_sample_idx: usize,
) -> Result<LinearWeightRow, SparseWeightRowsError> {
    if sample_len < 2 {
        return Ok(LinearWeightRow::unit(output_sample_idx, 1.0));
    }
    let scale = if output_sample_idx.is_multiple_of(2) {
        INV_KAPPA
    } else {
        KAPPA
    };
    let mut row = LinearWeightRow::unit(output_sample_idx, scale);
    reverse_even_stage(&mut row, sample_len, DELTA)?;
    reverse_odd_stage(&mut row, sample_len, GAMMA)?;
    reverse_even_stage(&mut row, sample_len, BETA)?;
    reverse_odd_stage(&mut row, sample_len, ALPHA)?;
    Ok(row)
}

fn reverse_odd_stage(
    row: &mut LinearWeightRow,
    sample_len: usize,
    coefficient: f64,
) -> Result<(), SparseWeightRowsError> {
    let last_even = if sample_len.is_multiple_of(2) {
        sample_len - 2
    } else {
        sample_len - 1
    };
    let (targets, count) = row.targets(true);
    for target in &targets[..count] {
        let left = target.sample_idx - 1;
        let right = if target.sample_idx + 1 < sample_len {
            target.sample_idx + 1
        } else {
            last_even
        };
        row.add(left, target.weight * coefficient)?;
        row.add(right, target.weight * coefficient)?;
    }
    Ok(())
}

fn reverse_even_stage(
    row: &mut LinearWeightRow,
    sample_len: usize,
    coefficient: f64,
) -> Result<(), SparseWeightRowsError> {
    let (targets, count) = row.targets(false);
    for target in &targets[..count] {
        let left = if target.sample_idx > 0 {
            target.sample_idx - 1
        } else {
            1
        };
        let right = if target.sample_idx + 1 < sample_len {
            target.sample_idx + 1
        } else {
            left
        };
        row.add(left, target.weight * coefficient)?;
        row.add(right, target.weight * coefficient)?;
    }
    Ok(())
}
