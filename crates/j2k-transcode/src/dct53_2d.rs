// SPDX-License-Identifier: MIT OR Apache-2.0

//! Constrained 2D DCT to 5/3 wavelet experiments.
//!
//! The direct float path projects an 8x8 DCT block into one separable
//! single-level 5/3 result without first storing the 8x8 spatial samples. The
//! reference path materializes samples to keep the oracle easy to audit.

use j2k_codec_math::dwt::{
    linearized_dwt53_row, Dwt53Band, DWT53_MAX_HIGH_LINEAR_TAPS, DWT53_MAX_LINEAR_TAPS,
};

use crate::allocation::{
    checked_add_allocation_bytes, checked_allocation_bytes, checked_allocation_len,
    checked_capacity_bytes, try_vec_filled, try_vec_reserve_len, try_vec_resize_with,
    try_vec_with_capacity, TranscodeAllocationError,
};
use crate::dct_grid::{high_len, idct8_basis, low_len, validate_dct_block_grid};
use crate::{DctTransformError, Dwt53TwoDimensional};

#[cfg(test)]
impl Dwt53TwoDimensional<f64> {
    /// Maximum absolute coefficient difference across matching bands.
    #[must_use]
    pub(crate) fn max_abs_diff(&self, other: &Self) -> f64 {
        assert_eq!(self.low_width, other.low_width);
        assert_eq!(self.low_height, other.low_height);
        assert_eq!(self.high_width, other.high_width);
        assert_eq!(self.high_height, other.high_height);

        self.ll
            .iter()
            .zip(other.ll.iter())
            .chain(self.hl.iter().zip(other.hl.iter()))
            .chain(self.lh.iter().zip(other.lh.iter()))
            .chain(self.hh.iter().zip(other.hh.iter()))
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max)
    }
}

/// Scratch storage for repeated DCT-grid to 5/3 projection calls.
///
/// Reuse one value per worker when transforming many components or tiles with
/// matching geometry. The scratch caches linearized 5/3 weight rows; it does
/// not store spatial samples.
#[derive(Debug, Default)]
pub(crate) struct Dct53GridScratch {
    geometry: Option<(usize, usize)>,
    x_weights: Dwt53WeightRows,
    y_weights: Dwt53WeightRows,
}

impl Dct53GridScratch {
    pub(crate) fn retained_bytes(&self) -> Result<usize, TranscodeAllocationError> {
        checked_add_allocation_bytes(
            self.x_weights.retained_bytes()?,
            self.y_weights.retained_bytes()?,
        )
    }

    #[cfg(test)]
    fn weight_row_capacity(&self) -> usize {
        self.x_weights.weight_capacity() + self.y_weights.weight_capacity()
    }
}

/// Map an adjacent 8x8 DCT block grid directly into a linearized one-level 2D
/// 5/3 result for the logical component dimensions.
///
/// Padded JPEG edge samples outside `width x height` are ignored.
pub fn dct8x8_blocks_to_dwt53_float_linear(
    blocks: &[[[f64; 8]; 8]],
    block_cols: usize,
    block_rows: usize,
    width: usize,
    height: usize,
) -> Result<Dwt53TwoDimensional<f64>, DctTransformError> {
    let mut scratch = Dct53GridScratch::default();
    dct8x8_blocks_to_dwt53_float_linear_with_scratch(
        blocks,
        block_cols,
        block_rows,
        width,
        height,
        &mut scratch,
    )
}

/// Map an adjacent 8x8 DCT block grid directly into a linearized one-level 2D
/// 5/3 result using caller-owned scratch for reusable weight rows.
pub(crate) fn dct8x8_blocks_to_dwt53_float_linear_with_scratch(
    blocks: &[[[f64; 8]; 8]],
    block_cols: usize,
    block_rows: usize,
    width: usize,
    height: usize,
    scratch: &mut Dct53GridScratch,
) -> Result<Dwt53TwoDimensional<f64>, DctTransformError> {
    validate_grid(blocks.len(), block_cols, block_rows, width, height)?;
    validate_direct_workspace(width, height)?;

    if scratch.geometry != Some((width, height)) {
        *scratch = Dct53GridScratch::default();
        scratch.geometry = Some((width, height));
    }

    let low_width = low_len(width);
    let low_height = low_len(height);
    let high_width = high_len(width);
    let high_height = high_len(height);
    scratch.x_weights.ensure_sample_len(width)?;
    scratch.y_weights.ensure_sample_len(height)?;
    let x_weights = &scratch.x_weights;
    let y_weights = &scratch.y_weights;

    let mut ll = try_vec_with_capacity(checked_allocation_len::<f64>(low_width, low_height)?)?;
    let mut hl = try_vec_with_capacity(checked_allocation_len::<f64>(high_width, low_height)?)?;
    let mut lh = try_vec_with_capacity(checked_allocation_len::<f64>(low_width, high_height)?)?;
    let mut hh = try_vec_with_capacity(checked_allocation_len::<f64>(high_width, high_height)?)?;

    for y in 0..low_height {
        for x in 0..low_width {
            ll.push(project_dct_grid(
                blocks,
                block_cols,
                &y_weights.low[y].taps,
                &x_weights.low[x].taps,
            ));
        }
        for x in 0..high_width {
            hl.push(project_dct_grid(
                blocks,
                block_cols,
                &y_weights.low[y].taps,
                &x_weights.high[x].taps,
            ));
        }
    }

    for y in 0..high_height {
        for x in 0..low_width {
            lh.push(project_dct_grid(
                blocks,
                block_cols,
                &y_weights.high[y].taps,
                &x_weights.low[x].taps,
            ));
        }
        for x in 0..high_width {
            hh.push(project_dct_grid(
                blocks,
                block_cols,
                &y_weights.high[y].taps,
                &x_weights.high[x].taps,
            ));
        }
    }

    Ok(Dwt53TwoDimensional {
        ll,
        hl,
        lh,
        hh,
        low_width,
        low_height,
        high_width,
        high_height,
    })
}

/// Reference path for a DCT block grid:
/// DCT coefficients -> float IDCT samples -> separable linearized 5/3.
pub fn dct8x8_blocks_then_dwt53_float(
    blocks: &[[[f64; 8]; 8]],
    block_cols: usize,
    block_rows: usize,
    width: usize,
    height: usize,
) -> Result<Dwt53TwoDimensional<f64>, DctTransformError> {
    validate_grid(blocks.len(), block_cols, block_rows, width, height)?;
    let sample_count = checked_allocation_len::<f64>(width, height)?;
    validate_reference_workspace(sample_count, height)?;

    let mut samples = try_vec_with_capacity(sample_count)?;
    for y in 0..height {
        let block_y = y / 8;
        let local_y = y % 8;
        for x in 0..width {
            let block_x = x / 8;
            let local_x = x % 8;
            let block = &blocks[block_y * block_cols + block_x];
            samples.push(idct8x8_sample(block, local_x, local_y));
        }
    }

    linearized_53_2d_from_plane(&samples, width, height)
}

fn project_dct_grid(
    blocks: &[[[f64; 8]; 8]],
    block_cols: usize,
    y_weights: &[SparseWeightTap],
    x_weights: &[SparseWeightTap],
) -> f64 {
    let mut output = 0.0;

    for &SparseWeightTap {
        sample_idx: sample_y,
        weight: y_weight,
    } in y_weights
    {
        let block_y = sample_y / 8;
        let local_y = sample_y % 8;

        for &SparseWeightTap {
            sample_idx: sample_x,
            weight: x_weight,
        } in x_weights
        {
            let block_x = sample_x / 8;
            let local_x = sample_x % 8;
            let block = &blocks[block_y * block_cols + block_x];
            let sample_weight = y_weight * x_weight;

            for (freq_y, coefficient_row) in block.iter().enumerate() {
                let y_basis = idct8_basis(local_y, freq_y);
                for (freq_x, coefficient) in coefficient_row.iter().copied().enumerate() {
                    output += sample_weight * y_basis * idct8_basis(local_x, freq_x) * coefficient;
                }
            }
        }
    }

    output
}

fn idct8x8_sample(block: &[[f64; 8]; 8], x: usize, y: usize) -> f64 {
    let mut sample = 0.0;
    for (freq_y, row) in block.iter().enumerate() {
        let y_basis = idct8_basis(y, freq_y);
        for (freq_x, coefficient) in row.iter().copied().enumerate() {
            sample += coefficient * y_basis * idct8_basis(x, freq_x);
        }
    }
    sample
}

pub(crate) fn linearized_53_2d_from_plane(
    samples: &[f64],
    width: usize,
    height: usize,
) -> Result<Dwt53TwoDimensional<f64>, DctTransformError> {
    if width == 0 || height == 0 {
        return Err(DctTransformError::InvalidSamplePlaneDimensions { width, height });
    }
    let sample_count = checked_allocation_len::<f64>(width, height)?;
    if samples.len() != sample_count {
        return Err(DctTransformError::SamplePlaneLengthMismatch {
            sample_count: samples.len(),
            width,
            height,
        });
    }
    validate_plane_workspace(sample_count, height)?;

    let low_width = low_len(width);
    let low_height = low_len(height);
    let high_width = high_len(width);
    let high_height = high_len(height);

    let mut row_low = try_vec_with_capacity(checked_allocation_len::<f64>(height, low_width)?)?;
    let mut row_high = try_vec_with_capacity(checked_allocation_len::<f64>(height, high_width)?)?;
    {
        let mut transformed = Dwt53OneDimensional {
            low: try_vec_with_capacity(low_width)?,
            high: try_vec_with_capacity(high_width)?,
        };
        for row in samples.chunks_exact(width) {
            linearized_53_into(row, &mut transformed);
            row_low.extend_from_slice(&transformed.low);
            row_high.extend_from_slice(&transformed.high);
        }
    }

    let mut ll = try_vec_filled(checked_allocation_len::<f64>(low_width, low_height)?, 0.0)?;
    let mut lh = try_vec_filled(checked_allocation_len::<f64>(low_width, high_height)?, 0.0)?;
    let mut hl = try_vec_filled(checked_allocation_len::<f64>(high_width, low_height)?, 0.0)?;
    let mut hh = try_vec_filled(checked_allocation_len::<f64>(high_width, high_height)?, 0.0)?;
    let mut column = try_vec_with_capacity(height)?;
    let mut transformed = Dwt53OneDimensional {
        low: try_vec_with_capacity(low_height)?,
        high: try_vec_with_capacity(high_height)?,
    };
    for x in 0..low_width {
        fill_column(&row_low, low_width, x, height, &mut column);
        linearized_53_into(&column, &mut transformed);
        store_column(&transformed.low, low_width, x, &mut ll);
        store_column(&transformed.high, low_width, x, &mut lh);
    }

    for x in 0..high_width {
        fill_column(&row_high, high_width, x, height, &mut column);
        linearized_53_into(&column, &mut transformed);
        store_column(&transformed.low, high_width, x, &mut hl);
        store_column(&transformed.high, high_width, x, &mut hh);
    }

    Ok(Dwt53TwoDimensional {
        ll,
        hl,
        lh,
        hh,
        low_width,
        low_height,
        high_width,
        high_height,
    })
}

fn fill_column(rows: &[f64], stride: usize, x: usize, height: usize, column: &mut Vec<f64>) {
    column.clear();
    for y in 0..height {
        column.push(rows[y * stride + x]);
    }
}

fn store_column(column: &[f64], stride: usize, x: usize, band: &mut [f64]) {
    for (y, value) in column.iter().copied().enumerate() {
        band[y * stride + x] = value;
    }
}

fn linearized_53_into(samples: &[f64], output: &mut Dwt53OneDimensional) {
    output.high.clear();
    for odd_idx in (1..samples.len()).step_by(2) {
        let left = samples[odd_idx - 1];
        let right = samples.get(odd_idx + 1).copied().unwrap_or(left);
        output.high.push(samples[odd_idx] - ((left + right) * 0.5));
    }

    output.low.clear();
    for even_idx in (0..samples.len()).step_by(2) {
        let current = samples[even_idx];
        let even_output_idx = even_idx / 2;
        let left_high = even_output_idx
            .checked_sub(1)
            .and_then(|idx| output.high.get(idx));
        let right_high = output.high.get(even_output_idx);
        let update = match (left_high, right_high) {
            (Some(left), Some(right)) => (*left + *right) * 0.25,
            (None, Some(right)) => *right * 0.5,
            (Some(left), None) => *left * 0.5,
            (None, None) => 0.0,
        };
        output.low.push(current + update);
    }
}

fn validate_grid(
    block_count: usize,
    block_cols: usize,
    block_rows: usize,
    width: usize,
    height: usize,
) -> Result<(), DctTransformError> {
    validate_dct_block_grid(block_count, block_cols, block_rows, width, height)?;
    Ok(())
}

fn validate_direct_workspace(width: usize, height: usize) -> Result<(), DctTransformError> {
    let sample_count = checked_allocation_len::<f64>(width, height)?;
    let x_weight_bytes = weight_workspace_bytes(width)?;
    let y_weight_bytes = weight_workspace_bytes(height)?;
    let retained_weight_bytes = checked_add_allocation_bytes(x_weight_bytes, y_weight_bytes)?;

    let output_bytes = checked_allocation_bytes::<f64>(sample_count)?;
    checked_add_allocation_bytes(retained_weight_bytes, output_bytes)?;
    Ok(())
}

fn validate_reference_workspace(
    sample_count: usize,
    height: usize,
) -> Result<(), DctTransformError> {
    let sample_bytes = checked_allocation_bytes::<f64>(sample_count)?;
    checked_add_allocation_bytes(sample_bytes, plane_workspace_bytes(sample_count, height)?)?;
    Ok(())
}

fn validate_plane_workspace(sample_count: usize, height: usize) -> Result<(), DctTransformError> {
    plane_workspace_bytes(sample_count, height)?;
    Ok(())
}

fn plane_workspace_bytes(sample_count: usize, height: usize) -> Result<usize, DctTransformError> {
    let row_and_output_bytes = allocation_product_bytes::<f64>(sample_count, 2)?;
    let column_and_split_bytes = allocation_product_bytes::<f64>(height, 2)?;
    Ok(checked_add_allocation_bytes(
        row_and_output_bytes,
        column_and_split_bytes,
    )?)
}

fn weight_workspace_bytes(sample_len: usize) -> Result<usize, DctTransformError> {
    let row_bytes = checked_allocation_bytes::<SparseWeightRow>(sample_len)?;
    let low_tap_bytes =
        allocation_product_bytes::<SparseWeightTap>(low_len(sample_len), DWT53_MAX_LINEAR_TAPS)?;
    let high_tap_bytes = allocation_product_bytes::<SparseWeightTap>(
        high_len(sample_len),
        DWT53_MAX_HIGH_LINEAR_TAPS,
    )?;
    let tap_bytes = checked_add_allocation_bytes(low_tap_bytes, high_tap_bytes)?;
    Ok(checked_add_allocation_bytes(row_bytes, tap_bytes)?)
}

fn allocation_product_bytes<T>(left: usize, right: usize) -> Result<usize, DctTransformError> {
    let element_count = checked_allocation_len::<T>(left, right)?;
    Ok(checked_allocation_bytes::<T>(element_count)?)
}

#[derive(Debug, Default)]
struct Dwt53WeightRows {
    sample_len: Option<usize>,
    low: Vec<SparseWeightRow>,
    high: Vec<SparseWeightRow>,
}

impl Dwt53WeightRows {
    fn retained_bytes(&self) -> Result<usize, TranscodeAllocationError> {
        let mut total = checked_add_allocation_bytes(
            checked_capacity_bytes::<SparseWeightRow>(self.low.capacity())?,
            checked_capacity_bytes::<SparseWeightRow>(self.high.capacity())?,
        )?;
        for row in self.low.iter().chain(&self.high) {
            total = checked_add_allocation_bytes(
                total,
                checked_capacity_bytes::<SparseWeightTap>(row.taps.capacity())?,
            )?;
        }
        Ok(total)
    }

    fn ensure_sample_len(&mut self, sample_len: usize) -> Result<(), DctTransformError> {
        if self.sample_len == Some(sample_len) {
            return Ok(());
        }

        *self = Self::default();
        resize_weight_rows(&mut self.low, low_len(sample_len), DWT53_MAX_LINEAR_TAPS)?;
        resize_weight_rows(
            &mut self.high,
            high_len(sample_len),
            DWT53_MAX_HIGH_LINEAR_TAPS,
        )?;
        write_symbolic_weight_rows(&mut self.low, sample_len, Dwt53Band::Low)?;
        write_symbolic_weight_rows(&mut self.high, sample_len, Dwt53Band::High)?;

        self.sample_len = Some(sample_len);
        Ok(())
    }

    #[cfg(test)]
    fn weight_capacity(&self) -> usize {
        self.low
            .iter()
            .map(|row| row.taps.capacity())
            .sum::<usize>()
            + self
                .high
                .iter()
                .map(|row| row.taps.capacity())
                .sum::<usize>()
    }
}

fn write_symbolic_weight_rows(
    rows: &mut [SparseWeightRow],
    sample_len: usize,
    band: Dwt53Band,
) -> Result<(), DctTransformError> {
    for (output_index, row) in rows.iter_mut().enumerate() {
        let symbolic = linearized_dwt53_row(sample_len, band, output_index).ok_or(
            DctTransformError::SymbolicWeightIndexOutOfRange {
                sample_len,
                output_index,
                high_pass: matches!(band, Dwt53Band::High),
            },
        )?;
        for tap in symbolic.taps() {
            push_weight_tap(
                row,
                SparseWeightTap {
                    sample_idx: tap.sample_index(),
                    weight: tap.weight(),
                },
            )?;
        }
    }
    Ok(())
}

fn resize_weight_rows(
    rows: &mut Vec<SparseWeightRow>,
    row_count: usize,
    max_taps: usize,
) -> Result<(), DctTransformError> {
    try_vec_resize_with(rows, row_count, SparseWeightRow::default)?;
    for row in rows.iter_mut().take(row_count) {
        row.taps.clear();
        try_vec_reserve_len(&mut row.taps, max_taps)?;
    }
    rows.truncate(row_count);
    Ok(())
}

fn push_weight_tap(
    row: &mut SparseWeightRow,
    tap: SparseWeightTap,
) -> Result<(), DctTransformError> {
    let required_len =
        row.taps
            .len()
            .checked_add(1)
            .ok_or(DctTransformError::MemoryCapExceeded {
                requested: usize::MAX,
                cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            })?;
    try_vec_reserve_len(&mut row.taps, required_len)?;
    row.taps.push(tap);
    Ok(())
}

#[derive(Debug, Default)]
struct SparseWeightRow {
    taps: Vec<SparseWeightTap>,
}

#[derive(Debug, Clone, Copy)]
struct SparseWeightTap {
    sample_idx: usize,
    weight: f64,
}

struct Dwt53OneDimensional {
    low: Vec<f64>,
    high: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_workspace_rejects_aggregate_before_any_single_vector_hits_cap() {
        let cap = j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;
        let sample_count = cap / core::mem::size_of::<f64>() / 3 + 1;
        assert!(checked_allocation_bytes::<f64>(sample_count).is_ok());
        assert!(matches!(
            validate_reference_workspace(sample_count, 1),
            Err(DctTransformError::MemoryCapExceeded { requested, cap: limit })
                if requested > limit && limit == cap
        ));
    }

    #[test]
    fn dct8x8_grid_scratch_reuses_weight_rows_for_same_geometry() {
        let blocks = synthetic_grid_blocks(2, 2);
        let mut scratch = Dct53GridScratch::default();

        let direct =
            dct8x8_blocks_to_dwt53_float_linear_with_scratch(&blocks, 2, 2, 13, 11, &mut scratch)
                .expect("valid DCT grid");
        let stateless =
            dct8x8_blocks_to_dwt53_float_linear(&blocks, 2, 2, 13, 11).expect("valid DCT grid");
        let capacity_after_first = scratch.weight_row_capacity();

        let repeated =
            dct8x8_blocks_to_dwt53_float_linear_with_scratch(&blocks, 2, 2, 13, 11, &mut scratch)
                .expect("valid DCT grid");

        assert!(capacity_after_first > 0);
        assert_eq!(scratch.weight_row_capacity(), capacity_after_first);
        assert!(direct.max_abs_diff(&stateless) <= 1.0e-9);
        assert!(repeated.max_abs_diff(&stateless) <= 1.0e-9);
    }

    #[test]
    fn dct8x8_grid_scratch_uses_sparse_weight_rows_for_wsi_tile() {
        let dim = 224_usize;
        let block_cols = dim / 8;
        let block_rows = dim / 8;
        let blocks = vec![[[0.0; 8]; 8]; block_cols * block_rows];
        let mut scratch = Dct53GridScratch::default();

        dct8x8_blocks_to_dwt53_float_linear_with_scratch(
            &blocks,
            block_cols,
            block_rows,
            dim,
            dim,
            &mut scratch,
        )
        .expect("valid DCT grid");

        assert!(
            scratch.weight_row_capacity() <= dim * 10,
            "5/3 grid weights should stay sparse at WSI tile sizes, got capacity {}",
            scratch.weight_row_capacity()
        );
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "small deterministic test-grid indices are exactly representable in f64"
    )]
    fn synthetic_grid_blocks(block_cols: usize, block_rows: usize) -> Vec<[[f64; 8]; 8]> {
        let mut blocks = Vec::with_capacity(block_cols * block_rows);
        for block_y in 0..block_rows {
            for block_x in 0..block_cols {
                let mut block = [[0.0; 8]; 8];
                block[0][0] = 192.0 + (block_x * 17 + block_y * 23) as f64;
                block[0][1] = -31.0 + block_x as f64;
                block[1][0] = 27.0 - block_y as f64;
                block[2][3] = 9.0;
                block[7][7] = -6.0;
                blocks.push(block);
            }
        }
        blocks
    }
}
