// SPDX-License-Identifier: MIT OR Apache-2.0

use core::f64::consts::PI;
use core::fmt;
use std::sync::LazyLock;

pub(crate) const fn low_len(sample_len: usize) -> usize {
    sample_len.div_ceil(2)
}

pub(crate) const fn high_len(sample_len: usize) -> usize {
    sample_len / 2
}

pub(crate) fn idct8_basis(sample_idx: usize, freq: usize) -> f64 {
    debug_assert!(sample_idx < 8);
    debug_assert!(freq < 8);

    idct8_basis_table()[sample_idx][freq]
}

pub(crate) fn idct8_basis_table() -> &'static [[f64; 8]; 8] {
    static BASIS: LazyLock<[[f64; 8]; 8]> = LazyLock::new(|| {
        let mut basis = [[0.0; 8]; 8];
        for (sample_idx, row) in basis.iter_mut().enumerate() {
            for (freq, value) in row.iter_mut().enumerate() {
                *value = idct8_basis_uncached(sample_idx, freq);
            }
        }
        basis
    });
    &BASIS
}

fn idct8_basis_uncached(sample_idx: usize, freq: usize) -> f64 {
    let scale = if freq == 0 {
        (1.0_f64 / 8.0).sqrt()
    } else {
        (2.0_f64 / 8.0).sqrt()
    };
    scale * (((sample_idx as f64 + 0.5) * freq as f64 * PI) / 8.0).cos()
}

/// Error returned when a DCT block grid cannot cover the requested component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DctGridError {
    block_count: usize,
    block_cols: usize,
    block_rows: usize,
    width: usize,
    height: usize,
}

impl DctGridError {
    const fn new(
        block_count: usize,
        block_cols: usize,
        block_rows: usize,
        width: usize,
        height: usize,
    ) -> Self {
        Self {
            block_count,
            block_cols,
            block_rows,
            width,
            height,
        }
    }

    /// Number of supplied 8x8 DCT blocks.
    #[must_use]
    pub const fn block_count(self) -> usize {
        self.block_count
    }

    /// Declared block columns.
    #[must_use]
    pub const fn block_cols(self) -> usize {
        self.block_cols
    }

    /// Declared block rows.
    #[must_use]
    pub const fn block_rows(self) -> usize {
        self.block_rows
    }

    /// Requested component width.
    #[must_use]
    pub const fn width(self) -> usize {
        self.width
    }

    /// Requested component height.
    #[must_use]
    pub const fn height(self) -> usize {
        self.height
    }
}

impl fmt::Display for DctGridError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DCT grid has {} blocks for {}x{} grid covering requested {}x{} samples",
            self.block_count, self.block_cols, self.block_rows, self.width, self.height
        )
    }
}

impl std::error::Error for DctGridError {}

pub(crate) fn validate_dct_block_grid(
    block_count: usize,
    block_cols: usize,
    block_rows: usize,
    width: usize,
    height: usize,
) -> Result<(), DctGridError> {
    let expected_blocks = block_cols
        .checked_mul(block_rows)
        .ok_or_else(|| DctGridError::new(block_count, block_cols, block_rows, width, height))?;
    let covered_width = block_cols
        .checked_mul(8)
        .ok_or_else(|| DctGridError::new(block_count, block_cols, block_rows, width, height))?;
    let covered_height = block_rows
        .checked_mul(8)
        .ok_or_else(|| DctGridError::new(block_count, block_cols, block_rows, width, height))?;

    if block_count != expected_blocks
        || width == 0
        || height == 0
        || width > covered_width
        || height > covered_height
    {
        return Err(DctGridError::new(
            block_count,
            block_cols,
            block_rows,
            width,
            height,
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{high_len, idct8_basis, low_len, validate_dct_block_grid};

    #[test]
    fn band_lengths_split_even_and_odd_samples() {
        assert_eq!((low_len(0), high_len(0)), (0, 0));
        assert_eq!((low_len(1), high_len(1)), (1, 0));
        assert_eq!((low_len(8), high_len(8)), (4, 4));
        assert_eq!((low_len(9), high_len(9)), (5, 4));
    }

    #[test]
    fn idct8_basis_is_orthonormal_for_dc_and_first_ac() {
        assert!((idct8_basis(0, 0) - (1.0_f64 / 8.0).sqrt()).abs() < 1e-12);
        assert!((idct8_basis(0, 1) - 0.490_392_640_201_615_2).abs() < 1e-12);
    }

    #[test]
    fn validates_non_empty_grid_covered_by_blocks() {
        assert_eq!(validate_dct_block_grid(6, 3, 2, 24, 16), Ok(()));
    }

    #[test]
    fn rejects_overflowing_grid_dimensions() {
        let err = validate_dct_block_grid(1, usize::MAX, usize::MAX, 8, 8)
            .expect_err("overflowing dimensions must fail");
        assert_eq!(err.block_count(), 1);
        assert_eq!(err.block_cols(), usize::MAX);
        assert_eq!(err.block_rows(), usize::MAX);
    }

    #[test]
    fn rejects_zero_image_extent() {
        assert!(validate_dct_block_grid(1, 1, 1, 0, 8).is_err());
        assert!(validate_dct_block_grid(1, 1, 1, 8, 0).is_err());
    }
}
