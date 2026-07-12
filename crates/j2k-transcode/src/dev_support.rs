// SPDX-License-Identifier: MIT OR Apache-2.0

//! Feature-gated support for workspace tests and benchmarks.
//!
//! This module is not part of the default production API. It exposes narrow
//! adapters around internal scratch storage and shared transform primitives so
//! dev-only consumers do not compile whole production modules by path.

use crate::{DctTransformError, Dwt53TwoDimensional, Dwt97TwoDimensional};

/// Opaque caller-owned scratch for repeated direct 5/3 grid projections.
#[derive(Debug, Default)]
pub struct Dct53GridScratch(crate::dct53_2d::Dct53GridScratch);

/// Opaque caller-owned scratch for repeated 9/7 reference transforms.
#[derive(Debug, Default)]
pub struct Dct97GridScratch(crate::dct97_2d::Dct97GridScratch);

/// Direct 5/3 grid projection with caller-owned scratch.
#[inline]
pub fn dct8x8_blocks_to_dwt53_float_linear_with_scratch(
    blocks: &[[[f64; 8]; 8]],
    block_cols: usize,
    block_rows: usize,
    width: usize,
    height: usize,
    scratch: &mut Dct53GridScratch,
) -> Result<Dwt53TwoDimensional<f64>, DctTransformError> {
    crate::dct53_2d::dct8x8_blocks_to_dwt53_float_linear_with_scratch(
        blocks,
        block_cols,
        block_rows,
        width,
        height,
        &mut scratch.0,
    )
}

/// Reference 9/7 transform with caller-owned scratch.
#[inline]
pub fn dct8x8_blocks_then_dwt97_float_with_scratch(
    blocks: &[[[f64; 8]; 8]],
    block_cols: usize,
    block_rows: usize,
    width: usize,
    height: usize,
    scratch: &mut Dct97GridScratch,
) -> Result<Dwt97TwoDimensional<f64>, DctTransformError> {
    crate::dct97_2d::dct8x8_blocks_then_dwt97_float_with_scratch(
        blocks,
        block_cols,
        block_rows,
        width,
        height,
        &mut scratch.0,
    )
}

/// Low-pass output length for a one-level wavelet split.
#[must_use]
pub const fn low_len(sample_len: usize) -> usize {
    crate::dct_grid::low_len(sample_len)
}

/// High-pass output length for a one-level wavelet split.
#[must_use]
pub const fn high_len(sample_len: usize) -> usize {
    crate::dct_grid::high_len(sample_len)
}

/// Cached orthonormal 8-point inverse-DCT basis coefficient.
#[must_use]
pub fn idct8_basis(sample_idx: usize, frequency: usize) -> f64 {
    crate::dct_grid::idct8_basis(sample_idx, frequency)
}

/// Apply the reversible integer 5/3 lift in place.
pub fn reversible_lift_53_i32(values: &mut [i32]) {
    crate::reversible53::reversible_lift_53_i32(values);
}

/// Apply a conventional linearized 5/3 transform to a sample plane.
pub fn linearized_53_2d_from_plane(
    samples: &[f64],
    width: usize,
    height: usize,
) -> Result<Dwt53TwoDimensional<f64>, DctTransformError> {
    crate::dct53_2d::linearized_53_2d_from_plane(samples, width, height)
}

#[cfg(test)]
mod tests {
    use super::{
        dct8x8_blocks_then_dwt97_float_with_scratch,
        dct8x8_blocks_to_dwt53_float_linear_with_scratch, Dct53GridScratch, Dct97GridScratch,
    };

    #[test]
    fn scratch_adapters_match_stateless_transform_paths() {
        let mut block = [[0.0; 8]; 8];
        block[0][0] = 384.0;
        block[0][1] = -31.0;
        block[1][0] = 27.0;
        block[7][7] = -6.0;
        let blocks = [block];

        let expected_53 =
            crate::dct8x8_blocks_to_dwt53_float_linear(&blocks, 1, 1, 8, 8).expect("valid grid");
        let actual_53 = dct8x8_blocks_to_dwt53_float_linear_with_scratch(
            &blocks,
            1,
            1,
            8,
            8,
            &mut Dct53GridScratch::default(),
        )
        .expect("valid grid");
        assert!(actual_53.max_abs_diff(&expected_53) <= f64::EPSILON);

        let expected_97 =
            crate::dct8x8_blocks_then_dwt97_float(&blocks, 1, 1, 8, 8).expect("valid grid");
        let actual_97 = dct8x8_blocks_then_dwt97_float_with_scratch(
            &blocks,
            1,
            1,
            8,
            8,
            &mut Dct97GridScratch::default(),
        )
        .expect("valid grid");
        assert!(actual_97.max_abs_diff(&expected_97) <= 1.0e-9);
    }
}
