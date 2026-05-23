// SPDX-License-Identifier: Apache-2.0

//! Optional acceleration hooks for coefficient-domain transform stages.
//!
//! These hooks are intentionally narrow: accelerated backends may replace the
//! direct DCT-grid to one-level wavelet projection, while the scalar path
//! remains the default oracle and fallback.

use crate::dct53_2d::Dwt53TwoDimensional;
use crate::dct97_2d::Dwt97TwoDimensional;

/// Direct DCT-grid to one-level 5/3 projection job.
#[derive(Debug, Clone, Copy)]
pub struct DctGridToDwt53Job<'a> {
    /// Natural-order, dequantized 8x8 DCT blocks.
    pub blocks: &'a [[[f64; 8]; 8]],
    /// Number of DCT block columns in `blocks`.
    pub block_cols: usize,
    /// Number of DCT block rows in `blocks`.
    pub block_rows: usize,
    /// Logical component width in samples.
    pub width: usize,
    /// Logical component height in samples.
    pub height: usize,
}

/// Direct DCT-grid to one-level 9/7 projection job.
#[derive(Debug, Clone, Copy)]
pub struct DctGridToDwt97Job<'a> {
    /// Natural-order, dequantized 8x8 DCT blocks.
    pub blocks: &'a [[[f64; 8]; 8]],
    /// Number of DCT block columns in `blocks`.
    pub block_cols: usize,
    /// Number of DCT block rows in `blocks`.
    pub block_rows: usize,
    /// Logical component width in samples.
    pub width: usize,
    /// Logical component height in samples.
    pub height: usize,
}

/// Optional backend for SIMD, GPU, or other accelerated transform stages.
pub trait DctToWaveletStageAccelerator {
    /// Optionally compute the direct DCT-grid to one-level 5/3 projection.
    ///
    /// Return `Ok(Some(output))` when the backend handled the job. Return
    /// `Ok(None)` to use the scalar fallback.
    fn dct_grid_to_dwt53(
        &mut self,
        _job: DctGridToDwt53Job<'_>,
    ) -> Result<Option<Dwt53TwoDimensional<f64>>, &'static str> {
        Ok(None)
    }

    /// Optionally compute the direct DCT-grid to one-level 9/7 projection.
    ///
    /// Return `Ok(Some(output))` when the backend handled the job. Return
    /// `Ok(None)` to use the scalar fallback.
    fn dct_grid_to_dwt97(
        &mut self,
        _job: DctGridToDwt97Job<'_>,
    ) -> Result<Option<Dwt97TwoDimensional<f64>>, &'static str> {
        Ok(None)
    }
}

/// Accelerator that always uses the scalar CPU fallback.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuOnlyDctToWaveletStageAccelerator;

impl DctToWaveletStageAccelerator for CpuOnlyDctToWaveletStageAccelerator {}
