// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    validate_dct_block_grid, validation::ensure_transcode_runtime_ptx_available,
    CudaTranscodeDwt97Bands, DctBlockGrid,
};
use crate::{
    allocation::HostPhaseBudget, context::CudaContext, error::CudaError, memory::CudaDeviceBuffer,
};

impl CudaContext {
    /// Compute one irreversible single-level 9/7 transform directly from
    /// dequantized 8x8 DCT blocks (`block_cols * block_rows` blocks of 64 `f32`
    /// natural-order coefficients), matching the `j2k-transcode` scalar
    /// oracle within f32 tolerance.
    #[doc(hidden)]
    pub fn j2k_transcode_dwt97(
        &self,
        blocks: &[f32],
        block_cols: usize,
        block_rows: usize,
        width: usize,
        height: usize,
    ) -> Result<CudaTranscodeDwt97Bands, CudaError> {
        self.j2k_transcode_dwt97_and_live_host_bytes(
            blocks, block_cols, block_rows, width, height, 0,
        )
    }

    /// Compute one irreversible level while accounting caller-live host owners.
    #[doc(hidden)]
    pub fn j2k_transcode_dwt97_and_live_host_bytes(
        &self,
        blocks: &[f32],
        block_cols: usize,
        block_rows: usize,
        width: usize,
        height: usize,
        live_host_bytes: usize,
    ) -> Result<CudaTranscodeDwt97Bands, CudaError> {
        ensure_transcode_runtime_ptx_available()?;
        let grid = validate_dct_block_grid(
            block_cols,
            block_rows,
            width,
            height,
            1,
            blocks.len(),
            "9/7 transcode job has unsupported grid geometry",
        )?;
        let DctBlockGrid {
            expected_coeffs: _,
            low_width,
            low_height,
            high_width,
            high_height,
            dims,
            ..
        } = grid;

        self.inner.set_current()?;

        let alloc_f32 = |count: usize| -> Result<CudaDeviceBuffer, CudaError> {
            let bytes = count
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or(CudaError::LengthTooLarge { len: count })?;
            self.allocate(bytes)
        };
        let spatial = alloc_f32(width * height)?;
        let row_low = alloc_f32(height * low_width)?;
        let row_high = alloc_f32(height * high_width)?;
        let ll = alloc_f32(low_width * low_height)?;
        let lh = alloc_f32(low_width * high_height)?;
        let hl = alloc_f32(high_width * low_height)?;
        let hh = alloc_f32(high_width * high_height)?;

        let blocks_dev = self.upload_f32(blocks)?;

        self.launch_transcode_dwt97_idct(dims, &blocks_dev, &spatial)?;
        self.launch_transcode_dwt97_row_lift(dims, &spatial, &row_low, &row_high)?;
        if dims.low_width > 0 {
            self.launch_transcode_dwt97_column_lift(
                &row_low,
                dims.low_width,
                dims.height,
                &ll,
                &lh,
            )?;
        }
        if dims.high_width > 0 {
            self.launch_transcode_dwt97_column_lift(
                &row_high,
                dims.high_width,
                dims.height,
                &hl,
                &hh,
            )?;
        }

        let mut host_budget =
            HostPhaseBudget::with_live_bytes("CUDA 9/7 subband readback", live_host_bytes)?;
        let ll = Self::download_f32_band(&ll, low_width * low_height, &mut host_budget)?;
        let hl = Self::download_f32_band(&hl, high_width * low_height, &mut host_budget)?;
        let lh = Self::download_f32_band(&lh, low_width * high_height, &mut host_budget)?;
        let hh = Self::download_f32_band(&hh, high_width * high_height, &mut host_budget)?;
        Ok(CudaTranscodeDwt97Bands {
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
}
