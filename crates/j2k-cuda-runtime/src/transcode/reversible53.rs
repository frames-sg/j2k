// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    checked_i32, validate_dct_block_grid, validation::ensure_transcode_runtime_ptx_available,
    CudaTranscodeReversible53Bands, DctBlockGrid,
};
use crate::{
    context::CudaContext, error::CudaError, kernels::CudaKernel, memory::CudaDeviceBuffer,
};

impl CudaContext {
    /// Compute one reversible integer 5/3 level directly from dequantized 8x8
    /// DCT blocks, bit-exact with the `j2k-transcode` scalar oracle.
    ///
    /// `dequantized_blocks` holds `block_cols * block_rows` natural-order blocks
    /// of 64 `i16` coefficients. `width`/`height` are the logical component
    /// dimensions (<= `block_cols*8` / `block_rows*8`).
    #[doc(hidden)]
    pub fn j2k_transcode_reversible_dwt53(
        &self,
        dequantized_blocks: &[i16],
        block_cols: usize,
        block_rows: usize,
        width: usize,
        height: usize,
    ) -> Result<CudaTranscodeReversible53Bands, CudaError> {
        ensure_transcode_runtime_ptx_available()?;
        let grid = validate_dct_block_grid(
            block_cols,
            block_rows,
            width,
            height,
            1,
            dequantized_blocks.len(),
            "reversible 5/3 transcode job has unsupported grid geometry",
        )?;
        let DctBlockGrid {
            block_count,
            expected_coeffs,
            low_width,
            low_height,
            high_width,
            high_height,
            dims,
        } = grid;

        self.inner.set_current()?;

        let alloc_i32 = |count: usize| -> Result<CudaDeviceBuffer, CudaError> {
            let bytes = count
                .checked_mul(std::mem::size_of::<i32>())
                .ok_or(CudaError::LengthTooLarge { len: count })?;
            self.allocate(bytes)
        };
        let samples = alloc_i32(expected_coeffs)?;
        let v_low = alloc_i32(width * low_height)?;
        let v_high = alloc_i32(width * high_height)?;
        let ll = alloc_i32(low_width * low_height)?;
        let hl = alloc_i32(high_width * low_height)?;
        let lh = alloc_i32(low_width * high_height)?;
        let hh = alloc_i32(high_width * high_height)?;

        // SAFETY: `dequantized_blocks` is a live `&[i16]`; reinterpreting it as a
        // byte slice of `len * 2` bytes for upload is a read-only view with the
        // same lifetime and no alignment requirement on the destination.
        let block_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                dequantized_blocks.as_ptr().cast::<u8>(),
                std::mem::size_of_val(dequantized_blocks),
            )
        };
        let blocks_dev = self.upload(block_bytes)?;

        self.launch_transcode_reversible53_idct(&blocks_dev, &samples, block_count)?;
        if low_height > 0 {
            self.launch_transcode_reversible53_vertical(
                CudaKernel::TranscodeReversible53VerticalLow,
                &samples,
                dims,
                &v_low,
                checked_i32(low_height)?,
            )?;
            self.launch_transcode_reversible53_horizontal(
                CudaKernel::TranscodeReversible53HorizontalLow,
                &v_low,
                dims,
                checked_i32(low_height)?,
                &ll,
                &hl,
            )?;
        }
        if high_height > 0 {
            self.launch_transcode_reversible53_vertical(
                CudaKernel::TranscodeReversible53VerticalHigh,
                &samples,
                dims,
                &v_high,
                checked_i32(high_height)?,
            )?;
            self.launch_transcode_reversible53_horizontal(
                CudaKernel::TranscodeReversible53HorizontalHigh,
                &v_high,
                dims,
                checked_i32(high_height)?,
                &lh,
                &hh,
            )?;
        }

        Ok(CudaTranscodeReversible53Bands {
            ll: Self::download_i32_band(&ll, low_width * low_height)?,
            hl: Self::download_i32_band(&hl, high_width * low_height)?,
            lh: Self::download_i32_band(&lh, low_width * high_height)?,
            hh: Self::download_i32_band(&hh, high_width * high_height)?,
            low_width,
            low_height,
            high_width,
            high_height,
        })
    }
}
