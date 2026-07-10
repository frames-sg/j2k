// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    checked_i32,
    types::{Dwt97BatchDeviceRequest, Dwt97ColumnLiftBatchLaunch},
    validate_dct_block_grid,
    validation::ensure_transcode_runtime_ptx_available,
    CudaDwt97BatchGeometry, CudaDwt97BatchWithPoolRequest, CudaTranscodeDwt97Bands, DctBlockGrid,
    Dwt97BatchDeviceBands, Dwt97BatchInput,
};
use crate::{
    context::CudaContext,
    error::CudaError,
    j2k_encode::CudaDwt97BatchStageTimings,
    kernels::CudaKernel,
    memory::{pooled_device_buffer, CudaDeviceBuffer, CudaPooledDeviceBuffer},
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

        Ok(CudaTranscodeDwt97Bands {
            ll: Self::download_f32_band(&ll, low_width * low_height)?,
            hl: Self::download_f32_band(&hl, high_width * low_height)?,
            lh: Self::download_f32_band(&lh, low_width * high_height)?,
            hh: Self::download_f32_band(&hh, high_width * high_height)?,
            low_width,
            low_height,
            high_width,
            high_height,
        })
    }
    /// Compute a same-geometry batch of irreversible single-level 9/7 transforms
    /// while reusing device buffers from `pool` for transient stage storage.
    #[expect(
        clippy::similar_names,
        reason = "LL/LH/HL/HH identifiers are the four distinct JPEG 2000 subband identities"
    )]
    #[doc(hidden)]
    pub fn j2k_transcode_dwt97_batch_with_pool(
        &self,
        request: CudaDwt97BatchWithPoolRequest<'_>,
    ) -> Result<(Vec<CudaTranscodeDwt97Bands>, CudaDwt97BatchStageTimings), CudaError> {
        let CudaDwt97BatchWithPoolRequest {
            blocks,
            geometry,
            pool,
        } = request;
        let CudaDwt97BatchGeometry { item_count, .. } = geometry;
        let (bands, pack_upload_us, idct_row_lift_us, column_lift_us) = self
            .transcode_dwt97_batch_input_to_device(Dwt97BatchDeviceRequest {
                input: Dwt97BatchInput::F32(blocks),
                geometry,
                pool,
            })?;
        let Dwt97BatchDeviceBands {
            ll,
            lh,
            hl,
            hh,
            low_width,
            low_height,
            high_width,
            high_height,
        } = bands;

        let ll_size = low_width * low_height;
        let lh_size = low_width * high_height;
        let hl_size = high_width * low_height;
        let hh_size = high_width * high_height;

        let (outputs, readback_us) = self.time_default_stream_us(|| {
            let ll_all = Self::download_pooled_f32_band(&ll, item_count * ll_size)?;
            let lh_all = Self::download_pooled_f32_band(&lh, item_count * lh_size)?;
            let hl_all = Self::download_pooled_f32_band(&hl, item_count * hl_size)?;
            let hh_all = Self::download_pooled_f32_band(&hh, item_count * hh_size)?;
            let mut outputs = Vec::with_capacity(item_count);
            for item in 0..item_count {
                outputs.push(CudaTranscodeDwt97Bands {
                    ll: ll_all[item * ll_size..(item + 1) * ll_size].to_vec(),
                    hl: hl_all[item * hl_size..(item + 1) * hl_size].to_vec(),
                    lh: lh_all[item * lh_size..(item + 1) * lh_size].to_vec(),
                    hh: hh_all[item * hh_size..(item + 1) * hh_size].to_vec(),
                    low_width,
                    low_height,
                    high_width,
                    high_height,
                });
            }
            Ok(outputs)
        })?;

        Ok((
            outputs,
            CudaDwt97BatchStageTimings {
                pack_upload_us,
                idct_row_lift_us,
                column_lift_us,
                quantize_codeblock_us: 0,
                ht_encode_us: 0,
                ht_codeblock_dispatches: 0,
                readback_us,
            },
        ))
    }
    /// Run the shared staged 9/7 batch pipeline (alloc + upload, batched IDCT +
    /// row lift, batched column lift) and return the device-resident bands plus
    /// the three pre-readback stage timings.
    #[expect(
        clippy::too_many_lines,
        reason = "the method preserves three separately timed CUDA-resident stages and their order"
    )]
    pub(super) fn transcode_dwt97_batch_input_to_device(
        &self,
        request: Dwt97BatchDeviceRequest<'_>,
    ) -> Result<(Dwt97BatchDeviceBands, u128, u128, u128), CudaError> {
        ensure_transcode_runtime_ptx_available()?;
        let input = request.input;
        let pool = request.pool;
        let CudaDwt97BatchGeometry {
            item_count,
            block_cols,
            block_rows,
            width,
            height,
        } = request.geometry;
        let grid = validate_dct_block_grid(
            block_cols,
            block_rows,
            width,
            height,
            item_count,
            input.len(),
            "9/7 transcode batch has unsupported grid geometry",
        )?;
        let DctBlockGrid {
            block_count,
            low_width,
            low_height,
            high_width,
            high_height,
            dims,
            ..
        } = grid;
        let items =
            u32::try_from(item_count).map_err(|_| CudaError::LengthTooLarge { len: item_count })?;
        let blocks_per_item = checked_i32(block_count)?;
        let low_height_i32 = checked_i32(low_height)?;
        let high_height_i32 = checked_i32(high_height)?;

        self.inner.set_current()?;

        let alloc_f32 = |count: usize| -> Result<CudaPooledDeviceBuffer, CudaError> {
            let bytes = count
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or(CudaError::LengthTooLarge { len: count })?;
            pool.take(bytes)
        };

        // Stage: allocate batch buffers and upload all blocks.
        let (buffers, pack_upload_us) = self.time_default_stream_us(|| {
            let spatial = alloc_f32(item_count * width * height)?;
            let row_low = alloc_f32(item_count * height * low_width)?;
            let row_high = alloc_f32(item_count * height * high_width)?;
            let ll = alloc_f32(item_count * low_width * low_height)?;
            let lh = alloc_f32(item_count * low_width * high_height)?;
            let hl = alloc_f32(item_count * high_width * low_height)?;
            let hh = alloc_f32(item_count * high_width * high_height)?;
            let blocks_dev = input.upload(pool)?;
            Ok((spatial, row_low, row_high, ll, lh, hl, hh, blocks_dev))
        })?;
        let (spatial, row_low, row_high, ll, lh, hl, hh, blocks_dev) = buffers;

        // Stage: batched separable IDCT then horizontal 9/7 row lift.
        let ((), idct_row_lift_us) = self.time_default_stream_us(|| {
            let idct_kernel = match input {
                Dwt97BatchInput::F32(_) => CudaKernel::TranscodeDwt97IdctBatch,
                Dwt97BatchInput::I16(_) => CudaKernel::TranscodeDwt97IdctI16Batch,
            };
            self.launch_transcode_dwt97_idct_batch_kernel(
                idct_kernel,
                dims,
                blocks_per_item,
                items,
                pooled_device_buffer(&blocks_dev)?,
                pooled_device_buffer(&spatial)?,
            )?;
            self.launch_transcode_dwt97_row_lift_batch(
                dims,
                items,
                pooled_device_buffer(&spatial)?,
                pooled_device_buffer(&row_low)?,
                pooled_device_buffer(&row_high)?,
            )?;
            Ok(())
        })?;

        // Stage: batched vertical 9/7 column lift for both low and high rows.
        let ((), column_lift_us) = self.time_default_stream_us(|| {
            if dims.low_width > 0 {
                self.launch_transcode_dwt97_column_lift_batch(&Dwt97ColumnLiftBatchLaunch {
                    rows_buffer: pooled_device_buffer(&row_low)?,
                    band_width: dims.low_width,
                    height: dims.height,
                    low_height: low_height_i32,
                    high_height: high_height_i32,
                    items,
                    low_out: pooled_device_buffer(&ll)?,
                    high_out: pooled_device_buffer(&lh)?,
                })?;
            }
            if dims.high_width > 0 {
                self.launch_transcode_dwt97_column_lift_batch(&Dwt97ColumnLiftBatchLaunch {
                    rows_buffer: pooled_device_buffer(&row_high)?,
                    band_width: dims.high_width,
                    height: dims.height,
                    low_height: low_height_i32,
                    high_height: high_height_i32,
                    items,
                    low_out: pooled_device_buffer(&hl)?,
                    high_out: pooled_device_buffer(&hh)?,
                })?;
            }
            Ok(())
        })?;

        Ok((
            Dwt97BatchDeviceBands {
                ll,
                lh,
                hl,
                hh,
                low_width,
                low_height,
                high_width,
                high_height,
            },
            pack_upload_us,
            idct_row_lift_us,
            column_lift_us,
        ))
    }
}
