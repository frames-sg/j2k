// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    checked_i32,
    types::{
        Dwt97BatchDeviceRequest, Dwt97ColumnLiftBatchLaunch,
        Dwt97ColumnLiftQuantizeCodeblocksBatchLaunch, Htj2k97I16ResidentFusedRequest,
    },
    validate_dct_block_grid,
    validation::{ensure_transcode_runtime_ptx_available, validate_transcode_pool_context},
    CudaDwt97BatchGeometry, CudaHtj2k97CodeblockBatchWithPoolRequest,
    CudaHtj2k97DeviceCodeblockBands, CudaHtj2k97I16CodeblockBatchWithPoolRequest, DctBlockGrid,
    Dwt97BatchInput, Dwt97CodeblockBandBuffers,
};
use crate::{
    build_flags::dwt97_fused_column_quantize_disabled,
    context::CudaContext,
    error::CudaError,
    j2k_encode::CudaDwt97BatchStageTimings,
    kernels::CudaKernel,
    memory::{pooled_device_buffer, CudaPooledDeviceBuffer},
};

impl CudaContext {
    /// Compute a same-geometry batch directly into device-resident
    /// prequantized HTJ2K code-block coefficients while reusing transient stage
    /// buffers from `pool`.
    /// The pool must belong to this context.
    #[expect(
        clippy::similar_names,
        reason = "LL/LH/HL/HH identifiers are the four distinct JPEG 2000 subband identities"
    )]
    #[doc(hidden)]
    pub fn j2k_transcode_htj2k97_codeblock_batch_resident_with_pool(
        &self,
        request: CudaHtj2k97CodeblockBatchWithPoolRequest<'_>,
    ) -> Result<(CudaHtj2k97DeviceCodeblockBands, CudaDwt97BatchStageTimings), CudaError> {
        validate_transcode_pool_context(self, request.pool)?;
        let CudaHtj2k97CodeblockBatchWithPoolRequest {
            blocks,
            geometry,
            params,
            pool,
        } = request;
        let CudaDwt97BatchGeometry { item_count, .. } = geometry;
        let (bands, pack_upload_us, idct_row_lift_us, column_lift_us) = self
            .transcode_dwt97_batch_input_to_device(Dwt97BatchDeviceRequest {
                input: Dwt97BatchInput::F32(blocks),
                geometry,
                pool,
            })?;
        let low_width = bands.low_width;
        let low_height = bands.low_height;
        let high_width = bands.high_width;
        let high_height = bands.high_height;
        let items =
            u32::try_from(item_count).map_err(|_| CudaError::LengthTooLarge { len: item_count })?;

        let alloc_i32 = |count: usize| -> Result<CudaPooledDeviceBuffer, CudaError> {
            let bytes = count
                .checked_mul(std::mem::size_of::<i32>())
                .ok_or(CudaError::LengthTooLarge { len: count })?;
            pool.take(bytes)
        };
        let ll_size = low_width * low_height;
        let lh_size = low_width * high_height;
        let hl_size = high_width * low_height;
        let hh_size = high_width * high_height;

        let ll_q = alloc_i32(item_count * ll_size)?;
        let lh_q = alloc_i32(item_count * lh_size)?;
        let hl_q = alloc_i32(item_count * hl_size)?;
        let hh_q = alloc_i32(item_count * hh_size)?;

        let ((), quantize_codeblock_us) = self.time_default_stream_us(|| {
            self.launch_transcode_dwt97_quantize_codeblock_bands(
                &bands,
                Dwt97CodeblockBandBuffers {
                    ll: pooled_device_buffer(&ll_q)?,
                    hl: pooled_device_buffer(&hl_q)?,
                    lh: pooled_device_buffer(&lh_q)?,
                    hh: pooled_device_buffer(&hh_q)?,
                },
                params,
                items,
            )
        })?;

        Ok((
            CudaHtj2k97DeviceCodeblockBands {
                ll: ll_q,
                hl: hl_q,
                lh: lh_q,
                hh: hh_q,
                item_count,
                low_width,
                low_height,
                high_width,
                high_height,
            },
            CudaDwt97BatchStageTimings {
                pack_upload_us,
                idct_row_lift_us,
                column_lift_us,
                quantize_codeblock_us,
                ht_encode_us: 0,
                ht_codeblock_dispatches: 0,
                readback_us: 0,
            },
        ))
    }

    /// Compute a same-geometry i16 batch directly into device-resident
    /// prequantized HTJ2K code-block coefficients while reusing transient stage
    /// buffers from `pool`.
    /// The pool must belong to this context.
    #[expect(
        clippy::similar_names,
        reason = "LL/LH/HL/HH identifiers are the four distinct JPEG 2000 subband identities"
    )]
    #[doc(hidden)]
    pub fn j2k_transcode_htj2k97_codeblock_i16_batch_resident_with_pool(
        &self,
        request: CudaHtj2k97I16CodeblockBatchWithPoolRequest<'_>,
    ) -> Result<(CudaHtj2k97DeviceCodeblockBands, CudaDwt97BatchStageTimings), CudaError> {
        // Validate before selecting the fused path: both implementations allocate
        // from and launch work against the caller-provided pool.
        validate_transcode_pool_context(self, request.pool)?;
        let CudaHtj2k97I16CodeblockBatchWithPoolRequest {
            blocks,
            geometry,
            params,
            pool,
        } = request;
        let CudaDwt97BatchGeometry { item_count, .. } = geometry;
        if !dwt97_fused_column_quantize_disabled() {
            return self.j2k_transcode_htj2k97_codeblock_i16_batch_resident_fused_with_pool(
                Htj2k97I16ResidentFusedRequest {
                    blocks,
                    geometry,
                    params,
                    pool,
                },
            );
        }

        let (bands, pack_upload_us, idct_row_lift_us, column_lift_us) = self
            .transcode_dwt97_batch_input_to_device(Dwt97BatchDeviceRequest {
                input: Dwt97BatchInput::I16(blocks),
                geometry,
                pool,
            })?;
        let low_width = bands.low_width;
        let low_height = bands.low_height;
        let high_width = bands.high_width;
        let high_height = bands.high_height;
        let items =
            u32::try_from(item_count).map_err(|_| CudaError::LengthTooLarge { len: item_count })?;

        let alloc_i32 = |count: usize| -> Result<CudaPooledDeviceBuffer, CudaError> {
            let bytes = count
                .checked_mul(std::mem::size_of::<i32>())
                .ok_or(CudaError::LengthTooLarge { len: count })?;
            pool.take(bytes)
        };
        let ll_size = low_width * low_height;
        let lh_size = low_width * high_height;
        let hl_size = high_width * low_height;
        let hh_size = high_width * high_height;

        let ll_q = alloc_i32(item_count * ll_size)?;
        let lh_q = alloc_i32(item_count * lh_size)?;
        let hl_q = alloc_i32(item_count * hl_size)?;
        let hh_q = alloc_i32(item_count * hh_size)?;

        let ((), quantize_codeblock_us) = self.time_default_stream_us(|| {
            self.launch_transcode_dwt97_quantize_codeblock_bands(
                &bands,
                Dwt97CodeblockBandBuffers {
                    ll: pooled_device_buffer(&ll_q)?,
                    hl: pooled_device_buffer(&hl_q)?,
                    lh: pooled_device_buffer(&lh_q)?,
                    hh: pooled_device_buffer(&hh_q)?,
                },
                params,
                items,
            )
        })?;

        Ok((
            CudaHtj2k97DeviceCodeblockBands {
                ll: ll_q,
                hl: hl_q,
                lh: lh_q,
                hh: hh_q,
                item_count,
                low_width,
                low_height,
                high_width,
                high_height,
            },
            CudaDwt97BatchStageTimings {
                pack_upload_us,
                idct_row_lift_us,
                column_lift_us,
                quantize_codeblock_us,
                ht_encode_us: 0,
                ht_codeblock_dispatches: 0,
                readback_us: 0,
            },
        ))
    }

    #[expect(
        clippy::similar_names,
        clippy::too_many_lines,
        reason = "the fused path keeps four named subbands in one ordered timed CUDA dispatch"
    )]
    fn j2k_transcode_htj2k97_codeblock_i16_batch_resident_fused_with_pool(
        &self,
        request: Htj2k97I16ResidentFusedRequest<'_>,
    ) -> Result<(CudaHtj2k97DeviceCodeblockBands, CudaDwt97BatchStageTimings), CudaError> {
        let Htj2k97I16ResidentFusedRequest {
            blocks,
            geometry,
            params,
            pool,
        } = request;
        let CudaDwt97BatchGeometry {
            item_count,
            block_cols,
            block_rows,
            width,
            height,
        } = geometry;
        ensure_transcode_runtime_ptx_available()?;
        let grid = validate_dct_block_grid(
            block_cols,
            block_rows,
            width,
            height,
            item_count,
            blocks.len(),
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
        let cb_w = checked_i32(params.cb_width)?;
        let cb_h = checked_i32(params.cb_height)?;

        self.inner.set_current()?;

        let alloc_f32 = |count: usize| -> Result<CudaPooledDeviceBuffer, CudaError> {
            let bytes = count
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or(CudaError::LengthTooLarge { len: count })?;
            pool.take(bytes)
        };
        let alloc_i32 = |count: usize| -> Result<CudaPooledDeviceBuffer, CudaError> {
            let bytes = count
                .checked_mul(std::mem::size_of::<i32>())
                .ok_or(CudaError::LengthTooLarge { len: count })?;
            pool.take(bytes)
        };
        let (buffers, pack_upload_us) = self.time_default_stream_us(|| {
            let spatial = alloc_f32(item_count * width * height)?;
            let row_low = alloc_f32(item_count * height * low_width)?;
            let row_high = alloc_f32(item_count * height * high_width)?;
            let blocks_dev = Dwt97BatchInput::I16(blocks).upload(pool)?;
            Ok((spatial, row_low, row_high, blocks_dev))
        })?;
        let (spatial, row_low, row_high, blocks_dev) = buffers;

        let ll_size = low_width * low_height;
        let lh_size = low_width * high_height;
        let hl_size = high_width * low_height;
        let hh_size = high_width * high_height;

        let ll_q = alloc_i32(item_count * ll_size)?;
        let lh_q = alloc_i32(item_count * lh_size)?;
        let hl_q = alloc_i32(item_count * hl_size)?;
        let hh_q = alloc_i32(item_count * hh_size)?;

        let ((), idct_row_lift_us) = self.time_default_stream_us(|| {
            self.launch_transcode_dwt97_idct_batch_kernel(
                CudaKernel::TranscodeDwt97IdctI16Batch,
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

        let ((), column_quantize_us) = self.time_default_stream_us(|| {
            if dims.low_width > 0 {
                self.launch_transcode_dwt97_column_lift_quantize_codeblocks_batch(
                    &Dwt97ColumnLiftQuantizeCodeblocksBatchLaunch {
                        column: Dwt97ColumnLiftBatchLaunch {
                            rows_buffer: pooled_device_buffer(&row_low)?,
                            band_width: dims.low_width,
                            height: dims.height,
                            low_height: low_height_i32,
                            high_height: high_height_i32,
                            items,
                            low_out: pooled_device_buffer(&ll_q)?,
                            high_out: pooled_device_buffer(&lh_q)?,
                        },
                        cb_width: cb_w,
                        cb_height: cb_h,
                        inv_delta_low: params.inv_delta_ll,
                        inv_delta_high: params.inv_delta_lh,
                    },
                )?;
            }
            if dims.high_width > 0 {
                self.launch_transcode_dwt97_column_lift_quantize_codeblocks_batch(
                    &Dwt97ColumnLiftQuantizeCodeblocksBatchLaunch {
                        column: Dwt97ColumnLiftBatchLaunch {
                            rows_buffer: pooled_device_buffer(&row_high)?,
                            band_width: dims.high_width,
                            height: dims.height,
                            low_height: low_height_i32,
                            high_height: high_height_i32,
                            items,
                            low_out: pooled_device_buffer(&hl_q)?,
                            high_out: pooled_device_buffer(&hh_q)?,
                        },
                        cb_width: cb_w,
                        cb_height: cb_h,
                        inv_delta_low: params.inv_delta_hl,
                        inv_delta_high: params.inv_delta_hh,
                    },
                )?;
            }
            Ok(())
        })?;

        Ok((
            CudaHtj2k97DeviceCodeblockBands {
                ll: ll_q,
                hl: hl_q,
                lh: lh_q,
                hh: hh_q,
                item_count,
                low_width,
                low_height,
                high_width,
                high_height,
            },
            CudaDwt97BatchStageTimings {
                pack_upload_us,
                idct_row_lift_us,
                column_lift_us: 0,
                quantize_codeblock_us: column_quantize_us,
                ht_encode_us: 0,
                ht_codeblock_dispatches: 0,
                readback_us: 0,
            },
        ))
    }
}
