// SPDX-License-Identifier: MIT OR Apache-2.0

mod single;

use super::{
    checked_i32,
    types::{Dwt97BatchDeviceRequest, Dwt97ColumnLiftBatchLaunch},
    validate_dct_block_grid,
    validation::{ensure_transcode_runtime_ptx_available, validate_transcode_pool_context},
    CudaDwt97BatchGeometry, CudaDwt97BatchWithPoolRequest, CudaTranscodeDwt97Bands, DctBlockGrid,
    Dwt97BatchDeviceBands, Dwt97BatchInput,
};
use crate::{
    allocation::HostPhaseBudget,
    context::CudaContext,
    error::CudaError,
    j2k_encode::CudaDwt97BatchStageTimings,
    kernels::CudaKernel,
    memory::{pooled_device_buffer, CudaPooledDeviceBuffer},
};

impl CudaContext {
    /// Compute a same-geometry batch of irreversible single-level 9/7 transforms
    /// while reusing device buffers from `pool` for transient stage storage.
    /// The pool must belong to this context.
    #[doc(hidden)]
    pub fn j2k_transcode_dwt97_batch_with_pool(
        &self,
        request: CudaDwt97BatchWithPoolRequest<'_>,
    ) -> Result<(Vec<CudaTranscodeDwt97Bands>, CudaDwt97BatchStageTimings), CudaError> {
        self.j2k_transcode_dwt97_batch_with_pool_and_live_host_bytes(request, 0)
    }

    /// Compute a 9/7 batch while accounting caller-live host staging.
    #[doc(hidden)]
    pub fn j2k_transcode_dwt97_batch_with_pool_and_live_host_bytes(
        &self,
        request: CudaDwt97BatchWithPoolRequest<'_>,
        live_host_bytes: usize,
    ) -> Result<(Vec<CudaTranscodeDwt97Bands>, CudaDwt97BatchStageTimings), CudaError> {
        validate_transcode_pool_context(self, request.pool)?;
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

        let low_low_count = low_width * low_height;
        let low_high_count = low_width * high_height;
        let high_low_count = high_width * low_height;
        let high_high_count = high_width * high_height;

        let (outputs, readback_us) = self.time_default_stream_us(|| {
            let mut host_budget =
                HostPhaseBudget::with_live_bytes("CUDA 9/7 batch readback", live_host_bytes)?;
            let low_low_values =
                Self::download_pooled_f32_band(&ll, item_count * low_low_count, &mut host_budget)?;
            let low_high_values =
                Self::download_pooled_f32_band(&lh, item_count * low_high_count, &mut host_budget)?;
            let high_low_values =
                Self::download_pooled_f32_band(&hl, item_count * high_low_count, &mut host_budget)?;
            let high_high_values = Self::download_pooled_f32_band(
                &hh,
                item_count * high_high_count,
                &mut host_budget,
            )?;
            let mut outputs = host_budget.try_vec_with_capacity(item_count)?;
            for item in 0..item_count {
                outputs.push(CudaTranscodeDwt97Bands {
                    ll: host_budget.try_vec_from_slice(
                        &low_low_values[item * low_low_count..(item + 1) * low_low_count],
                    )?,
                    hl: host_budget.try_vec_from_slice(
                        &high_low_values[item * high_low_count..(item + 1) * high_low_count],
                    )?,
                    lh: host_budget.try_vec_from_slice(
                        &low_high_values[item * low_high_count..(item + 1) * low_high_count],
                    )?,
                    hh: host_budget.try_vec_from_slice(
                        &high_high_values[item * high_high_count..(item + 1) * high_high_count],
                    )?,
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
