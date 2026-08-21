// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    types::Dwt97BatchDeviceRequest, validation::validate_transcode_pool_context,
    CudaDwt97BatchGeometry, CudaHtj2k97CodeblockBatchWithPoolRequest,
    CudaHtj2k97DeviceCodeblockBands, CudaHtj2k97I16CodeblockBatchWithPoolRequest, Dwt97BatchInput,
    Dwt97CodeblockBandBuffers,
};
use crate::{
    error::CudaError,
    memory::{pooled_device_buffer, CudaPooledDeviceBuffer},
    transcode::CudaDwt97BatchStageTimings,
    CudaTranscodeEngine,
};

impl CudaTranscodeEngine<'_> {
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
        validate_transcode_pool_context(self, request.pool)?;
        let CudaHtj2k97I16CodeblockBatchWithPoolRequest {
            blocks,
            geometry,
            params,
            pool,
        } = request;
        let CudaDwt97BatchGeometry { item_count, .. } = geometry;
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
}
