// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    types::Dwt97BatchDeviceRequest, validation::validate_transcode_pool_context,
    CudaDwt97BatchGeometry, CudaHtj2k97CodeblockBands, CudaHtj2k97CodeblockBatchWithPoolRequest,
    Dwt97BatchInput, Dwt97CodeblockBandBuffers,
};
use crate::{
    allocation::HostPhaseBudget, context::CudaContext, error::CudaError,
    j2k_encode::CudaDwt97BatchStageTimings, memory::CudaDeviceBuffer,
};

impl CudaContext {
    /// Compute a same-geometry batch directly into host-owned prequantized
    /// HTJ2K code-block coefficients while reusing transient stage buffers
    /// from `pool`.
    /// The pool must belong to this context.
    #[doc(hidden)]
    pub fn j2k_transcode_htj2k97_codeblock_batch_with_pool(
        &self,
        request: CudaHtj2k97CodeblockBatchWithPoolRequest<'_>,
    ) -> Result<(CudaHtj2k97CodeblockBands, CudaDwt97BatchStageTimings), CudaError> {
        self.j2k_transcode_htj2k97_codeblock_batch_with_pool_and_live_host_bytes(request, 0)
    }

    /// Compute and read back a code-block batch while accounting caller-live staging.
    #[doc(hidden)]
    pub fn j2k_transcode_htj2k97_codeblock_batch_with_pool_and_live_host_bytes(
        &self,
        request: CudaHtj2k97CodeblockBatchWithPoolRequest<'_>,
        live_host_bytes: usize,
    ) -> Result<(CudaHtj2k97CodeblockBands, CudaDwt97BatchStageTimings), CudaError> {
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

        let alloc_i32 = |count: usize| -> Result<CudaDeviceBuffer, CudaError> {
            let bytes = count
                .checked_mul(std::mem::size_of::<i32>())
                .ok_or(CudaError::LengthTooLarge { len: count })?;
            self.allocate(bytes)
        };
        let low_low_count = low_width * low_height;
        let low_high_count = low_width * high_height;
        let high_low_count = high_width * low_height;
        let high_high_count = high_width * high_height;

        let quantized_low_low = alloc_i32(item_count * low_low_count)?;
        let quantized_low_high = alloc_i32(item_count * low_high_count)?;
        let quantized_high_low = alloc_i32(item_count * high_low_count)?;
        let quantized_high_high = alloc_i32(item_count * high_high_count)?;

        let ((), quantize_codeblock_us) = self.time_default_stream_us(|| {
            self.launch_transcode_dwt97_quantize_codeblock_bands(
                &bands,
                Dwt97CodeblockBandBuffers {
                    ll: &quantized_low_low,
                    hl: &quantized_high_low,
                    lh: &quantized_low_high,
                    hh: &quantized_high_high,
                },
                params,
                items,
            )
        })?;

        let (codeblocks, readback_us) = self.time_default_stream_us(|| {
            let mut host_budget = HostPhaseBudget::with_live_bytes(
                "CUDA HTJ2K 9/7 code-block readback",
                live_host_bytes,
            )?;
            let ll = Self::download_i32_band(
                &quantized_low_low,
                item_count * low_low_count,
                &mut host_budget,
            )?;
            let hl = Self::download_i32_band(
                &quantized_high_low,
                item_count * high_low_count,
                &mut host_budget,
            )?;
            let lh = Self::download_i32_band(
                &quantized_low_high,
                item_count * low_high_count,
                &mut host_budget,
            )?;
            let hh = Self::download_i32_band(
                &quantized_high_high,
                item_count * high_high_count,
                &mut host_budget,
            )?;
            Ok(CudaHtj2k97CodeblockBands {
                ll,
                hl,
                lh,
                hh,
                item_count,
                low_width,
                low_height,
                high_width,
                high_height,
            })
        })?;

        Ok((
            codeblocks,
            CudaDwt97BatchStageTimings {
                pack_upload_us,
                idct_row_lift_us,
                column_lift_us,
                quantize_codeblock_us,
                ht_encode_us: 0,
                ht_codeblock_dispatches: 0,
                readback_us,
            },
        ))
    }
}
