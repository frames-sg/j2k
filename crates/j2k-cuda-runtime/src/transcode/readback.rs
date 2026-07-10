// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    types::Dwt97BatchDeviceRequest, CudaDwt97BatchGeometry, CudaHtj2k97CodeblockBands,
    CudaHtj2k97CodeblockBatchWithPoolRequest, Dwt97BatchInput, Dwt97CodeblockBandBuffers,
};
use crate::{
    context::CudaContext, error::CudaError, j2k_encode::CudaDwt97BatchStageTimings,
    memory::CudaDeviceBuffer,
};

impl CudaContext {
    /// Compute a same-geometry batch directly into host-owned prequantized
    /// HTJ2K code-block coefficients while reusing transient stage buffers
    /// from `pool`.
    #[expect(
        clippy::similar_names,
        reason = "LL/LH/HL/HH identifiers are the four distinct JPEG 2000 subband identities"
    )]
    #[doc(hidden)]
    pub fn j2k_transcode_htj2k97_codeblock_batch_with_pool(
        &self,
        request: CudaHtj2k97CodeblockBatchWithPoolRequest<'_>,
    ) -> Result<(CudaHtj2k97CodeblockBands, CudaDwt97BatchStageTimings), CudaError> {
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
                    ll: &ll_q,
                    hl: &hl_q,
                    lh: &lh_q,
                    hh: &hh_q,
                },
                params,
                items,
            )
        })?;

        let (codeblocks, readback_us) = self.time_default_stream_us(|| {
            Ok(CudaHtj2k97CodeblockBands {
                ll: Self::download_i32_band(&ll_q, item_count * ll_size)?,
                hl: Self::download_i32_band(&hl_q, item_count * hl_size)?,
                lh: Self::download_i32_band(&lh_q, item_count * lh_size)?,
                hh: Self::download_i32_band(&hh_q, item_count * hh_size)?,
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
