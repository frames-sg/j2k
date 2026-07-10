// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "cuda-runtime")]
use j2k_core::PixelFormat;
#[cfg(feature = "cuda-runtime")]
use j2k_cuda_runtime::{
    CudaDeviceBuffer, CudaJpegBaselineEncodeHuffmanTable, CudaJpegBaselineEncodeParams,
    CudaJpegBaselineEntropyEncodeBatchJob, CudaJpegBaselineEntropyEncodeJob,
};
#[cfg(feature = "cuda-runtime")]
use j2k_jpeg::adapter::{
    encode_jpeg_baseline_gpu_batch, encode_jpeg_baseline_gpu_tile, JpegBaselineEncodeTables,
    JpegBaselineGpuEncodeBatchPlan, JpegBaselineGpuEncodeError, JpegBaselineGpuEncodeHostAdapter,
    JpegBaselineGpuEncodeParams, JpegBaselineGpuEncodeTile, JpegBaselineGpuEncodeTilePlan,
    JpegBaselineHuffmanTable,
};
use j2k_jpeg::{EncodedJpeg, JpegEncodeOptions};
#[cfg(feature = "cuda-runtime")]
use j2k_jpeg::{JpegBackend, JpegEncodeError};

#[cfg(feature = "cuda-runtime")]
use crate::runtime::cuda_error;

#[cfg(feature = "cuda-runtime")]
#[derive(Debug, Clone, Copy)]
/// CUDA buffer and layout metadata for one baseline JPEG encode tile.
pub struct JpegBaselineCudaEncodeTile<'a> {
    /// Source CUDA buffer containing RGB8 or Gray8 pixels.
    pub buffer: &'a CudaDeviceBuffer,
    /// Byte offset of the first source pixel in `buffer`.
    pub byte_offset: usize,
    /// Width of the valid input region in pixels.
    pub width: u32,
    /// Height of the valid input region in pixels.
    pub height: u32,
    /// Number of bytes between consecutive input rows.
    pub pitch_bytes: usize,
    /// Encoded frame width in pixels.
    pub output_width: u32,
    /// Encoded frame height in pixels.
    pub output_height: u32,
    /// Pixel format of the source buffer.
    pub format: PixelFormat,
}

#[cfg(not(feature = "cuda-runtime"))]
#[derive(Debug, Clone, Copy)]
/// Placeholder encode tile type for builds without `cuda-runtime`.
pub struct JpegBaselineCudaEncodeTile<'a> {
    _private: core::marker::PhantomData<&'a ()>,
}

#[cfg(feature = "cuda-runtime")]
/// Encode one CUDA-resident tile as a baseline JPEG frame.
pub fn encode_jpeg_baseline_from_cuda_buffer(
    tile: JpegBaselineCudaEncodeTile<'_>,
    options: JpegEncodeOptions,
    session: &mut crate::CudaSession,
) -> Result<EncodedJpeg, crate::Error> {
    let _ = session;
    let mut adapter = CudaJpegBaselineEncodeAdapter;
    encode_jpeg_baseline_gpu_tile(tile, options, &mut adapter)
}

#[cfg(feature = "cuda-runtime")]
/// Encode multiple CUDA-resident tiles as baseline JPEG frames.
///
/// Consecutive tiles that share a source CUDA buffer are submitted through a
/// single entropy-kernel batch. The returned frames preserve input order.
pub fn encode_jpeg_baseline_batch_from_cuda_buffers(
    tiles: &[JpegBaselineCudaEncodeTile<'_>],
    options: JpegEncodeOptions,
    session: &mut crate::CudaSession,
) -> Result<Vec<EncodedJpeg>, crate::Error> {
    let _ = session;
    let mut adapter = CudaJpegBaselineEncodeAdapter;
    encode_jpeg_baseline_gpu_batch(tiles, options, &mut adapter)
}

#[cfg(not(feature = "cuda-runtime"))]
/// Return `Error::CudaUnavailable` for batch CUDA encode requests without `cuda-runtime`.
pub fn encode_jpeg_baseline_batch_from_cuda_buffers(
    tiles: &[JpegBaselineCudaEncodeTile<'_>],
    options: JpegEncodeOptions,
    session: &mut crate::CudaSession,
) -> Result<Vec<EncodedJpeg>, crate::Error> {
    let _ = (tiles, options, session);
    Err(crate::Error::CudaUnavailable)
}

#[cfg(not(feature = "cuda-runtime"))]
/// Return `Error::CudaUnavailable` for CUDA encode requests without `cuda-runtime`.
pub fn encode_jpeg_baseline_from_cuda_buffer(
    tile: JpegBaselineCudaEncodeTile<'_>,
    options: JpegEncodeOptions,
    session: &mut crate::CudaSession,
) -> Result<EncodedJpeg, crate::Error> {
    let _ = (tile, options, session);
    Err(crate::Error::CudaUnavailable)
}

#[cfg(feature = "cuda-runtime")]
fn compute_huffman_table(source: &JpegBaselineHuffmanTable) -> CudaJpegBaselineEncodeHuffmanTable {
    CudaJpegBaselineEncodeHuffmanTable {
        codes: source.codes,
        lens: source.lens,
    }
}

#[cfg(feature = "cuda-runtime")]
struct CudaJpegBaselineEncodeAdapter;

#[cfg(feature = "cuda-runtime")]
impl<'a> JpegBaselineGpuEncodeHostAdapter<JpegBaselineCudaEncodeTile<'a>>
    for CudaJpegBaselineEncodeAdapter
{
    type Error = crate::Error;
    type SourceKey = u64;

    fn backend(&self) -> JpegBackend {
        JpegBackend::Cuda
    }

    fn source_key(&self, tile: &JpegBaselineCudaEncodeTile<'a>) -> Self::SourceKey {
        tile.buffer.device_ptr()
    }

    fn gpu_tile(
        &self,
        tile: JpegBaselineCudaEncodeTile<'a>,
    ) -> Result<JpegBaselineGpuEncodeTile, Self::Error> {
        Ok(cuda_gpu_tile(tile))
    }

    fn map_plan_error(&self, error: JpegBaselineGpuEncodeError) -> Self::Error {
        cuda_gpu_encode_error(error)
    }

    fn encode_tile_entropy(
        &mut self,
        tile: JpegBaselineCudaEncodeTile<'a>,
        tables: &JpegBaselineEncodeTables,
        plan: JpegBaselineGpuEncodeTilePlan,
    ) -> Result<Vec<u8>, Self::Error> {
        tile.buffer
            .context()
            .encode_jpeg_baseline_entropy(&CudaJpegBaselineEntropyEncodeJob {
                input: tile.buffer,
                input_offset: tile.byte_offset,
                params: cuda_encode_params(plan.params),
                q_luma: tables.q_luma,
                q_chroma: tables.q_chroma,
                huff_dc_luma: compute_huffman_table(&tables.huff_dc_luma),
                huff_ac_luma: compute_huffman_table(&tables.huff_ac_luma),
                huff_dc_chroma: compute_huffman_table(&tables.huff_dc_chroma),
                huff_ac_chroma: compute_huffman_table(&tables.huff_ac_chroma),
                entropy_capacity: plan.entropy_capacity,
            })
            .map_err(cuda_error)
    }

    fn encode_batch_entropy(
        &mut self,
        tiles: &[JpegBaselineCudaEncodeTile<'a>],
        tables: &JpegBaselineEncodeTables,
        plan: JpegBaselineGpuEncodeBatchPlan,
    ) -> Result<Vec<Vec<u8>>, Self::Error> {
        let params = plan.params.into_iter().map(cuda_encode_params).collect();
        tiles[0]
            .buffer
            .context()
            .encode_jpeg_baseline_entropy_batch(&CudaJpegBaselineEntropyEncodeBatchJob {
                input: tiles[0].buffer,
                params,
                q_luma: tables.q_luma,
                q_chroma: tables.q_chroma,
                huff_dc_luma: compute_huffman_table(&tables.huff_dc_luma),
                huff_ac_luma: compute_huffman_table(&tables.huff_ac_luma),
                huff_dc_chroma: compute_huffman_table(&tables.huff_dc_chroma),
                huff_ac_chroma: compute_huffman_table(&tables.huff_ac_chroma),
                entropy_capacity: plan.total_entropy_capacity,
            })
            .map_err(cuda_error)
    }
}

#[cfg(feature = "cuda-runtime")]
fn cuda_gpu_tile(tile: JpegBaselineCudaEncodeTile<'_>) -> JpegBaselineGpuEncodeTile {
    JpegBaselineGpuEncodeTile {
        byte_offset: tile.byte_offset,
        width: tile.width,
        height: tile.height,
        pitch_bytes: tile.pitch_bytes,
        output_width: tile.output_width,
        output_height: tile.output_height,
        format: tile.format,
        buffer_len: tile.buffer.byte_len(),
    }
}

#[cfg(feature = "cuda-runtime")]
fn cuda_encode_params(params: JpegBaselineGpuEncodeParams) -> CudaJpegBaselineEncodeParams {
    CudaJpegBaselineEncodeParams {
        input_offset_bytes: params.input_offset_bytes,
        input_width: params.input_width,
        input_height: params.input_height,
        output_width: params.output_width,
        output_height: params.output_height,
        pitch_bytes: params.pitch_bytes,
        mcus_per_row: params.mcus_per_row,
        mcu_rows: params.mcu_rows,
        restart_interval_mcus: params.restart_interval_mcus,
        format: params.format,
        components: params.components,
        max_h: params.max_h,
        max_v: params.max_v,
        h0: params.h0,
        v0: params.v0,
        h1: params.h1,
        v1: params.v1,
        h2: params.h2,
        v2: params.v2,
        entropy_offset_bytes: params.entropy_offset_bytes,
        entropy_capacity: params.entropy_capacity,
    }
}

#[cfg(feature = "cuda-runtime")]
fn cuda_gpu_encode_error(error: JpegBaselineGpuEncodeError) -> crate::Error {
    match error {
        JpegBaselineGpuEncodeError::Encode(error) => error.into(),
        JpegBaselineGpuEncodeError::UnsupportedBackend { requested, .. } => {
            let reason = match requested {
                JpegBackend::Cpu => "JPEG Baseline CUDA encode does not accept Cpu backend",
                JpegBackend::Metal => "JPEG Baseline CUDA encode does not accept Metal backend",
                JpegBackend::Auto | JpegBackend::Cuda => {
                    "JPEG Baseline CUDA encode backend request is inconsistent"
                }
            };
            crate::Error::UnsupportedCudaRequest { reason }
        }
        JpegBaselineGpuEncodeError::InputExceedsOutputDimensions => {
            crate::Error::UnsupportedCudaRequest {
                reason: "JPEG Baseline CUDA encode input cannot exceed output dimensions",
            }
        }
        JpegBaselineGpuEncodeError::UnsupportedPixelFormat { .. } => {
            crate::Error::UnsupportedCudaRequest {
                reason: "JPEG Baseline CUDA encode supports only Gray8 and Rgb8 input buffers",
            }
        }
        JpegBaselineGpuEncodeError::IncompatibleSubsampling {
            subsampling,
            samples,
        } => JpegEncodeError::IncompatibleSubsampling {
            subsampling,
            samples,
        }
        .into(),
        JpegBaselineGpuEncodeError::RowByteCountOverflow => {
            cuda_request_error("JPEG Baseline CUDA encode row byte count overflow")
        }
        JpegBaselineGpuEncodeError::PitchTooShort { .. } => crate::Error::UnsupportedCudaRequest {
            reason: "JPEG Baseline CUDA encode pitch is shorter than one row",
        },
        JpegBaselineGpuEncodeError::InputRangeOverflow => {
            cuda_request_error("JPEG Baseline CUDA encode input range overflow")
        }
        JpegBaselineGpuEncodeError::InputRangeExceedsBuffer { .. } => {
            crate::Error::UnsupportedCudaRequest {
                reason: "JPEG Baseline CUDA encode input range exceeds buffer length",
            }
        }
        JpegBaselineGpuEncodeError::PitchTooLarge => crate::Error::UnsupportedCudaRequest {
            reason: "JPEG Baseline CUDA encode pitch exceeds CUDA kernel limits",
        },
        JpegBaselineGpuEncodeError::InputOffsetTooLarge => crate::Error::UnsupportedCudaRequest {
            reason: "JPEG Baseline CUDA encode input offset exceeds CUDA kernel limits",
        },
        JpegBaselineGpuEncodeError::EntropyOffsetTooLarge => crate::Error::UnsupportedCudaRequest {
            reason: "JPEG Baseline CUDA encode entropy offset exceeds CUDA kernel limits",
        },
        JpegBaselineGpuEncodeError::EntropyCapacityTooLarge => {
            crate::Error::UnsupportedCudaRequest {
                reason: "JPEG Baseline CUDA encode entropy capacity exceeds CUDA kernel limits",
            }
        }
        JpegBaselineGpuEncodeError::BatchEntropyCapacityOverflow => {
            cuda_request_error("JPEG Baseline CUDA batch entropy capacity overflow")
        }
    }
}

#[cfg(feature = "cuda-runtime")]
fn cuda_request_error(reason: &'static str) -> crate::Error {
    crate::Error::UnsupportedCudaRequest { reason }
}
