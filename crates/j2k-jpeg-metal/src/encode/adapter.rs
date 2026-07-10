// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_jpeg::adapter::{
    JpegBaselineEncodeTables, JpegBaselineGpuEncodeBatchPlan, JpegBaselineGpuEncodeError,
    JpegBaselineGpuEncodeHostAdapter, JpegBaselineGpuEncodeParams, JpegBaselineGpuEncodeTile,
    JpegBaselineGpuEncodeTilePlan, JpegBaselineHuffmanTable,
};
use j2k_jpeg::{JpegBackend, JpegEncodeError};

use super::{JpegBaselineMetalEncodeTile, MetalJpegBaselineEncodeAdapter};
use crate::compute;

fn compute_huffman_table(
    source: &JpegBaselineHuffmanTable,
) -> compute::JpegBaselineEncodeHuffmanTable {
    compute::JpegBaselineEncodeHuffmanTable {
        codes: source.codes,
        lens: source.lens,
    }
}

impl<'tile> JpegBaselineGpuEncodeHostAdapter<JpegBaselineMetalEncodeTile<'tile>>
    for MetalJpegBaselineEncodeAdapter<'_>
{
    type Error = crate::Error;
    type SourceKey = u64;

    fn backend(&self) -> JpegBackend {
        JpegBackend::Metal
    }

    fn source_key(&self, tile: &JpegBaselineMetalEncodeTile<'tile>) -> Self::SourceKey {
        tile.buffer_trusted().gpu_address()
    }

    fn gpu_tile(
        &self,
        tile: JpegBaselineMetalEncodeTile<'tile>,
    ) -> Result<JpegBaselineGpuEncodeTile, Self::Error> {
        metal_gpu_tile(tile)
    }

    fn map_plan_error(&self, error: JpegBaselineGpuEncodeError) -> Self::Error {
        metal_gpu_encode_error(error)
    }

    fn encode_tile_entropy(
        &mut self,
        tile: JpegBaselineMetalEncodeTile<'tile>,
        tables: &JpegBaselineEncodeTables,
        plan: JpegBaselineGpuEncodeTilePlan,
    ) -> Result<Vec<u8>, Self::Error> {
        compute::encode_jpeg_baseline_entropy_with_session(
            self.session,
            &compute::JpegBaselineEntropyEncodeJob {
                input: tile.buffer_trusted(),
                input_offset: tile.byte_offset,
                params: metal_encode_params(plan.params),
                q_luma: tables.q_luma,
                q_chroma: tables.q_chroma,
                huff_dc_luma: compute_huffman_table(&tables.huff_dc_luma),
                huff_ac_luma: compute_huffman_table(&tables.huff_ac_luma),
                huff_dc_chroma: compute_huffman_table(&tables.huff_dc_chroma),
                huff_ac_chroma: compute_huffman_table(&tables.huff_ac_chroma),
                entropy_capacity: plan.entropy_capacity,
            },
        )
    }

    fn encode_batch_entropy(
        &mut self,
        tiles: &[JpegBaselineMetalEncodeTile<'tile>],
        tables: &JpegBaselineEncodeTables,
        plan: JpegBaselineGpuEncodeBatchPlan,
    ) -> Result<Vec<Vec<u8>>, Self::Error> {
        let params = plan.params.into_iter().map(metal_encode_params).collect();
        compute::encode_jpeg_baseline_entropy_batch_with_session(
            self.session,
            &compute::JpegBaselineEntropyEncodeBatchJob {
                input: tiles[0].buffer_trusted(),
                params,
                q_luma: tables.q_luma,
                q_chroma: tables.q_chroma,
                huff_dc_luma: compute_huffman_table(&tables.huff_dc_luma),
                huff_ac_luma: compute_huffman_table(&tables.huff_ac_luma),
                huff_dc_chroma: compute_huffman_table(&tables.huff_dc_chroma),
                huff_ac_chroma: compute_huffman_table(&tables.huff_ac_chroma),
                entropy_capacity: plan.total_entropy_capacity,
            },
        )
    }
}

fn metal_gpu_tile(
    tile: JpegBaselineMetalEncodeTile<'_>,
) -> Result<JpegBaselineGpuEncodeTile, crate::Error> {
    let buffer_len =
        usize::try_from(tile.buffer_trusted().length()).map_err(|_| crate::Error::MetalKernel {
            message: "JPEG Baseline Metal encode buffer length exceeds usize".to_string(),
        })?;
    Ok(JpegBaselineGpuEncodeTile {
        byte_offset: tile.byte_offset,
        width: tile.width,
        height: tile.height,
        pitch_bytes: tile.pitch_bytes,
        output_width: tile.output_width,
        output_height: tile.output_height,
        format: tile.format,
        buffer_len,
    })
}

fn metal_encode_params(params: JpegBaselineGpuEncodeParams) -> compute::JpegBaselineEncodeParams {
    compute::JpegBaselineEncodeParams {
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

fn metal_gpu_encode_error(error: JpegBaselineGpuEncodeError) -> crate::Error {
    match error {
        JpegBaselineGpuEncodeError::Encode(error) => error.into(),
        JpegBaselineGpuEncodeError::UnsupportedBackend { requested, .. } => {
            let reason = match requested {
                JpegBackend::Cpu => "JPEG Baseline Metal encode does not accept Cpu backend",
                JpegBackend::Cuda => "JPEG Baseline Metal encode does not accept Cuda backend",
                JpegBackend::Auto | JpegBackend::Metal => {
                    "JPEG Baseline Metal encode backend request is inconsistent"
                }
            };
            crate::Error::UnsupportedMetalRequest { reason }
        }
        JpegBaselineGpuEncodeError::InputExceedsOutputDimensions => {
            crate::Error::UnsupportedMetalRequest {
                reason: "JPEG Baseline Metal encode input cannot exceed output dimensions",
            }
        }
        JpegBaselineGpuEncodeError::UnsupportedPixelFormat { .. } => {
            crate::Error::UnsupportedMetalRequest {
                reason: "JPEG Baseline Metal encode supports only Gray8 and Rgb8 input buffers",
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
            metal_kernel_error("JPEG Baseline Metal encode row byte count overflow")
        }
        JpegBaselineGpuEncodeError::PitchTooShort {
            row_bytes,
            pitch_bytes,
        } => crate::Error::MetalKernel {
            message: format!(
                "JPEG Baseline Metal encode pitch is shorter than one row: need {row_bytes}, got {pitch_bytes}"
            ),
        },
        JpegBaselineGpuEncodeError::InputRangeOverflow => {
            metal_kernel_error("JPEG Baseline Metal encode input range overflow")
        }
        JpegBaselineGpuEncodeError::InputRangeExceedsBuffer {
            required_end,
            buffer_len,
        } => crate::Error::MetalKernel {
            message: format!(
                "JPEG Baseline Metal encode input range exceeds buffer length: need {required_end}, buffer has {buffer_len}"
            ),
        },
        JpegBaselineGpuEncodeError::PitchTooLarge => crate::Error::MetalKernel {
            message: "JPEG Baseline Metal encode pitch exceeds u32".to_string(),
        },
        JpegBaselineGpuEncodeError::InputOffsetTooLarge => crate::Error::MetalKernel {
            message: "JPEG Baseline Metal batch input offset exceeds u32".to_string(),
        },
        JpegBaselineGpuEncodeError::EntropyOffsetTooLarge => crate::Error::MetalKernel {
            message: "JPEG Baseline Metal batch entropy offset exceeds u32".to_string(),
        },
        JpegBaselineGpuEncodeError::EntropyCapacityTooLarge => {
            crate::Error::UnsupportedMetalRequest {
                reason: "JPEG Baseline Metal encode entropy capacity exceeds Metal kernel limits",
            }
        }
        JpegBaselineGpuEncodeError::BatchEntropyCapacityOverflow => {
            metal_kernel_error("JPEG Baseline Metal batch entropy capacity overflow")
        }
    }
}

fn metal_kernel_error(message: &'static str) -> crate::Error {
    crate::Error::MetalKernel {
        message: message.to_string(),
    }
}
