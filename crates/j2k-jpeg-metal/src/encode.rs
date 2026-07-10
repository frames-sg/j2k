// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::similar_names)]

#[cfg(target_os = "macos")]
use j2k_core::PixelFormat;
#[cfg(target_os = "macos")]
use j2k_jpeg::adapter::{
    encode_jpeg_baseline_gpu_batch, encode_jpeg_baseline_gpu_tile, JpegBaselineEncodeTables,
    JpegBaselineGpuEncodeBatchPlan, JpegBaselineGpuEncodeError, JpegBaselineGpuEncodeHostAdapter,
    JpegBaselineGpuEncodeParams, JpegBaselineGpuEncodeTile, JpegBaselineGpuEncodeTilePlan,
    JpegBaselineHuffmanTable,
};
use j2k_jpeg::{EncodedJpeg, JpegEncodeOptions};
#[cfg(target_os = "macos")]
use j2k_jpeg::{JpegBackend, JpegEncodeError};
#[cfg(target_os = "macos")]
use metal::{Buffer, BufferRef};

#[cfg(target_os = "macos")]
use crate::compute;

#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy)]
/// Metal buffer and layout metadata for one baseline JPEG encode tile.
pub struct JpegBaselineMetalEncodeTile<'a> {
    buffer: &'a Buffer,
    byte_offset: usize,
    width: u32,
    height: u32,
    pitch_bytes: usize,
    output_width: u32,
    output_height: u32,
    format: PixelFormat,
}

#[cfg(target_os = "macos")]
impl<'a> JpegBaselineMetalEncodeTile<'a> {
    /// Describe one Metal-resident source tile for baseline JPEG encoding.
    ///
    /// # Safety
    ///
    /// All commands that write the described source range must have completed
    /// before construction. The caller must keep that range immutable to both
    /// CPU and GPU writers while this tile or any copy can be used, and through
    /// actual completion of every GPU read submitted from one. The provided
    /// encode functions are synchronous and wait for those reads before
    /// returning. The buffer must be usable by the device behind each session
    /// passed to the safe encode functions.
    pub unsafe fn new(
        buffer: &'a Buffer,
        byte_offset: usize,
        dimensions: (u32, u32),
        pitch_bytes: usize,
        output_dimensions: (u32, u32),
        format: PixelFormat,
    ) -> Self {
        Self {
            buffer,
            byte_offset,
            width: dimensions.0,
            height: dimensions.1,
            pitch_bytes,
            output_width: output_dimensions.0,
            output_height: output_dimensions.1,
            format,
        }
    }

    /// Byte offset of the first source pixel.
    pub fn byte_offset(&self) -> usize {
        self.byte_offset
    }

    /// Dimensions of the valid input region.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Number of bytes between consecutive input rows.
    pub fn pitch_bytes(&self) -> usize {
        self.pitch_bytes
    }

    /// Dimensions of the encoded JPEG frame.
    pub fn output_dimensions(&self) -> (u32, u32) {
        (self.output_width, self.output_height)
    }

    /// Pixel format of the source buffer.
    pub fn pixel_format(&self) -> PixelFormat {
        self.format
    }

    /// Return the raw Metal source buffer.
    ///
    /// # Safety
    ///
    /// The caller must preserve the synchronization and immutability contract
    /// established by [`JpegBaselineMetalEncodeTile::new`].
    pub unsafe fn buffer(&self) -> &BufferRef {
        self.buffer_trusted().as_ref()
    }

    pub(crate) fn buffer_trusted(&self) -> &'a Buffer {
        self.buffer
    }
}

#[cfg(not(target_os = "macos"))]
#[derive(Debug, Clone, Copy)]
/// Placeholder encode tile type for non-macOS builds.
pub struct JpegBaselineMetalEncodeTile<'a> {
    _private: core::marker::PhantomData<&'a ()>,
}

#[cfg(target_os = "macos")]
/// Encode one Metal-resident tile as a baseline JPEG frame.
pub fn encode_jpeg_baseline_from_metal_buffer(
    tile: JpegBaselineMetalEncodeTile<'_>,
    options: JpegEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<EncodedJpeg, crate::Error> {
    let mut adapter = MetalJpegBaselineEncodeAdapter { session };
    encode_jpeg_baseline_gpu_tile(tile, options, &mut adapter)
}

#[cfg(target_os = "macos")]
/// Encode multiple Metal-resident tiles as baseline JPEG frames.
///
/// Consecutive tiles that share a source Metal buffer are submitted through a
/// single entropy-kernel batch where possible. The returned frames preserve the
/// input order.
pub fn encode_jpeg_baseline_batch_from_metal_buffers(
    tiles: &[JpegBaselineMetalEncodeTile<'_>],
    options: JpegEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<EncodedJpeg>, crate::Error> {
    let mut adapter = MetalJpegBaselineEncodeAdapter { session };
    encode_jpeg_baseline_gpu_batch(tiles, options, &mut adapter)
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for batch Metal encode requests on non-macOS hosts.
pub fn encode_jpeg_baseline_batch_from_metal_buffers(
    tiles: &[JpegBaselineMetalEncodeTile<'_>],
    options: JpegEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<EncodedJpeg>, crate::Error> {
    let _ = (tiles, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for Metal encode requests on non-macOS hosts.
pub fn encode_jpeg_baseline_from_metal_buffer(
    tile: JpegBaselineMetalEncodeTile<'_>,
    options: JpegEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<EncodedJpeg, crate::Error> {
    let _ = (tile, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(target_os = "macos")]
fn compute_huffman_table(
    source: &JpegBaselineHuffmanTable,
) -> compute::JpegBaselineEncodeHuffmanTable {
    compute::JpegBaselineEncodeHuffmanTable {
        codes: source.codes,
        lens: source.lens,
    }
}

#[cfg(target_os = "macos")]
struct MetalJpegBaselineEncodeAdapter<'a> {
    session: &'a crate::MetalBackendSession,
}

#[cfg(target_os = "macos")]
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

#[cfg(target_os = "macos")]
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

#[cfg(target_os = "macos")]
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

#[cfg(target_os = "macos")]
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

#[cfg(target_os = "macos")]
fn metal_kernel_error(message: &'static str) -> crate::Error {
    crate::Error::MetalKernel {
        message: message.to_string(),
    }
}
