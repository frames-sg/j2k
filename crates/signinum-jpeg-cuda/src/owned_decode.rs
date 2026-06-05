// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "cuda-runtime")]
use signinum_core::{BackendKind, PixelFormat};

use crate::{CudaSession, Error, Surface};

#[cfg(feature = "cuda-runtime")]
use signinum_cuda_runtime::{
    CudaError, CudaJpeg420Rgb8DecodePlan, CudaJpegEntropyCheckpoint, CudaJpegHuffmanTable,
};
#[cfg(feature = "cuda-runtime")]
use signinum_jpeg::adapter::{
    build_metal_fast420_packet, JpegMetalEntropyCheckpointV1, MetalHuffmanTable,
};

#[cfg(feature = "cuda-runtime")]
use crate::surface::{CudaJpegDecodePath, CudaSurfaceStats, Storage};

pub(crate) fn unsupported_owned_cuda_output_format() -> Error {
    Error::UnsupportedCudaRequest {
        reason: "Signinum CUDA JPEG owned decode currently supports full-frame RGB8 output only",
    }
}

#[cfg(feature = "cuda-runtime")]
pub(crate) fn decode_owned_cuda_rgb8(
    bytes: &[u8],
    dimensions: (u32, u32),
    session: &mut CudaSession,
) -> Result<Surface, Error> {
    let packet = build_metal_fast420_packet(bytes).map_err(|_| Error::UnsupportedCudaRequest {
        reason:
            "Signinum CUDA JPEG decode currently supports baseline 8-bit YCbCr 4:2:0 RGB8 output",
    })?;
    if packet.dimensions != dimensions {
        return Err(Error::UnsupportedCudaRequest {
            reason: "Signinum CUDA JPEG packet dimensions do not match decoder metadata",
        });
    }

    let checkpoints: Vec<CudaJpegEntropyCheckpoint> = packet
        .entropy_checkpoints
        .iter()
        .copied()
        .map(cuda_entropy_checkpoint)
        .collect();
    let plan = CudaJpeg420Rgb8DecodePlan {
        dimensions,
        mcus_per_row: packet.mcus_per_row,
        mcu_rows: packet.mcu_rows,
        entropy_bytes: &packet.entropy_bytes,
        entropy_checkpoints: &checkpoints,
        y_quant: packet.y_quant,
        cb_quant: packet.cb_quant,
        cr_quant: packet.cr_quant,
        y_dc_table: cuda_huffman_table(&packet.y_dc_table)?,
        y_ac_table: cuda_huffman_table(&packet.y_ac_table)?,
        cb_dc_table: cuda_huffman_table(&packet.cb_dc_table)?,
        cb_ac_table: cuda_huffman_table(&packet.cb_ac_table)?,
        cr_dc_table: cuda_huffman_table(&packet.cr_dc_table)?,
        cr_ac_table: cuda_huffman_table(&packet.cr_ac_table)?,
    };
    let context = session.cuda_context()?;
    let output = context
        .decode_jpeg_420_rgb8_owned(&plan)
        .map_err(cuda_owned_decode_error)?;
    let (buffer, stats) = output.into_parts();
    Ok(Surface {
        backend: BackendKind::Cuda,
        dimensions,
        fmt: PixelFormat::Rgb8,
        pitch_bytes: dimensions.0 as usize * PixelFormat::Rgb8.bytes_per_pixel(),
        stats: CudaSurfaceStats {
            kernel_dispatches: stats.kernel_dispatches(),
            copy_kernel_dispatches: stats.copy_kernel_dispatches(),
            decode_kernel_dispatches: stats.decode_kernel_dispatches(),
            hardware_decode: false,
            decode_path: CudaJpegDecodePath::OwnedCuda,
        },
        storage: Storage::Cuda(buffer),
    })
}

#[cfg(not(feature = "cuda-runtime"))]
pub(crate) fn decode_owned_cuda_rgb8(
    _bytes: &[u8],
    _dimensions: (u32, u32),
    _session: &mut CudaSession,
) -> Result<Surface, Error> {
    Err(Error::CudaUnavailable)
}

#[cfg(feature = "cuda-runtime")]
fn cuda_huffman_table(table: &MetalHuffmanTable) -> Result<CudaJpegHuffmanTable, Error> {
    CudaJpegHuffmanTable::from_jpeg_bits_values(table.bits, table.values_len, table.values)
        .map_err(cuda_owned_decode_error)
}

#[cfg(feature = "cuda-runtime")]
fn cuda_owned_decode_error(error: CudaError) -> Error {
    match error {
        CudaError::Unavailable { .. } => Error::CudaUnavailable,
        CudaError::InvalidArgument { .. } => Error::UnsupportedCudaRequest {
            reason: "Signinum CUDA JPEG owned decode cannot handle this image or runtime build",
        },
        other => Error::CudaRuntime {
            message: other.to_string(),
        },
    }
}

#[cfg(feature = "cuda-runtime")]
fn cuda_entropy_checkpoint(value: JpegMetalEntropyCheckpointV1) -> CudaJpegEntropyCheckpoint {
    CudaJpegEntropyCheckpoint {
        mcu_index: value.mcu_index,
        entropy_pos: value.entropy_pos,
        bit_acc: value.bit_acc,
        bit_count: value.bit_count,
        y_prev_dc: value.y_prev_dc,
        cb_prev_dc: value.cb_prev_dc,
        cr_prev_dc: value.cr_prev_dc,
        reserved: value.reserved,
    }
}
