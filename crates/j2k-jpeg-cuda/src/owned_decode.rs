// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "cuda-runtime")]
use std::sync::Arc;

#[cfg(feature = "cuda-runtime")]
use j2k_core::{BackendKind, PixelFormat};

use crate::{CudaSession, Error, Surface};

#[cfg(feature = "cuda-runtime")]
use j2k_cuda_runtime::{
    CudaDeviceBuffer, CudaError, CudaJpegEntropyCheckpoint, CudaJpegHuffmanTable,
    CudaJpegRgb8DecodePlan, CudaJpegRgb8Sampling,
};
#[cfg(feature = "cuda-runtime")]
use j2k_jpeg::adapter::{
    JpegEntropyCheckpointV1, JpegFast420PacketV1, JpegFast422PacketV1, JpegFast444PacketV1,
    JpegHuffmanTable,
};
#[cfg(feature = "cuda-runtime")]
use j2k_jpeg::{JpegCapabilityReport, JpegCapabilityRequest, JpegDecodeOp};

#[cfg(feature = "cuda-runtime")]
use crate::surface::{CudaJpegDecodePath, CudaSurfaceStats, Storage};

pub(crate) fn unsupported_owned_cuda_output_format() -> Error {
    Error::UnsupportedCudaRequest {
        reason: "J2K CUDA JPEG owned decode currently supports full-frame RGB8 output only",
    }
}

#[cfg(feature = "cuda-runtime")]
const UNSUPPORTED_CHUNKED_ENTROPY_DIAGNOSTIC_INPUT: &str =
    "J2K CUDA JPEG chunked entropy diagnostic currently supports baseline 8-bit YCbCr 4:2:0 RGB8 inputs only";

#[cfg(feature = "cuda-runtime")]
const INVALID_CHUNKED_ENTROPY_DIAGNOSTIC_ARGUMENT: &str =
    "J2K CUDA JPEG chunked entropy diagnostic config or input is invalid";

#[cfg(feature = "cuda-runtime")]
pub(crate) fn decode_owned_cuda_rgb8(
    bytes: &[u8],
    dimensions: (u32, u32),
    session: &mut CudaSession,
) -> Result<Surface, Error> {
    let packet = resolve_owned_rgb8_packet(bytes, session)?;
    if packet.dimensions() != dimensions {
        return Err(Error::UnsupportedCudaRequest {
            reason: "J2K CUDA JPEG packet dimensions do not match decoder metadata",
        });
    }

    let checkpoints = cuda_entropy_checkpoints(packet.entropy_checkpoints());
    let plan = match &packet {
        OwnedFastRgb8Packet::Fast420(packet) => cuda_decode_plan(
            CudaJpegRgb8Sampling::Fast420,
            packet.as_ref(),
            dimensions,
            &checkpoints,
        )?,
        OwnedFastRgb8Packet::Fast422(packet) => cuda_decode_plan(
            CudaJpegRgb8Sampling::Fast422,
            packet.as_ref(),
            dimensions,
            &checkpoints,
        )?,
        OwnedFastRgb8Packet::Fast444(packet) => cuda_decode_plan(
            CudaJpegRgb8Sampling::Fast444,
            packet.as_ref(),
            dimensions,
            &checkpoints,
        )?,
    };
    let context = session.cuda_context()?;
    let output = context
        .decode_jpeg_rgb8_owned(&plan)
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

#[cfg(feature = "cuda-runtime")]
pub(crate) fn decode_owned_cuda_rgb8_into(
    bytes: &[u8],
    dimensions: (u32, u32),
    session: &mut CudaSession,
    output: &CudaDeviceBuffer,
    pitch_bytes: usize,
) -> Result<CudaSurfaceStats, Error> {
    let packet = resolve_owned_rgb8_packet(bytes, session)?;
    if packet.dimensions() != dimensions {
        return Err(Error::UnsupportedCudaRequest {
            reason: "J2K CUDA JPEG packet dimensions do not match decoder metadata",
        });
    }
    let checkpoints = cuda_entropy_checkpoints(packet.entropy_checkpoints());
    let plan = match &packet {
        OwnedFastRgb8Packet::Fast420(packet) => cuda_decode_plan(
            CudaJpegRgb8Sampling::Fast420,
            packet.as_ref(),
            dimensions,
            &checkpoints,
        )?,
        OwnedFastRgb8Packet::Fast422(packet) => cuda_decode_plan(
            CudaJpegRgb8Sampling::Fast422,
            packet.as_ref(),
            dimensions,
            &checkpoints,
        )?,
        OwnedFastRgb8Packet::Fast444(packet) => cuda_decode_plan(
            CudaJpegRgb8Sampling::Fast444,
            packet.as_ref(),
            dimensions,
            &checkpoints,
        )?,
    };
    let context = session.cuda_context()?;
    let stats = context
        .decode_jpeg_rgb8_owned_into(&plan, output, pitch_bytes)
        .map_err(cuda_owned_decode_error)?;
    Ok(CudaSurfaceStats {
        kernel_dispatches: stats.kernel_dispatches(),
        copy_kernel_dispatches: stats.copy_kernel_dispatches(),
        decode_kernel_dispatches: stats.decode_kernel_dispatches(),
        hardware_decode: false,
        decode_path: CudaJpegDecodePath::OwnedCuda,
    })
}

#[cfg(feature = "cuda-runtime")]
pub(crate) fn diagnose_owned_cuda_420_entropy(
    bytes: &[u8],
    config: j2k_cuda_runtime::CudaJpegChunkedEntropyConfig,
    session: &mut CudaSession,
) -> Result<j2k_cuda_runtime::CudaJpegChunkedEntropyReport, Error> {
    validate_chunked_entropy_diagnostic_input(bytes)?;
    config
        .validate()
        .map_err(cuda_chunked_entropy_diagnostic_error)?;
    let packet = session.resolve_owned_fast420_packet(bytes)?;
    let plan = j2k_cuda_runtime::CudaJpegChunkedEntropyPlan {
        config,
        entropy_bytes: &packet.entropy_bytes,
        y_dc_table: cuda_huffman_table(&packet.y_dc_table)?,
        y_ac_table: cuda_huffman_table(&packet.y_ac_table)?,
        cb_dc_table: cuda_huffman_table(&packet.cb_dc_table)?,
        cb_ac_table: cuda_huffman_table(&packet.cb_ac_table)?,
        cr_dc_table: cuda_huffman_table(&packet.cr_dc_table)?,
        cr_ac_table: cuda_huffman_table(&packet.cr_ac_table)?,
    };
    session
        .cuda_context()?
        .diagnose_jpeg_420_entropy_self_sync(&plan)
        .map_err(cuda_chunked_entropy_diagnostic_error)
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
enum OwnedFastRgb8Packet {
    Fast420(Arc<JpegFast420PacketV1>),
    Fast422(Arc<JpegFast422PacketV1>),
    Fast444(Arc<JpegFast444PacketV1>),
}

#[cfg(feature = "cuda-runtime")]
impl OwnedFastRgb8Packet {
    fn dimensions(&self) -> (u32, u32) {
        match self {
            Self::Fast420(packet) => packet.dimensions,
            Self::Fast422(packet) => packet.dimensions,
            Self::Fast444(packet) => packet.dimensions,
        }
    }

    fn entropy_checkpoints(&self) -> &[JpegEntropyCheckpointV1] {
        match self {
            Self::Fast420(packet) => &packet.entropy_checkpoints,
            Self::Fast422(packet) => &packet.entropy_checkpoints,
            Self::Fast444(packet) => &packet.entropy_checkpoints,
        }
    }
}

#[cfg(feature = "cuda-runtime")]
trait FastRgb8Packet {
    fn mcus_per_row(&self) -> u32;
    fn mcu_rows(&self) -> u32;
    fn entropy_bytes(&self) -> &[u8];
    fn y_quant(&self) -> [u16; 64];
    fn cb_quant(&self) -> [u16; 64];
    fn cr_quant(&self) -> [u16; 64];
    fn y_dc_table(&self) -> &JpegHuffmanTable;
    fn y_ac_table(&self) -> &JpegHuffmanTable;
    fn cb_dc_table(&self) -> &JpegHuffmanTable;
    fn cb_ac_table(&self) -> &JpegHuffmanTable;
    fn cr_dc_table(&self) -> &JpegHuffmanTable;
    fn cr_ac_table(&self) -> &JpegHuffmanTable;
}

#[cfg(feature = "cuda-runtime")]
macro_rules! impl_fast_rgb8_packet {
    ($packet:ty) => {
        impl FastRgb8Packet for $packet {
            fn mcus_per_row(&self) -> u32 {
                self.mcus_per_row
            }

            fn mcu_rows(&self) -> u32 {
                self.mcu_rows
            }

            fn entropy_bytes(&self) -> &[u8] {
                &self.entropy_bytes
            }

            fn y_quant(&self) -> [u16; 64] {
                self.y_quant
            }

            fn cb_quant(&self) -> [u16; 64] {
                self.cb_quant
            }

            fn cr_quant(&self) -> [u16; 64] {
                self.cr_quant
            }

            fn y_dc_table(&self) -> &JpegHuffmanTable {
                &self.y_dc_table
            }

            fn y_ac_table(&self) -> &JpegHuffmanTable {
                &self.y_ac_table
            }

            fn cb_dc_table(&self) -> &JpegHuffmanTable {
                &self.cb_dc_table
            }

            fn cb_ac_table(&self) -> &JpegHuffmanTable {
                &self.cb_ac_table
            }

            fn cr_dc_table(&self) -> &JpegHuffmanTable {
                &self.cr_dc_table
            }

            fn cr_ac_table(&self) -> &JpegHuffmanTable {
                &self.cr_ac_table
            }
        }
    };
}

#[cfg(feature = "cuda-runtime")]
impl_fast_rgb8_packet!(JpegFast420PacketV1);
#[cfg(feature = "cuda-runtime")]
impl_fast_rgb8_packet!(JpegFast422PacketV1);
#[cfg(feature = "cuda-runtime")]
impl_fast_rgb8_packet!(JpegFast444PacketV1);

#[cfg(feature = "cuda-runtime")]
fn resolve_owned_rgb8_packet(
    bytes: &[u8],
    session: &mut CudaSession,
) -> Result<OwnedFastRgb8Packet, Error> {
    let report = JpegCapabilityReport::inspect(
        bytes,
        JpegCapabilityRequest {
            op: JpegDecodeOp::Full,
            fmt: PixelFormat::Rgb8,
        },
    )?;
    if !report.owned_cuda.eligible {
        return Err(Error::UnsupportedCudaRequest {
            reason: report.owned_cuda.reason.unwrap_or(
                "J2K CUDA JPEG decode currently supports baseline 8-bit YCbCr 4:2:0, 4:2:2, or 4:4:4 RGB8 output",
            ),
        });
    }
    if report.device.matches_fast_420 {
        return session
            .resolve_owned_fast420_packet(bytes)
            .map(OwnedFastRgb8Packet::Fast420);
    }
    if report.device.matches_fast_422 {
        return session
            .resolve_owned_fast422_packet(bytes)
            .map(OwnedFastRgb8Packet::Fast422);
    }
    if report.device.matches_fast_444 {
        return session
            .resolve_owned_fast444_packet(bytes)
            .map(OwnedFastRgb8Packet::Fast444);
    }
    Err(Error::UnsupportedCudaRequest {
        reason: "J2K CUDA JPEG decode currently supports baseline 8-bit YCbCr 4:2:0, 4:2:2, or 4:4:4 RGB8 output",
    })
}

#[cfg(feature = "cuda-runtime")]
fn validate_chunked_entropy_diagnostic_input(bytes: &[u8]) -> Result<(), Error> {
    let report = JpegCapabilityReport::inspect(
        bytes,
        JpegCapabilityRequest {
            op: JpegDecodeOp::Full,
            fmt: PixelFormat::Rgb8,
        },
    )?;
    if report.owned_cuda.eligible && report.device.matches_fast_420 {
        return Ok(());
    }
    Err(Error::UnsupportedCudaRequest {
        reason: UNSUPPORTED_CHUNKED_ENTROPY_DIAGNOSTIC_INPUT,
    })
}

#[cfg(feature = "cuda-runtime")]
fn cuda_decode_plan<'a>(
    sampling: CudaJpegRgb8Sampling,
    packet: &'a impl FastRgb8Packet,
    dimensions: (u32, u32),
    checkpoints: &'a [CudaJpegEntropyCheckpoint],
) -> Result<CudaJpegRgb8DecodePlan<'a>, Error> {
    Ok(CudaJpegRgb8DecodePlan {
        sampling,
        dimensions,
        mcus_per_row: packet.mcus_per_row(),
        mcu_rows: packet.mcu_rows(),
        entropy_bytes: packet.entropy_bytes(),
        entropy_checkpoints: checkpoints,
        y_quant: packet.y_quant(),
        cb_quant: packet.cb_quant(),
        cr_quant: packet.cr_quant(),
        y_dc_table: cuda_huffman_table(packet.y_dc_table())?,
        y_ac_table: cuda_huffman_table(packet.y_ac_table())?,
        cb_dc_table: cuda_huffman_table(packet.cb_dc_table())?,
        cb_ac_table: cuda_huffman_table(packet.cb_ac_table())?,
        cr_dc_table: cuda_huffman_table(packet.cr_dc_table())?,
        cr_ac_table: cuda_huffman_table(packet.cr_ac_table())?,
    })
}

#[cfg(feature = "cuda-runtime")]
fn cuda_entropy_checkpoints(
    checkpoints: &[JpegEntropyCheckpointV1],
) -> Vec<CudaJpegEntropyCheckpoint> {
    checkpoints
        .iter()
        .copied()
        .map(cuda_entropy_checkpoint)
        .collect()
}

#[cfg(feature = "cuda-runtime")]
fn cuda_huffman_table(table: &JpegHuffmanTable) -> Result<CudaJpegHuffmanTable, Error> {
    CudaJpegHuffmanTable::from_jpeg_bits_values(table.bits, table.values_len, table.values)
        .map_err(cuda_owned_decode_error)
}

#[cfg(feature = "cuda-runtime")]
fn cuda_owned_decode_error(error: CudaError) -> Error {
    match error {
        CudaError::Unavailable { .. } => Error::CudaUnavailable,
        CudaError::InvalidArgument { .. } => Error::UnsupportedCudaRequest {
            reason: "J2K CUDA JPEG owned decode cannot handle this image or runtime build",
        },
        other => Error::CudaRuntime {
            message: other.to_string(),
        },
    }
}

#[cfg(feature = "cuda-runtime")]
fn cuda_chunked_entropy_diagnostic_error(error: CudaError) -> Error {
    match error {
        CudaError::Unavailable { .. } => Error::CudaUnavailable,
        CudaError::InvalidArgument { .. } => Error::UnsupportedCudaRequest {
            reason: INVALID_CHUNKED_ENTROPY_DIAGNOSTIC_ARGUMENT,
        },
        other => Error::CudaRuntime {
            message: other.to_string(),
        },
    }
}

#[cfg(feature = "cuda-runtime")]
fn cuda_entropy_checkpoint(value: JpegEntropyCheckpointV1) -> CudaJpegEntropyCheckpoint {
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
