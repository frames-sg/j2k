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
#[derive(Clone, Copy)]
struct FastRgb8PacketParts<'a> {
    sampling: CudaJpegRgb8Sampling,
    dimensions: (u32, u32),
    mcus_per_row: u32,
    mcu_rows: u32,
    entropy_bytes: &'a [u8],
    entropy_checkpoints: &'a [JpegEntropyCheckpointV1],
    y_quant: &'a [u16; 64],
    cb_quant: &'a [u16; 64],
    cr_quant: &'a [u16; 64],
    y_dc_table: &'a JpegHuffmanTable,
    y_ac_table: &'a JpegHuffmanTable,
    cb_dc_table: &'a JpegHuffmanTable,
    cb_ac_table: &'a JpegHuffmanTable,
    cr_dc_table: &'a JpegHuffmanTable,
    cr_ac_table: &'a JpegHuffmanTable,
}

#[cfg(feature = "cuda-runtime")]
#[derive(Debug)]
struct CudaRgb8PlanData<'a> {
    sampling: CudaJpegRgb8Sampling,
    dimensions: (u32, u32),
    mcus_per_row: u32,
    mcu_rows: u32,
    entropy_bytes: &'a [u8],
    entropy_checkpoints: Vec<CudaJpegEntropyCheckpoint>,
    y_quant: [u16; 64],
    cb_quant: [u16; 64],
    cr_quant: [u16; 64],
    y_dc_table: CudaJpegHuffmanTable,
    y_ac_table: CudaJpegHuffmanTable,
    cb_dc_table: CudaJpegHuffmanTable,
    cb_ac_table: CudaJpegHuffmanTable,
    cr_dc_table: CudaJpegHuffmanTable,
    cr_ac_table: CudaJpegHuffmanTable,
}

#[cfg(feature = "cuda-runtime")]
impl CudaRgb8PlanData<'_> {
    fn as_plan(&self) -> CudaJpegRgb8DecodePlan<'_> {
        CudaJpegRgb8DecodePlan {
            sampling: self.sampling,
            dimensions: self.dimensions,
            mcus_per_row: self.mcus_per_row,
            mcu_rows: self.mcu_rows,
            entropy_bytes: self.entropy_bytes,
            entropy_checkpoints: &self.entropy_checkpoints,
            y_quant: self.y_quant,
            cb_quant: self.cb_quant,
            cr_quant: self.cr_quant,
            y_dc_table: self.y_dc_table,
            y_ac_table: self.y_ac_table,
            cb_dc_table: self.cb_dc_table,
            cb_ac_table: self.cb_ac_table,
            cr_dc_table: self.cr_dc_table,
            cr_ac_table: self.cr_ac_table,
        }
    }
}

#[cfg(feature = "cuda-runtime")]
fn build_cuda_rgb8_plan_data<'a>(
    packet: &FastRgb8PacketParts<'a>,
    dimensions: (u32, u32),
) -> Result<CudaRgb8PlanData<'a>, Error> {
    if packet.dimensions != dimensions {
        return Err(Error::UnsupportedCudaRequest {
            reason: "J2K CUDA JPEG packet dimensions do not match decoder metadata",
        });
    }
    Ok(CudaRgb8PlanData {
        sampling: packet.sampling,
        dimensions,
        mcus_per_row: packet.mcus_per_row,
        mcu_rows: packet.mcu_rows,
        entropy_bytes: packet.entropy_bytes,
        entropy_checkpoints: cuda_entropy_checkpoints(packet.entropy_checkpoints),
        y_quant: *packet.y_quant,
        cb_quant: *packet.cb_quant,
        cr_quant: *packet.cr_quant,
        y_dc_table: cuda_huffman_table(packet.y_dc_table)?,
        y_ac_table: cuda_huffman_table(packet.y_ac_table)?,
        cb_dc_table: cuda_huffman_table(packet.cb_dc_table)?,
        cb_ac_table: cuda_huffman_table(packet.cb_ac_table)?,
        cr_dc_table: cuda_huffman_table(packet.cr_dc_table)?,
        cr_ac_table: cuda_huffman_table(packet.cr_ac_table)?,
    })
}

#[cfg(feature = "cuda-runtime")]
pub(crate) fn decode_owned_cuda_rgb8(
    bytes: &[u8],
    dimensions: (u32, u32),
    session: &mut CudaSession,
) -> Result<Surface, Error> {
    let packet = resolve_owned_rgb8_packet(bytes, session)?;
    let packet_parts = packet.parts();
    let plan_data = build_cuda_rgb8_plan_data(&packet_parts, dimensions)?;
    let plan = plan_data.as_plan();
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
    let packet_parts = packet.parts();
    let plan_data = build_cuda_rgb8_plan_data(&packet_parts, dimensions)?;
    let plan = plan_data.as_plan();
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
macro_rules! fast_rgb8_packet_parts {
    ($sampling:expr, $packet:expr $(,)?) => {{
        let packet = $packet;
        FastRgb8PacketParts {
            sampling: $sampling,
            dimensions: packet.dimensions,
            mcus_per_row: packet.mcus_per_row,
            mcu_rows: packet.mcu_rows,
            entropy_bytes: &packet.entropy_bytes,
            entropy_checkpoints: &packet.entropy_checkpoints,
            y_quant: &packet.y_quant,
            cb_quant: &packet.cb_quant,
            cr_quant: &packet.cr_quant,
            y_dc_table: &packet.y_dc_table,
            y_ac_table: &packet.y_ac_table,
            cb_dc_table: &packet.cb_dc_table,
            cb_ac_table: &packet.cb_ac_table,
            cr_dc_table: &packet.cr_dc_table,
            cr_ac_table: &packet.cr_ac_table,
        }
    }};
}

#[cfg(feature = "cuda-runtime")]
impl OwnedFastRgb8Packet {
    fn parts(&self) -> FastRgb8PacketParts<'_> {
        match self {
            Self::Fast420(packet) => {
                fast_rgb8_packet_parts!(CudaJpegRgb8Sampling::Fast420, packet)
            }
            Self::Fast422(packet) => {
                fast_rgb8_packet_parts!(CudaJpegRgb8Sampling::Fast422, packet)
            }
            Self::Fast444(packet) => {
                fast_rgb8_packet_parts!(CudaJpegRgb8Sampling::Fast444, packet)
            }
        }
    }
}

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

#[cfg(all(test, feature = "cuda-runtime"))]
mod tests {
    use super::{
        build_cuda_rgb8_plan_data, resolve_owned_rgb8_packet, CudaJpegRgb8Sampling, CudaSession,
        Error,
    };

    const BASELINE_420: &[u8] = include_bytes!("../fixtures/jpeg/baseline_420_16x16.jpg");
    const BASELINE_422: &[u8] = include_bytes!("../fixtures/jpeg/baseline_422_16x8.jpg");
    const BASELINE_444: &[u8] = include_bytes!("../fixtures/jpeg/baseline_444_8x8.jpg");

    #[test]
    fn packet_plan_helper_preserves_sampling_entropy_and_checkpoints() {
        for (input, dimensions, expected_sampling) in [
            (BASELINE_420, (16, 16), CudaJpegRgb8Sampling::Fast420),
            (BASELINE_422, (16, 8), CudaJpegRgb8Sampling::Fast422),
            (BASELINE_444, (8, 8), CudaJpegRgb8Sampling::Fast444),
        ] {
            let mut session = CudaSession::default();
            let packet = resolve_owned_rgb8_packet(input, &mut session).expect("owned packet");
            let parts = packet.parts();
            let expected_checkpoint = parts.entropy_checkpoints[0];
            let expected_entropy = parts.entropy_bytes;
            let plan_data = build_cuda_rgb8_plan_data(&parts, dimensions).expect("CUDA plan data");
            let plan = plan_data.as_plan();

            assert_eq!(plan.sampling, expected_sampling);
            assert_eq!(plan.dimensions, dimensions);
            assert_eq!(plan.entropy_bytes, expected_entropy);
            assert_eq!(
                plan.entropy_checkpoints.len(),
                packet.parts().entropy_checkpoints.len()
            );
            let actual_checkpoint = plan.entropy_checkpoints[0];
            assert_eq!(actual_checkpoint.mcu_index, expected_checkpoint.mcu_index);
            assert_eq!(
                actual_checkpoint.entropy_pos,
                expected_checkpoint.entropy_pos
            );
            assert_eq!(actual_checkpoint.bit_acc, expected_checkpoint.bit_acc);
            assert_eq!(actual_checkpoint.bit_count, expected_checkpoint.bit_count);
            assert_eq!(actual_checkpoint.y_prev_dc, expected_checkpoint.y_prev_dc);
            assert_eq!(actual_checkpoint.cb_prev_dc, expected_checkpoint.cb_prev_dc);
            assert_eq!(actual_checkpoint.cr_prev_dc, expected_checkpoint.cr_prev_dc);
            assert_eq!(actual_checkpoint.reserved, expected_checkpoint.reserved);
        }
    }

    #[test]
    fn packet_plan_helper_rejects_decoder_dimension_mismatch() {
        let mut session = CudaSession::default();
        let packet =
            resolve_owned_rgb8_packet(BASELINE_420, &mut session).expect("owned fast420 packet");
        let packet_parts = packet.parts();
        let error = build_cuda_rgb8_plan_data(&packet_parts, (15, 16))
            .expect_err("metadata mismatch must fail closed");

        assert!(matches!(
            error,
            Error::UnsupportedCudaRequest {
                reason: "J2K CUDA JPEG packet dimensions do not match decoder metadata"
            }
        ));
    }
}
