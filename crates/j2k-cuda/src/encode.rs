// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "cuda-runtime")]
use j2k::{
    J2kEncodeDispatchReport, J2kEncodeStageAccelerator, J2kForwardDwt53Level,
    J2kForwardDwt53Output, J2kForwardDwt97Level, J2kForwardDwt97Output, J2kHtj2kTileEncodeJob,
};
#[cfg(feature = "cuda-runtime")]
use j2k_core::BackendKind;
#[cfg(feature = "cuda-runtime")]
use j2k_core::{DeviceSubmission, DeviceSubmitSession, PixelFormat, ReadySubmission};
#[cfg(feature = "cuda-runtime")]
use j2k_cuda_runtime::{CudaContext, CudaDwt53Output, CudaDwt97Output, CudaHtj2kEncodeResources};
#[cfg(feature = "cuda-runtime")]
use std::time::{Duration, Instant};

#[cfg(feature = "cuda-runtime")]
use crate::{runtime::cuda_error, session::CudaSession};

mod api;
pub use self::api::{encode_j2k_lossless_with_cuda, encode_j2k_lossless_with_cuda_and_profile};
#[cfg(feature = "cuda-runtime")]
use self::api::{reject_non_cuda_encode_backend, strict_cuda_encode_options};
#[cfg(feature = "cuda-runtime")]
mod resident;
#[cfg(feature = "cuda-runtime")]
pub use self::resident::{
    CudaEncodedJ2k, CudaLosslessBufferEncodeOutcome, CudaLosslessEncodeOutcome,
    CudaLosslessEncodeResidency, CudaLosslessEncodeTile, CudaResidentCodestreamBuffer,
    SubmittedJ2kLosslessCudaEncode, SubmittedJ2kLosslessCudaEncodeBatch,
};
#[cfg(feature = "cuda-runtime")]
mod htj2k;
#[cfg(feature = "cuda-runtime")]
use self::htj2k::{cuda_encode_htj2k_device_tile_body, cuda_htj2k_encode_tables};
mod packetization;
mod stage;
#[cfg(feature = "cuda-runtime")]
use self::stage::time_cuda_stage;
pub use self::stage::{CudaEncodeStageAccelerator, CudaEncodeStageTimings};

#[cfg(feature = "cuda-runtime")]
/// Encode one CUDA-resident tile into host codestream bytes.
pub fn encode_lossless_from_cuda_buffer(
    tile: CudaLosslessEncodeTile<'_>,
    options: &j2k::J2kLosslessEncodeOptions,
    session: &mut CudaSession,
) -> Result<j2k::EncodedJ2k, crate::Error> {
    submit_lossless_from_cuda_buffer(tile, options, session)?.wait()
}

#[cfg(feature = "cuda-runtime")]
/// Submit one CUDA-resident tile encode for later host-byte collection.
pub fn submit_lossless_from_cuda_buffer(
    tile: CudaLosslessEncodeTile<'_>,
    options: &j2k::J2kLosslessEncodeOptions,
    session: &mut CudaSession,
) -> Result<SubmittedJ2kLosslessCudaEncode, crate::Error> {
    let result = encode_lossless_from_cuda_buffer_with_report(tile, options, session)
        .map(|outcome| outcome.encoded);
    Ok(SubmittedJ2kLosslessCudaEncode {
        inner: ReadySubmission::from_result(result),
    })
}

#[cfg(feature = "cuda-runtime")]
/// Encode one CUDA-resident tile and return a host-byte timing report.
#[doc(hidden)]
pub fn encode_lossless_from_cuda_buffer_with_report(
    tile: CudaLosslessEncodeTile<'_>,
    options: &j2k::J2kLosslessEncodeOptions,
    session: &mut CudaSession,
) -> Result<CudaLosslessEncodeOutcome, crate::Error> {
    validate_cuda_encode_options(*options)?;
    validate_cuda_encode_tile(tile)?;
    session.record_submit();
    encode_lossless_cuda_tile_with_report(tile, *options)
}

#[cfg(feature = "cuda-runtime")]
/// Encode one CUDA-resident tile and return CUDA-resident codestream bytes.
pub fn encode_lossless_from_cuda_buffer_to_cuda_buffer(
    tile: CudaLosslessEncodeTile<'_>,
    options: &j2k::J2kLosslessEncodeOptions,
    session: &mut CudaSession,
) -> Result<CudaEncodedJ2k, crate::Error> {
    Ok(
        encode_lossless_from_cuda_buffer_to_cuda_buffer_with_report(tile, options, session)?
            .encoded,
    )
}

#[cfg(feature = "cuda-runtime")]
/// Encode one CUDA-resident tile and return CUDA-resident codestream bytes with timings.
#[doc(hidden)]
pub fn encode_lossless_from_cuda_buffer_to_cuda_buffer_with_report(
    tile: CudaLosslessEncodeTile<'_>,
    options: &j2k::J2kLosslessEncodeOptions,
    session: &mut CudaSession,
) -> Result<CudaLosslessBufferEncodeOutcome, crate::Error> {
    let host_outcome = encode_lossless_from_cuda_buffer_with_report(tile, options, session)?;
    cuda_resident_codestream_outcome(tile, host_outcome)
}

#[cfg(feature = "cuda-runtime")]
/// Encode multiple CUDA-resident tiles into host codestream bytes.
pub fn encode_lossless_from_cuda_buffers(
    tiles: &[CudaLosslessEncodeTile<'_>],
    options: &j2k::J2kLosslessEncodeOptions,
    session: &mut CudaSession,
) -> Result<Vec<j2k::EncodedJ2k>, crate::Error> {
    submit_lossless_from_cuda_buffers(tiles, options, session)?.wait()
}

#[cfg(feature = "cuda-runtime")]
/// Submit multiple CUDA-resident tile encodes for later host-byte collection.
pub fn submit_lossless_from_cuda_buffers(
    tiles: &[CudaLosslessEncodeTile<'_>],
    options: &j2k::J2kLosslessEncodeOptions,
    session: &mut CudaSession,
) -> Result<SubmittedJ2kLosslessCudaEncodeBatch, crate::Error> {
    let result =
        encode_lossless_from_cuda_buffers_with_report(tiles, options, session).map(|outcomes| {
            outcomes
                .into_iter()
                .map(|outcome| outcome.encoded)
                .collect()
        });
    Ok(SubmittedJ2kLosslessCudaEncodeBatch {
        inner: ReadySubmission::from_result(result),
    })
}

#[cfg(feature = "cuda-runtime")]
/// Encode multiple CUDA-resident tiles and return host-byte timing reports.
#[doc(hidden)]
pub fn encode_lossless_from_cuda_buffers_with_report(
    tiles: &[CudaLosslessEncodeTile<'_>],
    options: &j2k::J2kLosslessEncodeOptions,
    session: &mut CudaSession,
) -> Result<Vec<CudaLosslessEncodeOutcome>, crate::Error> {
    if tiles.is_empty() {
        return Err(crate::Error::UnsupportedCudaRequest {
            reason: "J2K CUDA encode received an empty tile batch",
        });
    }
    validate_cuda_encode_options(*options)?;
    tiles
        .iter()
        .copied()
        .map(|tile| {
            validate_cuda_encode_tile(tile)?;
            session.record_submit();
            encode_lossless_cuda_tile_with_report(tile, *options)
        })
        .collect()
}

#[cfg(feature = "cuda-runtime")]
/// Encode multiple CUDA-resident tiles and return CUDA-resident codestream bytes.
pub fn encode_lossless_from_cuda_buffers_to_cuda_buffers(
    tiles: &[CudaLosslessEncodeTile<'_>],
    options: &j2k::J2kLosslessEncodeOptions,
    session: &mut CudaSession,
) -> Result<Vec<CudaEncodedJ2k>, crate::Error> {
    Ok(
        encode_lossless_from_cuda_buffers_to_cuda_buffers_with_report(tiles, options, session)?
            .into_iter()
            .map(|outcome| outcome.encoded)
            .collect(),
    )
}

#[cfg(feature = "cuda-runtime")]
/// Encode multiple CUDA-resident tiles and return CUDA-resident codestream bytes with timings.
#[doc(hidden)]
pub fn encode_lossless_from_cuda_buffers_to_cuda_buffers_with_report(
    tiles: &[CudaLosslessEncodeTile<'_>],
    options: &j2k::J2kLosslessEncodeOptions,
    session: &mut CudaSession,
) -> Result<Vec<CudaLosslessBufferEncodeOutcome>, crate::Error> {
    let host_outcomes = encode_lossless_from_cuda_buffers_with_report(tiles, options, session)?;
    tiles
        .iter()
        .copied()
        .zip(host_outcomes)
        .map(|(tile, outcome)| cuda_resident_codestream_outcome(tile, outcome))
        .collect()
}

#[cfg(feature = "cuda-runtime")]
fn cuda_resident_codestream_outcome(
    tile: CudaLosslessEncodeTile<'_>,
    host_outcome: CudaLosslessEncodeOutcome,
) -> Result<CudaLosslessBufferEncodeOutcome, crate::Error> {
    let upload_started = Instant::now();
    let codestream_len = host_outcome.encoded.codestream.len();
    let buffer = tile
        .buffer
        .context()
        .upload_pinned(&host_outcome.encoded.codestream)
        .map_err(cuda_error)?;
    let codestream_upload_duration = upload_started.elapsed();
    let encoded = CudaEncodedJ2k {
        encoded: host_outcome.encoded.clone(),
        codestream: CudaResidentCodestreamBuffer {
            buffer,
            byte_len: codestream_len,
        },
    };
    Ok(CudaLosslessBufferEncodeOutcome {
        encoded,
        host_outcome,
        codestream_upload_duration,
    })
}

#[cfg(feature = "cuda-runtime")]
fn validate_cuda_encode_options(
    options: j2k::J2kLosslessEncodeOptions,
) -> Result<(), crate::Error> {
    if options.block_coding_mode != j2k::J2kBlockCodingMode::HighThroughput {
        return Err(crate::Error::UnsupportedCudaRequest {
            reason: "J2K CUDA device-buffer encode currently requires HTJ2K block coding",
        });
    }
    if options.validation != j2k::J2kEncodeValidation::External {
        return Err(crate::Error::UnsupportedCudaRequest {
            reason: "J2K CUDA device-buffer encode requires external validation to avoid host input readback",
        });
    }
    Ok(())
}

#[cfg(feature = "cuda-runtime")]
fn validate_cuda_encode_tile(tile: CudaLosslessEncodeTile<'_>) -> Result<(), crate::Error> {
    if tile.width == 0 || tile.height == 0 || tile.output_width == 0 || tile.output_height == 0 {
        return Err(crate::Error::UnsupportedCudaRequest {
            reason: "J2K CUDA encode tile dimensions must be nonzero",
        });
    }
    if tile.width != tile.output_width || tile.height != tile.output_height {
        return Err(crate::Error::UnsupportedCudaRequest {
            reason: "J2K CUDA device-buffer encode does not yet support input padding",
        });
    }
    let format = cuda_encode_format(tile.format)?;
    let row_bytes = (tile.width as usize)
        .checked_mul(format.bytes_per_pixel)
        .ok_or(crate::Error::UnsupportedCudaRequest {
            reason: "J2K CUDA encode row byte count overflow",
        })?;
    if tile.pitch_bytes < row_bytes {
        return Err(crate::Error::UnsupportedCudaRequest {
            reason: "J2K CUDA encode tile pitch is shorter than one row",
        });
    }
    let required_end = tile
        .byte_offset
        .checked_add(
            tile.pitch_bytes
                .checked_mul(tile.height.saturating_sub(1) as usize)
                .and_then(|prefix| prefix.checked_add(row_bytes))
                .ok_or(crate::Error::UnsupportedCudaRequest {
                    reason: "J2K CUDA encode input byte range overflow",
                })?,
        )
        .ok_or(crate::Error::UnsupportedCudaRequest {
            reason: "J2K CUDA encode input byte range overflow",
        })?;
    if required_end > tile.buffer.byte_len() {
        return Err(crate::Error::UnsupportedCudaRequest {
            reason: "J2K CUDA encode input byte range exceeds buffer length",
        });
    }
    Ok(())
}

#[cfg(feature = "cuda-runtime")]
#[derive(Debug, Clone, Copy)]
pub(super) struct CudaEncodeFormat {
    pub(super) components: u8,
    pub(super) bit_depth: u8,
    bytes_per_pixel: usize,
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn cuda_component_count_u8(
    num_components: u16,
    reason: &'static str,
) -> core::result::Result<u8, &'static str> {
    u8::try_from(num_components).map_err(|_| reason)
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn cuda_encode_format(format: PixelFormat) -> Result<CudaEncodeFormat, crate::Error> {
    let components =
        u8::try_from(format.channels()).map_err(|_| crate::Error::UnsupportedCudaRequest {
            reason: "J2K CUDA encode received a pixel format with too many components",
        })?;
    let bit_depth = match format.bytes_per_sample() {
        1 => 8,
        2 => 16,
        _ => {
            return Err(crate::Error::UnsupportedCudaRequest {
                reason: "J2K CUDA encode received an unsupported sample width",
            });
        }
    };
    Ok(CudaEncodeFormat {
        components,
        bit_depth,
        bytes_per_pixel: format.bytes_per_pixel(),
    })
}

#[cfg(feature = "cuda-runtime")]
fn encode_lossless_cuda_tile_with_report(
    tile: CudaLosslessEncodeTile<'_>,
    options: j2k::J2kLosslessEncodeOptions,
) -> Result<CudaLosslessEncodeOutcome, crate::Error> {
    let encode_started = Instant::now();
    let format = cuda_encode_format(tile.format)?;
    let dummy_len = (tile.output_width as usize)
        .checked_mul(tile.output_height as usize)
        .and_then(|pixels| pixels.checked_mul(format.bytes_per_pixel))
        .ok_or(crate::Error::UnsupportedCudaRequest {
            reason: "J2K CUDA encode sample descriptor length overflow",
        })?;
    let dummy = vec![0u8; dummy_len];
    let samples = j2k::J2kLosslessSamples::new(
        &dummy,
        tile.output_width,
        tile.output_height,
        u16::from(format.components),
        format.bit_depth,
        false,
    )?;
    let context = tile.buffer.context();
    let resources = context
        .upload_htj2k_encode_resources(cuda_htj2k_encode_tables())
        .map_err(cuda_error)?;
    let mut accelerator = CudaDeviceBufferEncodeAccelerator {
        tile,
        context,
        resources,
        dispatch: J2kEncodeDispatchReport::default(),
        stage_timings: CudaEncodeStageTimings::default(),
    };
    let encoded = j2k::encode_j2k_lossless_with_accelerator(
        samples,
        &strict_cuda_encode_options(options),
        BackendKind::Cuda,
        &mut accelerator,
    )?;
    reject_non_cuda_encode_backend(&encoded)?;
    Ok(CudaLosslessEncodeOutcome {
        encoded,
        input_copy_used: false,
        resident: CudaLosslessEncodeResidency {
            coefficient_prep_used: accelerator.dispatch.deinterleave > 0,
            packetization_used: accelerator.dispatch.packetization > 0,
            codestream_assembly_used: false,
        },
        input_copy_duration: Duration::ZERO,
        encode_duration: encode_started.elapsed(),
        gpu_duration: None,
        validation_duration: Duration::ZERO,
        host_readback_duration: Duration::ZERO,
        stage_timings: accelerator.stage_timings,
    })
}

#[cfg(feature = "cuda-runtime")]
struct CudaDeviceBufferEncodeAccelerator<'a> {
    tile: CudaLosslessEncodeTile<'a>,
    context: CudaContext,
    resources: CudaHtj2kEncodeResources,
    dispatch: J2kEncodeDispatchReport,
    stage_timings: CudaEncodeStageTimings,
}

#[cfg(feature = "cuda-runtime")]
impl J2kEncodeStageAccelerator for CudaDeviceBufferEncodeAccelerator<'_> {
    fn dispatch_report(&self) -> J2kEncodeDispatchReport {
        self.dispatch
    }

    fn encode_htj2k_tile(
        &mut self,
        job: J2kHtj2kTileEncodeJob<'_>,
    ) -> core::result::Result<Option<Vec<u8>>, &'static str> {
        let Some(encoded) = cuda_encode_htj2k_device_tile_body(
            &self.context,
            &self.resources,
            self.tile,
            job,
            true,
        )?
        else {
            return Ok(None);
        };
        self.dispatch.deinterleave = self
            .dispatch
            .deinterleave
            .saturating_add(encoded.deinterleave_dispatches);
        self.dispatch.forward_rct = self
            .dispatch
            .forward_rct
            .saturating_add(encoded.forward_rct_dispatches);
        self.dispatch.forward_ict = self
            .dispatch
            .forward_ict
            .saturating_add(encoded.forward_ict_dispatches);
        self.dispatch.forward_dwt53 = self
            .dispatch
            .forward_dwt53
            .saturating_add(encoded.forward_dwt53_dispatches);
        self.dispatch.forward_dwt97 = self
            .dispatch
            .forward_dwt97
            .saturating_add(encoded.forward_dwt97_dispatches);
        self.dispatch.quantize_subband = self
            .dispatch
            .quantize_subband
            .saturating_add(encoded.quantize_dispatches);
        self.dispatch.ht_code_block = self
            .dispatch
            .ht_code_block
            .saturating_add(encoded.ht_code_block_dispatches);
        self.dispatch.packetization = self
            .dispatch
            .packetization
            .saturating_add(encoded.packetization_dispatches);
        self.stage_timings = self.stage_timings.saturating_add(encoded.timings);
        Ok(Some(encoded.tile_data))
    }
}

#[cfg(feature = "cuda-runtime")]
fn cuda_dwt53_output_to_j2k(
    output: &CudaDwt53Output,
) -> core::result::Result<J2kForwardDwt53Output, &'static str> {
    let (ll_width, ll_height) = output.ll_dimensions();
    let transformed = output.transformed();
    let full_width = output
        .levels()
        .first()
        .map_or(ll_width, |level| level.width) as usize;
    let mut ll = Vec::with_capacity((ll_width as usize) * (ll_height as usize));
    for y in 0..ll_height as usize {
        let row_start = y
            .checked_mul(full_width)
            .ok_or("CUDA DWT LL row offset overflow")?;
        ll.extend_from_slice(&transformed[row_start..row_start + ll_width as usize]);
    }

    let mut levels = Vec::with_capacity(output.levels().len());
    for shape in output.levels() {
        levels.push(J2kForwardDwt53Level {
            hl: extract_cuda_subband(
                transformed,
                full_width,
                shape.low_width,
                0,
                shape.high_width,
                shape.low_height,
            )?,
            lh: extract_cuda_subband(
                transformed,
                full_width,
                0,
                shape.low_height,
                shape.low_width,
                shape.high_height,
            )?,
            hh: extract_cuda_subband(
                transformed,
                full_width,
                shape.low_width,
                shape.low_height,
                shape.high_width,
                shape.high_height,
            )?,
            width: shape.width,
            height: shape.height,
            low_width: shape.low_width,
            low_height: shape.low_height,
            high_width: shape.high_width,
            high_height: shape.high_height,
        });
    }
    levels.reverse();

    Ok(J2kForwardDwt53Output {
        ll,
        ll_width,
        ll_height,
        levels,
    })
}

#[cfg(feature = "cuda-runtime")]
fn cuda_dwt97_output_to_j2k(
    output: &CudaDwt97Output,
) -> core::result::Result<J2kForwardDwt97Output, &'static str> {
    let (ll_width, ll_height) = output.ll_dimensions();
    let transformed = output.transformed();
    let full_width = output
        .levels()
        .first()
        .map_or(ll_width, |level| level.width) as usize;
    let mut ll = Vec::with_capacity((ll_width as usize) * (ll_height as usize));
    for y in 0..ll_height as usize {
        let row_start = y
            .checked_mul(full_width)
            .ok_or("CUDA DWT LL row offset overflow")?;
        ll.extend_from_slice(&transformed[row_start..row_start + ll_width as usize]);
    }

    let mut levels = Vec::with_capacity(output.levels().len());
    for shape in output.levels() {
        levels.push(J2kForwardDwt97Level {
            hl: extract_cuda_subband(
                transformed,
                full_width,
                shape.low_width,
                0,
                shape.high_width,
                shape.low_height,
            )?,
            lh: extract_cuda_subband(
                transformed,
                full_width,
                0,
                shape.low_height,
                shape.low_width,
                shape.high_height,
            )?,
            hh: extract_cuda_subband(
                transformed,
                full_width,
                shape.low_width,
                shape.low_height,
                shape.high_width,
                shape.high_height,
            )?,
            width: shape.width,
            height: shape.height,
            low_width: shape.low_width,
            low_height: shape.low_height,
            high_width: shape.high_width,
            high_height: shape.high_height,
        });
    }
    levels.reverse();

    Ok(J2kForwardDwt97Output {
        ll,
        ll_width,
        ll_height,
        levels,
    })
}

#[cfg(feature = "cuda-runtime")]
fn extract_cuda_subband(
    transformed: &[f32],
    full_width: usize,
    x0: u32,
    y0: u32,
    width: u32,
    height: u32,
) -> core::result::Result<Vec<f32>, &'static str> {
    let mut out = Vec::with_capacity((width as usize) * (height as usize));
    for y in 0..height as usize {
        let row_start = (y0 as usize)
            .checked_add(y)
            .and_then(|row| row.checked_mul(full_width))
            .and_then(|row| row.checked_add(x0 as usize))
            .ok_or("CUDA DWT subband offset overflow")?;
        out.extend_from_slice(&transformed[row_start..row_start + width as usize]);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "cuda-runtime")]
    use super::cuda_htj2k_encode_tables;
    use super::packetization::{
        cuda_ht_segment_lengths, flatten_cuda_htj2k_packetization_job,
        CudaHtj2kPacketizationPlanTagNodeState,
    };
    #[cfg(feature = "cuda-runtime")]
    use super::{
        cuda_dwt53_output_to_j2k, encode_lossless_from_cuda_buffer_to_cuda_buffer_with_report,
        encode_lossless_from_cuda_buffers_to_cuda_buffers_with_report, CudaLosslessEncodeTile,
    };
    use super::{
        encode_j2k_lossless_with_cuda, encode_j2k_lossless_with_cuda_and_profile,
        CudaEncodeStageAccelerator,
    };
    #[cfg(feature = "cuda-runtime")]
    use crate::CudaSession;
    #[cfg(feature = "cuda-runtime")]
    use j2k::{encode_j2k_lossy_with_accelerator, J2kLossyEncodeOptions, J2kLossySamples};
    use j2k::{
        EncodeBackendPreference, J2kBlockCodingMode, J2kEncodeValidation, J2kLosslessEncodeOptions,
        J2kLosslessSamples,
    };
    #[cfg(feature = "cuda-runtime")]
    use j2k::{J2kDeinterleaveToF32Job, J2kHtCodeBlockEncodeJob};
    use j2k::{
        J2kEncodeStageAccelerator, J2kHtSubbandEncodeJob, J2kPacketizationBlockCodingMode,
        J2kPacketizationCodeBlock, J2kPacketizationEncodeJob, J2kPacketizationPacketDescriptor,
        J2kPacketizationProgressionOrder, J2kPacketizationResolution, J2kPacketizationSubband,
        J2kQuantizeSubbandJob,
    };
    use j2k_core::CodecError;
    #[cfg(feature = "cuda-runtime")]
    use j2k_core::{BackendKind, PixelFormat};
    #[cfg(feature = "cuda-runtime")]
    use j2k_cuda_runtime::{
        CudaContext, CudaHtj2kEncodeCodeBlockJob, CudaHtj2kEncodeCodeBlockRegionJob,
        CudaJ2kQuantizeJob,
    };
    #[cfg(feature = "cuda-runtime")]
    use j2k_native::forward_dwt53_reference;
    use j2k_native::{
        encode_with_accelerator as encode_with_native_accelerator, DecodeSettings, EncodeOptions,
        Image,
    };

    fn assert_strict_cuda_classic_tier1_error<E: CodecError + ?Sized>(err: &E, context: &str) {
        assert!(err.is_unsupported());
        let message = err.to_string();
        assert!(
            message.contains("tier1_code_block") || message.contains("deinterleave"),
            "expected {context} error to mention either the missing classic tier-1 stage or unavailable CUDA deinterleave, got {message}"
	        );
    }

    #[cfg(feature = "cuda-runtime")]
    fn strict_cuda_resident_lossless_options() -> J2kLosslessEncodeOptions {
        J2kLosslessEncodeOptions::default()
            .with_backend(EncodeBackendPreference::RequireDevice)
            .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
            .with_max_decomposition_levels(Some(0))
            .with_validation(J2kEncodeValidation::External)
    }

    struct CudaTestEncodeRequest<'a> {
        pixels: &'a [u8],
        width: u32,
        height: u32,
        components: u8,
        bit_depth: u8,
        signed: bool,
        options: &'a EncodeOptions,
        accelerator: &'a mut CudaEncodeStageAccelerator,
    }

    fn encode_with_cuda_test_accelerator(
        request: CudaTestEncodeRequest<'_>,
    ) -> core::result::Result<Vec<u8>, &'static str> {
        let CudaTestEncodeRequest {
            pixels,
            width,
            height,
            components,
            bit_depth,
            signed,
            options,
            accelerator,
        } = request;
        encode_with_native_accelerator(
            pixels,
            width,
            height,
            u16::from(components),
            bit_depth,
            signed,
            options,
            accelerator,
        )
    }

    #[test]
    fn cuda_lossless_encode_auto_errors_for_unsupported_classic_tier1() {
        let pixels: Vec<u8> = (0u32..128 * 128)
            .map(|value| u8::try_from((value * 17 + 5) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let samples =
            J2kLosslessSamples::new(&pixels, 128, 128, 1, 8, false).expect("valid gray8 samples");
        let options = J2kLosslessEncodeOptions::default()
            .with_backend(EncodeBackendPreference::Auto)
            .with_block_coding_mode(J2kBlockCodingMode::Classic)
            .with_max_decomposition_levels(Some(0))
            .with_validation(J2kEncodeValidation::CpuRoundTrip);

        let err = encode_j2k_lossless_with_cuda(samples, &options)
            .expect_err("CUDA-named encode must not silently return CPU fallback");

        assert_strict_cuda_classic_tier1_error(&err, "strict CUDA encode");
    }

    #[test]
    fn cuda_lossless_encode_profile_auto_errors_for_unsupported_classic_tier1() {
        let pixels: Vec<u8> = (0u32..128 * 128)
            .map(|value| u8::try_from((value * 19 + 7) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let samples =
            J2kLosslessSamples::new(&pixels, 128, 128, 1, 8, false).expect("valid gray8 samples");
        let options = J2kLosslessEncodeOptions::default()
            .with_backend(EncodeBackendPreference::Auto)
            .with_block_coding_mode(J2kBlockCodingMode::Classic)
            .with_max_decomposition_levels(Some(0))
            .with_validation(J2kEncodeValidation::External);

        let err = encode_j2k_lossless_with_cuda_and_profile(samples, &options)
            .expect_err("profiled CUDA encode must not silently return CPU fallback");

        assert_strict_cuda_classic_tier1_error(&err, "profiled strict CUDA encode");
    }

    #[test]
    fn cuda_lossless_encode_require_device_errors_for_unsupported_classic_tier1() {
        let pixels: Vec<u8> = (0u32..128 * 128)
            .map(|value| u8::try_from((value * 29 + 11) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let samples =
            J2kLosslessSamples::new(&pixels, 128, 128, 1, 8, false).expect("valid gray8 samples");
        let options = J2kLosslessEncodeOptions::default()
            .with_backend(EncodeBackendPreference::RequireDevice)
            .with_block_coding_mode(J2kBlockCodingMode::Classic)
            .with_max_decomposition_levels(Some(0))
            .with_validation(J2kEncodeValidation::External);

        let err = encode_j2k_lossless_with_cuda(samples, &options)
            .expect_err("strict CUDA encode must not silently fall back to CPU");

        assert_strict_cuda_classic_tier1_error(&err, "strict CUDA encode");
    }

    #[test]
    fn cuda_packetization_flatten_accepts_cleanup_only_single_block_packet() {
        let payload = [0x12, 0x34, 0x56, 0x78];
        let code_block = J2kPacketizationCodeBlock {
            data: &payload,
            ht_cleanup_length: 0,
            ht_refinement_length: 0,
            num_coding_passes: 1,
            num_zero_bitplanes: 2,
            previously_included: false,
            l_block: 3,
            block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
        };
        let subband = J2kPacketizationSubband {
            code_blocks: vec![code_block],
            num_cbs_x: 1,
            num_cbs_y: 1,
        };
        let resolution = J2kPacketizationResolution {
            subbands: vec![subband],
        };
        let descriptor = J2kPacketizationPacketDescriptor {
            packet_index: 0,
            state_index: 0,
            layer: 0,
            resolution: 0,
            component: 0,
            precinct: 0,
        };
        let job = J2kPacketizationEncodeJob {
            resolution_count: 1,
            num_layers: 1,
            num_components: 1,
            code_block_count: 1,
            progression_order: J2kPacketizationProgressionOrder::Lrcp,
            packet_descriptors: &[descriptor],
            resolutions: &[resolution],
        };

        let plan = flatten_cuda_htj2k_packetization_job(job).expect("supported CUDA packetization");

        assert_eq!(plan.payload, payload);
        assert_eq!(plan.packets.len(), 1);
        assert_eq!(plan.subbands.len(), 1);
        assert_eq!(plan.blocks.len(), 1);
        assert_eq!(plan.packets[0].block_start, 0);
        assert_eq!(plan.packets[0].block_count, 1);
        assert_eq!(plan.packets[0].subband_start, 0);
        assert_eq!(plan.packets[0].subband_count, 1);
        assert_eq!(plan.subbands[0].block_start, 0);
        assert_eq!(plan.subbands[0].block_count, 1);
        let payload_len = u32::try_from(payload.len()).expect("test payload length fits in u32");
        assert!(plan.packets[0].output_capacity >= payload_len + 256);
        assert_eq!(plan.blocks[0].data_offset, 0);
        assert_eq!(plan.blocks[0].data_len, payload_len);
        assert_eq!(plan.blocks[0].num_coding_passes, 1);
        assert_eq!(plan.blocks[0].num_zero_bitplanes, 2);
    }

    #[test]
    fn cuda_packetization_flatten_accepts_cleanup_only_multi_block_packet() {
        let payloads = vec![
            vec![0x10, 0x11, 0x12],
            vec![0x20, 0x21],
            vec![0x30, 0x31, 0x32, 0x33],
            vec![0x40],
        ];
        let code_blocks = payloads
            .iter()
            .enumerate()
            .map(|(idx, payload)| J2kPacketizationCodeBlock {
                data: payload.as_slice(),
                ht_cleanup_length: 0,
                ht_refinement_length: 0,
                num_coding_passes: 1,
                num_zero_bitplanes: u8::try_from(idx + 1).expect("test zbp fits in u8"),
                previously_included: false,
                l_block: 3,
                block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
            })
            .collect();
        let subband = J2kPacketizationSubband {
            code_blocks,
            num_cbs_x: 2,
            num_cbs_y: 2,
        };
        let resolution = J2kPacketizationResolution {
            subbands: vec![subband],
        };
        let descriptor = J2kPacketizationPacketDescriptor {
            packet_index: 0,
            state_index: 0,
            layer: 0,
            resolution: 0,
            component: 0,
            precinct: 0,
        };
        let job = J2kPacketizationEncodeJob {
            resolution_count: 1,
            num_layers: 1,
            num_components: 1,
            code_block_count: 4,
            progression_order: J2kPacketizationProgressionOrder::Lrcp,
            packet_descriptors: &[descriptor],
            resolutions: &[resolution],
        };

        let plan =
            flatten_cuda_htj2k_packetization_job(job).expect("multi-block CUDA packetization");

        assert_eq!(plan.packets.len(), 1);
        assert_eq!(plan.subbands.len(), 1);
        assert_eq!(plan.blocks.len(), 4);
        assert_eq!(plan.packets[0].block_start, 0);
        assert_eq!(plan.packets[0].block_count, 4);
        assert_eq!(plan.packets[0].subband_start, 0);
        assert_eq!(plan.packets[0].subband_count, 1);
        assert_eq!(plan.subbands[0].block_start, 0);
        assert_eq!(plan.subbands[0].block_count, 4);
        assert_eq!(plan.subbands[0].num_cbs_x, 2);
        assert_eq!(plan.subbands[0].num_cbs_y, 2);
        assert_eq!(
            plan.payload,
            payloads.into_iter().flatten().collect::<Vec<_>>()
        );
        assert_eq!(plan.blocks[2].num_zero_bitplanes, 3);
    }

    #[test]
    fn cuda_packetization_flatten_accepts_ht_refinement_pass_packet() {
        let payload = [0x12, 0x34, 0x56, 0x78, 0x9a];
        let code_block = J2kPacketizationCodeBlock {
            data: &payload,
            ht_cleanup_length: 3,
            ht_refinement_length: 2,
            num_coding_passes: 3,
            num_zero_bitplanes: 2,
            previously_included: false,
            l_block: 3,
            block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
        };
        let subband = J2kPacketizationSubband {
            code_blocks: vec![code_block],
            num_cbs_x: 1,
            num_cbs_y: 1,
        };
        let resolution = J2kPacketizationResolution {
            subbands: vec![subband],
        };
        let descriptor = J2kPacketizationPacketDescriptor {
            packet_index: 0,
            state_index: 0,
            layer: 0,
            resolution: 0,
            component: 0,
            precinct: 0,
        };
        let job = J2kPacketizationEncodeJob {
            resolution_count: 1,
            num_layers: 1,
            num_components: 1,
            code_block_count: 1,
            progression_order: J2kPacketizationProgressionOrder::Lrcp,
            packet_descriptors: &[descriptor],
            resolutions: &[resolution],
        };

        let plan = flatten_cuda_htj2k_packetization_job(job).expect("HT refinement packetization");

        assert_eq!(plan.payload, payload);
        assert_eq!(plan.blocks.len(), 1);
        assert_eq!(plan.blocks[0].num_coding_passes, 3);
        assert_eq!(
            plan.blocks[0].data_len,
            u32::try_from(payload.len()).expect("test payload length fits in u32")
        );
    }

    #[test]
    fn cuda_packetization_rejects_overflowing_ht_refinement_lengths() {
        let payload = [0x12];
        let code_block = J2kPacketizationCodeBlock {
            data: &payload,
            ht_cleanup_length: u32::MAX,
            ht_refinement_length: 1,
            num_coding_passes: 3,
            num_zero_bitplanes: 2,
            previously_included: false,
            l_block: 3,
            block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
        };

        let err = cuda_ht_segment_lengths(&code_block)
            .expect_err("overflowing CUDA HT segment lengths rejected");

        assert_eq!(err, "multi-pass HTJ2K packet contribution length overflow");
    }

    #[test]
    fn cuda_packetization_flatten_rejects_out_of_range_ht_pass_count() {
        let payload = [0u8; 1];
        let code_block = J2kPacketizationCodeBlock {
            data: &payload,
            ht_cleanup_length: 0,
            ht_refinement_length: 0,
            num_coding_passes: 165,
            num_zero_bitplanes: 2,
            previously_included: false,
            l_block: 3,
            block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
        };
        let subband = J2kPacketizationSubband {
            code_blocks: vec![code_block],
            num_cbs_x: 1,
            num_cbs_y: 1,
        };
        let resolution = J2kPacketizationResolution {
            subbands: vec![subband],
        };
        let descriptor = J2kPacketizationPacketDescriptor {
            packet_index: 0,
            state_index: 0,
            layer: 0,
            resolution: 0,
            component: 0,
            precinct: 0,
        };
        let job = J2kPacketizationEncodeJob {
            resolution_count: 1,
            num_layers: 1,
            num_components: 1,
            code_block_count: 1,
            progression_order: J2kPacketizationProgressionOrder::Lrcp,
            packet_descriptors: &[descriptor],
            resolutions: &[resolution],
        };

        let err = flatten_cuda_htj2k_packetization_job(job)
            .expect_err("invalid HT pass count must be rejected before CUDA launch");

        assert_eq!(
            err,
            "CUDA HTJ2K packetization coding pass count exceeds JPEG 2000 bounds"
        );
    }

    #[test]
    fn cuda_packetization_flatten_accepts_previously_included_second_layer_packet() {
        let first_payload = [0x11u8; 20];
        let second_payload = [0x22u8; 5];
        let first_block = J2kPacketizationCodeBlock {
            data: &first_payload,
            ht_cleanup_length: 0,
            ht_refinement_length: 0,
            num_coding_passes: 1,
            num_zero_bitplanes: 2,
            previously_included: false,
            l_block: 3,
            block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
        };
        let second_block = J2kPacketizationCodeBlock {
            data: &second_payload,
            ht_cleanup_length: 0,
            ht_refinement_length: 0,
            num_coding_passes: 1,
            num_zero_bitplanes: 2,
            previously_included: false,
            l_block: 3,
            block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
        };
        let first_resolution = J2kPacketizationResolution {
            subbands: vec![J2kPacketizationSubband {
                code_blocks: vec![first_block],
                num_cbs_x: 1,
                num_cbs_y: 1,
            }],
        };
        let second_resolution = J2kPacketizationResolution {
            subbands: vec![J2kPacketizationSubband {
                code_blocks: vec![second_block],
                num_cbs_x: 1,
                num_cbs_y: 1,
            }],
        };
        let descriptors = [
            J2kPacketizationPacketDescriptor {
                packet_index: 0,
                state_index: 0,
                layer: 0,
                resolution: 0,
                component: 0,
                precinct: 0,
            },
            J2kPacketizationPacketDescriptor {
                packet_index: 1,
                state_index: 0,
                layer: 1,
                resolution: 0,
                component: 0,
                precinct: 0,
            },
        ];
        let resolutions = [first_resolution, second_resolution];
        let job = J2kPacketizationEncodeJob {
            resolution_count: 2,
            num_layers: 2,
            num_components: 1,
            code_block_count: 2,
            progression_order: J2kPacketizationProgressionOrder::Lrcp,
            packet_descriptors: &descriptors,
            resolutions: &resolutions,
        };

        let plan =
            flatten_cuda_htj2k_packetization_job(job).expect("stateful CUDA packetization plan");

        assert_eq!(
            plan.payload,
            [first_payload.as_slice(), second_payload.as_slice()].concat()
        );
        assert_eq!(plan.packets.len(), 2);
        assert_eq!(plan.blocks.len(), 2);
        assert_eq!(plan.packets[0].layer, 0);
        assert_eq!(plan.packets[1].layer, 1);
        assert_eq!(plan.blocks[0].l_block, 3);
        assert_eq!(plan.blocks[0].previously_included, 0);
        assert_eq!(plan.blocks[1].previously_included, 1);
        assert_eq!(plan.blocks[0].inclusion_layer, 0);
        assert_eq!(plan.blocks[1].inclusion_layer, 0);
        assert_eq!(
            plan.blocks[1].l_block, 5,
            "first layer length must update L-block for later packet state"
        );
    }

    #[test]
    fn cuda_packetization_flatten_accepts_deferred_first_inclusion_second_layer_packet() {
        let payload = [0x44u8; 5];
        let first_block = J2kPacketizationCodeBlock {
            data: &[],
            ht_cleanup_length: 0,
            ht_refinement_length: 0,
            num_coding_passes: 0,
            num_zero_bitplanes: 2,
            previously_included: false,
            l_block: 3,
            block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
        };
        let second_block = J2kPacketizationCodeBlock {
            data: &payload,
            ht_cleanup_length: 0,
            ht_refinement_length: 0,
            num_coding_passes: 1,
            num_zero_bitplanes: 2,
            previously_included: false,
            l_block: 3,
            block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
        };
        let first_resolution = J2kPacketizationResolution {
            subbands: vec![J2kPacketizationSubband {
                code_blocks: vec![first_block],
                num_cbs_x: 1,
                num_cbs_y: 1,
            }],
        };
        let second_resolution = J2kPacketizationResolution {
            subbands: vec![J2kPacketizationSubband {
                code_blocks: vec![second_block],
                num_cbs_x: 1,
                num_cbs_y: 1,
            }],
        };
        let descriptors = [
            J2kPacketizationPacketDescriptor {
                packet_index: 0,
                state_index: 0,
                layer: 0,
                resolution: 0,
                component: 0,
                precinct: 0,
            },
            J2kPacketizationPacketDescriptor {
                packet_index: 1,
                state_index: 0,
                layer: 1,
                resolution: 0,
                component: 0,
                precinct: 0,
            },
        ];
        let resolutions = [first_resolution, second_resolution];
        let job = J2kPacketizationEncodeJob {
            resolution_count: 2,
            num_layers: 2,
            num_components: 1,
            code_block_count: 2,
            progression_order: J2kPacketizationProgressionOrder::Lrcp,
            packet_descriptors: &descriptors,
            resolutions: &resolutions,
        };

        let plan =
            flatten_cuda_htj2k_packetization_job(job).expect("deferred first inclusion plan");

        assert_eq!(plan.payload, payload);
        assert_eq!(plan.packets.len(), 2);
        assert_eq!(plan.blocks.len(), 2);
        assert_eq!(plan.packets[0].layer, 0);
        assert_eq!(plan.packets[1].layer, 1);
        assert_eq!(plan.blocks[0].previously_included, 0);
        assert_eq!(plan.blocks[1].previously_included, 0);
        assert_eq!(plan.blocks[0].inclusion_layer, 1);
        assert_eq!(plan.blocks[1].inclusion_layer, 1);
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "multi-packet deferred-inclusion fixture is one byte-structure regression"
    )]
    fn cuda_packetization_flatten_accepts_deferred_first_inclusion_after_non_empty_packet() {
        let first_payload = [0x11u8; 3];
        let second_payload = [0x22u8; 5];
        let first_resolution = J2kPacketizationResolution {
            subbands: vec![J2kPacketizationSubband {
                code_blocks: vec![
                    J2kPacketizationCodeBlock {
                        data: &first_payload,
                        ht_cleanup_length: 0,
                        ht_refinement_length: 0,
                        num_coding_passes: 1,
                        num_zero_bitplanes: 2,
                        previously_included: false,
                        l_block: 3,
                        block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
                    },
                    J2kPacketizationCodeBlock {
                        data: &[],
                        ht_cleanup_length: 0,
                        ht_refinement_length: 0,
                        num_coding_passes: 0,
                        num_zero_bitplanes: 2,
                        previously_included: false,
                        l_block: 3,
                        block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
                    },
                ],
                num_cbs_x: 2,
                num_cbs_y: 1,
            }],
        };
        let second_resolution = J2kPacketizationResolution {
            subbands: vec![J2kPacketizationSubband {
                code_blocks: vec![
                    J2kPacketizationCodeBlock {
                        data: &[],
                        ht_cleanup_length: 0,
                        ht_refinement_length: 0,
                        num_coding_passes: 0,
                        num_zero_bitplanes: 2,
                        previously_included: false,
                        l_block: 3,
                        block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
                    },
                    J2kPacketizationCodeBlock {
                        data: &second_payload,
                        ht_cleanup_length: 0,
                        ht_refinement_length: 0,
                        num_coding_passes: 1,
                        num_zero_bitplanes: 2,
                        previously_included: false,
                        l_block: 3,
                        block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
                    },
                ],
                num_cbs_x: 2,
                num_cbs_y: 1,
            }],
        };
        let descriptors = [
            J2kPacketizationPacketDescriptor {
                packet_index: 0,
                state_index: 0,
                layer: 0,
                resolution: 0,
                component: 0,
                precinct: 0,
            },
            J2kPacketizationPacketDescriptor {
                packet_index: 1,
                state_index: 0,
                layer: 1,
                resolution: 0,
                component: 0,
                precinct: 0,
            },
        ];
        let resolutions = [first_resolution, second_resolution];
        let job = J2kPacketizationEncodeJob {
            resolution_count: 2,
            num_layers: 2,
            num_components: 1,
            code_block_count: 4,
            progression_order: J2kPacketizationProgressionOrder::Lrcp,
            packet_descriptors: &descriptors,
            resolutions: &resolutions,
        };

        let plan = flatten_cuda_htj2k_packetization_job(job)
            .expect("persistent tag-tree state is flattened for CUDA packetization");

        assert_eq!(
            plan.payload,
            [first_payload.as_slice(), second_payload.as_slice()].concat()
        );
        assert_eq!(plan.packets.len(), 2);
        assert_eq!(plan.blocks.len(), 4);
        assert_eq!(plan.blocks[0].previously_included, 0);
        assert_eq!(plan.blocks[1].previously_included, 0);
        assert_eq!(plan.blocks[2].previously_included, 1);
        assert_eq!(plan.blocks[3].previously_included, 0);
        assert_eq!(plan.blocks[0].inclusion_layer, 0);
        assert_eq!(plan.blocks[1].inclusion_layer, 1);
        assert_eq!(plan.blocks[2].inclusion_layer, 0);
        assert_eq!(plan.blocks[3].inclusion_layer, 1);
        assert_eq!(plan.tag_states.len(), 2);
        assert_eq!(plan.tag_nodes.len(), 12);
        assert_eq!(plan.tag_states[1].inclusion_node_start, 6);
        assert_eq!(plan.tag_states[1].zero_bitplane_node_start, 9);
        assert_eq!(
            &plan.tag_nodes[6..9],
            &[
                CudaHtj2kPacketizationPlanTagNodeState {
                    current: 0,
                    known: 1,
                },
                CudaHtj2kPacketizationPlanTagNodeState {
                    current: 1,
                    known: 0,
                },
                CudaHtj2kPacketizationPlanTagNodeState {
                    current: 0,
                    known: 1,
                },
            ]
        );
        assert_eq!(
            &plan.tag_nodes[9..12],
            &[
                CudaHtj2kPacketizationPlanTagNodeState {
                    current: 2,
                    known: 1,
                },
                CudaHtj2kPacketizationPlanTagNodeState {
                    current: 0,
                    known: 0,
                },
                CudaHtj2kPacketizationPlanTagNodeState {
                    current: 2,
                    known: 1,
                },
            ]
        );
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_lossless_encode_require_device_dispatches_cleanup_packetization_when_runtime_required()
    {
        if !j2k_test_support::cuda_runtime_gate(module_path!()) {
            return;
        }

        let pixels: Vec<u8> = (0u16..8 * 8)
            .map(|value| u8::try_from((value * 31 + 7) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let samples =
            J2kLosslessSamples::new(&pixels, 8, 8, 1, 8, false).expect("valid gray8 samples");
        let options = J2kLosslessEncodeOptions::default()
            .with_backend(EncodeBackendPreference::RequireDevice)
            .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
            .with_max_decomposition_levels(Some(0))
            .with_validation(J2kEncodeValidation::CpuRoundTrip);

        let encoded = encode_j2k_lossless_with_cuda(samples, &options)
            .expect("strict CUDA single-pass HT encode should dispatch all required stages");
        let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(encoded.backend, BackendKind::Cuda);
        assert_eq!(decoded.data, pixels);
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_lossless_buffer_encode_returns_resident_codestream_when_runtime_required() {
        if !j2k_test_support::cuda_runtime_gate(module_path!()) {
            return;
        }

        let width = 64;
        let height = 64;
        let pixels: Vec<u8> = (0u32..width * height)
            .map(|value| u8::try_from((value * 23 + 11) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let mut session = CudaSession::default();
        let context = session.cuda_context().expect("CUDA context");
        let buffer = context.upload(&pixels).expect("resident source pixels");
        let tile = CudaLosslessEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width,
            height,
            pitch_bytes: width as usize,
            output_width: width,
            output_height: height,
            format: PixelFormat::Gray8,
        };

        let outcome = encode_lossless_from_cuda_buffer_to_cuda_buffer_with_report(
            tile,
            &strict_cuda_resident_lossless_options(),
            &mut session,
        )
        .expect("strict CUDA resident codestream encode");
        let downloaded = outcome
            .encoded
            .codestream
            .download()
            .expect("download resident codestream");
        let decoded = Image::new(&downloaded, &DecodeSettings::default())
            .expect("resident codestream parses")
            .decode_native()
            .expect("resident codestream decodes");

        assert_eq!(outcome.encoded.encoded.backend, BackendKind::Cuda);
        assert_eq!(outcome.encoded.codestream.byte_len(), downloaded.len());
        assert_eq!(
            downloaded.as_slice(),
            outcome.host_outcome.encoded.codestream.as_slice()
        );
        assert_eq!(
            outcome.encoded.encoded.codestream.as_slice(),
            downloaded.as_slice()
        );
        assert!(!outcome.host_outcome.resident.codestream_assembly_used);
        assert_eq!(decoded.data, pixels);
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_lossless_buffer_batch_encode_returns_resident_codestreams_in_order_when_runtime_required(
    ) {
        if !j2k_test_support::cuda_runtime_gate(module_path!()) {
            return;
        }

        let width = 32;
        let height = 32;
        let inputs = [
            (0u32..width * height)
                .map(|value| {
                    u8::try_from((value * 17 + 3) & 0xFF).expect("masked value fits in u8")
                })
                .collect::<Vec<_>>(),
            (0u32..width * height)
                .map(|value| {
                    u8::try_from((value * 31 + 97) & 0xFF).expect("masked value fits in u8")
                })
                .collect::<Vec<_>>(),
        ];
        let mut session = CudaSession::default();
        let context = session.cuda_context().expect("CUDA context");
        let buffers = inputs
            .iter()
            .map(|pixels| context.upload(pixels).expect("resident source pixels"))
            .collect::<Vec<_>>();
        let tiles = buffers
            .iter()
            .map(|buffer| CudaLosslessEncodeTile {
                buffer,
                byte_offset: 0,
                width,
                height,
                pitch_bytes: width as usize,
                output_width: width,
                output_height: height,
                format: PixelFormat::Gray8,
            })
            .collect::<Vec<_>>();

        let outcomes = encode_lossless_from_cuda_buffers_to_cuda_buffers_with_report(
            &tiles,
            &strict_cuda_resident_lossless_options(),
            &mut session,
        )
        .expect("strict CUDA resident codestream batch encode");

        assert_eq!(outcomes.len(), inputs.len());
        for (outcome, expected_pixels) in outcomes.iter().zip(inputs.iter()) {
            let downloaded = outcome
                .encoded
                .codestream
                .download()
                .expect("download resident codestream");
            let decoded = Image::new(&downloaded, &DecodeSettings::default())
                .expect("resident codestream parses")
                .decode_native()
                .expect("resident codestream decodes");

            assert_eq!(outcome.encoded.encoded.backend, BackendKind::Cuda);
            assert_eq!(outcome.encoded.codestream.byte_len(), downloaded.len());
            assert_eq!(
                downloaded.as_slice(),
                outcome.host_outcome.encoded.codestream.as_slice()
            );
            assert_eq!(decoded.data.as_slice(), expected_pixels.as_slice());
        }
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_deinterleave_stage_dispatches_when_runtime_required() {
        if !j2k_test_support::cuda_runtime_gate(module_path!()) {
            return;
        }

        let pixels = [0u8, 128, 255, 64, 32, 16];
        let mut accelerator = CudaEncodeStageAccelerator::default();
        let components = accelerator
            .encode_deinterleave(J2kDeinterleaveToF32Job {
                pixels: &pixels,
                num_pixels: 2,
                num_components: 3,
                bit_depth: 8,
                signed: false,
            })
            .expect("CUDA deinterleave hook")
            .expect("CUDA deinterleave dispatch");

        assert_eq!(accelerator.deinterleave_dispatches(), 1);
        assert_eq!(
            components,
            vec![vec![-128.0, -64.0], vec![0.0, -96.0], vec![127.0, -112.0]]
        );
    }

    #[test]
    fn prefer_cpu_ht_subband_declines_fused_subband_but_counts_attempts() {
        let mut accelerator = CudaEncodeStageAccelerator::default()
            .prefer_cpu_ht_subband(true)
            .prefer_cpu_quantize_subband(true);
        let output = accelerator
            .encode_ht_subband(J2kHtSubbandEncodeJob {
                coefficients: &[0.0; 16],
                width: 4,
                height: 4,
                step_exponent: 8,
                step_mantissa: 0,
                range_bits: 8,
                reversible: false,
                code_block_width: 4,
                code_block_height: 4,
                total_bitplanes: 9,
            })
            .expect("subband hook can decline");

        assert!(output.is_none());
        assert_eq!(accelerator.ht_subband_attempts(), 1);
        assert_eq!(accelerator.quantize_subband_attempts(), 1);
        assert_eq!(accelerator.ht_code_block_attempts(), 1);
        assert_eq!(accelerator.dispatch_report().total(), 0);

        let quantized = accelerator
            .encode_quantize_subband(J2kQuantizeSubbandJob {
                coefficients: &[0.0; 16],
                step_exponent: 8,
                step_mantissa: 0,
                range_bits: 8,
                reversible: false,
            })
            .expect("quantize hook can decline");
        assert!(quantized.is_none());
        assert_eq!(accelerator.quantize_subband_attempts(), 2);
        assert_eq!(accelerator.dispatch_report().total(), 0);
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_lossless_encode_require_device_dispatches_multi_block_cleanup_when_runtime_required() {
        if !j2k_test_support::cuda_runtime_gate(module_path!()) {
            return;
        }

        let pixels: Vec<u8> = (0u32..128 * 128)
            .map(|value| u8::try_from((value * 19 + 23) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let samples =
            J2kLosslessSamples::new(&pixels, 128, 128, 1, 8, false).expect("valid gray8 samples");
        let options = J2kLosslessEncodeOptions::default()
            .with_backend(EncodeBackendPreference::RequireDevice)
            .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
            .with_max_decomposition_levels(Some(0))
            .with_validation(J2kEncodeValidation::CpuRoundTrip);

        let encoded = encode_j2k_lossless_with_cuda(samples, &options)
            .expect("strict CUDA multi-block cleanup encode should dispatch all required stages");
        let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(encoded.backend, BackendKind::Cuda);
        assert_eq!(decoded.data, pixels);
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_lossless_encode_require_device_dispatches_dwt53_cleanup_when_runtime_required() {
        if !j2k_test_support::cuda_runtime_gate(module_path!()) {
            return;
        }

        let pixels: Vec<u8> = (0u32..128 * 128)
            .map(|value| u8::try_from((value * 37 + 41) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let samples =
            J2kLosslessSamples::new(&pixels, 128, 128, 1, 8, false).expect("valid gray8 samples");
        let options = J2kLosslessEncodeOptions::default()
            .with_backend(EncodeBackendPreference::RequireDevice)
            .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
            .with_max_decomposition_levels(Some(1))
            .with_validation(J2kEncodeValidation::CpuRoundTrip);

        let encoded = encode_j2k_lossless_with_cuda(samples, &options)
            .expect("strict CUDA DWT cleanup encode should dispatch all required stages");
        let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(encoded.backend, BackendKind::Cuda);
        assert_eq!(decoded.data, pixels);
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_lossless_encode_profile_reports_resident_stage_timings_when_runtime_required() {
        if !j2k_test_support::cuda_runtime_gate(module_path!()) {
            return;
        }

        let pixels: Vec<u8> = (0u32..128 * 128)
            .map(|value| u8::try_from((value * 43 + 29) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let samples =
            J2kLosslessSamples::new(&pixels, 128, 128, 1, 8, false).expect("valid gray8 samples");
        let options = J2kLosslessEncodeOptions::default()
            .with_backend(EncodeBackendPreference::RequireDevice)
            .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
            .with_max_decomposition_levels(Some(1))
            .with_validation(J2kEncodeValidation::CpuRoundTrip);

        let (encoded, report) = encode_j2k_lossless_with_cuda_and_profile(samples, &options)
            .expect("strict CUDA profiled DWT cleanup encode should dispatch all required stages");
        let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(encoded.backend, BackendKind::Cuda);
        assert_eq!(decoded.data, pixels);
        assert_eq!(report.backend, BackendKind::Cuda);
        assert_eq!(report.input_bytes, pixels.len());
        assert_eq!(report.codestream_bytes, encoded.codestream.len());
        assert!(report.dispatch_count > 0);
        assert!(report.block_count > 0);
        assert!(report.deinterleave_us > 0);
        assert_eq!(report.mct_us, 0);
        assert!(report.dwt_us > 0);
        assert!(report.quantize_us > 0);
        assert!(report.ht_encode_us > 0);
        assert!(report.packetize_us > 0);
        assert!(report.total_us > 0);
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_lossless_encode_require_device_dispatches_rgb_rct_cleanup_when_runtime_required() {
        if !j2k_test_support::cuda_runtime_gate(module_path!()) {
            return;
        }

        let pixels: Vec<u8> = (0u32..128 * 128 * 3)
            .map(|value| u8::try_from((value * 13 + 71) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let samples =
            J2kLosslessSamples::new(&pixels, 128, 128, 3, 8, false).expect("valid rgb8 samples");
        let options = J2kLosslessEncodeOptions::default()
            .with_backend(EncodeBackendPreference::RequireDevice)
            .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
            .with_max_decomposition_levels(Some(1))
            .with_validation(J2kEncodeValidation::CpuRoundTrip);

        let encoded = encode_j2k_lossless_with_cuda(samples, &options)
            .expect("strict CUDA RGB cleanup encode should dispatch all required stages");
        let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(encoded.backend, BackendKind::Cuda);
        assert_eq!(decoded.data, pixels);
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_lossy_htj2k_facade_require_device_dispatches_supported_stages_when_runtime_required() {
        if !j2k_test_support::cuda_runtime_gate(module_path!()) {
            return;
        }

        let pixels: Vec<u8> = (0u32..64 * 64)
            .map(|value| u8::try_from((value * 41 + 17) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let samples =
            J2kLossySamples::new(&pixels, 64, 64, 1, 8, false).expect("valid gray8 samples");
        let options = J2kLossyEncodeOptions::default()
            .with_backend(EncodeBackendPreference::RequireDevice)
            .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
            .with_max_decomposition_levels(Some(1))
            .with_validation(J2kEncodeValidation::CpuRoundTrip);
        let mut accelerator = CudaEncodeStageAccelerator::default();

        let encoded = encode_j2k_lossy_with_accelerator(
            samples,
            &options,
            BackendKind::Cuda,
            &mut accelerator,
        )
        .expect("strict CUDA HTJ2K lossy facade encode should dispatch supported stages");
        let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(encoded.backend, BackendKind::Cuda);
        assert_eq!(decoded.width, 64);
        assert_eq!(decoded.height, 64);
        assert_eq!(decoded.num_components, 1);
        assert_eq!(accelerator.deinterleave_dispatches(), 1);
        assert!(accelerator.forward_dwt97_dispatches() > 0);
        assert_eq!(accelerator.quantize_subband_dispatches(), 4);
        assert_eq!(accelerator.ht_code_block_dispatches(), 4);
        assert_eq!(accelerator.packetization_dispatches(), 1);
    }

    #[test]
    fn cuda_encode_stage_accelerator_preserves_cpu_codestream_validity() {
        let pixels: Vec<u8> = (0u8..192).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let mut accelerator = CudaEncodeStageAccelerator::default();

        let codestream = encode_with_cuda_test_accelerator(CudaTestEncodeRequest {
            pixels: &pixels,
            width: 8,
            height: 8,
            components: 3,
            bit_depth: 8,
            signed: false,
            options: &options,
            accelerator: &mut accelerator,
        })
        .expect("encode with CUDA stage accelerator");
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.width, 8);
        assert_eq!(decoded.height, 8);
        assert_eq!(decoded.num_components, 3);
        assert_eq!(decoded.bit_depth, 8);
        assert_eq!(accelerator.forward_rct_attempts(), 1);
        assert_eq!(accelerator.forward_dwt53_attempts(), 3);
        assert!(accelerator.tier1_code_block_attempts() > 0);
        assert_eq!(accelerator.packetization_attempts(), 1);
    }

    #[test]
    fn cuda_auto_host_output_declines_packetization_before_flattening() {
        let mut accelerator = CudaEncodeStageAccelerator::for_auto_host_output();
        let invalid_for_cuda_flattening = J2kPacketizationEncodeJob {
            resolution_count: 1,
            num_layers: 1,
            num_components: 3,
            code_block_count: 0,
            progression_order: J2kPacketizationProgressionOrder::Lrcp,
            packet_descriptors: &[],
            resolutions: &[],
        };

        let encoded = J2kEncodeStageAccelerator::encode_packetization(
            &mut accelerator,
            invalid_for_cuda_flattening,
        )
        .expect("Auto host-output CUDA packetization should decline to CPU");

        assert!(encoded.is_none());
        assert_eq!(accelerator.packetization_attempts(), 1);
        assert_eq!(accelerator.packetization_dispatches(), 0);
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_forward_rct_dispatches_when_runtime_required() {
        if !j2k_test_support::cuda_runtime_gate(module_path!()) {
            return;
        }

        let pixels: Vec<u8> = (0u16..7 * 5 * 3)
            .map(|i| u8::try_from((i * 17) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 0,
            ..EncodeOptions::default()
        };
        let mut accelerator = CudaEncodeStageAccelerator::default();

        let codestream = encode_with_cuda_test_accelerator(CudaTestEncodeRequest {
            pixels: &pixels,
            width: 7,
            height: 5,
            components: 3,
            bit_depth: 8,
            signed: false,
            options: &options,
            accelerator: &mut accelerator,
        })
        .expect("encode with CUDA forward RCT");
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.data, pixels);
        assert_eq!(accelerator.forward_rct_attempts(), 1);
        assert_eq!(accelerator.forward_rct_dispatches(), 1);
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_forward_ict_dispatches_when_runtime_required() {
        if !j2k_test_support::cuda_runtime_gate(module_path!()) {
            return;
        }

        let pixels: Vec<u8> = (0u32..32 * 32 * 3)
            .map(|i| u8::try_from((i * 23 + 19) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let options = EncodeOptions {
            reversible: false,
            use_ht_block_coding: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let mut accelerator = CudaEncodeStageAccelerator::default();

        let codestream = encode_with_cuda_test_accelerator(CudaTestEncodeRequest {
            pixels: &pixels,
            width: 32,
            height: 32,
            components: 3,
            bit_depth: 8,
            signed: false,
            options: &options,
            accelerator: &mut accelerator,
        })
        .expect("encode irreversible RGB with CUDA forward ICT");
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.data.len(), pixels.len());
        assert_eq!(accelerator.forward_ict_attempts(), 1);
        assert_eq!(accelerator.forward_ict_dispatches(), 1);
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_forward_dwt53_dispatches_when_runtime_required() {
        if !j2k_test_support::cuda_runtime_gate(module_path!()) {
            return;
        }

        let pixels: Vec<u8> = (0u16..8 * 8)
            .map(|i| u8::try_from((i * 5) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let mut accelerator = CudaEncodeStageAccelerator::default();

        let codestream = encode_with_cuda_test_accelerator(CudaTestEncodeRequest {
            pixels: &pixels,
            width: 8,
            height: 8,
            components: 1,
            bit_depth: 8,
            signed: false,
            options: &options,
            accelerator: &mut accelerator,
        })
        .expect("encode with CUDA forward DWT 5/3");
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.data, pixels);
        assert_eq!(accelerator.forward_dwt53_attempts(), 1);
        assert_eq!(accelerator.forward_dwt53_dispatches(), 2);
    }

    #[cfg(feature = "cuda-runtime")]
    fn assert_cuda_forward_dwt53_reshape_matches_native(width: u32, height: u32, num_levels: u8) {
        let samples: Vec<f32> = (0u32..width * height)
            .map(|i| {
                let value = i16::try_from((i * 7 + 3) % 256).expect("sample fits in i16") - 128;
                f32::from(value)
            })
            .collect();

        let native = forward_dwt53_reference(&samples, width, height, num_levels);
        let context = CudaContext::system_default().expect("CUDA context");
        let cuda_output = context
            .j2k_forward_dwt53(&samples, width, height, num_levels)
            .expect("CUDA forward DWT 5/3");
        let cuda_as_native = cuda_dwt53_output_to_j2k(&cuda_output)
            .expect("CUDA DWT output reshapes to native subbands");

        assert_eq!(
            cuda_as_native.levels.len(),
            native.levels.len(),
            "reshaped level count (levels={num_levels})"
        );
        assert_eq!(
            (cuda_as_native.ll_width, cuda_as_native.ll_height),
            (native.ll_width, native.ll_height),
            "reshaped LL dimensions (levels={num_levels})"
        );
        for (level_idx, (cuda_level, native_level)) in cuda_as_native
            .levels
            .iter()
            .zip(native.levels.iter())
            .enumerate()
        {
            assert_eq!(
                cuda_level.hl, native_level.hl,
                "levels={num_levels} level {level_idx} HL mismatch"
            );
            assert_eq!(
                cuda_level.lh, native_level.lh,
                "levels={num_levels} level {level_idx} LH mismatch"
            );
            assert_eq!(
                cuda_level.hh, native_level.hh,
                "levels={num_levels} level {level_idx} HH mismatch"
            );
        }
        assert_eq!(
            cuda_as_native.ll, native.ll,
            "levels={num_levels} final LL mismatch"
        );
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_forward_dwt53_private_reshape_matches_native_reference_when_required() {
        if !j2k_test_support::cuda_runtime_gate(module_path!()) {
            return;
        }

        assert_cuda_forward_dwt53_reshape_matches_native(40, 24, 1);
        assert_cuda_forward_dwt53_reshape_matches_native(40, 24, 2);
        assert_cuda_forward_dwt53_reshape_matches_native(40, 24, 3);
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_forward_dwt97_dispatches_when_runtime_required() {
        if !j2k_test_support::cuda_runtime_gate(module_path!()) {
            return;
        }

        let pixels: Vec<u8> = (0u16..32 * 32)
            .map(|i| u8::try_from((i * 7 + 13) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let options = EncodeOptions {
            reversible: false,
            use_ht_block_coding: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let mut accelerator = CudaEncodeStageAccelerator::default();

        let codestream = encode_with_cuda_test_accelerator(CudaTestEncodeRequest {
            pixels: &pixels,
            width: 32,
            height: 32,
            components: 1,
            bit_depth: 8,
            signed: false,
            options: &options,
            accelerator: &mut accelerator,
        })
        .expect("encode with CUDA forward DWT 9/7");
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.data.len(), pixels.len());
        assert_eq!(accelerator.forward_dwt97_attempts(), 1);
        assert_eq!(accelerator.forward_dwt97_dispatches(), 3);
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_quantize_subband_dispatches_when_runtime_required() {
        if !j2k_test_support::cuda_runtime_gate(module_path!()) {
            return;
        }

        let pixels: Vec<u8> = (0u16..32 * 32)
            .map(|i| u8::try_from((i * 19 + 5) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let options = EncodeOptions {
            reversible: false,
            use_ht_block_coding: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let mut accelerator = CudaEncodeStageAccelerator::default();

        let codestream = encode_with_cuda_test_accelerator(CudaTestEncodeRequest {
            pixels: &pixels,
            width: 32,
            height: 32,
            components: 1,
            bit_depth: 8,
            signed: false,
            options: &options,
            accelerator: &mut accelerator,
        })
        .expect("encode with CUDA quantization");
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.data.len(), pixels.len());
        assert_eq!(accelerator.quantize_subband_attempts(), 4);
        assert_eq!(accelerator.quantize_subband_dispatches(), 4);
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_encode_uses_resident_tile_body_when_runtime_required() {
        if !j2k_test_support::cuda_runtime_gate(module_path!()) {
            return;
        }

        let pixels: Vec<u8> = (0u16..32 * 32)
            .map(|i| u8::try_from((i * 23 + 11) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let options = EncodeOptions {
            reversible: true,
            use_ht_block_coding: true,
            num_decomposition_levels: 0,
            code_block_width_exp: 2,
            code_block_height_exp: 2,
            ..EncodeOptions::default()
        };
        let mut accelerator = CudaEncodeStageAccelerator::default();

        let codestream = encode_with_cuda_test_accelerator(CudaTestEncodeRequest {
            pixels: &pixels,
            width: 32,
            height: 32,
            components: 1,
            bit_depth: 8,
            signed: false,
            options: &options,
            accelerator: &mut accelerator,
        })
        .expect("encode HTJ2K through CUDA tile-body hook");
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.data, pixels);
        assert_eq!(accelerator.htj2k_tile_attempts(), 1);
        assert_eq!(accelerator.htj2k_tile_dispatches(), 1);
        assert_eq!(accelerator.ht_subband_attempts(), 0);
        assert_eq!(accelerator.ht_subband_dispatches(), 0);
        assert_eq!(accelerator.deinterleave_dispatches(), 1);
        assert_eq!(accelerator.quantize_subband_attempts(), 1);
        assert_eq!(accelerator.quantize_subband_dispatches(), 1);
        assert_eq!(accelerator.ht_code_block_attempts(), 4);
        assert_eq!(accelerator.ht_code_block_dispatches(), 1);
        assert_eq!(accelerator.packetization_attempts(), 1);
        assert_eq!(accelerator.packetization_dispatches(), 1);
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_encode_uses_resident_dwt_tile_body_when_runtime_required() {
        if !j2k_test_support::cuda_runtime_gate(module_path!()) {
            return;
        }

        let pixels: Vec<u8> = (0u16..32 * 32)
            .map(|i| u8::try_from((i * 29 + 5) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let options = EncodeOptions {
            reversible: true,
            use_ht_block_coding: true,
            num_decomposition_levels: 1,
            code_block_width_exp: 2,
            code_block_height_exp: 2,
            ..EncodeOptions::default()
        };
        let mut accelerator = CudaEncodeStageAccelerator::default();

        let codestream = encode_with_cuda_test_accelerator(CudaTestEncodeRequest {
            pixels: &pixels,
            width: 32,
            height: 32,
            components: 1,
            bit_depth: 8,
            signed: false,
            options: &options,
            accelerator: &mut accelerator,
        })
        .expect("encode HTJ2K DWT through CUDA tile-body hook");
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.data, pixels);
        assert_eq!(accelerator.htj2k_tile_attempts(), 1);
        assert_eq!(accelerator.htj2k_tile_dispatches(), 1);
        assert_eq!(accelerator.ht_subband_attempts(), 0);
        assert_eq!(accelerator.ht_subband_dispatches(), 0);
        assert_eq!(accelerator.forward_dwt53_attempts(), 1);
        assert!(accelerator.forward_dwt53_dispatches() > 0);
        assert_eq!(accelerator.quantize_subband_attempts(), 4);
        assert_eq!(accelerator.quantize_subband_dispatches(), 4);
        assert_eq!(accelerator.ht_code_block_attempts(), 4);
        assert_eq!(accelerator.ht_code_block_dispatches(), 4);
        assert_eq!(accelerator.packetization_attempts(), 1);
        assert_eq!(accelerator.packetization_dispatches(), 1);
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_encode_uses_resident_mct_dwt_tile_body_when_runtime_required() {
        if !j2k_test_support::cuda_runtime_gate(module_path!()) {
            return;
        }

        let pixels: Vec<u8> = (0u16..32 * 32 * 3)
            .map(|i| u8::try_from((i * 19 + 17) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let options = EncodeOptions {
            reversible: true,
            use_mct: true,
            use_ht_block_coding: true,
            num_decomposition_levels: 1,
            code_block_width_exp: 2,
            code_block_height_exp: 2,
            ..EncodeOptions::default()
        };
        let mut accelerator = CudaEncodeStageAccelerator::default();

        let codestream = encode_with_cuda_test_accelerator(CudaTestEncodeRequest {
            pixels: &pixels,
            width: 32,
            height: 32,
            components: 3,
            bit_depth: 8,
            signed: false,
            options: &options,
            accelerator: &mut accelerator,
        })
        .expect("encode HTJ2K RGB DWT through CUDA tile-body hook");
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.data, pixels);
        assert_eq!(accelerator.htj2k_tile_attempts(), 1);
        assert_eq!(accelerator.htj2k_tile_dispatches(), 1);
        assert_eq!(accelerator.ht_subband_attempts(), 0);
        assert_eq!(accelerator.forward_rct_attempts(), 1);
        assert_eq!(accelerator.forward_rct_dispatches(), 1);
        assert_eq!(accelerator.forward_dwt53_attempts(), 3);
        assert!(accelerator.forward_dwt53_dispatches() > 0);
        assert_eq!(accelerator.quantize_subband_attempts(), 12);
        assert_eq!(accelerator.quantize_subband_dispatches(), 12);
        assert_eq!(accelerator.ht_code_block_attempts(), 12);
        assert_eq!(accelerator.ht_code_block_dispatches(), 12);
        assert_eq!(accelerator.packetization_attempts(), 1);
        assert_eq!(accelerator.packetization_dispatches(), 1);
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_encode_uses_resident_dwt97_tile_body_when_runtime_required() {
        if !j2k_test_support::cuda_runtime_gate(module_path!()) {
            return;
        }

        let pixels: Vec<u8> = (0u16..32 * 32)
            .map(|i| u8::try_from((i * 31 + 7) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let options = EncodeOptions {
            reversible: false,
            use_ht_block_coding: true,
            num_decomposition_levels: 1,
            code_block_width_exp: 2,
            code_block_height_exp: 2,
            ..EncodeOptions::default()
        };
        let mut accelerator = CudaEncodeStageAccelerator::default();

        let codestream = encode_with_cuda_test_accelerator(CudaTestEncodeRequest {
            pixels: &pixels,
            width: 32,
            height: 32,
            components: 1,
            bit_depth: 8,
            signed: false,
            options: &options,
            accelerator: &mut accelerator,
        })
        .expect("encode irreversible HTJ2K DWT through CUDA tile-body hook");
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.width, 32);
        assert_eq!(decoded.height, 32);
        assert_eq!(decoded.num_components, 1);
        assert_eq!(accelerator.htj2k_tile_attempts(), 1);
        assert_eq!(accelerator.htj2k_tile_dispatches(), 1);
        assert_eq!(accelerator.ht_subband_attempts(), 0);
        assert_eq!(accelerator.forward_dwt97_attempts(), 1);
        assert!(accelerator.forward_dwt97_dispatches() > 0);
        assert_eq!(accelerator.quantize_subband_attempts(), 4);
        assert_eq!(accelerator.quantize_subband_dispatches(), 4);
        assert_eq!(accelerator.ht_code_block_attempts(), 4);
        assert_eq!(accelerator.ht_code_block_dispatches(), 4);
        assert_eq!(accelerator.packetization_attempts(), 1);
        assert_eq!(accelerator.packetization_dispatches(), 1);
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_htj2k_codeblock_dispatches_when_runtime_required() {
        if !j2k_test_support::cuda_runtime_gate(module_path!()) {
            return;
        }

        let pixels: Vec<u8> = (0u16..8 * 8)
            .map(|i| u8::try_from((i * 11 + 3) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let options = EncodeOptions {
            reversible: true,
            use_ht_block_coding: true,
            num_decomposition_levels: 0,
            code_block_width_exp: 2,
            code_block_height_exp: 2,
            ..EncodeOptions::default()
        };
        let mut accelerator = CudaEncodeStageAccelerator::default();

        let codestream = encode_with_cuda_test_accelerator(CudaTestEncodeRequest {
            pixels: &pixels,
            width: 8,
            height: 8,
            components: 1,
            bit_depth: 8,
            signed: false,
            options: &options,
            accelerator: &mut accelerator,
        })
        .expect("encode HTJ2K with CUDA HT codeblock kernel");
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.data, pixels);
        assert!(accelerator.ht_code_block_attempts() > 0);
        assert!(accelerator.ht_code_block_dispatches() > 0);
        assert!(accelerator.ht_code_block_dispatches() <= accelerator.ht_code_block_attempts());
        assert_eq!(
            accelerator.dispatch_report().ht_code_block,
            accelerator.ht_code_block_dispatches()
        );
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_htj2k_codeblock_preserves_requested_refinement_passes_when_runtime_required() {
        if !j2k_test_support::cuda_runtime_gate(module_path!()) {
            return;
        }

        let coefficients = [0, 3, -5, 3, 5, 0, -3, 3, 7, -3, 0, 3, 0, 0, 5, -5];
        let mut accelerator = CudaEncodeStageAccelerator::default();

        let encoded = accelerator
            .encode_ht_code_block(J2kHtCodeBlockEncodeJob {
                coefficients: &coefficients,
                width: 4,
                height: 4,
                total_bitplanes: 4,
                target_coding_passes: 2,
            })
            .expect("CUDA HTJ2K code-block encode hook")
            .expect("CUDA HTJ2K code-block encode output");

        assert_eq!(encoded.num_coding_passes, 2);
        assert_eq!(encoded.num_zero_bitplanes, 2);
        assert_eq!(encoded.refinement_length, 1);
        assert_eq!(
            encoded.cleanup_length + encoded.refinement_length,
            u32::try_from(encoded.data.len()).expect("test payload length fits u32")
        );
        assert_eq!(accelerator.ht_code_block_dispatches(), 1);
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_htj2k_codeblock_batch_uses_single_dispatch_when_runtime_required() {
        if !j2k_test_support::cuda_runtime_gate(module_path!()) {
            return;
        }

        let pixels: Vec<u8> = (0u16..32 * 32)
            .map(|i| u8::try_from((i * 17 + 9) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let options = EncodeOptions {
            reversible: true,
            use_ht_block_coding: true,
            num_decomposition_levels: 0,
            code_block_width_exp: 2,
            code_block_height_exp: 2,
            ..EncodeOptions::default()
        };
        let mut accelerator = CudaEncodeStageAccelerator::default();

        let codestream = encode_with_cuda_test_accelerator(CudaTestEncodeRequest {
            pixels: &pixels,
            width: 32,
            height: 32,
            components: 1,
            bit_depth: 8,
            signed: false,
            options: &options,
            accelerator: &mut accelerator,
        })
        .expect("encode HTJ2K with CUDA HT batch codeblock kernel");
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.data, pixels);
        assert!(accelerator.ht_code_block_attempts() > 1);
        assert_eq!(accelerator.ht_code_block_dispatches(), 1);
        assert!(
            accelerator.ht_code_block_dispatches() < accelerator.ht_code_block_attempts(),
            "batch encode must not launch one kernel per codeblock"
        );
        assert_eq!(
            accelerator.dispatch_report().ht_code_block,
            accelerator.ht_code_block_dispatches()
        );
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_resident_quantized_subband_feeds_resident_ht_batch_when_runtime_required() {
        if !j2k_test_support::cuda_runtime_gate(module_path!()) {
            return;
        }

        let samples = [-3.6f32, -2.5, -0.4, 0.0, 0.49, 1.5, 3.2, 9.9];
        let context = CudaContext::system_default().expect("CUDA context");
        let sample_buffer = context.upload_f32(&samples).expect("resident samples");
        let quantization = CudaJ2kQuantizeJob {
            step_exponent: 8,
            step_mantissa: 0,
            range_bits: 8,
            reversible: true,
        };
        let resident_quantized = context
            .j2k_quantize_subband_resident(&sample_buffer, samples.len(), quantization)
            .expect("resident quantization");
        let host_quantized = context
            .j2k_quantize_subband(&samples, quantization)
            .expect("host-staged quantization");
        let jobs = [CudaHtj2kEncodeCodeBlockJob {
            coefficient_offset: 0,
            width: 4,
            height: 2,
            total_bitplanes: 5,
            target_coding_passes: 1,
        }];

        let resident_encoded = context
            .encode_htj2k_codeblocks_resident(
                resident_quantized.buffer(),
                resident_quantized.coefficient_count(),
                &jobs,
                cuda_htj2k_encode_tables(),
            )
            .expect("resident HTJ2K encode");
        let staged_encoded = context
            .encode_htj2k_codeblocks(
                host_quantized.coefficients(),
                &jobs,
                cuda_htj2k_encode_tables(),
            )
            .expect("host-staged HTJ2K encode");

        assert_eq!(resident_quantized.coefficient_count(), samples.len());
        assert_eq!(resident_encoded.execution().kernel_dispatches(), 1);
        assert_eq!(
            resident_encoded.code_blocks().len(),
            staged_encoded.code_blocks().len()
        );
        for (resident, staged) in resident_encoded
            .code_blocks()
            .iter()
            .zip(staged_encoded.code_blocks())
        {
            assert_eq!(resident.data(), staged.data());
            assert_eq!(resident.cleanup_length(), staged.cleanup_length());
            assert_eq!(resident.refinement_length(), staged.refinement_length());
            assert_eq!(resident.num_coding_passes(), staged.num_coding_passes());
            assert_eq!(resident.num_zero_bitplanes(), staged.num_zero_bitplanes());
        }
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_resident_strided_codeblock_region_matches_host_gather_when_runtime_required() {
        if !j2k_test_support::cuda_runtime_gate(module_path!()) {
            return;
        }

        let samples: Vec<f32> = (0u16..16).map(|value| f32::from(value) - 8.0).collect();
        let context = CudaContext::system_default().expect("CUDA context");
        let sample_buffer = context.upload_f32(&samples).expect("resident samples");
        let quantization = CudaJ2kQuantizeJob {
            step_exponent: 8,
            step_mantissa: 0,
            range_bits: 8,
            reversible: true,
        };
        let resident_quantized = context
            .j2k_quantize_subband_resident(&sample_buffer, samples.len(), quantization)
            .expect("resident quantization");
        let quantized = resident_quantized
            .download_coefficients()
            .expect("download quantized coefficients");
        let gathered_codeblock = vec![quantized[5], quantized[6], quantized[9], quantized[10]];
        let region_jobs = [CudaHtj2kEncodeCodeBlockRegionJob {
            coefficient_offset: 5,
            coefficient_stride: 4,
            width: 2,
            height: 2,
            total_bitplanes: 5,
            target_coding_passes: 1,
        }];
        let contiguous_jobs = [CudaHtj2kEncodeCodeBlockJob {
            coefficient_offset: 0,
            width: 2,
            height: 2,
            total_bitplanes: 5,
            target_coding_passes: 1,
        }];

        let resident_encoded = context
            .encode_htj2k_codeblock_regions_resident(
                resident_quantized.buffer(),
                resident_quantized.coefficient_count(),
                &region_jobs,
                cuda_htj2k_encode_tables(),
            )
            .expect("resident strided HTJ2K encode");
        let staged_encoded = context
            .encode_htj2k_codeblocks(
                &gathered_codeblock,
                &contiguous_jobs,
                cuda_htj2k_encode_tables(),
            )
            .expect("host-gathered HTJ2K encode");

        assert_eq!(resident_encoded.execution().kernel_dispatches(), 1);
        assert_eq!(resident_encoded.code_blocks().len(), 1);
        assert_eq!(
            resident_encoded.code_blocks()[0].data(),
            staged_encoded.code_blocks()[0].data()
        );
        assert_eq!(
            resident_encoded.code_blocks()[0].num_zero_bitplanes(),
            staged_encoded.code_blocks()[0].num_zero_bitplanes()
        );
    }
}
