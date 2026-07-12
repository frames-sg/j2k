// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "cuda-runtime")]
use j2k::{
    J2kEncodeDispatchReport, J2kEncodeStageAccelerator, J2kResidentEncodeInput,
    J2kResidentEncodeInputError, J2kResidentHtj2kTileEncodeJob,
};
#[cfg(feature = "cuda-runtime")]
use j2k_core::BackendKind;
#[cfg(feature = "cuda-runtime")]
use j2k_core::{DeviceSubmission, DeviceSubmitSession, PixelFormat, ReadySubmission};
#[cfg(feature = "cuda-runtime")]
use j2k_cuda_runtime::{CudaContext, CudaHtj2kEncodeResources};
#[cfg(feature = "cuda-runtime")]
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

#[cfg(feature = "cuda-runtime")]
use crate::allocation::HostPhaseBudget;
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
    CudaEncodedJ2k, CudaEncodedJ2kMetadata, CudaLosslessBufferEncodeOutcome,
    CudaLosslessEncodeOutcome, CudaLosslessEncodeResidency, CudaLosslessEncodeTile,
    CudaResidentCodestreamBuffer, SubmittedJ2kLosslessCudaEncode,
    SubmittedJ2kLosslessCudaEncodeBatch,
};
#[cfg(feature = "cuda-runtime")]
mod htj2k;
#[cfg(feature = "cuda-runtime")]
use self::htj2k::cuda_encode_htj2k_device_tile_body;
#[cfg(feature = "cuda-runtime")]
pub(crate) use self::htj2k::cuda_htj2k_encode_tables;
mod packetization;
mod stage;
mod stage_error;
#[cfg(feature = "cuda-runtime")]
use self::stage::time_cuda_stage;
pub use self::stage::{CudaEncodeStageAccelerator, CudaEncodeStageTimings};
#[cfg(feature = "cuda-runtime")]
use self::stage_error::CudaStageResult;

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
    encode_lossless_cuda_tile_with_report(tile, *options, session)
}

#[cfg(feature = "cuda-runtime")]
/// Encode one CUDA-resident tile and return a CUDA-resident codestream copy.
///
/// Final codestream assembly currently occurs in host memory; the host bytes
/// are copied directly to CUDA and released before this function returns.
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
/// Encode one CUDA-resident tile and return a CUDA-resident codestream copy with timings.
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
    let result = (|| {
        let outcomes = encode_lossless_from_cuda_buffers_with_report(tiles, options, session)?;
        let mut host_budget =
            host_encode_outcome_budget(&outcomes, "j2k CUDA submitted batch codestreams")?;
        let mut encoded = host_budget.try_vec_with_capacity(outcomes.len())?;
        for outcome in outcomes {
            encoded.push(outcome.encoded);
        }
        Ok(encoded)
    })();
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
    let mut host_budget = HostPhaseBudget::new("j2k CUDA host batch encode outcomes");
    let mut outcomes = host_budget.try_vec_with_capacity(tiles.len())?;
    for tile in tiles.iter().copied() {
        let outcome = (|| {
            validate_cuda_encode_tile(tile)?;
            session.record_submit();
            encode_lossless_cuda_tile_with_report(tile, *options, session)
        })()?;
        host_budget.account_vec(&outcome.encoded.codestream)?;
        outcomes.push(outcome);
    }
    Ok(outcomes)
}

#[cfg(feature = "cuda-runtime")]
/// Encode multiple CUDA-resident tiles and return CUDA-resident codestream copies.
pub fn encode_lossless_from_cuda_buffers_to_cuda_buffers(
    tiles: &[CudaLosslessEncodeTile<'_>],
    options: &j2k::J2kLosslessEncodeOptions,
    session: &mut CudaSession,
) -> Result<Vec<CudaEncodedJ2k>, crate::Error> {
    let outcomes =
        encode_lossless_from_cuda_buffers_to_cuda_buffers_with_report(tiles, options, session)?;
    let mut host_budget = HostPhaseBudget::new("j2k CUDA resident batch codestreams");
    host_budget.account_vec(&outcomes)?;
    let mut encoded = host_budget.try_vec_with_capacity(outcomes.len())?;
    for outcome in outcomes {
        encoded.push(outcome.encoded);
    }
    Ok(encoded)
}

#[cfg(feature = "cuda-runtime")]
/// Encode multiple CUDA-resident tiles and return CUDA-resident codestream copies with timings.
#[doc(hidden)]
pub fn encode_lossless_from_cuda_buffers_to_cuda_buffers_with_report(
    tiles: &[CudaLosslessEncodeTile<'_>],
    options: &j2k::J2kLosslessEncodeOptions,
    session: &mut CudaSession,
) -> Result<Vec<CudaLosslessBufferEncodeOutcome>, crate::Error> {
    let host_outcomes = encode_lossless_from_cuda_buffers_with_report(tiles, options, session)?;
    let mut host_budget =
        host_encode_outcome_budget(&host_outcomes, "j2k CUDA resident batch encode outcomes")?;
    let mut outcomes = host_budget.try_vec_with_capacity(host_outcomes.len())?;
    for (tile, outcome) in tiles.iter().copied().zip(host_outcomes) {
        outcomes.push(cuda_resident_codestream_outcome(tile, outcome)?);
    }
    Ok(outcomes)
}

#[cfg(feature = "cuda-runtime")]
fn host_encode_outcome_budget(
    outcomes: &Vec<CudaLosslessEncodeOutcome>,
    what: &'static str,
) -> Result<HostPhaseBudget, crate::Error> {
    let mut budget = HostPhaseBudget::new(what);
    budget.account_vec(outcomes)?;
    for outcome in outcomes {
        budget.account_vec(&outcome.encoded.codestream)?;
    }
    Ok(budget)
}

#[cfg(feature = "cuda-runtime")]
fn cuda_resident_codestream_outcome(
    tile: CudaLosslessEncodeTile<'_>,
    host_outcome: CudaLosslessEncodeOutcome,
) -> Result<CudaLosslessBufferEncodeOutcome, crate::Error> {
    let upload_started = Instant::now();
    let codestream_len = host_outcome.encoded.codestream.len();
    let metadata = CudaEncodedJ2kMetadata::from_host_encoded(&host_outcome.encoded);
    let buffer = tile
        .buffer
        .context()
        .upload(&host_outcome.encoded.codestream)
        .map_err(cuda_error)?;
    let codestream_upload_duration = upload_started.elapsed();
    let CudaLosslessEncodeOutcome {
        encoded: host_encoded,
        input_copy_used,
        resident,
        input_copy_duration,
        encode_duration,
        gpu_duration,
        validation_duration,
        host_readback_duration,
        stage_timings,
    } = host_outcome;
    drop(host_encoded);
    let encoded = CudaEncodedJ2k {
        metadata,
        codestream: CudaResidentCodestreamBuffer {
            buffer,
            byte_len: codestream_len,
        },
    };
    Ok(CudaLosslessBufferEncodeOutcome {
        encoded,
        input_copy_used,
        resident,
        input_copy_duration,
        encode_duration,
        gpu_duration,
        validation_duration,
        host_readback_duration,
        stage_timings,
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
) -> CudaStageResult<u8> {
    u8::try_from(num_components).map_err(|_| j2k::J2kEncodeStageError::unsupported(reason))
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
    session: &mut CudaSession,
) -> Result<CudaLosslessEncodeOutcome, crate::Error> {
    let encode_started = Instant::now();
    let format = cuda_encode_format(tile.format)?;
    let input = J2kResidentEncodeInput::new(
        tile.output_width,
        tile.output_height,
        u16::from(format.components),
        format.bit_depth,
        false,
    )
    .map_err(cuda_resident_input_error)?;
    let context = tile.buffer.context();
    let resources = session.htj2k_encode_resources(&context)?;
    let mut accelerator = CudaDeviceBufferEncodeAccelerator {
        tile,
        context,
        resources,
        dispatch: J2kEncodeDispatchReport::default(),
        stage_timings: CudaEncodeStageTimings::default(),
    };
    let encoded = j2k::encode_j2k_lossless_resident_with_accelerator(
        input,
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
fn cuda_resident_input_error(error: J2kResidentEncodeInputError) -> crate::Error {
    let reason = match error {
        J2kResidentEncodeInputError::EmptyGeometry { .. }
        | J2kResidentEncodeInputError::ComponentCountOutOfRange { .. }
        | J2kResidentEncodeInputError::PrecisionOutOfRange { .. }
        | J2kResidentEncodeInputError::AddressSpaceOverflow => error.reason(),
        _ => "J2K CUDA resident input validation failed",
    };
    crate::Error::UnsupportedCudaRequest { reason }
}

#[cfg(feature = "cuda-runtime")]
struct CudaDeviceBufferEncodeAccelerator<'a> {
    tile: CudaLosslessEncodeTile<'a>,
    context: CudaContext,
    resources: Arc<CudaHtj2kEncodeResources>,
    dispatch: J2kEncodeDispatchReport,
    stage_timings: CudaEncodeStageTimings,
}

#[cfg(feature = "cuda-runtime")]
impl J2kEncodeStageAccelerator for CudaDeviceBufferEncodeAccelerator<'_> {
    fn dispatch_report(&self) -> J2kEncodeDispatchReport {
        self.dispatch
    }

    fn encode_resident_htj2k_tile(
        &mut self,
        job: J2kResidentHtj2kTileEncodeJob<'_>,
    ) -> CudaStageResult<Option<Vec<u8>>> {
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

#[cfg(test)]
mod tests;
