// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    profile, BlockCodingMode, EncodeOptions, J2kEncodeStageAccelerator, J2kResidentEncodeInput,
    NativeEncodePipelineError, NativeEncodeRetainedInput, NativeEncodeSession,
    ResidentHtj2kEncodeError, Vec, MAX_RAW_PIXEL_ENCODE_BIT_DEPTH,
};
use super::accelerator::encode_complete_resident_ht_tile;
use super::finalize::finalize_accelerated_codestream;
use super::plan::{
    build_single_tile_plan, validate_non_pixel_single_tile_request, NonPixelSingleTileRequest,
};

pub(in crate::j2c::encode) fn encode_resident_impl(
    input: J2kResidentEncodeInput,
    options: &EncodeOptions,
    block_coding_mode: BlockCodingMode,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, ResidentHtj2kEncodeError> {
    if block_coding_mode != BlockCodingMode::HighThroughput {
        return Err(ResidentHtj2kEncodeError::Unsupported(
            "resident encode requires HTJ2K block coding",
        ));
    }
    if options.validate_high_throughput_codestream {
        return Err(ResidentHtj2kEncodeError::Unsupported(
            "resident HTJ2K encode requires external validation",
        ));
    }
    if input.bit_depth() > MAX_RAW_PIXEL_ENCODE_BIT_DEPTH {
        return Err(ResidentHtj2kEncodeError::Unsupported(
            "resident HTJ2K encode supports at most 24 bits per sample",
        ));
    }

    let session = NativeEncodeSession::try_new(NativeEncodeRetainedInput::none())
        .map_err(resident_error_from_encode_error)?;
    let validated = validate_non_pixel_single_tile_request(&NonPixelSingleTileRequest {
        width: input.width(),
        height: input.height(),
        num_components: input.num_components(),
        bit_depth: input.bit_depth(),
        options,
        block_coding_mode,
        component_sample_info: &[],
        multi_tile_error: "resident HTJ2K encode requires a single whole-image tile",
        session: &session,
    })
    .map_err(NativeEncodePipelineError::into_resident_error)?;
    let profile_enabled = profile::profile_stages_enabled();
    let total_start = profile::profile_now(profile_enabled);
    let plan = build_single_tile_plan(
        validated,
        input.width(),
        input.height(),
        input.num_components(),
        input.bit_depth(),
        input.signed(),
        options,
        block_coding_mode,
        &[],
        &[],
        &session,
    )
    .map_err(NativeEncodePipelineError::into_resident_error)?;
    let (tile_data, tile_body_us) = encode_complete_resident_ht_tile(
        input,
        options,
        &plan,
        profile_enabled,
        &session,
        accelerator,
    )?;
    let final_plan = plan.into_codestream_final_plan();
    finalize_accelerated_codestream(
        &final_plan,
        &tile_data,
        tile_body_us,
        profile_enabled,
        total_start,
        &session,
    )
    .map_err(NativeEncodePipelineError::into_resident_error)
}

impl NativeEncodePipelineError {
    fn into_resident_error(self) -> ResidentHtj2kEncodeError {
        resident_error_from_encode_error(self.into_encode_error())
    }
}

pub(super) fn resident_error_from_encode_error(
    error: crate::EncodeError,
) -> ResidentHtj2kEncodeError {
    match error {
        crate::EncodeError::InvalidInput { what } => ResidentHtj2kEncodeError::InvalidInput(what),
        crate::EncodeError::Unsupported { what } => ResidentHtj2kEncodeError::Unsupported(what),
        crate::EncodeError::AllocationTooLarge { .. }
        | crate::EncodeError::HostAllocationFailed { .. } => {
            ResidentHtj2kEncodeError::Resource(error)
        }
        crate::EncodeError::Accelerator { source, .. } => {
            ResidentHtj2kEncodeError::Accelerator(source)
        }
        crate::EncodeError::ArithmeticOverflow { .. }
        | crate::EncodeError::CodestreamValidation { .. }
        | crate::EncodeError::InternalInvariant { .. } => ResidentHtj2kEncodeError::Backend(error),
    }
}
