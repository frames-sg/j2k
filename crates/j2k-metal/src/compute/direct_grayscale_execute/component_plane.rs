// SPDX-License-Identifier: MIT OR Apache-2.0

use metal::{Buffer, CommandBufferRef};

use crate::compute::{
    direct_buffers::copied_slice_buffer, direct_commands::new_compute_command_encoder,
    direct_plan_types::PreparedDirectGrayscalePlan, direct_profile::DirectHybridStageTimings,
    direct_roi::checked_f32_span, direct_scratch::DirectScratchBuffer,
    direct_status::DirectStatusCheck, direct_tier1::DirectTier1Mode, MetalRuntime,
};
use crate::{
    profile_env::{hybrid_stage_signpost, SIGNPOST_DECODE_HYBRID_COEFFICIENT_UPLOAD},
    Error,
};

mod execution;

pub(in crate::compute) use self::execution::encode_prepared_direct_component_plane_in_encoder;

#[cfg(target_os = "macos")]
pub(in crate::compute) fn checked_coefficient_len(
    width: u32,
    height: u32,
    message: &str,
) -> Result<usize, Error> {
    checked_f32_span(width as usize, height as usize, message).map(|span| span.elements)
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn upload_cpu_decoded_coefficients(
    runtime: &MetalRuntime,
    coefficients: &[f32],
    retained_buffers: &mut Vec<Buffer>,
) -> Result<Buffer, Error> {
    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_COEFFICIENT_UPLOAD);
    let buffer = copied_slice_buffer(&runtime.device, coefficients)?;
    retained_buffers.push(buffer.clone());
    Ok(buffer)
}

#[cfg(target_os = "macos")]
pub(in crate::compute) struct DirectComponentPlaneRequest<'a> {
    pub(in crate::compute) runtime: &'a MetalRuntime,
    pub(in crate::compute) command_buffer: &'a CommandBufferRef,
    pub(in crate::compute) plan: &'a PreparedDirectGrayscalePlan,
    pub(in crate::compute) tier1_mode: DirectTier1Mode,
    pub(in crate::compute) stage_timings: &'a mut DirectHybridStageTimings,
    pub(in crate::compute) retained_buffers: &'a mut Vec<Buffer>,
    pub(in crate::compute) status_checks: &'a mut Vec<DirectStatusCheck>,
    pub(in crate::compute) scratch_buffers: &'a mut Vec<DirectScratchBuffer>,
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn encode_prepared_direct_component_plane_in_command_buffer(
    request: DirectComponentPlaneRequest<'_>,
) -> Result<Buffer, Error> {
    let command_buffer = request.command_buffer;
    let encoder = new_compute_command_encoder(command_buffer)?;
    let result = encode_prepared_direct_component_plane_in_encoder(request, &encoder);
    encoder.end_encoding();
    result
}
