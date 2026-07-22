// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::{bail, DecodingError, Result};
use crate::{
    decode_ht_code_block_scalar_with_workspace, decode_j2k_code_block_scalar_with_workspace,
    HtCodeBlockDecodeJob, HtCodeBlockPayloadRanges, J2kCodeBlockDecodeJob, J2kDirectGrayscaleStep,
    J2kReferencedClassicPlan, J2kReferencedHtj2kPlan,
};

use super::super::referenced::payload_slice;
use super::super::{
    checked_sub_band_job_output_range, DirectComponentBandScratch, DirectCpuBand,
    J2kDirectCpuScratch, StagedDirectRoute, SubBandJobOutputRange,
};
use super::plan_access::{classic_tile_components, ht_tile_components};
use super::state::validate_active_tile;
use super::{J2kDirectCodeBlockIndex, J2kDirectCpuEntropyWorkspace};

/// Execute one flattened HT code block into its prepared image coefficient owner.
#[doc(hidden)]
pub fn execute_referenced_htj2k_entropy_job(
    plan: &J2kReferencedHtj2kPlan,
    index: J2kDirectCodeBlockIndex,
    payload_arena: &[u8],
    payload: HtCodeBlockPayloadRanges,
    image_scratch: &mut J2kDirectCpuScratch,
    worker_workspace: &mut J2kDirectCpuEntropyWorkspace,
) -> Result<()> {
    validate_active_tile(image_scratch, StagedDirectRoute::Htj2k, index.tile)?;
    let components = ht_tile_components(plan, index.tile)?;
    let component_plan = components
        .get(index.component)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    let Some(J2kDirectGrayscaleStep::HtSubBand(sub_band)) = component_plan.steps.get(index.step)
    else {
        bail!(DecodingError::CodeBlockDecodeFailure);
    };
    let job = sub_band
        .jobs
        .get(index.code_block)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    if payload.cleanup.length != job.cleanup_length as usize
        || payload.refinement.map_or(0, |range| range.length) != job.refinement_length as usize
    {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    let data = contiguous_ht_payload(payload_arena, payload)?;
    let output = code_block_output(
        image_scratch,
        index.component,
        sub_band.band_id,
        sub_band.width,
        sub_band.height,
        job.output_x,
        job.output_y,
        job.output_stride,
        job.width,
        job.height,
    )?;
    decode_ht_code_block_scalar_with_workspace(
        HtCodeBlockDecodeJob {
            data,
            cleanup_length: job.cleanup_length,
            refinement_length: job.refinement_length,
            width: job.width,
            height: job.height,
            output_stride: job.output_stride,
            missing_bit_planes: job.missing_bit_planes,
            number_of_coding_passes: job.number_of_coding_passes,
            num_bitplanes: job.num_bitplanes,
            roi_shift: job.roi_shift,
            stripe_causal: job.stripe_causal,
            strict: job.strict,
            dequantization_step: job.dequantization_step,
        },
        output,
        &mut worker_workspace.ht,
    )
}

/// Execute one flattened classic code block into its prepared image coefficient owner.
#[doc(hidden)]
pub fn execute_referenced_classic_entropy_job(
    plan: &J2kReferencedClassicPlan,
    index: J2kDirectCodeBlockIndex,
    payload_arena: &[u8],
    payload: crate::J2kCodestreamRange,
    image_scratch: &mut J2kDirectCpuScratch,
    worker_workspace: &mut J2kDirectCpuEntropyWorkspace,
) -> Result<()> {
    validate_active_tile(image_scratch, StagedDirectRoute::Classic, index.tile)?;
    let components = classic_tile_components(plan, index.tile)?;
    let component_plan = components
        .get(index.component)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    let Some(J2kDirectGrayscaleStep::ClassicSubBand(sub_band)) =
        component_plan.steps.get(index.step)
    else {
        bail!(DecodingError::CodeBlockDecodeFailure);
    };
    let job = sub_band
        .jobs
        .get(index.code_block)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    if !job.data.is_empty() {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    let data = payload_slice(payload_arena, payload)?;
    let output = code_block_output(
        image_scratch,
        index.component,
        sub_band.band_id,
        sub_band.width,
        sub_band.height,
        job.output_x,
        job.output_y,
        job.output_stride,
        job.width,
        job.height,
    )?;
    decode_j2k_code_block_scalar_with_workspace(
        J2kCodeBlockDecodeJob {
            data,
            segments: &job.segments,
            width: job.width,
            height: job.height,
            output_stride: job.output_stride,
            missing_bit_planes: job.missing_bit_planes,
            number_of_coding_passes: job.number_of_coding_passes,
            total_bitplanes: job.total_bitplanes,
            roi_shift: job.roi_shift,
            sub_band_type: job.sub_band_type,
            style: job.style,
            strict: job.strict,
            dequantization_step: job.dequantization_step,
        },
        output,
        &mut worker_workspace.classic,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "the code-block destination is validated against both sub-band and block geometry"
)]
fn code_block_output(
    scratch: &mut J2kDirectCpuScratch,
    component: usize,
    band_id: crate::J2kDirectBandId,
    band_width: u32,
    band_height: u32,
    output_x: u32,
    output_y: u32,
    output_stride: usize,
    width: u32,
    height: u32,
) -> Result<&mut [f32]> {
    let bands = scratch
        .component_band_sets
        .get_mut(component)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    let band = find_active_band_mut(bands, band_id)?;
    let sub_band_width =
        usize::try_from(band_width).map_err(|_| DecodingError::CodeBlockDecodeFailure)?;
    let output_range = checked_sub_band_job_output_range(&SubBandJobOutputRange {
        output_x,
        output_y,
        output_stride,
        width,
        height,
        sub_band_width,
        plan_width: band_width,
        plan_height: band_height,
        output_len: band.coefficients.len(),
    })?;
    Ok(&mut band.coefficients[output_range])
}

fn find_active_band_mut(
    bands: &mut DirectComponentBandScratch,
    band_id: crate::J2kDirectBandId,
) -> Result<&mut DirectCpuBand> {
    bands.bands[..bands.active_len]
        .iter_mut()
        .find(|band| band.band_id == band_id)
        .ok_or_else(|| DecodingError::CodeBlockDecodeFailure.into())
}

fn contiguous_ht_payload(input: &[u8], payload: HtCodeBlockPayloadRanges) -> Result<&[u8]> {
    let Some(refinement) = payload.refinement else {
        return payload_slice(input, payload.cleanup);
    };
    if payload.cleanup.end() != Some(refinement.offset) {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    let length = payload
        .cleanup
        .length
        .checked_add(refinement.length)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    payload_slice(
        input,
        crate::J2kCodestreamRange {
            offset: payload.cleanup.offset,
            length,
        },
    )
}
