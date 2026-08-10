// SPDX-License-Identifier: MIT OR Apache-2.0

//! HT and classic code-block payload construction for one direct-plan sub-band.

use super::super::{
    add_roi_shift_to_bitplanes, bail, classic_decode_job_parameters, code_block_required_by_index,
    collect_classic_code_block_data, collect_referenced_classic_code_block_data,
    count_classic_code_blocks, count_ht_code_blocks, ht_block_decode,
    ht_code_block_has_decodable_passes, sub_band_decode_parameters, ClassicPayloadCollector,
    ComponentInfo, DecodeAllocationBudget, DecodingError, DecompositionStorage, Header,
    HtCodeBlockPayloadRanges, HtOwnedCodeBlockBatchJob, HtOwnedSubBandPlan,
    J2kClassicCodeBlockPayload, J2kCodestreamRange, J2kDirectBandId, J2kDirectGrayscalePlan,
    J2kDirectGrayscaleStep, J2kOwnedCodeBlockBatchJob, J2kOwnedSubBandPlan, J2kRect,
    PayloadRangeOwner, Result, SubBand, SubBandDecodeParameters, ValidationError, Vec,
};
use crate::j2c::build::{CodeBlock, CodeBlockCoding};
use crate::{J2kCodeBlockSegment, J2kCodeBlockStyle, J2kSubBandType};

mod ht;
use self::ht::build_ht_sub_band_step;

#[expect(
    clippy::too_many_arguments,
    reason = "the sub-band boundary keeps validated band identity, storage, and payload collectors explicit"
)]
#[expect(
    clippy::too_many_lines,
    reason = "the ordered mixed-codec branch keeps one validated sub-band decision atomic"
)]
pub(super) fn build_grayscale_sub_band_steps(
    payload_range_owner: PayloadRangeOwner<'_>,
    sub_band: &SubBand,
    sub_band_idx: usize,
    band_id: J2kDirectBandId,
    resolution: u8,
    component_info: &ComponentInfo,
    storage: &DecompositionStorage<'_>,
    header: &Header<'_>,
    budget: &mut DecodeAllocationBudget,
    ht_payloads: Option<&mut Vec<HtCodeBlockPayloadRanges>>,
    classic_payloads: Option<&mut ClassicPayloadCollector<'_>>,
) -> Result<[Option<J2kDirectGrayscaleStep>; 2]> {
    let SubBandDecodeParameters {
        dequantization_step,
        irreversible_midpoint,
        num_bitplanes,
    } = sub_band_decode_parameters(sub_band, resolution, component_info)?;

    let code_block_style = component_info.coding_style.parameters.code_block_style;
    if code_block_style.allows_mixed_block_coding() {
        let classic_count = count_classic_code_blocks(sub_band_idx, sub_band, storage)?;
        let ht_count = count_ht_code_blocks(sub_band_idx, sub_band, storage)?;
        if classic_count > 0 && ht_count > 0 {
            let classic = build_classic_sub_band_step(
                payload_range_owner,
                sub_band,
                sub_band_idx,
                band_id,
                component_info,
                storage,
                header,
                budget,
                classic_payloads,
                dequantization_step,
                irreversible_midpoint,
                num_bitplanes,
            )?;
            let ht = build_ht_sub_band_step(
                payload_range_owner,
                sub_band,
                sub_band_idx,
                band_id,
                component_info,
                storage,
                header,
                budget,
                ht_payloads,
                dequantization_step,
                irreversible_midpoint,
                num_bitplanes,
            )?;
            return Ok([Some(classic), Some(ht)]);
        }
        if ht_count > 0 {
            return build_ht_sub_band_step(
                payload_range_owner,
                sub_band,
                sub_band_idx,
                band_id,
                component_info,
                storage,
                header,
                budget,
                ht_payloads,
                dequantization_step,
                irreversible_midpoint,
                num_bitplanes,
            )
            .map(|step| [Some(step), None]);
        }
        return build_classic_sub_band_step(
            payload_range_owner,
            sub_band,
            sub_band_idx,
            band_id,
            component_info,
            storage,
            header,
            budget,
            classic_payloads,
            dequantization_step,
            irreversible_midpoint,
            num_bitplanes,
        )
        .map(|step| [Some(step), None]);
    }

    if code_block_style.uses_high_throughput_block_coding() {
        build_ht_sub_band_step(
            payload_range_owner,
            sub_band,
            sub_band_idx,
            band_id,
            component_info,
            storage,
            header,
            budget,
            ht_payloads,
            dequantization_step,
            irreversible_midpoint,
            num_bitplanes,
        )
        .map(|step| [Some(step), None])
    } else {
        build_classic_sub_band_step(
            payload_range_owner,
            sub_band,
            sub_band_idx,
            band_id,
            component_info,
            storage,
            header,
            budget,
            classic_payloads,
            dequantization_step,
            irreversible_midpoint,
            num_bitplanes,
        )
        .map(|step| [Some(step), None])
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "classic job construction needs the validated band geometry, decode parameters, range owner, and shared budget"
)]
fn build_classic_sub_band_step(
    payload_range_owner: PayloadRangeOwner<'_>,
    sub_band: &SubBand,
    sub_band_idx: usize,
    band_id: J2kDirectBandId,
    component_info: &ComponentInfo,
    storage: &DecompositionStorage<'_>,
    header: &Header<'_>,
    budget: &mut DecodeAllocationBudget,
    mut classic_payloads: Option<&mut ClassicPayloadCollector<'_>>,
    dequantization_step: f32,
    irreversible_midpoint: bool,
    num_bitplanes: u8,
) -> Result<J2kDirectGrayscaleStep> {
    let (sub_band_type, style) =
        classic_decode_job_parameters(sub_band.sub_band_type, component_info);
    let job_capacity = direct_sub_band_job_capacity(sub_band, storage)?;
    let mut jobs = Vec::new();
    budget.reserve_new(&mut jobs, job_capacity)?;

    for precinct in sub_band
        .precincts
        .clone()
        .map(|idx| &storage.precincts[idx])
    {
        for code_block in precinct
            .code_blocks
            .clone()
            .map(|idx| &storage.code_blocks[idx])
        {
            if code_block.coding != Some(CodeBlockCoding::Classic)
                || !code_block_required_by_index(storage, sub_band_idx, code_block)
            {
                continue;
            }
            jobs.push(build_classic_code_block_job(
                payload_range_owner,
                code_block,
                sub_band,
                component_info,
                storage,
                header,
                budget,
                classic_payloads.as_deref_mut(),
                sub_band_type,
                style,
                dequantization_step,
                num_bitplanes,
            )?);
        }
    }

    Ok(J2kDirectGrayscaleStep::ClassicSubBand(
        J2kOwnedSubBandPlan {
            band_id,
            rect: J2kRect::from(sub_band.rect),
            width: sub_band.rect.width(),
            height: sub_band.rect.height(),
            irreversible_midpoint,
            jobs,
        },
    ))
}

#[expect(
    clippy::too_many_arguments,
    reason = "the classic job record combines codec metadata with its selected owned or referenced payload representation"
)]
fn build_classic_code_block_job(
    payload_range_owner: PayloadRangeOwner<'_>,
    code_block: &CodeBlock,
    sub_band: &SubBand,
    component_info: &ComponentInfo,
    storage: &DecompositionStorage<'_>,
    header: &Header<'_>,
    budget: &mut DecodeAllocationBudget,
    classic_payloads: Option<&mut ClassicPayloadCollector<'_>>,
    sub_band_type: J2kSubBandType,
    style: J2kCodeBlockStyle,
    dequantization_step: f32,
    num_bitplanes: u8,
) -> Result<J2kOwnedCodeBlockBatchJob> {
    let (data, segments) = collect_classic_payload(
        payload_range_owner,
        code_block,
        component_info,
        storage,
        budget,
        classic_payloads,
    )?;
    Ok(J2kOwnedCodeBlockBatchJob {
        output_x: code_block.rect.x0 - sub_band.rect.x0,
        output_y: code_block.rect.y0 - sub_band.rect.y0,
        data,
        segments,
        width: code_block.rect.width(),
        height: code_block.rect.height(),
        output_stride: sub_band.rect.width() as usize,
        missing_bit_planes: code_block.missing_bit_planes,
        number_of_coding_passes: code_block.number_of_coding_passes,
        total_bitplanes: num_bitplanes,
        roi_shift: component_info.roi_shift,
        sub_band_type,
        style,
        strict: header.strict,
        dequantization_step,
    })
}

fn collect_classic_payload(
    payload_range_owner: PayloadRangeOwner<'_>,
    code_block: &CodeBlock,
    component_info: &ComponentInfo,
    storage: &DecompositionStorage<'_>,
    budget: &mut DecodeAllocationBudget,
    classic_payloads: Option<&mut ClassicPayloadCollector<'_>>,
) -> Result<(Vec<u8>, Vec<J2kCodeBlockSegment>)> {
    let Some(collector) = classic_payloads else {
        return collect_classic_code_block_data(
            code_block,
            &component_info.coding_style.parameters.code_block_style,
            storage,
            budget,
        );
    };

    let first_range = collector.ranges.len();
    let (range_count, combined_length, segments) = collect_referenced_classic_code_block_data(
        code_block,
        &component_info.coding_style.parameters.code_block_style,
        storage,
        budget,
        |fragment| collector.push_range(encoded_input_range(payload_range_owner, fragment)?),
    )?;
    collector.push_payload(J2kClassicCodeBlockPayload {
        first_range,
        range_count,
        combined_length,
    })?;
    Ok((Vec::new(), segments))
}

pub(in crate::j2c::decode::direct_plan) fn strip_grayscale_payload_owners(
    plan: &mut J2kDirectGrayscalePlan,
    payloads: &[HtCodeBlockPayloadRanges],
) -> Result<usize> {
    let mut next_record = 0_usize;
    for step in &mut plan.steps {
        match step {
            J2kDirectGrayscaleStep::HtSubBand(sub_band) => {
                for job in &mut sub_band.jobs {
                    let first = *payloads
                        .get(next_record)
                        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
                    next_record = next_record
                        .checked_add(1)
                        .ok_or(ValidationError::ImageTooLarge)?;
                    if first.cleanup.length != job.cleanup_length as usize {
                        bail!(DecodingError::CodeBlockDecodeFailure);
                    }
                    let expected_refinement = job.refinement_length as usize;
                    let mut refinement = first.refinement.map_or(0, |range| range.length);
                    if refinement > expected_refinement {
                        bail!(DecodingError::CodeBlockDecodeFailure);
                    }
                    while refinement < expected_refinement {
                        let continuation = *payloads
                            .get(next_record)
                            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
                        next_record = next_record
                            .checked_add(1)
                            .ok_or(ValidationError::ImageTooLarge)?;
                        let length = continuation
                            .refinement
                            .ok_or(DecodingError::CodeBlockDecodeFailure)?
                            .length;
                        if continuation.cleanup.length != 0 || length == 0 {
                            bail!(DecodingError::CodeBlockDecodeFailure);
                        }
                        refinement = refinement
                            .checked_add(length)
                            .ok_or(ValidationError::ImageTooLarge)?;
                        if refinement > expected_refinement {
                            bail!(DecodingError::CodeBlockDecodeFailure);
                        }
                    }
                    job.data = Vec::new();
                }
            }
            J2kDirectGrayscaleStep::ClassicSubBand(_)
            | J2kDirectGrayscaleStep::Idwt(_)
            | J2kDirectGrayscaleStep::Store(_) => {}
        }
    }
    Ok(next_record)
}

pub(in crate::j2c::decode::direct_plan) fn strip_classic_payload_owners(
    plan: &mut J2kDirectGrayscalePlan,
) -> Result<usize> {
    let mut job_count = 0usize;
    for step in &mut plan.steps {
        match step {
            J2kDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                job_count = job_count
                    .checked_add(sub_band.jobs.len())
                    .ok_or(ValidationError::ImageTooLarge)?;
                for job in &mut sub_band.jobs {
                    job.data = Vec::new();
                }
            }
            J2kDirectGrayscaleStep::HtSubBand(_)
            | J2kDirectGrayscaleStep::Idwt(_)
            | J2kDirectGrayscaleStep::Store(_) => {}
        }
    }
    Ok(job_count)
}

fn encoded_input_range(owner: PayloadRangeOwner<'_>, payload: &[u8]) -> Result<J2kCodestreamRange> {
    let encoded_input_start = owner.encoded_input.as_ptr() as usize;
    let codestream_start = owner.codestream.as_ptr() as usize;
    let codestream_offset = codestream_start
        .checked_sub(encoded_input_start)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    let codestream_end = codestream_offset
        .checked_add(owner.codestream.len())
        .ok_or(ValidationError::ImageTooLarge)?;
    if codestream_end > owner.encoded_input.len() {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    let payload_start = payload.as_ptr() as usize;
    let payload_offset = payload_start
        .checked_sub(codestream_start)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    let payload_end = payload_offset
        .checked_add(payload.len())
        .ok_or(ValidationError::ImageTooLarge)?;
    if payload_end > owner.codestream.len() {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    let offset = codestream_offset
        .checked_add(payload_offset)
        .ok_or(ValidationError::ImageTooLarge)?;
    let end = offset
        .checked_add(payload.len())
        .ok_or(ValidationError::ImageTooLarge)?;
    if end > owner.encoded_input.len() {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    Ok(J2kCodestreamRange {
        offset,
        length: payload.len(),
    })
}

fn direct_sub_band_job_capacity(
    sub_band: &SubBand,
    storage: &DecompositionStorage<'_>,
) -> Result<usize> {
    sub_band
        .precincts
        .clone()
        .map(|idx| storage.precincts[idx].code_blocks.len())
        .try_fold(0_usize, |total, count| {
            total
                .checked_add(count)
                .ok_or(ValidationError::ImageTooLarge.into())
        })
}

#[cfg(test)]
mod tests;
