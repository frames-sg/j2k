// SPDX-License-Identifier: MIT OR Apache-2.0

//! HT code-block payload construction for one direct-plan sub-band.

use super::{
    add_roi_shift_to_bitplanes, code_block_required_by_index, direct_sub_band_job_capacity,
    encoded_input_range, ht_block_decode, ht_code_block_has_decodable_passes, ComponentInfo,
    DecodeAllocationBudget, DecompositionStorage, Header, HtCodeBlockPayloadRanges,
    HtOwnedCodeBlockBatchJob, HtOwnedSubBandPlan, J2kCodestreamRange, J2kDirectBandId,
    J2kDirectGrayscaleStep, J2kRect, PayloadRangeOwner, Result, SubBand, Vec,
};
use crate::j2c::build::CodeBlockCoding;

#[expect(
    clippy::too_many_arguments,
    reason = "HT job construction needs the validated band geometry, decode parameters, range owner, and shared budget"
)]
pub(super) fn build_ht_sub_band_step(
    payload_range_owner: PayloadRangeOwner<'_>,
    sub_band: &SubBand,
    sub_band_idx: usize,
    band_id: J2kDirectBandId,
    component_info: &ComponentInfo,
    storage: &DecompositionStorage<'_>,
    header: &Header<'_>,
    budget: &mut DecodeAllocationBudget,
    mut ht_payloads: Option<&mut Vec<HtCodeBlockPayloadRanges>>,
    dequantization_step: f32,
    irreversible_midpoint: bool,
    num_bitplanes: u8,
) -> Result<J2kDirectGrayscaleStep> {
    let coded_bitplanes = add_roi_shift_to_bitplanes(num_bitplanes, component_info.roi_shift, 31)?;
    let stripe_causal = component_info
        .coding_style
        .parameters
        .code_block_style
        .vertically_causal_context;
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
            if code_block.coding != Some(CodeBlockCoding::HighThroughput)
                || !code_block_required_by_index(storage, sub_band_idx, code_block)
                || !ht_code_block_has_decodable_passes(code_block, coded_bitplanes, header.strict)?
            {
                continue;
            }

            if let Some(payloads) = ht_payloads.as_deref_mut() {
                append_referenced_payload_records(
                    payloads,
                    payload_range_owner,
                    code_block,
                    storage,
                )?;
            }

            let combined = ht_block_decode::collect_code_block_data(code_block, storage, budget)?;
            jobs.push(HtOwnedCodeBlockBatchJob {
                output_x: code_block.rect.x0 - sub_band.rect.x0,
                output_y: code_block.rect.y0 - sub_band.rect.y0,
                data: combined.data,
                cleanup_length: combined.cleanup_length,
                refinement_length: combined.refinement_length,
                width: code_block.rect.width(),
                height: code_block.rect.height(),
                output_stride: sub_band.rect.width() as usize,
                missing_bit_planes: code_block.missing_bit_planes,
                number_of_coding_passes: code_block.number_of_coding_passes,
                num_bitplanes,
                roi_shift: component_info.roi_shift,
                stripe_causal,
                strict: header.strict,
                dequantization_step,
            });
        }
    }

    Ok(J2kDirectGrayscaleStep::HtSubBand(HtOwnedSubBandPlan {
        band_id,
        rect: J2kRect::from(sub_band.rect),
        width: sub_band.rect.width(),
        height: sub_band.rect.height(),
        irreversible_midpoint,
        jobs,
    }))
}

fn append_referenced_payload_records(
    payloads: &mut Vec<HtCodeBlockPayloadRanges>,
    payload_range_owner: PayloadRangeOwner<'_>,
    code_block: &crate::j2c::build::CodeBlock,
    storage: &DecompositionStorage<'_>,
) -> Result<()> {
    let first_record = payloads.len();
    ht_block_decode::visit_code_block_segments(code_block, storage, |kind, data| {
        let range = encoded_input_range(payload_range_owner, data)?;
        match kind {
            ht_block_decode::HtCodeBlockSegmentKind::Cleanup => {
                payloads.push(HtCodeBlockPayloadRanges {
                    cleanup: range,
                    refinement: None,
                });
            }
            ht_block_decode::HtCodeBlockSegmentKind::Refinement => {
                let first = payloads
                    .get_mut(first_record)
                    .ok_or(crate::DecodingError::CodeBlockDecodeFailure)?;
                if first.refinement.is_none() {
                    first.refinement = Some(range);
                } else {
                    payloads.push(HtCodeBlockPayloadRanges {
                        cleanup: J2kCodestreamRange {
                            offset: range.offset,
                            length: 0,
                        },
                        refinement: Some(range),
                    });
                }
            }
        }
        Ok(())
    })
}
