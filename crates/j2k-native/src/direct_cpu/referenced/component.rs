// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::{bail, DecodingError, Result};
use crate::{
    decode_ht_code_block_scalar_with_workspace,
    decode_ht_code_block_scalar_with_workspace_midpoint, HtCodeBlockDecodeJob,
    HtCodeBlockDecodeWorkspace, HtOwnedSubBandPlan, J2kDirectGrayscalePlan, J2kDirectGrayscaleStep,
    J2kWaveletTransform,
};

use super::super::referenced_classic::{execute_classic_sub_band_referenced, ClassicPayloadCursor};
use super::super::{
    apply_inverse_mct_region, checked_sub_band_job_output_range, execute_idwt_step,
    prepare_sub_band_output, store_component, DirectComponentBandScratch, DirectComponentPlane,
    SubBandJobOutputRange,
};
use super::payload::ReferencedPayloadCursor;

#[expect(
    clippy::too_many_arguments,
    reason = "the parse-free RGB-family executor keeps geometry, transform metadata, retained owners, payload cursor, and scalar workspace explicit"
)]
pub(super) fn execute_color_components_referenced(
    component_plans: &[J2kDirectGrayscalePlan],
    expected_component_count: usize,
    rgb_bit_depths: [u8; 3],
    mct: bool,
    transform: J2kWaveletTransform,
    signed: bool,
    component_band_sets: &mut [DirectComponentBandScratch],
    reconstructed_planes: &mut [DirectComponentPlane],
    payloads: &mut ReferencedPayloadCursor<'_>,
    classic_payloads: &mut ClassicPayloadCursor<'_>,
    compressed_payload: &mut alloc::vec::Vec<u8>,
    classic_workspace: &mut crate::J2kCodeBlockDecodeWorkspace,
    ht_workspace: &mut HtCodeBlockDecodeWorkspace,
    output_initialized: &mut [bool; 4],
    destination: crate::J2kRect,
) -> Result<()> {
    if component_plans.len() != expected_component_count
        || !matches!(expected_component_count, 3 | 4)
        || component_band_sets.len() < component_plans.len()
        || reconstructed_planes.len() < component_plans.len()
    {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    for (component_index, component_plan) in component_plans.iter().enumerate() {
        execute_component_plan_referenced(
            component_plan,
            &mut component_band_sets[component_index],
            &mut reconstructed_planes[component_index],
            payloads,
            classic_payloads,
            compressed_payload,
            classic_workspace,
            ht_workspace,
            &mut output_initialized[component_index],
        )?;
    }
    if mct {
        let [plane0, plane1, plane2, ..] = reconstructed_planes else {
            bail!(DecodingError::CodeBlockDecodeFailure);
        };
        apply_inverse_mct_region(
            transform,
            rgb_bit_depths,
            signed,
            false,
            destination,
            [plane0, plane1, plane2],
        )?;
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "mixed HT/classic execution keeps both validated payload cursors and retained decode workspaces explicit"
)]
pub(super) fn execute_component_plan_referenced(
    plan: &J2kDirectGrayscalePlan,
    bands: &mut DirectComponentBandScratch,
    output: &mut DirectComponentPlane,
    payloads: &mut ReferencedPayloadCursor<'_>,
    classic_payloads: &mut ClassicPayloadCursor<'_>,
    compressed_payload: &mut alloc::vec::Vec<u8>,
    classic_workspace: &mut crate::J2kCodeBlockDecodeWorkspace,
    ht_workspace: &mut HtCodeBlockDecodeWorkspace,
    output_initialized: &mut bool,
) -> Result<()> {
    bands.reset();
    let mut stored = false;
    for step in &plan.steps {
        match step {
            J2kDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                execute_classic_sub_band_referenced(
                    sub_band,
                    bands,
                    classic_payloads,
                    compressed_payload,
                    classic_workspace,
                )?;
            }
            J2kDirectGrayscaleStep::HtSubBand(sub_band) => {
                execute_ht_sub_band_referenced(
                    sub_band,
                    bands,
                    payloads,
                    compressed_payload,
                    ht_workspace,
                )?;
            }
            J2kDirectGrayscaleStep::Idwt(step) => execute_idwt_step(step, bands)?,
            J2kDirectGrayscaleStep::Store(store) => {
                store_component(store, bands.active(), output, output_initialized, false)?;
                stored = true;
            }
        }
    }
    if stored {
        Ok(())
    } else {
        Err(DecodingError::CodeBlockDecodeFailure.into())
    }
}

fn execute_ht_sub_band_referenced(
    plan: &HtOwnedSubBandPlan,
    bands: &mut DirectComponentBandScratch,
    payloads: &mut ReferencedPayloadCursor<'_>,
    compressed_payload: &mut alloc::vec::Vec<u8>,
    workspace: &mut HtCodeBlockDecodeWorkspace,
) -> Result<()> {
    let (output, sub_band_width) =
        prepare_sub_band_output(bands, plan.band_id, plan.rect, plan.width, plan.height)?;
    for job in &plan.jobs {
        let output_range = checked_sub_band_job_output_range(&SubBandJobOutputRange {
            output_x: job.output_x,
            output_y: job.output_y,
            output_stride: job.output_stride,
            width: job.width,
            height: job.height,
            sub_band_width,
            plan_width: plan.width,
            plan_height: plan.height,
            output_len: output.len(),
        })?;
        let data = payloads.next_data(
            job.cleanup_length,
            job.refinement_length,
            compressed_payload,
        )?;
        let code_block = HtCodeBlockDecodeJob {
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
        };
        let decode = if plan.irreversible_midpoint {
            decode_ht_code_block_scalar_with_workspace_midpoint
        } else {
            decode_ht_code_block_scalar_with_workspace
        };
        decode(code_block, &mut output[output_range], workspace)?;
    }
    Ok(())
}
