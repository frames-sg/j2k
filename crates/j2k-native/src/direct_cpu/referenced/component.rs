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
    prepare_sub_band_output, rounds_at_store, store_component, DirectComponentBandScratch,
    DirectComponentPlane, DirectWorkspaceBudget, SubBandJobOutputRange,
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
    ht: &mut HtWorkspaces<'_>,
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
            ht,
            &mut output_initialized[component_index],
            rounds_at_store(mct, component_index),
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
            true,
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
    ht: &mut HtWorkspaces<'_>,
    output_initialized: &mut bool,
    round_irreversible_output: bool,
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
                execute_ht_sub_band_referenced(sub_band, bands, payloads, compressed_payload, ht)?;
            }
            J2kDirectGrayscaleStep::Idwt(step) => execute_idwt_step(step, bands, ht.parallel)?,
            J2kDirectGrayscaleStep::Store(store) => {
                store_component(
                    store,
                    bands.active(),
                    output,
                    output_initialized,
                    round_irreversible_output,
                )?;
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

/// HT code-block workspaces for one referenced execution: the retained main
/// workspace, plus one per extra Rayon worker when `parallel` is set.
pub(super) struct HtWorkspaces<'a> {
    pub(super) main: &'a mut HtCodeBlockDecodeWorkspace,
    #[cfg_attr(
        not(feature = "parallel"),
        expect(dead_code, reason = "extra workspaces serve parallel execution only")
    )]
    pub(super) tasks: &'a mut [HtCodeBlockDecodeWorkspace],
    pub(super) parallel: bool,
    #[cfg_attr(
        not(feature = "parallel"),
        expect(
            dead_code,
            reason = "the workspace budget validates parallel dispatch owners"
        )
    )]
    pub(super) budget: DirectWorkspaceBudget,
}

fn execute_ht_sub_band_referenced(
    plan: &HtOwnedSubBandPlan,
    bands: &mut DirectComponentBandScratch,
    payloads: &mut ReferencedPayloadCursor<'_>,
    compressed_payload: &mut alloc::vec::Vec<u8>,
    ht: &mut HtWorkspaces<'_>,
) -> Result<()> {
    let (output, sub_band_width) =
        prepare_sub_band_output(bands, plan.band_id, plan.rect, plan.width, plan.height)?;
    #[cfg(feature = "parallel")]
    if ht.parallel
        && !ht.tasks.is_empty()
        && decode_ht_sub_band_rows_parallel(plan, output, sub_band_width, payloads, ht)?
    {
        return Ok(());
    }
    let workspace = &mut *ht.main;
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

#[cfg(feature = "parallel")]
type RowJob<'p, 'i> = super::rows::RowJob<'p, 'i, crate::HtOwnedCodeBlockBatchJob>;

/// Bytes owned together while a referenced HT sub-band is dispatched in
/// parallel. Planning and execution share this helper so the private dispatch
/// types cannot drift from their accounting.
#[cfg(feature = "parallel")]
pub(in crate::direct_cpu) fn parallel_ht_decode_transient_bytes(
    entry_capacity: usize,
    row_capacity: usize,
    workspace_capacity: usize,
) -> Result<usize> {
    super::rows::row_dispatch_transient_bytes::<
        crate::HtOwnedCodeBlockBatchJob,
        HtCodeBlockDecodeWorkspace,
    >(entry_capacity, row_capacity, workspace_capacity)
}

/// Decode a sub-band's rows of code blocks in parallel, each writing its own
/// rows of `output`. Returns `false` without consuming payloads when a block's
/// bytes are not one contiguous input slice or the blocks do not tile the
/// sub-band in rows; the serial path then decodes the sub-band.
#[cfg(feature = "parallel")]
fn decode_ht_sub_band_rows_parallel(
    plan: &HtOwnedSubBandPlan,
    output: &mut [f32],
    sub_band_width: usize,
    payloads: &mut ReferencedPayloadCursor<'_>,
    ht: &mut HtWorkspaces<'_>,
) -> Result<bool> {
    let checkpoint = payloads.position();
    let Some(mut entries) =
        collect_row_jobs(plan, output.len(), sub_band_width, payloads, ht.budget)?
    else {
        payloads.rewind(checkpoint);
        return Ok(false);
    };
    entries.sort_unstable_by_key(|entry| (entry.job.output_y, entry.job.output_x));
    let entry_capacity = entries.capacity();
    let Some(rows) = super::rows::carve_rows::<_, HtCodeBlockDecodeWorkspace>(
        &entries,
        output,
        sub_band_width,
        entry_capacity,
        ht.budget,
    )?
    else {
        payloads.rewind(checkpoint);
        return Ok(false);
    };
    let irreversible_midpoint = plan.irreversible_midpoint;
    super::rows::decode_rows(
        rows,
        entry_capacity,
        &mut *ht.main,
        ht.tasks,
        ht.budget,
        |entry, output, workspace| {
            let job = entry.job;
            let code_block = HtCodeBlockDecodeJob {
                data: entry.data,
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
            if irreversible_midpoint {
                decode_ht_code_block_scalar_with_workspace_midpoint(code_block, output, workspace)
            } else {
                decode_ht_code_block_scalar_with_workspace(code_block, output, workspace)
            }
        },
    )?;
    Ok(true)
}

/// Validate every job's output range and borrow its payload, or `None` when
/// a payload is not one contiguous input slice.
#[cfg(feature = "parallel")]
fn collect_row_jobs<'p, 'i>(
    plan: &'p HtOwnedSubBandPlan,
    output_len: usize,
    sub_band_width: usize,
    payloads: &mut ReferencedPayloadCursor<'i>,
    budget: DirectWorkspaceBudget,
) -> Result<Option<alloc::vec::Vec<RowJob<'p, 'i>>>> {
    let mut entries = alloc::vec::Vec::new();
    crate::try_reserve_decode_elements(&mut entries, plan.jobs.len())?;
    budget.validate_workspace(parallel_ht_decode_transient_bytes(
        entries.capacity(),
        0,
        0,
    )?)?;
    for job in &plan.jobs {
        let range = checked_sub_band_job_output_range(&SubBandJobOutputRange {
            output_x: job.output_x,
            output_y: job.output_y,
            output_stride: job.output_stride,
            width: job.width,
            height: job.height,
            sub_band_width,
            plan_width: plan.width,
            plan_height: plan.height,
            output_len,
        })?;
        if job.output_stride != sub_band_width {
            return Ok(None);
        }
        let Some(data) = payloads.next_contiguous(job.cleanup_length, job.refinement_length)?
        else {
            return Ok(None);
        };
        entries.push(RowJob {
            job,
            data,
            start: range.start,
            end: range.end,
        });
    }
    Ok(Some(entries))
}
