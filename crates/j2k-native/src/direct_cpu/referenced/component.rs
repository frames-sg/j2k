// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "parallel")]
use crate::error::ValidationError;
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

/// One code block of a parallel sub-band decode with its borrowed payload and
/// output range in the sub-band.
#[cfg(feature = "parallel")]
struct RowJob<'p, 'i> {
    job: &'p crate::HtOwnedCodeBlockBatchJob,
    data: &'i [u8],
    start: usize,
    end: usize,
}

/// Bytes owned together while a referenced HT sub-band is dispatched in
/// parallel. Planning and execution share this helper so the private dispatch
/// types cannot drift from their accounting.
#[cfg(feature = "parallel")]
pub(in crate::direct_cpu) fn parallel_ht_decode_transient_bytes(
    entry_capacity: usize,
    row_capacity: usize,
    workspace_capacity: usize,
) -> Result<usize> {
    entry_capacity
        .checked_mul(core::mem::size_of::<RowJob<'static, 'static>>())
        .and_then(|bytes| {
            row_capacity
                .checked_mul(core::mem::size_of::<
                    CodeBlockRow<'static, 'static, 'static, 'static>,
                >())
                .and_then(|rows| bytes.checked_add(rows))
        })
        .and_then(|bytes| {
            workspace_capacity
                .checked_mul(core::mem::size_of::<&mut HtCodeBlockDecodeWorkspace>())
                .and_then(|workspaces| bytes.checked_add(workspaces))
        })
        .ok_or_else(|| ValidationError::ImageTooLarge.into())
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
    let Some(rows) = carve_rows(&entries, output, sub_band_width, entry_capacity, ht.budget)?
    else {
        payloads.rewind(checkpoint);
        return Ok(false);
    };
    decode_rows(rows, entry_capacity, plan.irreversible_midpoint, ht)?;
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

/// A row of code blocks: its output rows, its jobs, and its first sample.
#[cfg(feature = "parallel")]
type CodeBlockRow<'o, 'j, 'p, 'i> = (&'o mut [f32], &'j [RowJob<'p, 'i>], usize);

/// Split `output` into one slice per row of code blocks, or `None` when the
/// sorted jobs do not form disjoint rows of equal-height blocks.
#[cfg(feature = "parallel")]
fn carve_rows<'o, 'j, 'p, 'i>(
    entries: &'j [RowJob<'p, 'i>],
    output: &'o mut [f32],
    sub_band_width: usize,
    entry_capacity: usize,
    budget: DirectWorkspaceBudget,
) -> Result<Option<alloc::vec::Vec<CodeBlockRow<'o, 'j, 'p, 'i>>>> {
    let mut rows = alloc::vec::Vec::new();
    crate::try_reserve_decode_elements(&mut rows, entries.len())?;
    budget.validate_workspace(parallel_ht_decode_transient_bytes(
        entry_capacity,
        rows.capacity(),
        0,
    )?)?;
    let mut rest = output;
    let mut consumed = 0usize;
    let mut index = 0usize;
    while index < entries.len() {
        let first = &entries[index];
        let (y, height) = (first.job.output_y as usize, first.job.height as usize);
        let mut end = index;
        let mut next_x = 0u32;
        while end < entries.len() && entries[end].job.output_y as usize == y {
            let job = entries[end].job;
            if job.height as usize != height || job.output_x < next_x {
                return Ok(None);
            }
            next_x = job.output_x + job.width;
            end += 1;
        }
        let row_start = y * sub_band_width;
        let row_end = (y + height) * sub_band_width;
        if row_start < consumed || row_end > consumed + rest.len() {
            return Ok(None);
        }
        let tail = core::mem::take(&mut rest);
        let (_, tail) = tail.split_at_mut(row_start - consumed);
        let (row, tail) = tail.split_at_mut(row_end - row_start);
        rest = tail;
        consumed = row_end;
        rows.push((row, &entries[index..end], row_start));
        index = end;
    }
    Ok(Some(rows))
}

/// Decode rows of code blocks on the Rayon pool, each with a pooled workspace.
#[cfg(feature = "parallel")]
fn decode_rows(
    rows: alloc::vec::Vec<CodeBlockRow<'_, '_, '_, '_>>,
    entry_capacity: usize,
    irreversible_midpoint: bool,
    ht: &mut HtWorkspaces<'_>,
) -> Result<()> {
    use rayon::prelude::*;
    use std::sync::Mutex;

    let budget = ht.budget;
    let row_capacity = rows.capacity();
    let workspace_count = ht
        .tasks
        .len()
        .checked_add(1)
        .ok_or(ValidationError::ImageTooLarge)?;
    let mut workspaces = alloc::vec::Vec::new();
    crate::try_reserve_decode_elements(&mut workspaces, workspace_count)?;
    workspaces.push(&mut *ht.main);
    workspaces.extend(ht.tasks.iter_mut());
    budget.validate_workspace(parallel_ht_decode_transient_bytes(
        entry_capacity,
        row_capacity,
        workspaces.capacity(),
    )?)?;
    let pool = Mutex::new(workspaces);
    rows.into_par_iter()
        .with_max_len(1)
        .try_for_each(|(row, jobs, row_start)| -> Result<()> {
            // Bind the popped workspace first so the guard drops before the
            // workspace is returned.
            let pooled = pool
                .lock()
                .map_err(|_| DecodingError::CodeBlockDecodeFailure)?
                .pop();
            let mut decode_row = |workspace: &mut HtCodeBlockDecodeWorkspace| -> Result<()> {
                jobs.iter().try_for_each(|entry| {
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
                    let output = &mut row[entry.start - row_start..entry.end - row_start];
                    if irreversible_midpoint {
                        decode_ht_code_block_scalar_with_workspace_midpoint(
                            code_block, output, workspace,
                        )
                    } else {
                        decode_ht_code_block_scalar_with_workspace(code_block, output, workspace)
                    }
                })
            };
            let workspace = pooled.ok_or(DecodingError::CodeBlockDecodeFailure)?;
            let result = decode_row(workspace);
            pool.lock()
                .map_err(|_| DecodingError::CodeBlockDecodeFailure)?
                .push(workspace);
            result
        })
}
