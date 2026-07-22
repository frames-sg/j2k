// SPDX-License-Identifier: MIT OR Apache-2.0

use metal::{Buffer, ComputeCommandEncoderRef};

use crate::batch_allocation::BatchMetadataBudget;

use super::super::{
    metal_profile_stages_enabled, DirectColorBatchCommandBuffers, DirectHybridStageTimings,
    DirectScratchBuffer, DirectStatusCheck, DirectTier1Mode, Error, FlattenedCpuTier1Cache,
    MetalRuntime, PreparedDirectGrayscalePlan, PreparedDirectGrayscaleStep,
};
use super::resources::StackedComponentResources;
use super::validation::StackedComponentBatchPlan;
use super::StackedDirectComponentPlaneBatchRequest;

mod classic_tier1;
mod final_store;
mod ht_tier1;
mod reconstruction;

struct SubmissionContext<'a, 'p, 'r> {
    runtime: &'a MetalRuntime,
    command_buffers: DirectColorBatchCommandBuffers<'a>,
    compute_encoder: Option<&'a ComputeCommandEncoderRef>,
    plans: &'a [&'p PreparedDirectGrayscalePlan],
    component_idx: usize,
    flattened_cpu_tier1_cache: Option<&'a FlattenedCpuTier1Cache>,
    tier1_mode: DirectTier1Mode,
    stage_timings: &'a mut DirectHybridStageTimings,
    retained_buffers: &'a mut Vec<Buffer>,
    status_checks: &'a mut Vec<DirectStatusCheck>,
    scratch_buffers: &'a mut Vec<DirectScratchBuffer>,
    count: usize,
    broadcast_tier1_inputs: bool,
    profile_stages: bool,
    resources: &'r mut StackedComponentResources,
}

fn planned_cpu_input_count(
    mode: DirectTier1Mode,
    has_flattened_cache: bool,
    broadcast: bool,
    item_count: usize,
) -> usize {
    if mode != DirectTier1Mode::CpuUpload || has_flattened_cache {
        0
    } else if broadcast {
        item_count.min(1)
    } else {
        item_count
    }
}

fn try_collect_submission_items<T>(
    budget: &mut BatchMetadataBudget,
    mut items: impl ExactSizeIterator<Item = Result<T, Error>>,
    what: &'static str,
) -> Result<Vec<T>, Error> {
    let mut values = budget.try_vec(items.len(), what)?;
    for item in &mut items {
        values.push(item?);
    }
    Ok(values)
}

pub(super) fn submit_stacked_component_commands<'p>(
    request: StackedDirectComponentPlaneBatchRequest<'_, 'p>,
    plan: &StackedComponentBatchPlan<'p>,
    resources: &mut StackedComponentResources,
) -> Result<(), Error> {
    let StackedDirectComponentPlaneBatchRequest {
        runtime,
        command_buffers,
        compute_encoder,
        plans,
        component_idx,
        flattened_cpu_tier1_cache,
        tier1_mode,
        stage_timings,
        retained_buffers,
        status_checks,
        scratch_buffers,
    } = request;
    let profile_stages = tier1_mode == DirectTier1Mode::CpuUpload && metal_profile_stages_enabled();
    let mut context = SubmissionContext {
        runtime,
        command_buffers,
        compute_encoder,
        plans,
        component_idx,
        flattened_cpu_tier1_cache,
        tier1_mode,
        stage_timings,
        retained_buffers,
        status_checks,
        scratch_buffers,
        count: plan.count,
        broadcast_tier1_inputs: plan.broadcast_tier1_inputs,
        profile_stages,
        resources,
    };

    let first = plan.first;
    let mut step_idx = 0;
    while step_idx < first.steps.len() {
        if let Some(group) = first.classic_group_starting_at(step_idx) {
            context.submit_classic_group(first, step_idx, group)?;
            step_idx = group.end_step;
            continue;
        }
        if let Some(group) = first.ht_group_starting_at(step_idx) {
            context.submit_ht_group(first, step_idx, group)?;
            step_idx = group.end_step;
            continue;
        }

        match &first.steps[step_idx] {
            PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                context.submit_classic_sub_band(first, step_idx, sub_band)?;
            }
            PreparedDirectGrayscaleStep::HtSubBand(sub_band) => {
                context.submit_ht_sub_band(first, step_idx, sub_band)?;
            }
            PreparedDirectGrayscaleStep::Idwt(idwt) => {
                context.submit_idwt(step_idx, idwt)?;
            }
            PreparedDirectGrayscaleStep::Store(store) => {
                context.submit_store(store)?;
            }
        }
        step_idx += 1;
    }

    Ok(())
}
