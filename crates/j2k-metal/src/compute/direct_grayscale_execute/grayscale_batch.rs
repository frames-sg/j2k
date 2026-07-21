// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    allocate_direct_execution_metadata, commit_and_wait_metal,
    encode_prepared_direct_grayscale_plan_in_command_buffer,
    encode_repeated_direct_grayscale_plan_in_command_buffer,
    encode_repeated_gray_plane_to_surfaces_in_command_buffer,
    encode_stacked_direct_component_plane_batch, new_command_buffer, recycle_scratch_buffers,
    retire_direct_status_checks, supports_stacked_direct_component_plane_batch, with_runtime, Arc,
    DirectColorBatchCommandBuffers, DirectExecutionMetadata, DirectHybridStageTimings,
    DirectStatusRetirementMode, DirectTier1Mode, Error, PixelFormat, PreparedDirectGrayscalePlan,
    RepeatedDirectGrayscalePlanRequest, StackedDirectComponentPlaneBatchRequest, Surface,
};

pub(crate) fn execute_repeated_prepared_direct_grayscale_plan(
    plan: &PreparedDirectGrayscalePlan,
    fmt: PixelFormat,
    count: usize,
) -> Result<Vec<Surface>, Error> {
    with_runtime(|runtime| {
        let DirectExecutionMetadata {
            mut retained_buffers,
            mut status_checks,
            mut scratch_buffers,
        } = allocate_direct_execution_metadata(
            plan.steps.len(),
            0,
            crate::batch_allocation::BatchMetadataBudget::new(
                "J2K Metal repeated direct execution resources",
            ),
        )?;
        let command_buffer = new_command_buffer(&runtime.queue)?;
        let surfaces = encode_repeated_direct_grayscale_plan_in_command_buffer(
            RepeatedDirectGrayscalePlanRequest {
                runtime,
                command_buffer: &command_buffer,
                plan,
                fmt,
                count,
                retained_buffers: &mut retained_buffers,
                status_checks: &mut status_checks,
                scratch_buffers: &mut scratch_buffers,
            },
        )?;
        let completion = commit_and_wait_metal(&command_buffer);
        let status_retirement = retire_direct_status_checks(
            runtime,
            status_checks,
            if completion.is_ok() {
                DirectStatusRetirementMode::Validate
            } else {
                DirectStatusRetirementMode::RecycleWithoutRead
            },
        );
        drop(retained_buffers);
        let scratch_retirement = recycle_scratch_buffers(runtime, scratch_buffers);
        completion.and(status_retirement).and(scratch_retirement)?;
        Ok(surfaces)
    })
}

pub(crate) fn execute_prepared_direct_grayscale_plan_batch(
    plans: &[Arc<PreparedDirectGrayscalePlan>],
    fmt: PixelFormat,
) -> Result<Vec<Surface>, Error> {
    if plans.is_empty() {
        return Ok(Vec::new());
    }

    with_runtime(|runtime| {
        let step_count = crate::batch_allocation::checked_count_sum(
            plans.iter().map(|plan| plan.steps.len()),
            "J2K Metal direct grayscale batch step metadata",
        )?;
        let DirectExecutionMetadata {
            mut retained_buffers,
            mut status_checks,
            mut scratch_buffers,
        } = allocate_direct_execution_metadata(
            step_count,
            0,
            crate::batch_allocation::BatchMetadataBudget::new(
                "J2K Metal direct grayscale batch execution resources",
            ),
        )?;
        let command_buffer = new_command_buffer(&runtime.queue)?;
        let mut stage_timings = DirectHybridStageTimings::default();
        let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
            "J2K Metal direct grayscale batch execution",
        );
        budget.preflight(&[
            crate::batch_allocation::BatchMetadataRequest::of::<Surface>(plans.len()),
            crate::batch_allocation::BatchMetadataRequest::of::<&PreparedDirectGrayscalePlan>(
                plans.len(),
            ),
        ])?;
        let mut surfaces = budget.try_vec(plans.len(), "J2K Metal direct grayscale surfaces")?;
        let mut component_plan_refs = budget.try_vec(
            plans.len(),
            "J2K Metal direct grayscale component plan references",
        )?;
        component_plan_refs.extend(plans.iter().map(Arc::as_ref));
        if plans.len() > 1 && supports_stacked_direct_component_plane_batch(&component_plan_refs) {
            let stacked_plane = encode_stacked_direct_component_plane_batch(
                StackedDirectComponentPlaneBatchRequest {
                    runtime,
                    command_buffers: DirectColorBatchCommandBuffers::single(&command_buffer),
                    compute_encoder: None,
                    plans: &component_plan_refs,
                    component_idx: 0,
                    flattened_cpu_tier1_cache: None,
                    tier1_mode: DirectTier1Mode::Metal,
                    stage_timings: &mut stage_timings,
                    retained_buffers: &mut retained_buffers,
                    status_checks: &mut status_checks,
                    scratch_buffers: &mut scratch_buffers,
                },
            )?;
            let first = plans.first().expect("plans is not empty");
            if stacked_plane.dimensions == first.dimensions && stacked_plane.count == plans.len() {
                surfaces = encode_repeated_gray_plane_to_surfaces_in_command_buffer(
                    runtime,
                    &command_buffer,
                    &stacked_plane.buffer,
                    first.dimensions,
                    first.bit_depth,
                    fmt,
                    plans.len(),
                )?;
            }
        }

        if surfaces.is_empty() {
            for plan in plans {
                surfaces.push(encode_prepared_direct_grayscale_plan_in_command_buffer(
                    runtime,
                    &command_buffer,
                    plan,
                    fmt,
                    &mut retained_buffers,
                    &mut status_checks,
                    &mut scratch_buffers,
                )?);
            }
        }

        let completion = commit_and_wait_metal(&command_buffer);
        let status_retirement = retire_direct_status_checks(
            runtime,
            status_checks,
            if completion.is_ok() {
                DirectStatusRetirementMode::Validate
            } else {
                DirectStatusRetirementMode::RecycleWithoutRead
            },
        );
        drop(retained_buffers);
        let scratch_retirement = recycle_scratch_buffers(runtime, scratch_buffers);
        completion.and(status_retirement).and(scratch_retirement)?;
        Ok(surfaces)
    })
}
