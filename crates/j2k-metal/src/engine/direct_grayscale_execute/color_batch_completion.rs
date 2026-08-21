// SPDX-License-Identifier: MIT OR Apache-2.0

//! Completion timing and retained-resource retirement for direct color batches.

use std::time::Instant;

use crate::profile_env::{hybrid_stage_signpost, SIGNPOST_DECODE_HYBRID_COMMAND_WAIT};

use super::super::{
    completed_command_buffer_gpu_duration, elapsed_us, record_completed_decode_split_gpu_stages,
    recycle_scratch_buffers, retire_direct_status_checks, wait_for_completion_metal,
    CommandBufferRef, DecodeHybridSplitCommandBuffers, DirectHybridStageTimings,
    DirectStatusRetirementMode, Error, MetalRuntime,
};
use super::allocation::DirectExecutionMetadata;

pub(super) fn retire_direct_color_batch_resources(
    runtime: &MetalRuntime,
    completion: Result<(), Error>,
    status_mode: DirectStatusRetirementMode,
    metadata: &mut DirectExecutionMetadata,
) -> Result<(), Error> {
    let status_retirement = retire_direct_status_checks(
        runtime,
        core::mem::take(&mut metadata.status_checks),
        status_mode,
    );
    metadata.retained_buffers.clear();
    let scratch_retirement =
        recycle_scratch_buffers(runtime, core::mem::take(&mut metadata.scratch_buffers));
    completion.and(status_retirement).and(scratch_retirement)
}

pub(super) fn complete_direct_color_batch_command(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    profile_hybrid_stages: bool,
    stage_timings: &mut DirectHybridStageTimings,
    metadata: &mut DirectExecutionMetadata,
) -> Result<(), Error> {
    let wait_started = profile_hybrid_stages.then(Instant::now);
    let _wait_signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_COMMAND_WAIT);
    let completion = wait_for_completion_metal(command_buffer);
    if completion.is_ok() {
        if let Some(started) = wait_started {
            stage_timings.command_wait += elapsed_us(started);
        }
        if profile_hybrid_stages {
            if let Some(duration) = completed_command_buffer_gpu_duration(command_buffer) {
                stage_timings.gpu_command += duration.as_micros();
            }
        }
    }
    let status_mode = if completion.is_ok() {
        DirectStatusRetirementMode::Validate
    } else {
        DirectStatusRetirementMode::RecycleWithoutRead
    };
    retire_direct_color_batch_resources(runtime, completion, status_mode, metadata)
}

pub(super) fn complete_split_direct_color_batch_command(
    runtime: &MetalRuntime,
    command_buffers: &DecodeHybridSplitCommandBuffers,
    stage_timings: &mut DirectHybridStageTimings,
    metadata: &mut DirectExecutionMetadata,
) -> Result<(), Error> {
    let wait_started = Instant::now();
    let _wait_signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_COMMAND_WAIT);
    let completion = wait_for_completion_metal(&command_buffers.mct_pack);
    if completion.is_ok() {
        stage_timings.command_wait += elapsed_us(wait_started);
        record_completed_decode_split_gpu_stages(stage_timings, command_buffers);
    }
    let status_mode = if completion.is_ok() {
        DirectStatusRetirementMode::Validate
    } else {
        DirectStatusRetirementMode::RecycleWithoutRead
    };
    retire_direct_color_batch_resources(runtime, completion, status_mode, metadata)
}
