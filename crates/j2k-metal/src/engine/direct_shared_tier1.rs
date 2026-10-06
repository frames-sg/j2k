// SPDX-License-Identifier: MIT OR Apache-2.0

//! One classic Tier-1 dispatch shared by several stacked color groups.
//!
//! Groups submitted one after another run their Tier-1 dispatches
//! concurrently, and each dispatch lasts as long as its slowest SIMD group.
//! Decoding every group's code blocks in one earlier dispatch lets the dense
//! kernel pack jobs against the whole GPU. Each group then reads its
//! coefficients from the shared output and runs its inverse transforms and
//! store unchanged.
//!
//! The group command buffers are committed after the shared one on the same
//! queue. Pooled buffers use Metal's default tracked hazard mode, which keeps
//! later commands from reading the output until the dispatch has written it.

#[cfg(target_os = "macos")]
use crate::metal_types::prelude::*;

use std::rc::Rc;
use std::sync::Arc;

use crate::metal_types::{Buffer, CommandBuffer};
use crate::profile_env::label_command_buffer;

use super::decode_dispatch::{
    encode_distinct_classic_batches_to_buffer_in_encoder, DistinctClassicBatch,
};
use super::direct_grayscale_execute::checked_coefficient_len;
use super::{
    new_command_buffer, new_compute_command_encoder, recycle_scratch_buffers,
    take_f32_scratch_buffer, wait_for_completion_metal, DirectScratchBuffer, DirectStatusCheck,
    Error, MetalRuntime, PreparedDirectGrayscalePlan, PreparedDirectGrayscaleStep,
};

/// A committed shared Tier-1 command buffer and the resources it uses.
pub(super) struct SharedClassicTier1Pass {
    runtime: Arc<MetalRuntime>,
    command_buffer: CommandBuffer,
    retained_buffers: Vec<Buffer>,
    scratch_buffers: Vec<DirectScratchBuffer>,
}

impl SharedClassicTier1Pass {
    /// Waits for the shared dispatch. Code-block statuses are validated by
    /// each consuming group.
    pub(super) fn wait(&self) -> Result<(), Error> {
        wait_for_completion_metal(&self.command_buffer)
    }

    /// Drops one consumer's reference. The last one returns the scratch
    /// buffers to the pool once the dispatch has completed.
    pub(super) fn release(pass: Rc<Self>) -> Result<(), Error> {
        Rc::into_inner(pass).map_or(Ok(()), |mut pass| pass.retire())
    }

    fn retire(&mut self) -> Result<(), Error> {
        let completion = wait_for_completion_metal(&self.command_buffer);
        self.retained_buffers.clear();
        let recycled =
            recycle_scratch_buffers(&self.runtime, core::mem::take(&mut self.scratch_buffers));
        completion.and(recycled)
    }
}

impl Drop for SharedClassicTier1Pass {
    fn drop(&mut self) {
        // Reached when the last holder never submitted, e.g. a group that
        // failed to encode. Submitted groups report a dispatch failure
        // through `wait`; nothing reads the output after this point.
        let _ = self.retire();
    }
}

struct SharedClassicTier1Step {
    step_idx: usize,
    per_instance_len: usize,
    instance_count: usize,
    offset_bytes: usize,
    /// Positions of this step's jobs in the shared status buffer, and the
    /// stack instance each job belongs to.
    status_indices: Vec<usize>,
    instances: Vec<usize>,
    consumed: bool,
}

/// One group's coefficients inside a shared Tier-1 output.
pub(super) struct SharedClassicTier1Group {
    pass: Rc<SharedClassicTier1Pass>,
    output: Buffer,
    status_buffer: Buffer,
    status_len: usize,
    steps: Vec<SharedClassicTier1Step>,
}

impl SharedClassicTier1Group {
    /// Returns the buffer and byte offset holding the coefficients of the
    /// Tier-1 step at `step_idx`, laid out as the group's own dispatch would
    /// have written them, and moves the step's status check into
    /// `status_checks`.
    pub(super) fn take_step(
        &mut self,
        step_idx: usize,
        per_instance_len: usize,
        instance_count: usize,
        status_checks: &mut Vec<DirectStatusCheck>,
    ) -> Result<(Buffer, usize), Error> {
        let step = self
            .steps
            .iter_mut()
            .find(|step| step.step_idx == step_idx)
            .ok_or(Error::MetalStateInvariant {
                state: "J2K Metal shared classic Tier-1",
                reason: "group requested a step the shared dispatch did not decode",
            })?;
        if step.per_instance_len != per_instance_len || step.instance_count != instance_count {
            return Err(Error::MetalStateInvariant {
                state: "J2K Metal shared classic Tier-1",
                reason: "group step shape differs from the shared dispatch",
            });
        }
        if step.consumed {
            return Err(Error::MetalStateInvariant {
                state: "J2K Metal shared classic Tier-1",
                reason: "group step was consumed twice",
            });
        }
        crate::batch_allocation::try_reserve_for_push(
            status_checks,
            "J2K Metal shared classic Tier-1 status checks",
        )?;
        step.consumed = true;
        status_checks.push(DirectStatusCheck::ClassicShared {
            buffer: self.status_buffer.clone(),
            buffer_len: self.status_len,
            status_indices: core::mem::take(&mut step.status_indices),
            source_indices: Some(core::mem::take(&mut step.instances)),
        });
        Ok((self.output.clone(), step.offset_bytes))
    }

    /// Confirms every decoded step was consumed and returns the pass the
    /// group's submission must wait for.
    pub(super) fn finish(self) -> Result<Rc<SharedClassicTier1Pass>, Error> {
        if self.steps.iter().any(|step| !step.consumed) {
            return Err(Error::MetalStateInvariant {
                state: "J2K Metal shared classic Tier-1",
                reason: "group encoding skipped a step the shared dispatch decoded",
            });
        }
        Ok(self.pass)
    }
}

struct BatchOwner {
    group: usize,
    step: usize,
    instance: usize,
}

struct SharedTier1Inputs<'p> {
    batches: Vec<DistinctClassicBatch<'p>>,
    owners: Vec<BatchOwner>,
    steps: Vec<Vec<SharedClassicTier1Step>>,
    total_len: usize,
}

/// Whether the stacked component plans decode Tier-1 only with classic
/// code blocks, so a shared classic dispatch covers all of it.
pub(super) fn stacked_plans_use_classic_tier1_only(plans: &[&PreparedDirectGrayscalePlan]) -> bool {
    plans.first().is_some_and(|first| {
        first.ht_groups.is_empty()
            && !first
                .steps
                .iter()
                .any(|step| matches!(step, PreparedDirectGrayscaleStep::HtSubBand(_)))
    })
}

/// Decodes the classic Tier-1 steps of every group in one committed command
/// buffer. Each entry of `groups` lists one group's component plans in the
/// order its stacked batch uses. Returns `None` when the combined output does
/// not fit the kernel's 32-bit coefficient offsets.
pub(super) fn encode_shared_classic_tier1(
    runtime: &Arc<MetalRuntime>,
    groups: &[Vec<&PreparedDirectGrayscalePlan>],
) -> Result<Option<Vec<SharedClassicTier1Group>>, Error> {
    let inputs = collect_shared_tier1_inputs(groups)?;
    if u32::try_from(inputs.total_len).is_err() {
        return Ok(None);
    }
    let SharedTier1Inputs {
        batches,
        owners,
        mut steps,
        total_len,
    } = inputs;

    let output = take_f32_scratch_buffer(runtime, total_len)?;
    let output_buffer = output.buffer.clone();
    let mut scratch_buffers =
        crate::batch_allocation::try_vec(3, "J2K Metal shared classic Tier-1 scratch buffers")?;
    scratch_buffers.push(output);
    let command_buffer = new_command_buffer(&runtime.queue)?;
    label_command_buffer(&command_buffer, "j2k shared classic Tier-1");
    let encoder = new_compute_command_encoder(&command_buffer)?;
    let dispatch = encode_distinct_classic_batches_to_buffer_in_encoder(
        runtime,
        &encoder,
        batches.iter().copied(),
        &output_buffer,
        &mut scratch_buffers,
    );
    encoder.endEncoding();
    let (retained_buffers, status) = dispatch?;
    let DirectStatusCheck::Classic {
        buffer: status_buffer,
        len: status_len,
        source_indices: Some(batch_indices),
    } = status
    else {
        return Err(Error::MetalStateInvariant {
            state: "J2K Metal shared classic Tier-1",
            reason: "shared dispatch did not attribute statuses to batches",
        });
    };
    attach_group_statuses(&batch_indices, &owners, &mut steps)?;
    command_buffer.commit();
    #[cfg(test)]
    super::test_counters::record_shared_classic_tier1_pass();

    let pass = Rc::new(SharedClassicTier1Pass {
        runtime: runtime.clone(),
        command_buffer,
        retained_buffers,
        scratch_buffers,
    });
    let mut shared =
        crate::batch_allocation::try_vec(groups.len(), "J2K Metal shared classic Tier-1 groups")?;
    shared.extend(steps.into_iter().map(|steps| SharedClassicTier1Group {
        pass: pass.clone(),
        output: output_buffer.clone(),
        status_buffer: status_buffer.clone(),
        status_len,
        steps,
    }));
    Ok(Some(shared))
}

fn collect_shared_tier1_inputs<'p>(
    groups: &[Vec<&'p PreparedDirectGrayscalePlan>],
) -> Result<SharedTier1Inputs<'p>, Error> {
    let mut budget =
        crate::batch_allocation::BatchMetadataBudget::new("J2K Metal shared classic Tier-1");
    let batch_capacity = crate::batch_allocation::checked_count_sum(
        groups.iter().map(|plans| {
            plans
                .len()
                .saturating_mul(plans.first().map_or(0, |first| first.steps.len()))
        }),
        "J2K Metal shared classic Tier-1 batches",
    )?;
    let mut inputs = SharedTier1Inputs {
        batches: budget.try_vec(batch_capacity, "J2K Metal shared classic Tier-1 batches")?,
        owners: budget.try_vec(batch_capacity, "J2K Metal shared classic Tier-1 owners")?,
        steps: budget.try_vec(groups.len(), "J2K Metal shared classic Tier-1 group steps")?,
        total_len: 0,
    };
    for (group_index, plans) in groups.iter().enumerate() {
        let first = plans.first().ok_or(Error::MetalStateInvariant {
            state: "J2K Metal shared classic Tier-1",
            reason: "group has no component plans",
        })?;
        inputs
            .steps
            .push(budget.try_vec(first.steps.len(), "J2K Metal shared classic Tier-1 steps")?);
        let mut step_idx = 0;
        while step_idx < first.steps.len() {
            if let Some(group) = first.classic_group_starting_at(step_idx) {
                let per_instance_len = group.total_coefficients;
                let start =
                    inputs.begin_step(group_index, step_idx, per_instance_len, plans.len())?;
                for (instance, plan) in plans.iter().enumerate() {
                    let other = plan.classic_group_starting_at(step_idx).ok_or(
                        Error::MetalStateInvariant {
                            state: "J2K Metal shared classic Tier-1",
                            reason: "stacked plans disagree on a classic group step",
                        },
                    )?;
                    inputs.push_batch(
                        group_index,
                        instance,
                        start,
                        per_instance_len,
                        (
                            &other.coded_data,
                            &other.jobs,
                            &other.segments,
                            other.zero_fill,
                        ),
                    )?;
                }
                step_idx = group.end_step;
                continue;
            }
            if let PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) = &first.steps[step_idx] {
                let per_instance_len = checked_coefficient_len(
                    sub_band.width,
                    sub_band.height,
                    "J2K Metal shared classic Tier-1 sub-band size overflow",
                )?;
                let start =
                    inputs.begin_step(group_index, step_idx, per_instance_len, plans.len())?;
                for (instance, plan) in plans.iter().enumerate() {
                    let Some(PreparedDirectGrayscaleStep::ClassicSubBand(other)) =
                        plan.steps.get(step_idx)
                    else {
                        return Err(Error::MetalStateInvariant {
                            state: "J2K Metal shared classic Tier-1",
                            reason: "stacked plans disagree on a classic sub-band step",
                        });
                    };
                    inputs.push_batch(
                        group_index,
                        instance,
                        start,
                        per_instance_len,
                        (
                            &other.coded_data,
                            &other.jobs,
                            &other.segments,
                            other.zero_fill,
                        ),
                    )?;
                }
            }
            step_idx += 1;
        }
    }
    Ok(inputs)
}

impl<'p> SharedTier1Inputs<'p> {
    /// Reserves `instance_count` consecutive coefficient planes for one step
    /// and returns the element offset of the first.
    fn begin_step(
        &mut self,
        group: usize,
        step_idx: usize,
        per_instance_len: usize,
        instance_count: usize,
    ) -> Result<usize, Error> {
        let start = self.total_len;
        let step_len = crate::batch_allocation::checked_count_product(
            per_instance_len,
            instance_count,
            "J2K Metal shared classic Tier-1 step coefficients",
        )?;
        self.total_len = crate::batch_allocation::checked_count_sum(
            [start, step_len],
            "J2K Metal shared classic Tier-1 coefficients",
        )?;
        let offset_bytes = crate::batch_allocation::checked_count_product(
            start,
            size_of::<f32>(),
            "J2K Metal shared classic Tier-1 step offset",
        )?;
        self.steps[group].push(SharedClassicTier1Step {
            step_idx,
            per_instance_len,
            instance_count,
            offset_bytes,
            status_indices: Vec::new(),
            instances: Vec::new(),
            consumed: false,
        });
        Ok(start)
    }

    fn push_batch(
        &mut self,
        group: usize,
        instance: usize,
        step_start: usize,
        per_instance_len: usize,
        (coded_data, jobs, segments, zero_fill): (
            &'p [u8],
            &'p [super::abi::J2kClassicCleanupBatchJob],
            &'p [super::abi::J2kClassicSegment],
            bool,
        ),
    ) -> Result<(), Error> {
        let output_base = crate::batch_allocation::checked_count_product(
            instance,
            per_instance_len,
            "J2K Metal shared classic Tier-1 instance offset",
        )
        .and_then(|offset| {
            crate::batch_allocation::checked_count_sum(
                [step_start, offset],
                "J2K Metal shared classic Tier-1 instance offset",
            )
        })?;
        self.batches.push(DistinctClassicBatch {
            coded_data,
            jobs,
            segments,
            output_base,
            output_len: per_instance_len,
            zero_fill,
        });
        self.owners.push(BatchOwner {
            group,
            step: self.steps[group].len() - 1,
            instance,
        });
        Ok(())
    }
}

/// Records each job's status position on its group step. Jobs keep their
/// instance index within the group's stack, as the group's own dispatch
/// would report it, so the group's source remapping still applies.
fn attach_group_statuses(
    batch_indices: &[usize],
    owners: &[BatchOwner],
    steps: &mut [Vec<SharedClassicTier1Step>],
) -> Result<(), Error> {
    for (status_index, &batch_index) in batch_indices.iter().enumerate() {
        let owner = owners.get(batch_index).ok_or(Error::MetalStateInvariant {
            state: "J2K Metal shared classic Tier-1",
            reason: "status names a batch outside the shared dispatch",
        })?;
        let step = &mut steps[owner.group][owner.step];
        crate::batch_allocation::try_reserve_for_push(
            &mut step.status_indices,
            "J2K Metal shared classic status indices",
        )?;
        crate::batch_allocation::try_reserve_for_push(
            &mut step.instances,
            "J2K Metal shared classic status sources",
        )?;
        step.status_indices.push(status_index);
        step.instances.push(owner.instance);
    }
    Ok(())
}
