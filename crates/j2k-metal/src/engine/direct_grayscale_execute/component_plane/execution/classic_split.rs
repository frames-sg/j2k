// SPDX-License-Identifier: MIT OR Apache-2.0

//! Classic Tier-1 of a single-image colour decode, shared by the GPU and the
//! CPU pool, which decode code blocks at comparable rates. Each component's
//! finest sub-bands go to the pool, whole, up to the runtime's CPU share of
//! the coded bytes; the GPU decodes the other jobs in a command buffer
//! committed first. Both write one shared buffer: every component's GPU
//! coefficients, then every component's CPU coefficients.
//!
//! The share does not change the outcome: the plan was checked against the
//! GPU's limits before execution, and a CPU decode error leaves the GPU to
//! decode every job and report its own error. Other work on rayon's global
//! pool delays the CPU part (and lowers the share).

#[cfg(target_os = "macos")]
use crate::metal_types::prelude::*;

use std::time::{Duration, Instant};

use rayon::prelude::*;

use super::DecodedFirstGroup;
use crate::engine::{
    abi::J2kClassicCleanupBatchJob,
    decode_dispatch::{encode_distinct_classic_batches_to_buffer_in_encoder, DistinctClassicBatch},
    direct_cpu::decode_classic_jobs_on_pool,
    direct_grayscale_execute::extend_preallocated_retained_buffers,
    direct_plan_types::{PreparedClassicSubBandGroup, PreparedDirectGrayscalePlan},
    direct_roi::checked_f32_span,
    direct_scratch::DirectScratchBuffer,
    direct_stacked_batch::push_member_band_slices,
    direct_status::{validate_direct_status_contents, DirectStatusCheck},
    gpu_timing::completed_command_buffer_gpu_duration,
    new_command_buffer, new_compute_command_encoder, wait_for_completion_metal, MetalRuntime,
};
use crate::metal_types::Buffer;
use crate::profile_env::label_command_buffer;
use crate::{batch_allocation::try_vec, error::metal_kernel_support_error, Error};
use j2k_native::J2kWaveletTransform;

/// Decodes each component's first classic group, which covers all of its code
/// blocks, on the GPU and the CPU pool together. `None` when a component has
/// no such group, the caller runs on the pool, [`plan_classic_splits`] finds
/// no split, or the CPU part fails.
pub(in crate::engine) fn encode_split_classic_first_groups(
    runtime: &MetalRuntime,
    components: &[PreparedDirectGrayscalePlan],
    transform: J2kWaveletTransform,
    retained_buffers: &mut Vec<Buffer>,
    status_checks: &mut Vec<DirectStatusCheck>,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<Option<Vec<DecodedFirstGroup>>, Error> {
    let Some(groups) = components
        .iter()
        .map(|plan| plan.classic_group_starting_at(0))
        .collect::<Option<Vec<_>>>()
    else {
        return Ok(None);
    };
    // On a pool thread, the CPU part would queue behind the caller's own work.
    if rayon::current_num_threads() < 2 || rayon::current_thread_index().is_some() {
        return Ok(None);
    }
    let total: u64 = groups.iter().map(|group| coded_bytes(&group.jobs)).sum();
    let share = runtime.classic_cpu_share(transform);
    let Some(splits) = plan_classic_splits(&groups, share, total)? else {
        return Ok(None);
    };
    let cpu_jobs = groups
        .iter()
        .zip(&splits)
        .map(|(group, split)| &group.jobs[split.0..]);
    let cpu_bytes: u64 = cpu_jobs.map(coded_bytes).sum();
    // Each group's (GPU, CPU) base: the buffer offset of its coefficient 0 in
    // either part.
    let gpu_len: usize = splits.iter().map(|split| split.1).sum();
    let (mut bases, mut gpu_base, mut len) =
        (try_vec(groups.len(), "J2K Metal split")?, 0, gpu_len);
    for (group, &(_, cpu_start)) in groups.iter().zip(&splits) {
        bases.push((gpu_base, len - cpu_start));
        (gpu_base, len) = (
            gpu_base + cpu_start,
            len + group.total_coefficients - cpu_start,
        );
    }
    let bytes = checked_f32_span(len, 1, "J2K Metal split coefficients")?.bytes;
    let output = runtime
        .ht_input_heap
        .allocate_single(&runtime.device, bytes)?;

    let command_buffer = new_command_buffer(&runtime.queue)?;
    label_command_buffer(&command_buffer, "j2k decode split classic Tier-1");
    let encoder = new_compute_command_encoder(&command_buffer)?;
    let batches = (groups.iter().zip(&splits).zip(&bases)).map(|((group, split), base)| {
        DistinctClassicBatch {
            coded_data: &group.coded_data,
            jobs: &group.jobs[..split.0],
            segments: &group.segments,
            output_base: base.0,
            output_len: split.1,
            zero_fill: group.zero_fill,
        }
    });
    let dispatched = encode_distinct_classic_batches_to_buffer_in_encoder(
        runtime,
        &encoder,
        batches,
        &output,
        scratch_buffers,
    );
    encoder.endEncoding();
    let (buffers, status_check) = dispatched?;
    extend_preallocated_retained_buffers(retained_buffers, buffers)?;
    command_buffer.commit();

    let cpu = decode_cpu_share(&groups, &splits, &output, gpu_len, len);
    wait_for_completion_metal(&command_buffer)?;
    let Ok(cpu_time) = cpu else {
        return Ok(None);
    };
    if validate_direct_status_contents(&status_check).is_ok() {
        if let Some(gpu_time) = completed_command_buffer_gpu_duration(&command_buffer) {
            let gpu_bytes = total - cpu_bytes;
            runtime.record_classic_split(transform, (cpu_bytes, cpu_time), (gpu_bytes, gpu_time));
        }
    }
    status_checks.push(status_check);
    crate::batch_allocation::try_reserve_for_push(retained_buffers, "J2K Metal split output")?;
    retained_buffers.push(output.clone());

    let mut decoded = try_vec(groups.len(), "J2K Metal split groups")?;
    for ((group, &(_, cpu_start)), &(gpu_base, cpu_base)) in groups.iter().zip(&splits).zip(&bases)
    {
        let mut bands = try_vec(group.members.len(), "J2K Metal split bands")?;
        push_member_band_slices(
            &mut bands,
            &output,
            group.members.iter().map(|member| {
                let offset = member.offset_elements;
                let base = if offset < cpu_start {
                    gpu_base
                } else {
                    cpu_base
                };
                (member.band_id, base + offset, member.window)
            }),
        )?;
        decoded.push(DecodedFirstGroup {
            bands,
            band_cells: Vec::new(),
            end_step: group.end_step,
        });
    }
    Ok(Some(decoded))
}

/// Each group's first CPU job and coefficient. Round by round, each group
/// gives the CPU its finest remaining member unless that would take the CPU
/// past `share` of `total` bytes; the CPU gets at least the smallest such
/// non-empty member, so its rate is always measured, and each group's first
/// member stays on the GPU. `None` when a group's jobs are not in output
/// order (e.g. precinct-partitioned).
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the byte target is a rough balance"
)]
fn plan_classic_splits(
    groups: &[&PreparedClassicSubBandGroup],
    share: f64,
    total: u64,
) -> Result<Option<Vec<(usize, usize)>>, Error> {
    let mut member_starts = try_vec(groups.len(), "J2K Metal split members")?;
    for group in groups {
        let jobs = &group.jobs;
        // Unordered jobs or members keep the unsplit route.
        if !jobs.is_sorted_by(|a, b| a.output_offset < b.output_offset)
            || !group
                .members
                .is_sorted_by_key(|member| member.offset_elements)
        {
            return Ok(None);
        }
        let mut starts = try_vec(group.members.len() + 1, "J2K Metal split member starts")?;
        starts.extend(group.members.iter().map(|member| {
            jobs.partition_point(|job| (job.output_offset as usize) < member.offset_elements)
        }));
        starts.push(jobs.len());
        member_starts.push(starts);
    }
    let member_bytes = |index: usize, member: usize| {
        let starts = &member_starts[index];
        coded_bytes(&groups[index].jobs[starts[member]..starts[member + 1]])
    };
    let target = (share * total as f64) as u64;
    let mut firsts = try_vec(groups.len(), "J2K Metal split firsts")?;
    firsts.extend(groups.iter().map(|group| group.members.len()));
    let (mut cpu_bytes, mut grew) = (0, true);
    while grew {
        grew = false;
        for (index, first) in firsts.iter_mut().enumerate() {
            let next = if *first > 1 {
                member_bytes(index, *first - 1)
            } else {
                continue;
            };
            if cpu_bytes + next / 2 < target {
                (cpu_bytes, *first, grew) = (cpu_bytes + next, *first - 1, true);
            }
        }
    }
    if cpu_bytes == 0 {
        let smallest = (0..groups.len())
            .filter(|&index| firsts[index] > 1)
            .map(|index| (member_bytes(index, firsts[index] - 1), index))
            .filter(|&(bytes, _)| bytes > 0)
            .min();
        let Some((_, index)) = smallest else {
            return Ok(None);
        };
        firsts[index] -= 1;
    }
    let mut splits = try_vec(groups.len(), "J2K Metal split points")?;
    for ((group, starts), &first) in groups.iter().zip(&member_starts).zip(&firsts) {
        let member = group.members.get(first);
        let start = member.map_or(group.total_coefficients, |member| member.offset_elements);
        splits.push((starts[first], start));
    }
    Ok(Some(splits))
}

fn coded_bytes(jobs: &[J2kClassicCleanupBatchJob]) -> u64 {
    jobs.iter().map(|job| u64::from(job.coded_len)).sum()
}

/// Decodes the CPU's jobs into `output` from `gpu_len` to `len` elements:
/// each group's coefficients from its first CPU one on, in group order,
/// zeroed first where the group needs it. Returns the decode time without the
/// zeroing.
fn decode_cpu_share(
    groups: &[&PreparedClassicSubBandGroup],
    splits: &[(usize, usize)],
    output: &Buffer,
    gpu_len: usize,
    len: usize,
) -> Result<Duration, Error> {
    let offset = checked_f32_span(gpu_len, 1, "J2K Metal split CPU coefficients")?.bytes;
    // SAFETY: the split's command buffer writes only the GPU coefficients
    // before `gpu_len`, and no other code uses `output` until this returns.
    unsafe {
        j2k_metal_support::with_checked_buffer_slice_mut::<f32, _>(
            output,
            offset,
            len - gpu_len,
            |values| {
                let (mut outputs, mut rest) =
                    (try_vec(groups.len(), "J2K Metal split CPU")?, values);
                for (group, &(cpu_job, start)) in groups.iter().zip(splits) {
                    let (output, tail) = rest.split_at_mut(group.total_coefficients - start);
                    rest = tail;
                    if group.zero_fill {
                        output
                            .par_chunks_mut(1 << 16)
                            .for_each(|chunk| chunk.fill(0.0));
                    }
                    if !output.is_empty() {
                        outputs.push((*group, cpu_job, start, output));
                    }
                }
                let started = Instant::now();
                decode_classic_jobs_on_pool(outputs)?;
                Ok::<_, Error>(started.elapsed())
            },
        )
    }
    .map_err(|source| metal_kernel_support_error("J2K Metal split CPU output", source))?
}
