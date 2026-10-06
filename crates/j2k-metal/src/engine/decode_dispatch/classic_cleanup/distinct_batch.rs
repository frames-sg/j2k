// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use crate::metal_types::prelude::*;

use super::super::ht_subband::dispatch_zero_u32_buffer_in_encoder;
use super::{
    classic_batch_uses_plain_fast_path, dispatch_classic_cleanup_batched_in_encoder,
    dispatch_classic_cleanup_plain_dense_batched_in_encoder,
    distinct_allocation::{allocate_distinct_classic_metadata, DistinctClassicMetadata},
    ClassicCleanupBatchDispatch,
};
use crate::engine::abi::{J2kClassicCleanupBatchJob, J2kClassicSegment};
use crate::engine::direct_plan_validation::classic_prepared_job_supports_runtime;
use crate::engine::{
    copied_slice_buffer, new_compute_command_encoder, new_shared_buffer,
    take_classic_coefficients_scratch_buffer, take_classic_states_scratch_buffer, Buffer,
    CommandBufferRef, ComputeCommandEncoderRef, DirectScratchBuffer, DirectStatusCheck, Error,
    MetalRuntime, PreparedClassicSubBand, PreparedClassicSubBandGroup,
};

const CLASSIC_PLAIN_DENSE_MIN_JOBS: usize = 1024;

pub(in crate::engine) fn encode_distinct_classic_sub_bands_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    sub_bands: &[&PreparedClassicSubBand],
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let encoder = new_compute_command_encoder(command_buffer)?;
    let result = encode_distinct_classic_sub_bands_to_buffer_in_encoder(
        runtime,
        &encoder,
        sub_bands,
        output,
        scratch_buffers,
    );
    encoder.endEncoding();
    result
}

pub(in crate::engine) fn encode_distinct_classic_sub_bands_to_buffer_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    sub_bands: &[&PreparedClassicSubBand],
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let Some(first) = sub_bands.first() else {
        let empty = new_shared_buffer(&runtime.device, 1)?;
        return Ok((
            crate::batch_allocation::try_vec_from_array(
                [empty.clone()],
                "J2K Metal empty classic retained buffer",
            )?,
            DirectStatusCheck::Classic {
                buffer: empty,
                len: 0,
                source_indices: Some(Vec::new()),
            },
        ));
    };
    let per_instance_len = first.width as usize * first.height as usize;
    encode_distinct_classic_batches_to_buffer_in_encoder(
        runtime,
        encoder,
        sub_bands
            .iter()
            .enumerate()
            .map(|(index, sub_band)| DistinctClassicBatch {
                coded_data: &sub_band.coded_data,
                jobs: &sub_band.jobs,
                segments: &sub_band.segments,
                output_base: index * per_instance_len,
                output_len: per_instance_len,
                zero_fill: sub_band.zero_fill,
            }),
        output,
        scratch_buffers,
    )
}

pub(in crate::engine) fn encode_distinct_classic_sub_band_groups_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    groups: &[&PreparedClassicSubBandGroup],
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let encoder = new_compute_command_encoder(command_buffer)?;
    let result = encode_distinct_classic_sub_band_groups_to_buffer_in_encoder(
        runtime,
        &encoder,
        groups,
        output,
        scratch_buffers,
    );
    encoder.endEncoding();
    result
}

pub(in crate::engine) fn encode_distinct_classic_sub_band_groups_to_buffer_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    groups: &[&PreparedClassicSubBandGroup],
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let Some(first) = groups.first() else {
        let empty = new_shared_buffer(&runtime.device, 1)?;
        return Ok((
            crate::batch_allocation::try_vec_from_array(
                [empty.clone()],
                "J2K Metal empty classic retained buffer",
            )?,
            DirectStatusCheck::Classic {
                buffer: empty,
                len: 0,
                source_indices: Some(Vec::new()),
            },
        ));
    };
    let per_instance_len = first.total_coefficients;
    encode_distinct_classic_batches_to_buffer_in_encoder(
        runtime,
        encoder,
        groups
            .iter()
            .enumerate()
            .map(|(index, group)| DistinctClassicBatch {
                coded_data: &group.coded_data,
                jobs: &group.jobs,
                segments: &group.segments,
                output_base: index * per_instance_len,
                output_len: per_instance_len,
                zero_fill: group.zero_fill,
            }),
        output,
        scratch_buffers,
    )
}

/// Code blocks of one sub-band or sub-band group whose coefficients start at
/// `output_base` elements in the dispatch output.
#[derive(Clone, Copy)]
pub(in crate::engine) struct DistinctClassicBatch<'a> {
    pub(in crate::engine) coded_data: &'a [u8],
    pub(in crate::engine) jobs: &'a [J2kClassicCleanupBatchJob],
    pub(in crate::engine) segments: &'a [J2kClassicSegment],
    pub(in crate::engine) output_base: usize,
    pub(in crate::engine) output_len: usize,
    pub(in crate::engine) zero_fill: bool,
}

fn append_distinct_classic_batch(
    metadata: &mut DistinctClassicMetadata,
    source_index: usize,
    batch: DistinctClassicBatch<'_>,
) -> Result<(), Error> {
    let coded_base = u32::try_from(metadata.coded_data.len()).map_err(|_| Error::MetalKernel {
        message: "classic J2K MetalDirect distinct color coded payload exceeds u32".to_string(),
    })?;
    let segment_base = u32::try_from(metadata.segments.len()).map_err(|_| Error::MetalKernel {
        message: "classic J2K MetalDirect distinct color segment table exceeds u32".to_string(),
    })?;
    metadata.coded_data.extend_from_slice(batch.coded_data);
    for segment in batch.segments {
        let mut adjusted = *segment;
        adjusted.data_offset = adjusted
            .data_offset
            .checked_add(coded_base)
            .ok_or_else(|| Error::MetalKernel {
                message: "classic J2K MetalDirect distinct color segment offset overflow"
                    .to_string(),
            })?;
        metadata.segments.push(adjusted);
    }
    let output_base = u32::try_from(batch.output_base).map_err(|_| Error::MetalKernel {
        message: "classic J2K MetalDirect distinct color output offset exceeds u32".to_string(),
    })?;
    for job in batch.jobs {
        let mut adjusted = *job;
        adjusted.coded_offset = adjusted
            .coded_offset
            .checked_add(coded_base)
            .ok_or_else(|| Error::MetalKernel {
                message: "classic J2K MetalDirect distinct color job coded offset overflow"
                    .to_string(),
            })?;
        adjusted.segment_offset = adjusted
            .segment_offset
            .checked_add(segment_base)
            .ok_or_else(|| Error::MetalKernel {
                message: "classic J2K MetalDirect distinct color job segment offset overflow"
                    .to_string(),
            })?;
        adjusted.output_offset =
            adjusted
                .output_offset
                .checked_add(output_base)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K MetalDirect distinct color job output offset overflow"
                        .to_string(),
                })?;
        metadata.jobs.push(adjusted);
        metadata.source_indices.push(source_index);
    }
    Ok(())
}

// Each dense lane runs one serial MQ stream, and lanes sharing a SIMD group
// diverge, so a group lasts roughly as long as its heaviest job times a
// slowdown that grows with the number of jobs in the group. A kernel dispatch
// lasts as long as its slowest group. Packing 32 heavy jobs per group therefore
// stretches the whole dispatch, while one job per group wastes throughput.
//
// Slowdown x100 for (jobs per SIMD group), measured on an M4 Pro.
const DENSE_GROUP_SLOWDOWN: [(u32, u64); 6] =
    [(32, 435), (16, 350), (8, 260), (4, 193), (2, 137), (1, 100)];
// Upper bound on concurrent SIMD groups implied by a packing, expressed in
// units of the target duration. Calibrated on a 16-core M4 Pro GPU; above it
// the GPU is throughput-bound and wider groups win.
const DENSE_OCCUPANCY_LIMIT: u128 = 360;

/// Estimated decode work for one code block, in arbitrary units. Fitted to
/// single-thread decode time (r = 0.985 on lossless 5/3 photographs): coded
/// bytes dominate, plus one unit per coefficient visit per coding pass.
fn classic_dense_job_cost(job: &J2kClassicCleanupBatchJob) -> u64 {
    634 * u64::from(job.coded_len)
        + 4 * u64::from(job.width) * u64::from(job.height) * u64::from(job.number_of_coding_passes)
        + 113_860
}

/// Number of jobs per SIMD group and their summed `slowdown * heaviest cost`
/// when every job may stretch to at most `target_x100 / 100` times the
/// heaviest job's cost. Costs must be sorted in decreasing order.
fn dense_lane_groups(costs: &[u64], target_x100: u64, mut emit: impl FnMut(usize)) -> u128 {
    let target = u128::from(target_x100) * u128::from(costs.first().copied().unwrap_or(0));
    let mut occupancy = 0u128;
    let mut start = 0;
    while start < costs.len() {
        let head = u128::from(costs[start]);
        let (width, slowdown) = DENSE_GROUP_SLOWDOWN
            .iter()
            .copied()
            .find(|&(_, slowdown)| u128::from(slowdown) * head <= target)
            .unwrap_or((1, 100));
        let count = (width as usize).min(costs.len() - start);
        occupancy += u128::from(slowdown) * head;
        emit(count);
        start += count;
    }
    occupancy
}

/// Orders jobs heaviest first and splits them into SIMD groups of 1..=32 jobs,
/// returning `(first job, job count)` per group. The tightest duration target
/// whose implied occupancy fits the GPU is chosen; when none fits, every group holds 32 jobs as before.
fn pack_distinct_classic_jobs_for_dense_dispatch(
    jobs: &mut [J2kClassicCleanupBatchJob],
    source_indices: &mut [usize],
) -> Result<Vec<[u32; 2]>, Error> {
    if jobs.len() != source_indices.len() {
        return Err(Error::MetalStateInvariant {
            state: "classic dense SoA job ordering",
            reason: "job and status-source counts differ",
        });
    }
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "classic J2K MetalDirect dense SoA ordering",
    );
    let mut paired: Vec<(u64, J2kClassicCleanupBatchJob, usize)> = budget.try_vec(
        jobs.len(),
        "classic J2K MetalDirect dense SoA job/source pairs",
    )?;
    paired.extend(
        jobs.iter()
            .zip(source_indices.iter())
            .map(|(job, &source)| (classic_dense_job_cost(job), *job, source)),
    );
    // Stable on equal cost keeps the order deterministic.
    paired.sort_by_key(|(cost, _, _)| std::cmp::Reverse(*cost));
    let mut costs = budget.try_vec(paired.len(), "classic J2K MetalDirect dense job costs")?;
    costs.extend(paired.iter().map(|(cost, _, _)| *cost));
    let max_cost = u128::from(costs.first().copied().unwrap_or(0));
    let fits = |target: u64| {
        dense_lane_groups(&costs, target, |_| {})
            <= DENSE_OCCUPANCY_LIMIT * u128::from(target) * max_cost
    };
    let target_x100 = (100..=440)
        .step_by(20)
        .find(|&target| fits(target))
        .unwrap_or(440);
    // A group holds at least one job.
    let mut groups = budget.try_vec(paired.len(), "classic J2K MetalDirect dense SIMD groups")?;
    let mut start = 0usize;
    let mut overflow = false;
    dense_lane_groups(&costs, target_x100, |count| {
        match (u32::try_from(start), u32::try_from(count)) {
            (Ok(first), Ok(count)) => groups.push([first, count]),
            _ => overflow = true,
        }
        start += count;
    });
    if overflow {
        return Err(Error::MetalKernel {
            message: "classic J2K MetalDirect dense job index exceeds u32".to_string(),
        });
    }
    for ((job_slot, source_slot), (_, job, source_index)) in
        jobs.iter_mut().zip(source_indices.iter_mut()).zip(paired)
    {
        *job_slot = job;
        *source_slot = source_index;
    }
    Ok(groups)
}

/// Packs `jobs` for the dense kernel and uploads the SIMD-group table.
fn dense_lane_group_buffer(
    runtime: &MetalRuntime,
    jobs: &mut [J2kClassicCleanupBatchJob],
    source_indices: &mut [usize],
) -> Result<(Buffer, usize), Error> {
    let groups = pack_distinct_classic_jobs_for_dense_dispatch(jobs, source_indices)?;
    Ok((copied_slice_buffer(&runtime.device, &groups)?, groups.len()))
}

/// Decodes every batch in one Tier-1 dispatch. The status check attributes each
/// job to the index of its batch.
pub(in crate::engine) fn encode_distinct_classic_batches_to_buffer_in_encoder<'a>(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    batches: impl Iterator<Item = DistinctClassicBatch<'a>> + Clone,
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let zero_fill_word_count = batches.clone().try_fold(0usize, |word_count, batch| {
        if !batch.zero_fill && !batch.jobs.is_empty() {
            return Ok(word_count);
        }
        batch
            .output_base
            .checked_add(batch.output_len)
            .map(|end| word_count.max(end))
            .ok_or_else(|| Error::MetalKernel {
                message: "classic J2K MetalDirect distinct output span overflow".to_string(),
            })
    })?;
    let coded_len = crate::batch_allocation::checked_count_sum(
        batches.clone().map(|batch| batch.coded_data.len()),
        "classic J2K MetalDirect distinct color coded payload",
    )?;
    let job_count = crate::batch_allocation::checked_count_sum(
        batches.clone().map(|batch| batch.jobs.len()),
        "classic J2K MetalDirect distinct color jobs",
    )?;
    let segment_count = crate::batch_allocation::checked_count_sum(
        batches.clone().map(|batch| batch.segments.len()),
        "classic J2K MetalDirect distinct color segments",
    )?;
    let mut metadata = allocate_distinct_classic_metadata(
        coded_len,
        job_count,
        segment_count,
        crate::batch_allocation::BatchMetadataBudget::new(
            "classic J2K MetalDirect distinct color submission",
        ),
    )?;

    for (source_index, batch) in batches.enumerate() {
        append_distinct_classic_batch(&mut metadata, source_index, batch)?;
    }
    let DistinctClassicMetadata {
        coded_data,
        mut jobs,
        segments,
        mut source_indices,
    } = metadata;

    dispatch_zero_u32_buffer_in_encoder(runtime, encoder, output, zero_fill_word_count)?;

    if jobs.is_empty() {
        let empty = new_shared_buffer(&runtime.device, 1)?;
        return Ok((
            crate::batch_allocation::try_vec_from_array(
                [empty.clone()],
                "J2K Metal empty classic retained buffer",
            )?,
            DirectStatusCheck::Classic {
                buffer: empty,
                len: 0,
                source_indices: Some(Vec::new()),
            },
        ));
    }
    if zero_fill_word_count != 0 {
        encoder.memory_barrier_with_resources(&[output]);
    }

    let use_plain_fast_path = classic_batch_uses_plain_fast_path(&jobs, &segments)
        && runtime
            .decode()?
            .classic_cleanup_plain_batched
            .maxTotalThreadsPerThreadgroup()
            >= 32;
    let use_dense_plain_path = jobs.len() >= CLASSIC_PLAIN_DENSE_MIN_JOBS
        && classic_batch_uses_plain_fast_path(&jobs, &segments)
        && jobs
            .iter()
            .all(|job| classic_prepared_job_supports_runtime(job, &segments))
        && runtime
            .decode()?
            .classic_cleanup_plain_dense_batched
            .maxTotalThreadsPerThreadgroup()
            >= 32
        && runtime
            .decode()?
            .classic_cleanup_plain_dense_batched
            .threadExecutionWidth()
            == 32;
    let lane_groups = if use_dense_plain_path {
        Some(dense_lane_group_buffer(
            runtime,
            &mut jobs,
            &mut source_indices,
        )?)
    } else {
        None
    };
    let coded_buffer = copied_slice_buffer(&runtime.device, &coded_data)?;
    let jobs_buffer = copied_slice_buffer(&runtime.device, &jobs)?;
    let segments_buffer = copied_slice_buffer(&runtime.device, &segments)?;
    let coefficients_scratch = take_classic_coefficients_scratch_buffer(runtime, jobs.len())?;
    let flags_scratch = if use_dense_plain_path {
        Some(take_classic_states_scratch_buffer(runtime, jobs.len())?)
    } else {
        None
    };
    let dispatch = ClassicCleanupBatchDispatch {
        runtime,
        kernels: runtime.decode()?,
        coded_data: &coded_buffer,
        jobs: &jobs_buffer,
        job_count: jobs.len(),
        use_plain_fast_path,
        segments: &segments_buffer,
        decoded: output,
        coefficients_scratch: &coefficients_scratch.buffer,
    };
    let (status_check, states_scratch) = if let (Some(flags_scratch), Some((groups, group_count))) =
        (&flags_scratch, &lane_groups)
    {
        (
            dispatch_classic_cleanup_plain_dense_batched_in_encoder(
                encoder,
                dispatch,
                &flags_scratch.buffer,
                groups,
                *group_count,
                Some(source_indices),
            )?,
            None,
        )
    } else {
        dispatch_classic_cleanup_batched_in_encoder(encoder, dispatch, Some(source_indices))?
    };
    let mut retained_buffers =
        crate::batch_allocation::try_vec(4, "J2K Metal distinct classic retained buffers")?;
    retained_buffers.extend([coded_buffer, jobs_buffer, segments_buffer]);
    retained_buffers.extend(lane_groups.map(|(groups, _)| groups));
    scratch_buffers.push(coefficients_scratch);
    if let Some(flags_scratch) = flags_scratch {
        scratch_buffers.push(flags_scratch);
    }
    if let Some(states_scratch) = states_scratch {
        retained_buffers.push(states_scratch);
    }
    Ok((retained_buffers, status_check))
}

#[cfg(test)]
#[path = "distinct_metadata_tests.rs"]
mod tests;
