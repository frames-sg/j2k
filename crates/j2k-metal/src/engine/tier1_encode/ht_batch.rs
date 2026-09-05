// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host-staged Metal HT Tier-1 batches and candidate-set projection.

use j2k_metal_support::dispatch_1d_pipeline;

use crate::metal_types::prelude::*;
use crate::profile_env::{label_command_buffer, label_compute_encoder};

use super::super::abi::{J2kHtEncodeBatchJob, J2kHtEncodeStatus};
use super::super::{
    checked_buffer_slice, commit_and_wait_metal, copied_slice_buffer, ht_encode_output_capacity,
    new_command_buffer, new_compute_command_encoder, new_shared_buffer, with_runtime,
    zeroed_shared_buffer, EncodedHtJ2kCodeBlock, EncodedHtJ2kCodeBlockSet, Error,
    J2kHtCodeBlockEncodeJob, J2kHtCodeBlockSetEncodeJob, MetalRuntime,
};
use super::{checked_type_buffer_bytes, read_ht_encoded_code_block};

#[derive(Clone, Copy)]
struct MetalHtCodeBlockJob<'a> {
    coefficients: &'a [i32],
    width: u32,
    height: u32,
    total_bitplanes: u8,
    cleanup_bitplane: u8,
    target_coding_passes: u8,
}

trait MetalHtCodeBlockJobSource {
    fn metal_job(&self) -> MetalHtCodeBlockJob<'_>;
}

impl MetalHtCodeBlockJobSource for J2kHtCodeBlockEncodeJob<'_> {
    fn metal_job(&self) -> MetalHtCodeBlockJob<'_> {
        MetalHtCodeBlockJob {
            coefficients: self.coefficients,
            width: self.width,
            height: self.height,
            total_bitplanes: self.total_bitplanes,
            cleanup_bitplane: 0,
            target_coding_passes: self.target_coding_passes,
        }
    }
}

impl MetalHtCodeBlockJobSource for J2kHtCodeBlockSetEncodeJob<'_> {
    fn metal_job(&self) -> MetalHtCodeBlockJob<'_> {
        MetalHtCodeBlockJob {
            coefficients: self.coefficients,
            width: self.width,
            height: self.height,
            total_bitplanes: self.total_bitplanes,
            cleanup_bitplane: self.cleanup_bitplane,
            target_coding_passes: self.target_coding_passes,
        }
    }
}

pub(crate) fn encode_ht_cleanup_code_blocks(
    jobs: &[J2kHtCodeBlockEncodeJob<'_>],
) -> Result<Vec<EncodedHtJ2kCodeBlock>, Error> {
    with_runtime(|runtime| encode_ht_cleanup_code_blocks_with_runtime(runtime, jobs))
}

fn encode_ht_cleanup_code_blocks_with_runtime(
    runtime: &MetalRuntime,
    jobs: &[J2kHtCodeBlockEncodeJob<'_>],
) -> Result<Vec<EncodedHtJ2kCodeBlock>, Error> {
    let blocks = encode_ht_cleanup_code_blocks_with_runtime_and_statuses(runtime, jobs)?;
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "HTJ2K Metal encoded block projection metadata",
    );
    budget.account_capacity::<(EncodedHtJ2kCodeBlock, J2kHtEncodeStatus)>(blocks.capacity())?;
    budget.preflight(&[crate::batch_allocation::BatchMetadataRequest::of::<
        EncodedHtJ2kCodeBlock,
    >(blocks.len())])?;
    let mut encoded = budget.try_vec(blocks.len(), "HTJ2K Metal encoded block results")?;
    for (block, _status) in blocks {
        encoded.push(block);
    }
    Ok(encoded)
}

fn encode_ht_cleanup_code_blocks_with_runtime_and_statuses(
    runtime: &MetalRuntime,
    jobs: &[J2kHtCodeBlockEncodeJob<'_>],
) -> Result<Vec<(EncodedHtJ2kCodeBlock, J2kHtEncodeStatus)>, Error> {
    if jobs.is_empty() {
        return Ok(Vec::new());
    }
    if jobs.iter().any(|job| job.target_coding_passes != 1) {
        return Err(Error::MetalKernel {
            message: "HTJ2K Metal cleanup encode supports one coding pass".to_string(),
        });
    }

    encode_ht_code_block_jobs_with_runtime_and_statuses(runtime, jobs)
}

fn encode_ht_code_block_jobs_with_runtime_and_statuses<J: MetalHtCodeBlockJobSource>(
    runtime: &MetalRuntime,
    jobs: &[J],
) -> Result<Vec<(EncodedHtJ2kCodeBlock, J2kHtEncodeStatus)>, Error> {
    if jobs.is_empty() {
        return Ok(Vec::new());
    }

    let mut budget =
        crate::batch_allocation::BatchMetadataBudget::new("HTJ2K Metal Tier-1 encode batch");
    let HtBatchStaging {
        coefficients,
        batch_jobs,
        output_capacity_total,
    } = prepare_ht_batch(jobs, &mut budget)?;

    let coefficient_buffer = copied_slice_buffer(&runtime.device, &coefficients)?;
    let job_buffer = copied_slice_buffer(&runtime.device, &batch_jobs)?;
    let output = new_shared_buffer(&runtime.device, output_capacity_total.max(1))?;
    let status_buffer = zeroed_shared_buffer(
        &runtime.device,
        checked_type_buffer_bytes::<J2kHtEncodeStatus>(
            jobs.len(),
            "HTJ2K Metal encode status buffer",
        )?,
    )?;
    let job_count = u32::try_from(batch_jobs.len()).map_err(|_| Error::MetalKernel {
        message: "HTJ2K Metal encode job count exceeds u32".to_string(),
    })?;

    let command_buffer = new_command_buffer(&runtime.queue)?;
    label_command_buffer(&command_buffer, "j2k htj2k tier1 batch");
    let encoder = new_compute_command_encoder(&command_buffer)?;
    label_compute_encoder(&encoder, "HTJ2K Tier-1 encode");
    let pipeline = &runtime.encode()?.ht_encode_code_blocks;
    encoder.setComputePipelineState(pipeline);
    encoder.set_buffer(0, Some(&coefficient_buffer), 0);
    encoder.set_buffer(1, Some(&output), 0);
    encoder.set_buffer(2, Some(&job_buffer), 0);
    encoder.set_buffer(3, Some(&runtime.encode()?.ht_vlc_encode_table0), 0);
    encoder.set_buffer(4, Some(&runtime.encode()?.ht_vlc_encode_table1), 0);
    encoder.set_buffer(5, Some(&runtime.encode()?.ht_uvlc_encode_table), 0);
    encoder.set_buffer(6, Some(&status_buffer), 0);
    encoder.set_bytes::<u32>(7, &job_count);
    dispatch_1d_pipeline(&encoder, pipeline, u64::from(job_count));
    encoder.endEncoding();
    commit_and_wait_metal(&command_buffer)?;

    let statuses = checked_buffer_slice::<J2kHtEncodeStatus>(
        &status_buffer,
        jobs.len(),
        "HT encode statuses",
    )?;
    let mut results = budget.try_vec(jobs.len(), "J2K Metal HT Tier-1 encoded blocks")?;
    for (index, status) in statuses.iter().copied().enumerate() {
        let batch_job = batch_jobs[index];
        let encoded_block = read_ht_encoded_code_block(
            status,
            &output,
            usize::try_from(batch_job.output_offset).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal encode output offset exceeds usize".to_string(),
            })?,
            usize::try_from(batch_job.output_capacity).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal encode output capacity exceeds usize".to_string(),
            })?,
        )?;
        results.push((encoded_block, status));
    }

    Ok(results)
}

struct HtBatchStaging {
    coefficients: Vec<i32>,
    batch_jobs: Vec<J2kHtEncodeBatchJob>,
    output_capacity_total: usize,
}

fn prepare_ht_batch<J: MetalHtCodeBlockJobSource>(
    jobs: &[J],
    budget: &mut crate::batch_allocation::BatchMetadataBudget,
) -> Result<HtBatchStaging, Error> {
    let coefficient_count = jobs.iter().try_fold(0usize, |total, source| {
        total
            .checked_add(ht_job_coefficient_count(source.metal_job())?)
            .ok_or_else(|| {
                Error::from(j2k_core::BatchInfrastructureError::AllocationTooLarge {
                    what: "HTJ2K Metal encode coefficients",
                    requested: usize::MAX,
                    cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                })
            })
    })?;
    let mut coefficients = budget.try_vec(coefficient_count, "HTJ2K Metal encode coefficients")?;
    let mut batch_jobs = budget.try_vec(jobs.len(), "HTJ2K Metal encode batch jobs")?;
    let mut output_capacity_total = 0usize;

    for source in jobs {
        let job = source.metal_job();
        let expected_coefficients = ht_job_coefficient_count(job)?;
        if job.coefficients.len() < expected_coefficients {
            return Err(Error::MetalKernel {
                message: "HTJ2K Metal encode coefficient slice is too small".to_string(),
            });
        }
        let output_capacity = ht_encode_output_capacity(job.width, job.height)?;
        let coefficient_offset = checked_u32(
            coefficients.len(),
            "HTJ2K Metal encode coefficient table exceeds u32",
        )?;
        let output_offset = checked_u32(
            output_capacity_total,
            "HTJ2K Metal encode output table exceeds u32",
        )?;
        batch_jobs.push(J2kHtEncodeBatchJob {
            coefficient_offset,
            output_offset,
            width: job.width,
            height: job.height,
            total_bitplanes: u32::from(job.total_bitplanes),
            cleanup_bitplane: u32::from(job.cleanup_bitplane),
            target_coding_passes: u32::from(job.target_coding_passes),
            output_capacity: checked_u32(
                output_capacity,
                "HTJ2K Metal encode output capacity exceeds u32",
            )?,
        });
        coefficients.extend_from_slice(&job.coefficients[..expected_coefficients]);
        output_capacity_total = output_capacity_total
            .checked_add(output_capacity)
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K Metal encode output buffer overflow".to_string(),
            })?;
    }
    Ok(HtBatchStaging {
        coefficients,
        batch_jobs,
        output_capacity_total,
    })
}

fn ht_job_coefficient_count(job: MetalHtCodeBlockJob<'_>) -> Result<usize, Error> {
    usize::try_from(job.width)
        .ok()
        .and_then(|width| {
            usize::try_from(job.height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .ok_or_else(|| Error::MetalKernel {
            message: "HTJ2K Metal encode coefficient count overflow".to_string(),
        })
}

fn checked_u32(value: usize, message: &'static str) -> Result<u32, Error> {
    u32::try_from(value).map_err(|_| Error::MetalKernel {
        message: message.to_string(),
    })
}

pub(crate) fn encode_ht_code_block_sets(
    jobs: &[J2kHtCodeBlockSetEncodeJob<'_>],
) -> Result<Vec<EncodedHtJ2kCodeBlockSet>, Error> {
    with_runtime(|runtime| {
        let blocks = encode_ht_code_block_jobs_with_runtime_and_statuses(runtime, jobs)?;
        let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
            "HTJ2K Metal candidate-set projection",
        );
        budget.account_capacity::<(EncodedHtJ2kCodeBlock, J2kHtEncodeStatus)>(blocks.capacity())?;
        let mut output = budget.try_vec(blocks.len(), "HTJ2K Metal candidate-set results")?;
        for (block, status) in blocks {
            output.push(EncodedHtJ2kCodeBlockSet {
                data: block.data,
                cleanup_length: block.cleanup_length,
                sigprop_length: status.sigprop_length,
                magref_length: status.magref_length,
                num_coding_passes: block.num_coding_passes,
                num_zero_bitplanes: block.num_zero_bitplanes,
            });
        }
        Ok(output)
    })
}
