// SPDX-License-Identifier: MIT OR Apache-2.0

mod execution;
mod prepared;

#[cfg(test)]
mod tests;

use core::{mem::size_of, num::NonZeroUsize};

use j2k_core::{
    plan_ht_gpu_job_chunks, HtGpuJobChunkLimits, HtGpuJobChunkPlan, HtGpuJobChunkPlanError,
    HtGpuJobChunkRequest, HtGpuJobPassBucket,
};
use j2k_native::HtCodeBlockPayloadRanges;

use std::sync::Arc;

use super::{Error, J2kHtCleanupBatchJob};
use crate::compute::{PreparedHtExecutionOwner, PreparedHtPayloadSource};

pub(in crate::compute) use execution::{
    encode_metal_ht_batches_in_encoder, encode_repeated_metal_ht_batch_in_command_buffer,
};
pub(in crate::compute) use prepared::{prepared_metal_ht_execution, PreparedMetalHtExecutionCache};

const MAX_METAL_HT_JOBS_PER_CHUNK: usize = 16_384;
const MAX_METAL_HT_PAYLOAD_BYTES_PER_CHUNK: usize = 64 * 1024 * 1024;
const MAX_METAL_HT_DESCRIPTOR_BYTES_PER_CHUNK: usize =
    MAX_METAL_HT_JOBS_PER_CHUNK * size_of::<J2kHtCleanupBatchJob>();

#[derive(Clone, Copy)]
pub(in crate::compute) struct HtBatchInput<'a> {
    pub(in crate::compute) source_index: usize,
    pub(in crate::compute) payload: HtPayloadSource<'a>,
    pub(in crate::compute) jobs: &'a [J2kHtCleanupBatchJob],
    pub(in crate::compute) output_base: usize,
    pub(in crate::compute) execution_owner: &'a Arc<PreparedHtExecutionOwner>,
}

#[derive(Clone, Copy)]
pub(in crate::compute) enum HtPayloadSource<'a> {
    Contiguous(&'a [u8]),
    Referenced {
        input: &'a Arc<[u8]>,
        ranges: &'a [HtCodeBlockPayloadRanges],
    },
}

impl PreparedHtPayloadSource {
    pub(in crate::compute) fn as_ht_payload_source(&self) -> HtPayloadSource<'_> {
        match self {
            Self::Contiguous(data) => HtPayloadSource::Contiguous(data),
            Self::Referenced { input, ranges } => HtPayloadSource::Referenced { input, ranges },
        }
    }
}

#[derive(Clone, Copy)]
struct FlattenedHtJob<'a> {
    source_index: usize,
    payload: FlattenedHtPayload<'a>,
    job: J2kHtCleanupBatchJob,
    output_base: usize,
}

#[derive(Clone, Copy)]
enum FlattenedHtPayload<'a> {
    Contiguous(&'a [u8]),
    Referenced {
        input: &'a [u8],
        ranges: HtCodeBlockPayloadRanges,
    },
}

pub(in crate::compute) struct MetalHtChunkPlan<'a> {
    jobs: Vec<FlattenedHtJob<'a>>,
    plan: HtGpuJobChunkPlan,
}

pub(in crate::compute) struct PackedMetalHtChunk {
    pub(in crate::compute) bucket: HtGpuJobPassBucket,
    pub(in crate::compute) coded_data: Vec<u8>,
    pub(in crate::compute) jobs: Vec<J2kHtCleanupBatchJob>,
    pub(in crate::compute) source_indices: Vec<usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::compute) enum MetalHtPipelineKind {
    CleanupOnly,
    SigProp,
    MagRef,
}

pub(in crate::compute) const fn metal_ht_pipeline_kind_for_bucket(
    bucket: HtGpuJobPassBucket,
) -> MetalHtPipelineKind {
    match bucket {
        HtGpuJobPassBucket::CleanupOnly => MetalHtPipelineKind::CleanupOnly,
        HtGpuJobPassBucket::SigProp => MetalHtPipelineKind::SigProp,
        HtGpuJobPassBucket::MagRef => MetalHtPipelineKind::MagRef,
    }
}

struct PackedMetalHtChunkOwners {
    coded_data: Vec<u8>,
    jobs: Vec<J2kHtCleanupBatchJob>,
    source_indices: Vec<usize>,
}

fn allocate_packed_metal_ht_chunk(
    payload_bytes: usize,
    job_count: usize,
    mut budget: crate::batch_allocation::BatchMetadataBudget,
) -> Result<PackedMetalHtChunkOwners, Error> {
    budget.preflight(&[
        crate::batch_allocation::BatchMetadataRequest::of::<u8>(payload_bytes),
        crate::batch_allocation::BatchMetadataRequest::of::<J2kHtCleanupBatchJob>(job_count),
        crate::batch_allocation::BatchMetadataRequest::of::<usize>(job_count),
    ])?;
    Ok(PackedMetalHtChunkOwners {
        coded_data: budget.try_vec(payload_bytes, "HTJ2K Metal packed chunk payload")?,
        jobs: budget.try_vec(job_count, "HTJ2K Metal packed chunk jobs")?,
        source_indices: budget.try_vec(job_count, "HTJ2K Metal packed chunk source indices")?,
    })
}

pub(in crate::compute) fn default_metal_ht_chunk_limits() -> HtGpuJobChunkLimits {
    HtGpuJobChunkLimits::new(
        NonZeroUsize::new(MAX_METAL_HT_JOBS_PER_CHUNK).unwrap_or(NonZeroUsize::MIN),
        MAX_METAL_HT_PAYLOAD_BYTES_PER_CHUNK,
        MAX_METAL_HT_DESCRIPTOR_BYTES_PER_CHUNK,
    )
}

pub(in crate::compute) fn plan_metal_ht_chunks<'a>(
    batches: &[HtBatchInput<'a>],
    limits: HtGpuJobChunkLimits,
) -> Result<MetalHtChunkPlan<'a>, Error> {
    let job_count = crate::batch_allocation::checked_count_sum(
        batches.iter().map(|batch| batch.jobs.len()),
        "HTJ2K Metal chunk planning jobs",
    )?;
    let mut budget =
        crate::batch_allocation::BatchMetadataBudget::new("HTJ2K Metal chunk planning metadata");
    let mut jobs = budget.try_vec(job_count, "HTJ2K Metal flattened chunk jobs")?;
    let mut requests = budget.try_vec(job_count, "HTJ2K Metal chunk planner requests")?;

    for batch in batches {
        if let HtPayloadSource::Referenced { ranges, .. } = batch.payload {
            if ranges.len() != batch.jobs.len() {
                return Err(Error::MetalStateInvariant {
                    state: "HTJ2K Metal referenced batch input",
                    reason: "payload range count does not match job count",
                });
            }
        }
        for (job_index, job) in batch.jobs.iter().enumerate() {
            let coding_passes =
                u8::try_from(job.number_of_coding_passes).map_err(|_| Error::MetalKernel {
                    message: format!(
                        "HTJ2K Metal source {} job coding-pass count {} exceeds u8",
                        batch.source_index, job.number_of_coding_passes
                    ),
                })?;
            let payload_bytes = usize::try_from(job.coded_len).map_err(|_| Error::MetalKernel {
                message: format!(
                    "HTJ2K Metal source {} job payload length exceeds usize",
                    batch.source_index
                ),
            })?;
            requests.push(HtGpuJobChunkRequest::new(
                batch.source_index,
                coding_passes,
                payload_bytes,
                size_of::<J2kHtCleanupBatchJob>(),
            ));
            let payload = match batch.payload {
                HtPayloadSource::Contiguous(coded_data) => {
                    FlattenedHtPayload::Contiguous(coded_data)
                }
                HtPayloadSource::Referenced { input, ranges } => {
                    let ranges = *ranges.get(job_index).ok_or(Error::MetalStateInvariant {
                        state: "HTJ2K Metal referenced batch input",
                        reason: "payload range index is outside the retained table",
                    })?;
                    validate_referenced_ht_payload(batch.source_index, input, ranges, job)?;
                    FlattenedHtPayload::Referenced {
                        input: input.as_ref(),
                        ranges,
                    }
                }
            };
            jobs.push(FlattenedHtJob {
                source_index: batch.source_index,
                payload,
                job: *job,
                output_base: batch.output_base,
            });
        }
    }

    let plan = plan_ht_gpu_job_chunks(&requests, limits).map_err(chunk_plan_error)?;
    Ok(MetalHtChunkPlan { jobs, plan })
}

impl MetalHtChunkPlan<'_> {
    pub(in crate::compute) fn chunk_count(&self) -> usize {
        self.plan.chunks().len()
    }

    pub(in crate::compute) fn job_count(&self) -> usize {
        self.jobs.len()
    }

    pub(in crate::compute) fn pack_chunk(
        &self,
        chunk_index: usize,
    ) -> Result<PackedMetalHtChunk, Error> {
        let chunk =
            self.plan
                .chunks()
                .get(chunk_index)
                .copied()
                .ok_or(Error::MetalStateInvariant {
                    state: "HTJ2K Metal chunk plan",
                    reason: "requested chunk index is outside the planned range",
                })?;
        let entries = self
            .plan
            .chunk_entries(chunk_index)
            .ok_or(Error::MetalStateInvariant {
                state: "HTJ2K Metal chunk plan",
                reason: "chunk entry range is outside the flattened job table",
            })?;
        let PackedMetalHtChunkOwners {
            mut coded_data,
            mut jobs,
            mut source_indices,
        } = allocate_packed_metal_ht_chunk(
            chunk.payload_bytes(),
            chunk.job_count(),
            crate::batch_allocation::BatchMetadataBudget::new("HTJ2K Metal packed chunk metadata"),
        )?;

        for entry in entries {
            let flattened =
                self.jobs
                    .get(entry.original_job_index())
                    .ok_or(Error::MetalStateInvariant {
                        state: "HTJ2K Metal chunk plan",
                        reason: "planned job index is outside the flattened job table",
                    })?;
            if flattened.source_index != entry.source_index() {
                return Err(Error::MetalStateInvariant {
                    state: "HTJ2K Metal chunk plan",
                    reason: "planned source identity does not match the flattened job",
                });
            }
            append_packed_ht_job(flattened, &mut coded_data, &mut jobs, &mut source_indices)?;
        }

        if coded_data.len() != chunk.payload_bytes()
            || jobs.len() != chunk.job_count()
            || source_indices.len() != chunk.job_count()
        {
            return Err(Error::MetalStateInvariant {
                state: "HTJ2K Metal packed chunk",
                reason: "packed arena lengths do not match the shared chunk plan",
            });
        }
        Ok(PackedMetalHtChunk {
            bucket: chunk.bucket(),
            coded_data,
            jobs,
            source_indices,
        })
    }
}

fn append_packed_ht_job(
    flattened: &FlattenedHtJob<'_>,
    coded_data: &mut Vec<u8>,
    jobs: &mut Vec<J2kHtCleanupBatchJob>,
    source_indices: &mut Vec<usize>,
) -> Result<(), Error> {
    let mut adjusted = flattened.job;
    adjusted.coded_offset = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
        message: "HTJ2K Metal packed chunk payload exceeds u32".to_string(),
    })?;
    let output_base = u32::try_from(flattened.output_base).map_err(|_| Error::MetalKernel {
        message: format!(
            "HTJ2K Metal source {} output base exceeds u32",
            flattened.source_index
        ),
    })?;
    adjusted.output_offset = adjusted
        .output_offset
        .checked_add(output_base)
        .ok_or_else(|| Error::MetalKernel {
            message: format!(
                "HTJ2K Metal source {} output offset overflow",
                flattened.source_index
            ),
        })?;
    match flattened.payload {
        FlattenedHtPayload::Contiguous(input) => {
            let coded_start =
                usize::try_from(flattened.job.coded_offset).map_err(|_| Error::MetalKernel {
                    message: format!(
                        "HTJ2K Metal source {} coded offset exceeds usize",
                        flattened.source_index
                    ),
                })?;
            let coded_len =
                usize::try_from(flattened.job.coded_len).map_err(|_| Error::MetalKernel {
                    message: format!(
                        "HTJ2K Metal source {} coded length exceeds usize",
                        flattened.source_index
                    ),
                })?;
            let coded_end =
                coded_start
                    .checked_add(coded_len)
                    .ok_or_else(|| Error::MetalKernel {
                        message: format!(
                            "HTJ2K Metal source {} coded payload range overflow",
                            flattened.source_index
                        ),
                    })?;
            let payload = input
                .get(coded_start..coded_end)
                .ok_or_else(|| Error::MetalKernel {
                    message: format!(
                        "HTJ2K Metal source {} coded payload range exceeds its input arena",
                        flattened.source_index
                    ),
                })?;
            coded_data.extend_from_slice(payload);
        }
        FlattenedHtPayload::Referenced { input, ranges } => {
            coded_data.extend_from_slice(referenced_payload_slice(
                flattened.source_index,
                input,
                ranges.cleanup,
            )?);
            if let Some(refinement) = ranges.refinement {
                coded_data.extend_from_slice(referenced_payload_slice(
                    flattened.source_index,
                    input,
                    refinement,
                )?);
            }
        }
    }
    jobs.push(adjusted);
    source_indices.push(flattened.source_index);
    Ok(())
}

fn validate_referenced_ht_payload(
    source_index: usize,
    input: &[u8],
    ranges: HtCodeBlockPayloadRanges,
    job: &J2kHtCleanupBatchJob,
) -> Result<(), Error> {
    referenced_payload_slice(source_index, input, ranges.cleanup)?;
    if let Some(refinement) = ranges.refinement {
        referenced_payload_slice(source_index, input, refinement)?;
    }
    let refinement_len = ranges.refinement.map_or(0, |range| range.length);
    let coded_len = ranges
        .cleanup
        .length
        .checked_add(refinement_len)
        .ok_or_else(|| Error::MetalKernel {
            message: format!(
                "HTJ2K Metal source {source_index} referenced payload length overflow"
            ),
        })?;
    if usize::try_from(job.cleanup_length).ok() != Some(ranges.cleanup.length)
        || usize::try_from(job.refinement_length).ok() != Some(refinement_len)
        || usize::try_from(job.coded_len).ok() != Some(coded_len)
    {
        return Err(Error::MetalKernel {
            message: format!(
                "HTJ2K Metal source {source_index} referenced payload lengths do not match its job"
            ),
        });
    }
    Ok(())
}

fn referenced_payload_slice(
    source_index: usize,
    input: &[u8],
    range: j2k_native::J2kCodestreamRange,
) -> Result<&[u8], Error> {
    let end = range.end().ok_or_else(|| Error::MetalKernel {
        message: format!("HTJ2K Metal source {source_index} referenced payload range overflow"),
    })?;
    input
        .get(range.offset..end)
        .ok_or_else(|| Error::MetalKernel {
            message: format!(
                "HTJ2K Metal source {source_index} referenced payload range exceeds retained input"
            ),
        })
}

fn chunk_plan_error(source: HtGpuJobChunkPlanError) -> Error {
    match source {
        HtGpuJobChunkPlanError::BatchInfrastructure(source) => Error::BatchInfrastructure(source),
        _ => Error::MetalKernel {
            message: format!("HTJ2K Metal chunk planning failed: {source}"),
        },
    }
}
