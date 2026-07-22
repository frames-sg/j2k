// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_native::{
    J2kClassicCodeBlockPayload, J2kCodestreamRange, J2kOwnedCodeBlockBatchJob, J2kOwnedSubBandPlan,
};

use super::{
    append_classic_job_metadata, checked_u32, checked_u64, invalid_classic_plan,
    validate_classic_job, CLASSIC_PLAN_INVALID,
};
use crate::direct_plan::{
    required_regions::RequiredBandRegions, shared::CudaPlanOwners, CudaClassicSubband,
    CudaHtj2kDecodePlan, Error, PLAN_PAYLOAD_TOO_LARGE,
};

impl CudaHtj2kDecodePlan {
    pub(crate) fn validate_referenced_classic_payload_sequence(
        payloads: &[J2kClassicCodeBlockPayload],
        ranges: &[J2kCodestreamRange],
    ) -> Result<(), Error> {
        validate_referenced_classic_payload_sequence(payloads, ranges)
    }
}

pub(in crate::direct_plan) fn append_referenced_classic_subband<'a>(
    owners: &mut CudaPlanOwners,
    subband: &J2kOwnedSubBandPlan,
    required_regions: Option<&RequiredBandRegions>,
    payloads: &mut impl Iterator<Item = &'a J2kClassicCodeBlockPayload>,
    ranges: &[J2kCodestreamRange],
    encoded: &[u8],
    shared_payload: &mut Vec<u8>,
) -> Result<(), Error> {
    let subband_index = checked_u32(owners.classic_subbands.len())?;
    let code_block_start = checked_u32(owners.classic_code_blocks.len())?;
    for job in &subband.jobs {
        let payload = payloads.next().ok_or(Error::UnsupportedCudaRequest {
            reason: CLASSIC_PLAN_INVALID,
        })?;
        let fragments = referenced_classic_ranges(encoded, *payload, ranges)?;
        validate_referenced_classic_job(job, payload.combined_length)?;
        if required_regions.is_some_and(|regions| {
            !regions.get(subband.band_id).is_some_and(|required| {
                required.intersects(job.output_x, job.output_y, job.width, job.height)
            })
        }) {
            continue;
        }
        let payload_offset = checked_u64(shared_payload.len())?;
        for range in fragments {
            shared_payload.extend_from_slice(referenced_classic_slice(encoded, *range)?);
        }
        append_classic_job_metadata(
            owners,
            subband_index,
            job,
            payload_offset,
            checked_u32(payload.combined_length)?,
        )?;
    }
    owners.classic_subbands.push(CudaClassicSubband {
        band_id: subband.band_id,
        width: subband.width,
        height: subband.height,
        code_block_start,
        code_block_count: checked_u32(
            owners.classic_code_blocks.len() - code_block_start as usize,
        )?,
    });
    Ok(())
}

pub(in crate::direct_plan) fn referenced_classic_payload_bytes(
    encoded: &[u8],
    payloads: &[J2kClassicCodeBlockPayload],
    ranges: &[J2kCodestreamRange],
) -> Result<usize, Error> {
    payloads.iter().try_fold(0usize, |total, payload| {
        referenced_classic_ranges(encoded, *payload, ranges)?;
        total
            .checked_add(payload.combined_length)
            .ok_or(Error::UnsupportedCudaRequest {
                reason: PLAN_PAYLOAD_TOO_LARGE,
            })
    })
}

pub(in crate::direct_plan) fn validate_referenced_classic_payload_sequence(
    payloads: &[J2kClassicCodeBlockPayload],
    ranges: &[J2kCodestreamRange],
) -> Result<(), Error> {
    let mut next_range = 0usize;
    for payload in payloads {
        if payload.first_range != next_range {
            return invalid_classic_plan();
        }
        next_range = payload.end_range().ok_or(Error::UnsupportedCudaRequest {
            reason: PLAN_PAYLOAD_TOO_LARGE,
        })?;
    }
    if next_range != ranges.len() {
        return invalid_classic_plan();
    }
    Ok(())
}

fn validate_referenced_classic_job(
    job: &J2kOwnedCodeBlockBatchJob,
    payload_len: usize,
) -> Result<(), Error> {
    if !job.data.is_empty() {
        return invalid_classic_plan();
    }
    validate_classic_job(job, payload_len)
}

fn referenced_classic_ranges<'a>(
    encoded: &[u8],
    payload: J2kClassicCodeBlockPayload,
    ranges: &'a [J2kCodestreamRange],
) -> Result<&'a [J2kCodestreamRange], Error> {
    let end_range = payload.end_range().ok_or(Error::UnsupportedCudaRequest {
        reason: PLAN_PAYLOAD_TOO_LARGE,
    })?;
    let selected =
        ranges
            .get(payload.first_range..end_range)
            .ok_or(Error::UnsupportedCudaRequest {
                reason: CLASSIC_PLAN_INVALID,
            })?;
    let mut combined = 0usize;
    for range in selected {
        combined = combined
            .checked_add(referenced_classic_slice(encoded, *range)?.len())
            .ok_or(Error::UnsupportedCudaRequest {
                reason: PLAN_PAYLOAD_TOO_LARGE,
            })?;
    }
    if combined != payload.combined_length {
        return invalid_classic_plan();
    }
    Ok(selected)
}

fn referenced_classic_slice(encoded: &[u8], range: J2kCodestreamRange) -> Result<&[u8], Error> {
    let end = range.end().ok_or(Error::UnsupportedCudaRequest {
        reason: PLAN_PAYLOAD_TOO_LARGE,
    })?;
    encoded
        .get(range.offset..end)
        .ok_or(Error::UnsupportedCudaRequest {
            reason: CLASSIC_PLAN_INVALID,
        })
}
