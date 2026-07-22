// SPDX-License-Identifier: MIT OR Apache-2.0

//! Referenced HTJ2K payload validation and sub-band preparation.

use std::sync::Arc;

use super::super::{
    Error, HtCodeBlockPayloadRanges, J2kCodestreamRange, J2kHtCleanupBatchJob,
    PreparedHtPayloadSource, PreparedHtSubBand,
};
use crate::compute::direct_plan_types::PreparedHtExecutionOwner;

#[cfg(target_os = "macos")]
pub(in crate::compute::direct_prepare) fn prepare_referenced_ht_sub_band(
    job: &j2k_native::HtOwnedSubBandPlan,
    input: &Arc<[u8]>,
    payloads: &[HtCodeBlockPayloadRanges],
    payload_cursor: &mut usize,
) -> Result<PreparedHtSubBand, Error> {
    let (job_payloads, payload_end) =
        referenced_payloads_for_sub_band(payloads, *payload_cursor, job.jobs.len())?;
    let coded_len = crate::batch_allocation::checked_count_sum(
        job_payloads.iter().flat_map(|payload| {
            core::iter::once(payload.cleanup.length)
                .chain(payload.refinement.map(|range| range.length))
        }),
        "HTJ2K MetalDirect referenced coded payload",
    )?;
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "HTJ2K MetalDirect referenced prepared sub-band",
    );
    let mut jobs = budget.try_vec(job.jobs.len(), "HTJ2K MetalDirect referenced jobs")?;
    let mut ranges = budget.try_vec(
        job.jobs.len(),
        "HTJ2K MetalDirect referenced payload ranges",
    )?;
    let mut logical_coded_len = 0usize;
    for (block, payload) in job.jobs.iter().zip(job_payloads.iter().copied()) {
        append_referenced_ht_job(
            &mut jobs,
            &mut ranges,
            &mut logical_coded_len,
            block,
            payload,
            input,
            job.width,
        )?;
    }
    if logical_coded_len != coded_len {
        return Err(Error::MetalStateInvariant {
            state: "HTJ2K referenced prepared sub-band",
            reason: "validated payload lengths do not match the planned logical arena",
        });
    }
    *payload_cursor = payload_end;

    Ok(PreparedHtSubBand {
        band_id: job.band_id,
        width: job.width,
        height: job.height,
        payload_source: PreparedHtPayloadSource::Referenced {
            input: input.clone(),
            ranges,
        },
        jobs,
        execution_owner: Arc::new(PreparedHtExecutionOwner),
    })
}

#[cfg(target_os = "macos")]
fn referenced_payloads_for_sub_band(
    payloads: &[HtCodeBlockPayloadRanges],
    payload_cursor: usize,
    job_count: usize,
) -> Result<(&[HtCodeBlockPayloadRanges], usize), Error> {
    let payload_end = payload_cursor
        .checked_add(job_count)
        .ok_or_else(|| Error::MetalKernel {
            message: "HTJ2K referenced payload cursor overflow".to_string(),
        })?;
    let job_payloads =
        payloads
            .get(payload_cursor..payload_end)
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K referenced plan has fewer payload ranges than code-block jobs"
                    .to_string(),
            })?;
    Ok((job_payloads, payload_end))
}

#[cfg(target_os = "macos")]
fn append_referenced_ht_job(
    jobs: &mut Vec<J2kHtCleanupBatchJob>,
    ranges: &mut Vec<HtCodeBlockPayloadRanges>,
    logical_coded_len: &mut usize,
    block: &j2k_native::HtOwnedCodeBlockBatchJob,
    payload: HtCodeBlockPayloadRanges,
    input: &[u8],
    output_stride: u32,
) -> Result<(), Error> {
    validate_referenced_ht_payload(block, payload, input)?;
    let refinement_len = payload.refinement.map_or(0, |range| range.length);
    let block_coded_len = payload
        .cleanup
        .length
        .checked_add(refinement_len)
        .ok_or_else(|| Error::MetalKernel {
            message: "HTJ2K referenced code-block payload length overflow".to_string(),
        })?;
    let coded_offset = u32::try_from(*logical_coded_len).map_err(|_| Error::MetalKernel {
        message: "HTJ2K MetalDirect referenced coded payload exceeds u32".to_string(),
    })?;
    *logical_coded_len = logical_coded_len
        .checked_add(block_coded_len)
        .ok_or_else(|| Error::MetalKernel {
            message: "HTJ2K referenced prepared payload length overflow".to_string(),
        })?;
    ranges.push(payload);
    jobs.push(referenced_ht_job(
        block,
        coded_offset,
        block_coded_len,
        output_stride,
    )?);
    Ok(())
}

#[cfg(target_os = "macos")]
fn validate_referenced_ht_payload(
    block: &j2k_native::HtOwnedCodeBlockBatchJob,
    payload: HtCodeBlockPayloadRanges,
    input: &[u8],
) -> Result<(), Error> {
    if !block.data.is_empty() {
        return Err(Error::MetalStateInvariant {
            state: "HTJ2K referenced direct plan",
            reason: "referenced plan geometry unexpectedly owns code-block payload bytes",
        });
    }
    let refinement_len = payload.refinement.map_or(0, |range| range.length);
    if usize::try_from(block.cleanup_length).ok() != Some(payload.cleanup.length)
        || usize::try_from(block.refinement_length).ok() != Some(refinement_len)
    {
        return Err(Error::MetalKernel {
            message: "HTJ2K referenced payload lengths do not match code-block geometry"
                .to_string(),
        });
    }
    referenced_codestream_slice(input, payload.cleanup)?;
    if let Some(refinement) = payload.refinement {
        referenced_codestream_slice(input, refinement)?;
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn referenced_ht_job(
    block: &j2k_native::HtOwnedCodeBlockBatchJob,
    coded_offset: u32,
    block_coded_len: usize,
    output_stride: u32,
) -> Result<J2kHtCleanupBatchJob, Error> {
    Ok(J2kHtCleanupBatchJob {
        coded_offset,
        width: block.width,
        height: block.height,
        coded_len: u32::try_from(block_coded_len).map_err(|_| Error::MetalKernel {
            message: "HTJ2K referenced code-block payload exceeds u32".to_string(),
        })?,
        cleanup_length: block.cleanup_length,
        refinement_length: block.refinement_length,
        missing_msbs: u32::from(block.missing_bit_planes),
        num_bitplanes: u32::from(block.num_bitplanes),
        roi_shift: u32::from(block.roi_shift),
        number_of_coding_passes: u32::from(block.number_of_coding_passes),
        output_stride,
        output_offset: block
            .output_y
            .checked_mul(output_stride)
            .and_then(|row| row.checked_add(block.output_x))
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K referenced output offset overflow".to_string(),
            })?,
        dequantization_step: block.dequantization_step,
        stripe_causal: u32::from(block.stripe_causal),
    })
}

#[cfg(target_os = "macos")]
fn referenced_codestream_slice(
    codestream: &[u8],
    range: J2kCodestreamRange,
) -> Result<&[u8], Error> {
    let end = range.end().ok_or_else(|| Error::MetalKernel {
        message: "HTJ2K referenced payload range overflows usize".to_string(),
    })?;
    codestream
        .get(range.offset..end)
        .ok_or_else(|| Error::MetalKernel {
            message: "HTJ2K referenced payload range exceeds retained codestream".to_string(),
        })
}
