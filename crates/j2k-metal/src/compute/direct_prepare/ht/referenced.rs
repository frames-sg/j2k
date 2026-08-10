// SPDX-License-Identifier: MIT OR Apache-2.0

//! Referenced HTJ2K payload validation and sub-band preparation.

use std::sync::Arc;

use super::super::{
    Error, HtCodeBlockPayloadRanges, J2kCodestreamRange, J2kHtCleanupBatchJob,
    PreparedHtPayloadSource, PreparedHtSubBand,
};
use crate::compute::direct_plan_types::PreparedHtExecutionOwner;

#[cfg(all(test, target_os = "macos"))]
mod tests;

#[cfg(target_os = "macos")]
pub(in crate::compute::direct_prepare) fn prepare_referenced_ht_sub_band(
    job: &j2k_native::HtOwnedSubBandPlan,
    input: &Arc<[u8]>,
    payloads: &[HtCodeBlockPayloadRanges],
    payload_cursor: &mut usize,
) -> Result<PreparedHtSubBand, Error> {
    let payload_start = *payload_cursor;
    let (payload_end, fragmented) =
        validate_referenced_sub_band_records(job, input, payloads, payload_start)?;
    let job_payloads =
        payloads
            .get(payload_start..payload_end)
            .ok_or(Error::MetalStateInvariant {
                state: "HTJ2K referenced prepared sub-band",
                reason: "validated payload span is outside the retained record table",
            })?;
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
    let mut ranges = if fragmented {
        Vec::new()
    } else {
        budget.try_vec(
            job.jobs.len(),
            "HTJ2K MetalDirect referenced payload ranges",
        )?
    };
    let mut coded_data = if fragmented {
        budget.try_vec(coded_len, "HTJ2K MetalDirect fragmented payload")?
    } else {
        Vec::new()
    };
    let mut logical_coded_len = 0usize;
    let mut record_cursor = payload_start;
    for block in &job.jobs {
        let records = referenced_records_for_job(payloads, &mut record_cursor, block)?;
        let block_coded_len = (block.cleanup_length as usize)
            .checked_add(block.refinement_length as usize)
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K referenced code-block payload length overflow".to_string(),
            })?;
        let coded_offset = u32::try_from(logical_coded_len).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect referenced coded payload exceeds u32".to_string(),
        })?;
        logical_coded_len = logical_coded_len
            .checked_add(block_coded_len)
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K referenced prepared payload length overflow".to_string(),
            })?;
        if fragmented {
            for record in records {
                coded_data.extend_from_slice(referenced_codestream_slice(input, record.cleanup)?);
                if let Some(refinement) = record.refinement {
                    coded_data.extend_from_slice(referenced_codestream_slice(input, refinement)?);
                }
            }
        } else {
            ranges.push(*records.first().ok_or(Error::MetalStateInvariant {
                state: "HTJ2K referenced prepared sub-band",
                reason: "validated code block has no payload record",
            })?);
        }
        jobs.push(referenced_ht_job(
            block,
            coded_offset,
            block_coded_len,
            job.width,
            job.irreversible_midpoint,
        )?);
    }
    if logical_coded_len != coded_len
        || record_cursor != payload_end
        || (fragmented && coded_data.len() != coded_len)
    {
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
        payload_source: if fragmented {
            PreparedHtPayloadSource::Contiguous(coded_data)
        } else {
            PreparedHtPayloadSource::Referenced {
                input: input.clone(),
                ranges,
            }
        },
        jobs,
        execution_owner: Arc::new(PreparedHtExecutionOwner),
    })
}

#[cfg(target_os = "macos")]
fn validate_referenced_sub_band_records(
    sub_band: &j2k_native::HtOwnedSubBandPlan,
    input: &[u8],
    payloads: &[HtCodeBlockPayloadRanges],
    payload_start: usize,
) -> Result<(usize, bool), Error> {
    let mut payload_end = payload_start;
    let mut fragmented = false;
    for block in &sub_band.jobs {
        let records = referenced_records_for_job(payloads, &mut payload_end, block)?;
        validate_referenced_ht_records(block, records, input)?;
        fragmented |= records.len() > 1;
    }
    Ok((payload_end, fragmented))
}

#[cfg(target_os = "macos")]
fn referenced_records_for_job<'a>(
    payloads: &'a [HtCodeBlockPayloadRanges],
    payload_cursor: &mut usize,
    block: &j2k_native::HtOwnedCodeBlockBatchJob,
) -> Result<&'a [HtCodeBlockPayloadRanges], Error> {
    let first_record = *payload_cursor;
    let first = payloads
        .get(first_record)
        .ok_or_else(|| Error::MetalKernel {
            message: "HTJ2K referenced plan has fewer payload records than code-block jobs"
                .to_string(),
        })?;
    *payload_cursor = payload_cursor
        .checked_add(1)
        .ok_or_else(|| Error::MetalKernel {
            message: "HTJ2K referenced payload cursor overflow".to_string(),
        })?;
    if first.cleanup.length != block.cleanup_length as usize {
        return Err(Error::MetalKernel {
            message: "HTJ2K referenced cleanup length does not match code-block geometry"
                .to_string(),
        });
    }
    let expected_refinement = block.refinement_length as usize;
    let mut refinement = first.refinement.map_or(0, |range| range.length);
    while refinement < expected_refinement {
        let continuation = payloads
            .get(*payload_cursor)
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K referenced plan is missing a refinement continuation record"
                    .to_string(),
            })?;
        *payload_cursor = payload_cursor
            .checked_add(1)
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K referenced payload cursor overflow".to_string(),
            })?;
        let length = continuation
            .refinement
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K refinement continuation record has no refinement range".to_string(),
            })?
            .length;
        if continuation.cleanup.length != 0 || length == 0 {
            return Err(Error::MetalKernel {
                message: "HTJ2K refinement continuation record is malformed".to_string(),
            });
        }
        refinement = refinement
            .checked_add(length)
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K referenced refinement length overflow".to_string(),
            })?;
    }
    if refinement != expected_refinement {
        return Err(Error::MetalKernel {
            message: "HTJ2K referenced refinement length does not match code-block geometry"
                .to_string(),
        });
    }
    payloads
        .get(first_record..*payload_cursor)
        .ok_or(Error::MetalStateInvariant {
            state: "HTJ2K referenced prepared sub-band",
            reason: "validated code-block payload span is outside the retained record table",
        })
}

#[cfg(target_os = "macos")]
fn validate_referenced_ht_records(
    block: &j2k_native::HtOwnedCodeBlockBatchJob,
    records: &[HtCodeBlockPayloadRanges],
    input: &[u8],
) -> Result<(), Error> {
    if !block.data.is_empty() {
        return Err(Error::MetalStateInvariant {
            state: "HTJ2K referenced direct plan",
            reason: "referenced plan geometry unexpectedly owns code-block payload bytes",
        });
    }
    for record in records {
        referenced_codestream_slice(input, record.cleanup)?;
        if let Some(refinement) = record.refinement {
            referenced_codestream_slice(input, refinement)?;
        }
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn referenced_ht_job(
    block: &j2k_native::HtOwnedCodeBlockBatchJob,
    coded_offset: u32,
    block_coded_len: usize,
    output_stride: u32,
    irreversible_midpoint: bool,
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
        irreversible_midpoint: u32::from(irreversible_midpoint),
    })
}

#[cfg(target_os = "macos")]
pub(super) fn referenced_codestream_slice(
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
