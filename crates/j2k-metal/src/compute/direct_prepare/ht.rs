// SPDX-License-Identifier: MIT OR Apache-2.0

//! HTJ2K sub-band and grouped payload preparation.

use super::{
    prepare_sub_band_groups, BandRequiredRegion, DirectTier1Mode, Error, HtCodeBlockPayloadRanges,
    J2kCodestreamRange, J2kHtCleanupBatchJob, PreparedDirectGrayscaleStep, PreparedHtPayloadSource,
    PreparedHtSubBand, PreparedHtSubBandGroup, PreparedHtSubBandGroupMember,
};
use crate::compute::direct_plan_types::PreparedHtExecutionOwner;
use std::sync::Arc;

#[cfg(target_os = "macos")]
pub(in crate::compute) fn prepare_ht_sub_band(
    job: &j2k_native::HtOwnedSubBandPlan,
    _tier1_prepare_mode: DirectTier1Mode,
) -> Result<PreparedHtSubBand, Error> {
    let coded_len = crate::batch_allocation::checked_count_sum(
        job.jobs.iter().map(|block| block.data.len()),
        "HTJ2K MetalDirect coded payload",
    )?;
    let mut budget =
        crate::batch_allocation::BatchMetadataBudget::new("HTJ2K MetalDirect prepared sub-band");
    let mut jobs = budget.try_vec(job.jobs.len(), "HTJ2K MetalDirect jobs")?;
    let mut coded_data = budget.try_vec(coded_len, "HTJ2K MetalDirect coded payload")?;
    for block in &job.jobs {
        let coded_offset = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect coded payload exceeds u32".to_string(),
        })?;
        coded_data.extend_from_slice(&block.data);
        jobs.push(J2kHtCleanupBatchJob {
            coded_offset,
            width: block.width,
            height: block.height,
            coded_len: u32::try_from(block.data.len()).map_err(|_| Error::MetalKernel {
                message: "HTJ2K MetalDirect coded payload exceeds u32".to_string(),
            })?,
            cleanup_length: block.cleanup_length,
            refinement_length: block.refinement_length,
            missing_msbs: u32::from(block.missing_bit_planes),
            num_bitplanes: u32::from(block.num_bitplanes),
            roi_shift: u32::from(block.roi_shift),
            number_of_coding_passes: u32::from(block.number_of_coding_passes),
            output_stride: job.width,
            output_offset: block
                .output_y
                .checked_mul(job.width)
                .and_then(|row| row.checked_add(block.output_x))
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K MetalDirect output offset overflow".to_string(),
                })?,
            dequantization_step: block.dequantization_step,
            stripe_causal: u32::from(block.stripe_causal),
        });
    }

    Ok(PreparedHtSubBand {
        band_id: job.band_id,
        width: job.width,
        height: job.height,
        payload_source: PreparedHtPayloadSource::Contiguous(coded_data),
        jobs,
        execution_owner: Arc::new(PreparedHtExecutionOwner),
    })
}

#[cfg(target_os = "macos")]
#[expect(
    clippy::too_many_lines,
    reason = "referenced preparation keeps payload validation and ABI job offsets synchronized"
)]
pub(super) fn prepare_referenced_ht_sub_band(
    job: &j2k_native::HtOwnedSubBandPlan,
    input: &Arc<[u8]>,
    payloads: &[HtCodeBlockPayloadRanges],
    payload_cursor: &mut usize,
) -> Result<PreparedHtSubBand, Error> {
    let payload_end =
        payload_cursor
            .checked_add(job.jobs.len())
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K referenced payload cursor overflow".to_string(),
            })?;
    let job_payloads =
        payloads
            .get(*payload_cursor..payload_end)
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K referenced plan has fewer payload ranges than code-block jobs"
                    .to_string(),
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
    let mut ranges = budget.try_vec(
        job.jobs.len(),
        "HTJ2K MetalDirect referenced payload ranges",
    )?;
    let mut logical_coded_len = 0usize;
    for (block, payload) in job.jobs.iter().zip(job_payloads) {
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
        let coded_offset = u32::try_from(logical_coded_len).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect referenced coded payload exceeds u32".to_string(),
        })?;
        referenced_codestream_slice(input, payload.cleanup)?;
        if let Some(refinement) = payload.refinement {
            referenced_codestream_slice(input, refinement)?;
        }
        let block_coded_len = payload
            .cleanup
            .length
            .checked_add(refinement_len)
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K referenced code-block payload length overflow".to_string(),
            })?;
        logical_coded_len = logical_coded_len
            .checked_add(block_coded_len)
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K referenced prepared payload length overflow".to_string(),
            })?;
        ranges.push(*payload);
        jobs.push(J2kHtCleanupBatchJob {
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
            output_stride: job.width,
            output_offset: block
                .output_y
                .checked_mul(job.width)
                .and_then(|row| row.checked_add(block.output_x))
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K referenced output offset overflow".to_string(),
                })?,
            dequantization_step: block.dequantization_step,
            stripe_causal: u32::from(block.stripe_causal),
        });
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

#[cfg(target_os = "macos")]
pub(in crate::compute) fn prepare_ht_sub_band_groups(
    steps: &[PreparedDirectGrayscaleStep],
    tier1_prepare_mode: DirectTier1Mode,
) -> Result<Vec<PreparedHtSubBandGroup>, Error> {
    prepare_sub_band_groups(
        steps,
        tier1_prepare_mode,
        |step| match step {
            PreparedDirectGrayscaleStep::HtSubBand(sub_band) => Some(sub_band),
            _ => None,
        },
        prepare_ht_sub_band_group,
    )
}

#[cfg(target_os = "macos")]
#[expect(
    clippy::too_many_lines,
    reason = "group assembly keeps payload ownership, rebased offsets, and arena lengths aligned"
)]
pub(super) fn prepare_ht_sub_band_group(
    start_step: usize,
    end_step: usize,
    sub_bands: &[&PreparedHtSubBand],
    _tier1_prepare_mode: DirectTier1Mode,
) -> Result<PreparedHtSubBandGroup, Error> {
    let job_count = crate::batch_allocation::checked_count_sum(
        sub_bands.iter().map(|sub_band| sub_band.jobs.len()),
        "HTJ2K MetalDirect grouped jobs",
    )?;
    let coded_len = crate::batch_allocation::checked_count_sum(
        sub_bands
            .iter()
            .flat_map(|sub_band| sub_band.jobs.iter())
            .map(|job| job.coded_len as usize),
        "HTJ2K MetalDirect grouped coded payload",
    )?;
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "HTJ2K MetalDirect prepared sub-band group",
    );
    let mut members = budget.try_vec(sub_bands.len(), "HTJ2K MetalDirect grouped members")?;
    let mut jobs = budget.try_vec(job_count, "HTJ2K MetalDirect grouped jobs")?;
    let first = sub_bands.first().ok_or(Error::MetalStateInvariant {
        state: "HTJ2K MetalDirect prepared sub-band group",
        reason: "group preparation received no sub-bands",
    })?;
    let mut payload_source = match &first.payload_source {
        PreparedHtPayloadSource::Contiguous(_) => PreparedHtPayloadSource::Contiguous(
            budget.try_vec(coded_len, "HTJ2K MetalDirect grouped coded payload")?,
        ),
        PreparedHtPayloadSource::Referenced { input, .. } => PreparedHtPayloadSource::Referenced {
            input: input.clone(),
            ranges: budget.try_vec(
                job_count,
                "HTJ2K MetalDirect grouped referenced payload ranges",
            )?,
        },
    };
    let mut output_base = 0usize;
    let mut logical_coded_base = 0usize;

    for sub_band in sub_bands {
        members.push(PreparedHtSubBandGroupMember {
            band_id: sub_band.band_id,
            offset_elements: output_base,
            window: BandRequiredRegion::full(sub_band.width, sub_band.height),
        });

        let coded_base = u32::try_from(logical_coded_base).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect grouped coded payload exceeds u32".to_string(),
        })?;
        let output_base_u32 = u32::try_from(output_base).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect grouped coefficient arena exceeds u32".to_string(),
        })?;
        for job in &sub_band.jobs {
            let mut grouped_job = *job;
            grouped_job.coded_offset =
                coded_base
                    .checked_add(job.coded_offset)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "HTJ2K MetalDirect grouped coded offset overflow".to_string(),
                    })?;
            grouped_job.output_offset =
                output_base_u32
                    .checked_add(job.output_offset)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "HTJ2K MetalDirect grouped output offset overflow".to_string(),
                    })?;
            jobs.push(grouped_job);
        }
        match (&mut payload_source, &sub_band.payload_source) {
            (
                PreparedHtPayloadSource::Contiguous(grouped),
                PreparedHtPayloadSource::Contiguous(source),
            ) => grouped.extend_from_slice(source),
            (
                PreparedHtPayloadSource::Referenced {
                    input: grouped_input,
                    ranges: grouped_ranges,
                },
                PreparedHtPayloadSource::Referenced {
                    input: source_input,
                    ranges: source_ranges,
                },
            ) => {
                if !Arc::ptr_eq(grouped_input, source_input) {
                    return Err(Error::MetalStateInvariant {
                        state: "HTJ2K MetalDirect referenced sub-band group",
                        reason: "grouped sub-bands reference different encoded input owners",
                    });
                }
                if source_ranges.len() != sub_band.jobs.len() {
                    return Err(Error::MetalStateInvariant {
                        state: "HTJ2K MetalDirect referenced sub-band group",
                        reason: "payload range count does not match grouped job count",
                    });
                }
                grouped_ranges.extend_from_slice(source_ranges);
            }
            _ => {
                return Err(Error::MetalStateInvariant {
                    state: "HTJ2K MetalDirect prepared sub-band group",
                    reason: "group mixes contiguous and referenced payload ownership",
                });
            }
        }
        logical_coded_base = sub_band
            .jobs
            .iter()
            .try_fold(logical_coded_base, |total, job| {
                total
                    .checked_add(job.coded_len as usize)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "HTJ2K MetalDirect grouped coded payload overflow".to_string(),
                    })
            })?;
        let sub_band_len =
            sub_band
                .width
                .checked_mul(sub_band.height)
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K MetalDirect grouped sub-band size overflow".to_string(),
                })? as usize;
        output_base = output_base
            .checked_add(sub_band_len)
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K MetalDirect grouped coefficient arena overflow".to_string(),
            })?;
    }
    if logical_coded_base != coded_len {
        return Err(Error::MetalStateInvariant {
            state: "HTJ2K MetalDirect prepared sub-band group",
            reason: "grouped logical payload length does not match its planned arena",
        });
    }
    match &payload_source {
        PreparedHtPayloadSource::Contiguous(data) if data.len() != coded_len => {
            return Err(Error::MetalStateInvariant {
                state: "HTJ2K MetalDirect prepared sub-band group",
                reason: "grouped contiguous payload length does not match its jobs",
            });
        }
        PreparedHtPayloadSource::Referenced { ranges, .. } if ranges.len() != job_count => {
            return Err(Error::MetalStateInvariant {
                state: "HTJ2K MetalDirect referenced sub-band group",
                reason: "grouped payload range count does not match its jobs",
            });
        }
        PreparedHtPayloadSource::Contiguous(_) | PreparedHtPayloadSource::Referenced { .. } => {}
    }

    Ok(PreparedHtSubBandGroup {
        start_step,
        end_step,
        total_coefficients: output_base,
        payload_source,
        jobs,
        members,
        execution_owner: Arc::new(PreparedHtExecutionOwner),
    })
}
