// SPDX-License-Identifier: MIT OR Apache-2.0

//! Homogeneous HTJ2K sub-band group preparation.

use std::sync::Arc;

use super::super::{
    prepare_sub_band_groups, BandRequiredRegion, DirectTier1Mode, Error, J2kHtCleanupBatchJob,
    PreparedDirectGrayscaleStep, PreparedHtPayloadSource, PreparedHtSubBand,
    PreparedHtSubBandGroup, PreparedHtSubBandGroupMember,
};
use crate::compute::direct_plan_types::PreparedHtExecutionOwner;

#[cfg(target_os = "macos")]
struct HtGroupOwners {
    members: Vec<PreparedHtSubBandGroupMember>,
    jobs: Vec<J2kHtCleanupBatchJob>,
    payload_source: PreparedHtPayloadSource,
    planned_coded_len: usize,
    planned_job_count: usize,
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
pub(super) fn prepare_ht_sub_band_group(
    start_step: usize,
    end_step: usize,
    sub_bands: &[&PreparedHtSubBand],
    _tier1_prepare_mode: DirectTier1Mode,
) -> Result<PreparedHtSubBandGroup, Error> {
    let mut owners = allocate_ht_group_owners(sub_bands)?;
    let mut output_base = 0usize;
    let mut logical_coded_base = 0usize;
    for sub_band in sub_bands {
        owners.members.push(PreparedHtSubBandGroupMember {
            band_id: sub_band.band_id,
            offset_elements: output_base,
            window: BandRequiredRegion::full(sub_band.width, sub_band.height),
        });
        append_grouped_ht_jobs(&mut owners.jobs, sub_band, logical_coded_base, output_base)?;
        append_grouped_ht_payload(&mut owners.payload_source, sub_band)?;
        logical_coded_base = checked_grouped_ht_coded_end(logical_coded_base, sub_band)?;
        output_base = checked_grouped_ht_output_end(output_base, sub_band)?;
    }
    validate_ht_group_owners(&owners, logical_coded_base)?;

    Ok(PreparedHtSubBandGroup {
        start_step,
        end_step,
        total_coefficients: output_base,
        payload_source: owners.payload_source,
        jobs: owners.jobs,
        members: owners.members,
        execution_owner: Arc::new(PreparedHtExecutionOwner),
    })
}

#[cfg(target_os = "macos")]
fn allocate_ht_group_owners(sub_bands: &[&PreparedHtSubBand]) -> Result<HtGroupOwners, Error> {
    let planned_job_count = crate::batch_allocation::checked_count_sum(
        sub_bands.iter().map(|sub_band| sub_band.jobs.len()),
        "HTJ2K MetalDirect grouped jobs",
    )?;
    let planned_coded_len = crate::batch_allocation::checked_count_sum(
        sub_bands
            .iter()
            .flat_map(|sub_band| sub_band.jobs.iter())
            .map(|job| job.coded_len as usize),
        "HTJ2K MetalDirect grouped coded payload",
    )?;
    let first = sub_bands.first().ok_or(Error::MetalStateInvariant {
        state: "HTJ2K MetalDirect prepared sub-band group",
        reason: "group preparation received no sub-bands",
    })?;
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "HTJ2K MetalDirect prepared sub-band group",
    );
    let payload_source = match &first.payload_source {
        PreparedHtPayloadSource::Contiguous(_) => PreparedHtPayloadSource::Contiguous(
            budget.try_vec(planned_coded_len, "HTJ2K MetalDirect grouped coded payload")?,
        ),
        PreparedHtPayloadSource::Referenced { input, .. } => PreparedHtPayloadSource::Referenced {
            input: input.clone(),
            ranges: budget.try_vec(
                planned_job_count,
                "HTJ2K MetalDirect grouped referenced payload ranges",
            )?,
        },
    };
    Ok(HtGroupOwners {
        members: budget.try_vec(sub_bands.len(), "HTJ2K MetalDirect grouped members")?,
        jobs: budget.try_vec(planned_job_count, "HTJ2K MetalDirect grouped jobs")?,
        payload_source,
        planned_coded_len,
        planned_job_count,
    })
}

#[cfg(target_os = "macos")]
fn append_grouped_ht_jobs(
    jobs: &mut Vec<J2kHtCleanupBatchJob>,
    sub_band: &PreparedHtSubBand,
    logical_coded_base: usize,
    output_base: usize,
) -> Result<(), Error> {
    let coded_base = u32::try_from(logical_coded_base).map_err(|_| Error::MetalKernel {
        message: "HTJ2K MetalDirect grouped coded payload exceeds u32".to_string(),
    })?;
    let output_base = u32::try_from(output_base).map_err(|_| Error::MetalKernel {
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
            output_base
                .checked_add(job.output_offset)
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K MetalDirect grouped output offset overflow".to_string(),
                })?;
        jobs.push(grouped_job);
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn append_grouped_ht_payload(
    payload_source: &mut PreparedHtPayloadSource,
    sub_band: &PreparedHtSubBand,
) -> Result<(), Error> {
    match (payload_source, &sub_band.payload_source) {
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
    Ok(())
}

#[cfg(target_os = "macos")]
fn checked_grouped_ht_coded_end(
    logical_coded_base: usize,
    sub_band: &PreparedHtSubBand,
) -> Result<usize, Error> {
    sub_band
        .jobs
        .iter()
        .try_fold(logical_coded_base, |total, job| {
            total
                .checked_add(job.coded_len as usize)
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K MetalDirect grouped coded payload overflow".to_string(),
                })
        })
}

#[cfg(target_os = "macos")]
fn checked_grouped_ht_output_end(
    output_base: usize,
    sub_band: &PreparedHtSubBand,
) -> Result<usize, Error> {
    let sub_band_len =
        sub_band
            .width
            .checked_mul(sub_band.height)
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K MetalDirect grouped sub-band size overflow".to_string(),
            })? as usize;
    output_base
        .checked_add(sub_band_len)
        .ok_or_else(|| Error::MetalKernel {
            message: "HTJ2K MetalDirect grouped coefficient arena overflow".to_string(),
        })
}

#[cfg(target_os = "macos")]
fn validate_ht_group_owners(owners: &HtGroupOwners, logical_coded_len: usize) -> Result<(), Error> {
    if logical_coded_len != owners.planned_coded_len {
        return Err(Error::MetalStateInvariant {
            state: "HTJ2K MetalDirect prepared sub-band group",
            reason: "grouped logical payload length does not match its planned arena",
        });
    }
    match &owners.payload_source {
        PreparedHtPayloadSource::Contiguous(data) if data.len() != owners.planned_coded_len => {
            Err(Error::MetalStateInvariant {
                state: "HTJ2K MetalDirect prepared sub-band group",
                reason: "grouped contiguous payload length does not match its jobs",
            })
        }
        PreparedHtPayloadSource::Referenced { ranges, .. }
            if ranges.len() != owners.planned_job_count =>
        {
            Err(Error::MetalStateInvariant {
                state: "HTJ2K MetalDirect referenced sub-band group",
                reason: "grouped payload range count does not match its jobs",
            })
        }
        PreparedHtPayloadSource::Contiguous(_) | PreparedHtPayloadSource::Referenced { .. } => {
            Ok(())
        }
    }
}
