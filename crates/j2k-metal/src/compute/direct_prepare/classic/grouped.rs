// SPDX-License-Identifier: MIT OR Apache-2.0

//! Consecutive sub-band grouping and classic JPEG 2000 group preparation.

use super::super::{
    prepare_direct_tier1_input_buffer, with_runtime, BandRequiredRegion, DirectTier1Mode, Error,
    J2kClassicCleanupBatchJob, J2kClassicSegment, PreparedClassicSubBand,
    PreparedClassicSubBandGroup, PreparedClassicSubBandGroupMember, PreparedDirectGrayscaleStep,
};

#[cfg(target_os = "macos")]
struct ClassicGroupOwners {
    members: Vec<PreparedClassicSubBandGroupMember>,
    jobs: Vec<J2kClassicCleanupBatchJob>,
    segments: Vec<J2kClassicSegment>,
    coded_data: Vec<u8>,
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn prepare_sub_band_groups<'a, SubBand: 'a, Group>(
    steps: &'a [PreparedDirectGrayscaleStep],
    tier1_prepare_mode: DirectTier1Mode,
    mut sub_band_for_step: impl FnMut(&'a PreparedDirectGrayscaleStep) -> Option<&'a SubBand>,
    mut prepare_group: impl FnMut(usize, usize, &[&'a SubBand], DirectTier1Mode) -> Result<Group, Error>,
) -> Result<Vec<Group>, Error> {
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "J2K MetalDirect prepared sub-band groups",
    );
    let mut groups = budget.try_vec(
        steps.len(),
        "J2K MetalDirect prepared sub-band group results",
    )?;
    let mut sub_bands =
        budget.try_vec(steps.len(), "J2K MetalDirect grouped sub-band references")?;
    let mut step_idx = 0;
    while step_idx < steps.len() {
        let start_step = step_idx;
        sub_bands.clear();
        while let Some(sub_band) = steps.get(step_idx).and_then(&mut sub_band_for_step) {
            sub_bands.push(sub_band);
            step_idx += 1;
        }
        if sub_bands.len() > 1 {
            groups.push(prepare_group(
                start_step,
                step_idx,
                &sub_bands,
                tier1_prepare_mode,
            )?);
        }
        if step_idx == start_step {
            step_idx += 1;
        }
    }
    Ok(groups)
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn prepare_classic_sub_band_groups(
    steps: &[PreparedDirectGrayscaleStep],
    tier1_prepare_mode: DirectTier1Mode,
) -> Result<Vec<PreparedClassicSubBandGroup>, Error> {
    prepare_sub_band_groups(
        steps,
        tier1_prepare_mode,
        |step| match step {
            PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => Some(sub_band),
            _ => None,
        },
        prepare_classic_sub_band_group,
    )
}

#[cfg(target_os = "macos")]
fn prepare_classic_sub_band_group(
    start_step: usize,
    end_step: usize,
    sub_bands: &[&PreparedClassicSubBand],
    tier1_prepare_mode: DirectTier1Mode,
) -> Result<PreparedClassicSubBandGroup, Error> {
    let mut owners = allocate_classic_group_owners(sub_bands)?;
    let mut output_base = 0usize;
    for sub_band in sub_bands {
        output_base = append_classic_group_sub_band(&mut owners, sub_band, output_base)?;
    }
    finish_classic_sub_band_group(
        start_step,
        end_step,
        output_base,
        sub_bands.iter().any(|sub_band| sub_band.zero_fill),
        tier1_prepare_mode,
        owners,
    )
}

#[cfg(target_os = "macos")]
fn allocate_classic_group_owners(
    sub_bands: &[&PreparedClassicSubBand],
) -> Result<ClassicGroupOwners, Error> {
    let job_count = crate::batch_allocation::checked_count_sum(
        sub_bands.iter().map(|sub_band| sub_band.jobs.len()),
        "classic J2K MetalDirect grouped jobs",
    )?;
    let segment_count = crate::batch_allocation::checked_count_sum(
        sub_bands.iter().map(|sub_band| sub_band.segments.len()),
        "classic J2K MetalDirect grouped segment table",
    )?;
    let coded_len = crate::batch_allocation::checked_count_sum(
        sub_bands.iter().map(|sub_band| sub_band.coded_data.len()),
        "classic J2K MetalDirect grouped coded payload",
    )?;
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "classic J2K MetalDirect prepared sub-band group",
    );
    Ok(ClassicGroupOwners {
        members: budget.try_vec(sub_bands.len(), "classic J2K MetalDirect grouped members")?,
        jobs: budget.try_vec(job_count, "classic J2K MetalDirect grouped jobs")?,
        segments: budget.try_vec(segment_count, "classic J2K MetalDirect grouped segments")?,
        coded_data: budget.try_vec(coded_len, "classic J2K MetalDirect grouped coded payload")?,
    })
}

#[cfg(target_os = "macos")]
fn append_classic_group_sub_band(
    owners: &mut ClassicGroupOwners,
    sub_band: &PreparedClassicSubBand,
    output_base: usize,
) -> Result<usize, Error> {
    owners.members.push(PreparedClassicSubBandGroupMember {
        band_id: sub_band.band_id,
        offset_elements: output_base,
        window: BandRequiredRegion::full(sub_band.width, sub_band.height),
    });
    let coded_base = u32::try_from(owners.coded_data.len()).map_err(|_| Error::MetalKernel {
        message: "classic J2K MetalDirect grouped coded payload exceeds u32".to_string(),
    })?;
    let segment_base = u32::try_from(owners.segments.len()).map_err(|_| Error::MetalKernel {
        message: "classic J2K MetalDirect grouped segment table exceeds u32".to_string(),
    })?;
    let output_base_u32 = u32::try_from(output_base).map_err(|_| Error::MetalKernel {
        message: "classic J2K MetalDirect grouped coefficient arena exceeds u32".to_string(),
    })?;
    append_grouped_classic_segments(&mut owners.segments, sub_band, coded_base)?;
    append_grouped_classic_jobs(
        &mut owners.jobs,
        sub_band,
        coded_base,
        segment_base,
        output_base_u32,
    )?;
    owners.coded_data.extend_from_slice(&sub_band.coded_data);
    checked_classic_group_output_end(output_base, sub_band)
}

#[cfg(target_os = "macos")]
fn append_grouped_classic_segments(
    segments: &mut Vec<J2kClassicSegment>,
    sub_band: &PreparedClassicSubBand,
    coded_base: u32,
) -> Result<(), Error> {
    for segment in &sub_band.segments {
        let mut grouped_segment = *segment;
        grouped_segment.data_offset =
            coded_base
                .checked_add(segment.data_offset)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K MetalDirect grouped segment offset overflow".to_string(),
                })?;
        segments.push(grouped_segment);
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn append_grouped_classic_jobs(
    jobs: &mut Vec<J2kClassicCleanupBatchJob>,
    sub_band: &PreparedClassicSubBand,
    coded_base: u32,
    segment_base: u32,
    output_base: u32,
) -> Result<(), Error> {
    for job in &sub_band.jobs {
        let mut grouped_job = *job;
        grouped_job.coded_offset =
            coded_base
                .checked_add(job.coded_offset)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K MetalDirect grouped job coded offset overflow"
                        .to_string(),
                })?;
        grouped_job.segment_offset =
            segment_base
                .checked_add(job.segment_offset)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K MetalDirect grouped job segment offset overflow"
                        .to_string(),
                })?;
        grouped_job.output_offset =
            output_base
                .checked_add(job.output_offset)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K MetalDirect grouped output offset overflow".to_string(),
                })?;
        jobs.push(grouped_job);
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn checked_classic_group_output_end(
    output_base: usize,
    sub_band: &PreparedClassicSubBand,
) -> Result<usize, Error> {
    let sub_band_len =
        sub_band
            .width
            .checked_mul(sub_band.height)
            .ok_or_else(|| Error::MetalKernel {
                message: "classic J2K MetalDirect grouped sub-band size overflow".to_string(),
            })? as usize;
    output_base
        .checked_add(sub_band_len)
        .ok_or_else(|| Error::MetalKernel {
            message: "classic J2K MetalDirect grouped coefficient arena overflow".to_string(),
        })
}

#[cfg(target_os = "macos")]
fn finish_classic_sub_band_group(
    start_step: usize,
    end_step: usize,
    total_coefficients: usize,
    zero_fill: bool,
    tier1_prepare_mode: DirectTier1Mode,
    owners: ClassicGroupOwners,
) -> Result<PreparedClassicSubBandGroup, Error> {
    with_runtime(|runtime| {
        let coded_buffer =
            prepare_direct_tier1_input_buffer(runtime, &owners.coded_data, tier1_prepare_mode)?;
        let jobs_buffer =
            prepare_direct_tier1_input_buffer(runtime, &owners.jobs, tier1_prepare_mode)?;
        let segments_buffer =
            prepare_direct_tier1_input_buffer(runtime, &owners.segments, tier1_prepare_mode)?;
        Ok(PreparedClassicSubBandGroup {
            start_step,
            end_step,
            total_coefficients,
            zero_fill,
            coded_data: owners.coded_data,
            coded_buffer,
            jobs: owners.jobs,
            jobs_buffer,
            segments: owners.segments,
            segments_buffer,
            members: owners.members,
        })
    })
}
