// SPDX-License-Identifier: MIT OR Apache-2.0

//! Classic JPEG 2000 sub-band payload packing and Metal buffer preparation.

use super::super::{
    classic_style_flags, prepare_direct_tier1_input_buffer, with_runtime, DirectTier1Mode, Error,
    J2kClassicCleanupBatchJob, J2kClassicSegment, PreparedClassicSubBand,
};

#[cfg(target_os = "macos")]
struct ClassicSubBandOwners {
    jobs: Vec<J2kClassicCleanupBatchJob>,
    coded_data: Vec<u8>,
    segments: Vec<J2kClassicSegment>,
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn prepare_classic_sub_band(
    job: &j2k_native::J2kOwnedSubBandPlan,
    tier1_prepare_mode: DirectTier1Mode,
) -> Result<PreparedClassicSubBand, Error> {
    let coded_len = crate::batch_allocation::checked_count_sum(
        job.jobs.iter().map(|block| block.data.len()),
        "classic J2K MetalDirect coded payload",
    )?;
    prepare_classic_sub_band_with_payloads(
        job,
        tier1_prepare_mode,
        coded_len,
        |block_index, coded_data| {
            let before = coded_data.len();
            coded_data.extend_from_slice(&job.jobs[block_index].data);
            Ok(coded_data.len() - before)
        },
    )
}

#[cfg(target_os = "macos")]
pub(super) fn prepare_classic_sub_band_with_payloads(
    job: &j2k_native::J2kOwnedSubBandPlan,
    tier1_prepare_mode: DirectTier1Mode,
    coded_len: usize,
    mut append_payload: impl FnMut(usize, &mut Vec<u8>) -> Result<usize, Error>,
) -> Result<PreparedClassicSubBand, Error> {
    let mut owners = allocate_classic_sub_band_owners(job, coded_len)?;
    for (block_index, block) in job.jobs.iter().enumerate() {
        append_classic_sub_band_job(
            &mut owners,
            block_index,
            block,
            job.width,
            &mut append_payload,
        )?;
    }
    if owners.coded_data.len() != coded_len {
        return Err(Error::MetalStateInvariant {
            state: "classic J2K MetalDirect prepared sub-band",
            reason: "appended payload bytes do not match the preflight allocation",
        });
    }
    let zero_fill = owners
        .jobs
        .iter()
        .any(|job| job.coded_len == 0 || job.number_of_coding_passes == 0);
    finish_classic_sub_band(job, tier1_prepare_mode, zero_fill, owners)
}

#[cfg(target_os = "macos")]
fn allocate_classic_sub_band_owners(
    job: &j2k_native::J2kOwnedSubBandPlan,
    coded_len: usize,
) -> Result<ClassicSubBandOwners, Error> {
    let segment_count = crate::batch_allocation::checked_count_sum(
        job.jobs.iter().map(|block| block.segments.len()),
        "classic J2K MetalDirect segment table",
    )?;
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "classic J2K MetalDirect prepared sub-band",
    );
    Ok(ClassicSubBandOwners {
        jobs: budget.try_vec(job.jobs.len(), "classic J2K MetalDirect jobs")?,
        coded_data: budget.try_vec(coded_len, "classic J2K MetalDirect coded payload")?,
        segments: budget.try_vec(segment_count, "classic J2K MetalDirect segment table")?,
    })
}

#[cfg(target_os = "macos")]
fn append_classic_sub_band_job(
    owners: &mut ClassicSubBandOwners,
    block_index: usize,
    block: &j2k_native::J2kOwnedCodeBlockBatchJob,
    output_stride: u32,
    append_payload: &mut impl FnMut(usize, &mut Vec<u8>) -> Result<usize, Error>,
) -> Result<(), Error> {
    let coded_offset = u32::try_from(owners.coded_data.len()).map_err(|_| Error::MetalKernel {
        message: "classic J2K MetalDirect coded payload exceeds u32".to_string(),
    })?;
    let block_coded_len = append_payload(block_index, &mut owners.coded_data)?;
    let segment_offset = append_classic_segments(&mut owners.segments, block, coded_offset)?;
    owners.jobs.push(classic_cleanup_job(
        block,
        coded_offset,
        block_coded_len,
        segment_offset,
        output_stride,
    )?);
    Ok(())
}

#[cfg(target_os = "macos")]
fn append_classic_segments(
    segments: &mut Vec<J2kClassicSegment>,
    block: &j2k_native::J2kOwnedCodeBlockBatchJob,
    coded_offset: u32,
) -> Result<u32, Error> {
    let segment_offset = u32::try_from(segments.len()).map_err(|_| Error::MetalKernel {
        message: "classic J2K MetalDirect segment table exceeds u32".to_string(),
    })?;
    for segment in &block.segments {
        let data_offset = coded_offset
            .checked_add(segment.data_offset)
            .ok_or_else(|| Error::MetalKernel {
                message: "classic J2K MetalDirect segment offset overflow".to_string(),
            })?;
        segments.push(J2kClassicSegment {
            data_offset,
            data_length: segment.data_length,
            start_coding_pass: u32::from(segment.start_coding_pass),
            end_coding_pass: u32::from(segment.end_coding_pass),
            use_arithmetic: u32::from(segment.use_arithmetic),
        });
    }
    Ok(segment_offset)
}

#[cfg(target_os = "macos")]
fn classic_cleanup_job(
    block: &j2k_native::J2kOwnedCodeBlockBatchJob,
    coded_offset: u32,
    block_coded_len: usize,
    segment_offset: u32,
    output_stride: u32,
) -> Result<J2kClassicCleanupBatchJob, Error> {
    Ok(J2kClassicCleanupBatchJob {
        coded_offset,
        coded_len: u32::try_from(block_coded_len).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect coded payload exceeds u32".to_string(),
        })?,
        segment_offset,
        segment_count: u32::try_from(block.segments.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect segment count exceeds u32".to_string(),
        })?,
        width: block.width,
        height: block.height,
        output_stride,
        output_offset: block
            .output_y
            .checked_mul(output_stride)
            .and_then(|row| row.checked_add(block.output_x))
            .ok_or_else(|| Error::MetalKernel {
                message: "classic J2K MetalDirect output offset overflow".to_string(),
            })?,
        missing_msbs: u32::from(block.missing_bit_planes),
        total_bitplanes: u32::from(block.total_bitplanes),
        roi_shift: u32::from(block.roi_shift),
        number_of_coding_passes: u32::from(block.number_of_coding_passes),
        sub_band_type: match block.sub_band_type {
            j2k_native::J2kSubBandType::LowLow => 0,
            j2k_native::J2kSubBandType::HighLow => 1,
            j2k_native::J2kSubBandType::LowHigh => 2,
            j2k_native::J2kSubBandType::HighHigh => 3,
        },
        style_flags: classic_style_flags(block.style),
        strict: u32::from(block.strict),
        dequantization_step: block.dequantization_step,
    })
}

#[cfg(target_os = "macos")]
fn finish_classic_sub_band(
    job: &j2k_native::J2kOwnedSubBandPlan,
    tier1_prepare_mode: DirectTier1Mode,
    zero_fill: bool,
    owners: ClassicSubBandOwners,
) -> Result<PreparedClassicSubBand, Error> {
    with_runtime(|runtime| {
        let coded_buffer =
            prepare_direct_tier1_input_buffer(runtime, &owners.coded_data, tier1_prepare_mode)?;
        let jobs_buffer =
            prepare_direct_tier1_input_buffer(runtime, &owners.jobs, tier1_prepare_mode)?;
        let segments_buffer =
            prepare_direct_tier1_input_buffer(runtime, &owners.segments, tier1_prepare_mode)?;
        Ok(PreparedClassicSubBand {
            band_id: job.band_id,
            width: job.width,
            height: job.height,
            zero_fill,
            coded_data: owners.coded_data,
            coded_buffer,
            jobs: owners.jobs,
            jobs_buffer,
            segments: owners.segments,
            segments_buffer,
        })
    })
}
