// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_native::{
    J2kCodeBlockSegment, J2kCodeBlockStyle, J2kOwnedCodeBlockBatchJob, J2kOwnedSubBandPlan,
    J2kSubBandType,
};

use super::{
    required_regions::RequiredBandRegions, shared::CudaPlanOwners, CudaClassicCodeBlock,
    CudaClassicSegment, CudaClassicSubband, Error, PLAN_PAYLOAD_TOO_LARGE,
};

const CLASSIC_PLAN_INVALID: &str = "strict CUDA classic Tier-1 plan is invalid";
const STYLE_RESET_CONTEXT_PROBABILITIES: u32 = 1 << 0;
const STYLE_TERMINATION_ON_EACH_PASS: u32 = 1 << 1;
const STYLE_VERTICALLY_CAUSAL_CONTEXT: u32 = 1 << 2;
const STYLE_SEGMENTATION_SYMBOLS: u32 = 1 << 3;
const STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS: u32 = 1 << 4;

pub(super) fn append_classic_subband(
    owners: &mut CudaPlanOwners,
    subband: &J2kOwnedSubBandPlan,
    required_regions: Option<&RequiredBandRegions>,
) -> Result<(), Error> {
    let subband_index = checked_u32(owners.classic_subbands.len())?;
    let code_block_start = checked_u32(owners.classic_code_blocks.len())?;
    for job in &subband.jobs {
        if required_regions.is_some_and(|regions| {
            !regions.get(subband.band_id).is_some_and(|required| {
                required.intersects(job.output_x, job.output_y, job.width, job.height)
            })
        }) {
            continue;
        }
        append_classic_job(owners, subband_index, job)?;
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

fn append_classic_job(
    owners: &mut CudaPlanOwners,
    subband_index: u32,
    job: &J2kOwnedCodeBlockBatchJob,
) -> Result<(), Error> {
    validate_classic_job(job)?;
    let payload_offset = checked_u64(owners.payload.len())?;
    let payload_len = checked_u32(job.data.len())?;
    let segment_start = checked_u32(owners.classic_segments.len())?;
    let output_stride = checked_u32(job.output_stride)?;
    owners.payload.extend_from_slice(&job.data);
    owners
        .classic_segments
        .extend(job.segments.iter().map(convert_classic_segment));
    owners.classic_code_blocks.push(CudaClassicCodeBlock {
        subband_index,
        payload_offset,
        payload_len,
        segment_start,
        segment_count: checked_u32(job.segments.len())?,
        output_x: job.output_x,
        output_y: job.output_y,
        width: job.width,
        height: job.height,
        output_stride,
        missing_bit_planes: job.missing_bit_planes,
        number_of_coding_passes: job.number_of_coding_passes,
        total_bitplanes: job.total_bitplanes,
        sub_band_type: classic_subband_type(job.sub_band_type),
        style_flags: classic_style_flags(job.style),
        strict: job.strict,
        dequantization_step: job.dequantization_step,
    });
    Ok(())
}

fn validate_classic_job(job: &J2kOwnedCodeBlockBatchJob) -> Result<(), Error> {
    if job.roi_shift != 0
        || !(1..=64).contains(&job.width)
        || !(1..=64).contains(&job.height)
        || !(1..=31).contains(&job.total_bitplanes)
        || job.missing_bit_planes >= job.total_bitplanes
    {
        return invalid_classic_plan();
    }
    let coded_bitplanes = job.total_bitplanes - job.missing_bit_planes;
    let max_passes = 1 + 3 * (coded_bitplanes - 1);
    if job.number_of_coding_passes > max_passes {
        return invalid_classic_plan();
    }
    let mut expected_pass = 0u8;
    let mut expected_offset = 0u32;
    for segment in &job.segments {
        if segment.start_coding_pass != expected_pass
            || segment.end_coding_pass < segment.start_coding_pass
            || segment.data_offset != expected_offset
        {
            return invalid_classic_plan();
        }
        expected_pass = segment.end_coding_pass;
        expected_offset = segment.data_offset.checked_add(segment.data_length).ok_or(
            Error::UnsupportedCudaRequest {
                reason: CLASSIC_PLAN_INVALID,
            },
        )?;
    }
    if expected_pass != job.number_of_coding_passes
        || usize::try_from(expected_offset).ok() != Some(job.data.len())
    {
        return invalid_classic_plan();
    }
    Ok(())
}

fn invalid_classic_plan<T>() -> Result<T, Error> {
    Err(Error::UnsupportedCudaRequest {
        reason: CLASSIC_PLAN_INVALID,
    })
}

fn convert_classic_segment(segment: &J2kCodeBlockSegment) -> CudaClassicSegment {
    CudaClassicSegment {
        data_offset: segment.data_offset,
        data_length: segment.data_length,
        start_coding_pass: segment.start_coding_pass,
        end_coding_pass: segment.end_coding_pass,
        use_arithmetic: segment.use_arithmetic,
    }
}

fn classic_subband_type(value: J2kSubBandType) -> u8 {
    match value {
        J2kSubBandType::LowLow => 0,
        J2kSubBandType::HighLow => 1,
        J2kSubBandType::LowHigh => 2,
        J2kSubBandType::HighHigh => 3,
    }
}

fn classic_style_flags(style: J2kCodeBlockStyle) -> u32 {
    (u32::from(style.reset_context_probabilities) * STYLE_RESET_CONTEXT_PROBABILITIES)
        | (u32::from(style.termination_on_each_pass) * STYLE_TERMINATION_ON_EACH_PASS)
        | (u32::from(style.vertically_causal_context) * STYLE_VERTICALLY_CAUSAL_CONTEXT)
        | (u32::from(style.segmentation_symbols) * STYLE_SEGMENTATION_SYMBOLS)
        | (u32::from(style.selective_arithmetic_coding_bypass)
            * STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS)
}

fn checked_u32(value: usize) -> Result<u32, Error> {
    u32::try_from(value).map_err(|_| Error::UnsupportedCudaRequest {
        reason: PLAN_PAYLOAD_TOO_LARGE,
    })
}

fn checked_u64(value: usize) -> Result<u64, Error> {
    u64::try_from(value).map_err(|_| Error::UnsupportedCudaRequest {
        reason: PLAN_PAYLOAD_TOO_LARGE,
    })
}
