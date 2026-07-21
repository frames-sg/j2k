// SPDX-License-Identifier: MIT OR Apache-2.0

use super::abi::{
    CudaClassicCodeBlockJob, CudaClassicDecodeTarget, CudaClassicKernelJob,
    CudaClassicKernelSegment, CudaClassicSegment,
};
use crate::{
    allocation::HostPhaseBudget,
    context::CudaContext,
    error::CudaError,
    htj2k_decode::{
        output_regions::{validate_disjoint_output_regions, Htj2kOutputRect, Htj2kOutputRegion},
        CudaHtj2kDecodeResources,
    },
    memory::{CheckedDeviceBufferRanges, CudaBufferPool},
};

const MAX_CODEBLOCK_DIMENSION: u32 = 64;
const MAX_BITPLANES: u32 = 31;
pub(super) const STYLE_TERMALL: u32 = 1 << 1;
const STYLE_BYPASS: u32 = 1 << 4;
const KNOWN_STYLE_FLAGS: u32 = 0x1f;

pub(super) struct PreparedClassicDecode {
    pub(super) jobs: Vec<CudaClassicKernelJob>,
    pub(super) segments: Vec<CudaClassicKernelSegment>,
    pub(super) scratch_words: usize,
}

pub(super) fn validate_classic_launch_owners(
    context: &CudaContext,
    resources: &CudaHtj2kDecodeResources,
    targets: &[CudaClassicDecodeTarget<'_>],
    pool: &CudaBufferPool,
) -> Result<(), CudaError> {
    if !pool.is_owned_by(context) || !resources.is_owned_by(context)? {
        return Err(invalid(
            "classic decode resources, targets, and pool must belong to the launch context",
        ));
    }
    let target_ranges = CheckedDeviceBufferRanges::from_same_context(
        context,
        targets
            .iter()
            .enumerate()
            .map(|(index, target)| (index, target.coefficients)),
    )?;
    if target_ranges.first_self_overlap().is_some() {
        return Err(invalid(
            "classic decode target allocations must be pairwise disjoint",
        ));
    }
    Ok(())
}

pub(super) fn prepare_classic_decode(
    payload_len: usize,
    targets: &[CudaClassicDecodeTarget<'_>],
    host_budget: &mut HostPhaseBudget,
) -> Result<PreparedClassicDecode, CudaError> {
    let total_jobs = targets.iter().try_fold(0usize, |count, target| {
        count
            .checked_add(target.jobs.len())
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })
    })?;
    let total_segments = targets.iter().try_fold(0usize, |count, target| {
        count
            .checked_add(target.segments.len())
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })
    })?;
    let mut jobs = host_budget.try_vec_with_capacity(total_jobs)?;
    let mut segments = host_budget.try_vec_with_capacity(total_segments)?;
    let mut scratch_words = 0usize;

    for target in targets {
        if target.output_words > target.coefficients.byte_len() / std::mem::size_of::<f32>() {
            return Err(invalid(
                "classic coefficient target is smaller than output_words",
            ));
        }
        for job in target.jobs {
            validate_classic_job(payload_len, target.segments, target.output_words, job)?;
        }
        validate_target_output_regions(target, host_budget)?;
        let mut expected_segment_start = 0u32;
        for job in target.jobs {
            if job.segment_start != expected_segment_start {
                return Err(invalid(
                    "classic job segment ranges must form a contiguous partition",
                ));
            }
            let segment_offset =
                u32::try_from(segments.len()).map_err(|_| CudaError::LengthTooLarge {
                    len: segments.len(),
                })?;
            let coded_offset = u32::try_from(job.payload_offset)
                .map_err(|_| CudaError::LengthTooLarge { len: payload_len })?;
            let scratch_offset = u32::try_from(scratch_words)
                .map_err(|_| CudaError::LengthTooLarge { len: scratch_words })?;
            scratch_words = scratch_words
                .checked_add((job.width as usize + 2) * (job.height as usize + 2))
                .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
            jobs.push(CudaClassicKernelJob {
                output_ptr: target.coefficients.device_ptr(),
                coded_offset,
                coded_len: job.payload_len,
                segment_offset,
                segment_count: job.segment_count,
                scratch_offset,
                width: job.width,
                height: job.height,
                output_stride: job.output_stride,
                output_offset: job.output_offset,
                missing_msbs: job.missing_bitplanes,
                total_bitplanes: job.total_bitplanes,
                number_of_coding_passes: job.number_of_coding_passes,
                sub_band_type: job.sub_band_type,
                style_flags: job.style_flags,
                strict: u32::from(job.strict),
                dequantization_step: job.dequantization_step,
            });
            let segment_end = job.segment_start.checked_add(job.segment_count).ok_or(
                CudaError::LengthTooLarge {
                    len: target.segments.len(),
                },
            )?;
            for segment in &target.segments[job.segment_start as usize..segment_end as usize] {
                let absolute = job
                    .payload_offset
                    .checked_add(u64::from(segment.data_offset))
                    .and_then(|value| u32::try_from(value).ok())
                    .ok_or(CudaError::LengthTooLarge { len: payload_len })?;
                segments.push(CudaClassicKernelSegment {
                    data_offset: absolute,
                    data_length: segment.data_length,
                    start_coding_pass: segment.start_coding_pass,
                    end_coding_pass: segment.end_coding_pass,
                    use_arithmetic: u32::from(segment.use_arithmetic),
                });
            }
            expected_segment_start = segment_end;
        }
        if expected_segment_start as usize != target.segments.len() {
            return Err(invalid(
                "classic job segment ranges do not cover the target segment slice",
            ));
        }
    }
    Ok(PreparedClassicDecode {
        jobs,
        segments,
        scratch_words,
    })
}

pub(super) fn validate_classic_job(
    payload_len: usize,
    segments: &[CudaClassicSegment],
    output_words: usize,
    job: &CudaClassicCodeBlockJob,
) -> Result<(), CudaError> {
    if !(1..=MAX_CODEBLOCK_DIMENSION).contains(&job.width)
        || !(1..=MAX_CODEBLOCK_DIMENSION).contains(&job.height)
        || !(1..=MAX_BITPLANES).contains(&job.total_bitplanes)
        || job.missing_bitplanes >= job.total_bitplanes
        || job.sub_band_type > 3
        || job.style_flags & !KNOWN_STYLE_FLAGS != 0
    {
        return Err(invalid(
            "classic code-block dimensions, bitplanes, or sub-band are invalid",
        ));
    }
    let coded_bitplanes = job.total_bitplanes - job.missing_bitplanes;
    if job.number_of_coding_passes > 1 + 3 * (coded_bitplanes - 1) {
        return Err(invalid(
            "classic code-block pass count exceeds its coded bitplanes",
        ));
    }
    let payload_end = job
        .payload_offset
        .checked_add(u64::from(job.payload_len))
        .ok_or(CudaError::LengthTooLarge { len: payload_len })?;
    if payload_end > payload_len as u64 {
        return Err(invalid("classic code-block payload range is out of bounds"));
    }
    let segment_end = (job.segment_start as usize)
        .checked_add(job.segment_count as usize)
        .ok_or(CudaError::LengthTooLarge {
            len: segments.len(),
        })?;
    let job_segments = segments
        .get(job.segment_start as usize..segment_end)
        .ok_or_else(|| invalid("classic code-block segment range is out of bounds"))?;
    let mut expected_pass = 0;
    let mut expected_offset = 0;
    for segment in job_segments {
        if segment.start_coding_pass != expected_pass
            || segment.end_coding_pass < segment.start_coding_pass
            || segment.data_offset != expected_offset
        {
            return Err(invalid("classic code-block segments are not contiguous"));
        }
        let pass_count = segment.end_coding_pass - segment.start_coding_pass;
        if job.style_flags & STYLE_TERMALL != 0 && pass_count > 1 {
            return Err(invalid(
                "classic TERMALL segments may cover at most one coding pass",
            ));
        }
        for pass in segment.start_coding_pass..segment.end_coding_pass {
            let expected_arithmetic =
                job.style_flags & STYLE_BYPASS == 0 || pass <= 9 || pass.is_multiple_of(3);
            if segment.use_arithmetic != expected_arithmetic {
                return Err(invalid(
                    "classic segment coding mode contradicts BYPASS pass boundaries",
                ));
            }
        }
        expected_pass = segment.end_coding_pass;
        expected_offset = segment
            .data_offset
            .checked_add(segment.data_length)
            .ok_or(CudaError::LengthTooLarge { len: payload_len })?;
    }
    if expected_pass != job.number_of_coding_passes || expected_offset != job.payload_len {
        return Err(invalid(
            "classic code-block segments do not cover its passes and payload",
        ));
    }
    if job.style_flags & (STYLE_TERMALL | STYLE_BYPASS) == 0 && job_segments.len() != 1 {
        return Err(invalid(
            "classic normal mode requires one arithmetic segment",
        ));
    }
    let output_end = u64::from(job.output_offset)
        .checked_add(u64::from(job.height - 1) * u64::from(job.output_stride))
        .and_then(|value| value.checked_add(u64::from(job.width)))
        .ok_or(CudaError::LengthTooLarge { len: output_words })?;
    if job.output_stride < job.width || output_end > output_words as u64 {
        return Err(invalid("classic code-block output range is out of bounds"));
    }
    Ok(())
}

fn validate_target_output_regions(
    target: &CudaClassicDecodeTarget<'_>,
    host_budget: &mut HostPhaseBudget,
) -> Result<(), CudaError> {
    let mut regions = host_budget.try_vec_with_capacity(target.jobs.len())?;
    for job in target.jobs {
        let stride = job.output_stride as usize;
        let width = job.width as usize;
        let height = job.height as usize;
        let start = job.output_offset as usize;
        if stride == 0 || width > stride {
            return Err(invalid(
                "classic output rows require a nonzero stride at least as wide as the block",
            ));
        }
        let column_start = start % stride;
        let column_end = column_start
            .checked_add(width)
            .ok_or(CudaError::LengthTooLarge {
                len: target.output_words,
            })?;
        if column_end > stride {
            return Err(invalid("classic output block crosses its row stride"));
        }
        let row_start = start / stride;
        let row_end = row_start
            .checked_add(height)
            .ok_or(CudaError::LengthTooLarge {
                len: target.output_words,
            })?;
        let end = start
            .checked_add(
                stride
                    .checked_mul(height - 1)
                    .ok_or(CudaError::LengthTooLarge {
                        len: target.output_words,
                    })?,
            )
            .and_then(|last_row| last_row.checked_add(width))
            .ok_or(CudaError::LengthTooLarge {
                len: target.output_words,
            })?;
        if end > target.output_words {
            return Err(invalid("classic code-block output range is out of bounds"));
        }
        regions.push(Htj2kOutputRegion {
            stride,
            rect: Htj2kOutputRect {
                row_start,
                row_end,
                column_start,
                column_end,
            },
            linear_start: start,
            linear_end: end,
        });
    }
    validate_disjoint_output_regions(&mut regions, host_budget.live_bytes())
}

pub(super) fn checked_bytes<T>(count: usize) -> Result<usize, CudaError> {
    count
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(CudaError::LengthTooLarge { len: count })
}

pub(super) fn invalid(message: &'static str) -> CudaError {
    CudaError::InvalidArgument {
        message: message.to_string(),
    }
}
