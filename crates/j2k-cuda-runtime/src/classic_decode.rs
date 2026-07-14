// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    allocation::HostPhaseBudget,
    bytes::{
        classic_jobs_as_bytes, classic_segments_as_bytes, classic_statuses_as_bytes_mut,
        classic_tables_as_bytes,
    },
    context::CudaContext,
    error::{select_resource_release_error, CudaError},
    execution::cuda_kernel_param,
    htj2k_decode::output_regions::{
        validate_disjoint_output_regions, Htj2kOutputRect, Htj2kOutputRegion,
    },
    htj2k_decode::CudaHtj2kDecodeResources,
    kernels::{j2k_classic_codeblock_launch_geometry, CudaKernel},
    memory::{pooled_device_buffer, CheckedDeviceBufferRanges, CudaBufferPool, CudaDeviceBuffer},
};
use j2k_codec_math::classic::{
    MQ_QE_VALUES, PACKED_MQ_TRANSITION_VALUES, PACKED_SIGN_CONTEXT_LOOKUP, ZERO_CTX_HH_LOOKUP,
    ZERO_CTX_HL_LOOKUP, ZERO_CTX_LL_LH_LOOKUP,
};
use std::time::Instant;

const CLASSIC_KERNEL_NAME: &str = "j2k_decode_classic_codeblocks_multi";
const MAX_CODEBLOCK_DIMENSION: u32 = 64;
const MAX_BITPLANES: u32 = 31;
const STYLE_TERMALL: u32 = 1 << 1;
const STYLE_BYPASS: u32 = 1 << 4;
const KNOWN_STYLE_FLAGS: u32 = 0x1f;

/// One classic JPEG 2000 Tier-1 code-block decode job.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CudaClassicCodeBlockJob {
    /// Byte offset of this code-block in the shared compressed payload.
    pub payload_offset: u64,
    /// Byte length of this code-block payload.
    pub payload_len: u32,
    /// First segment in the target's segment slice.
    pub segment_start: u32,
    /// Number of contiguous segment records.
    pub segment_count: u32,
    /// Code-block width.
    pub width: u32,
    /// Code-block height.
    pub height: u32,
    /// Output row stride in f32 coefficients.
    pub output_stride: u32,
    /// First output coefficient.
    pub output_offset: u32,
    /// Missing most-significant bitplanes.
    pub missing_bitplanes: u32,
    /// Total bitplanes in the sub-band.
    pub total_bitplanes: u32,
    /// Number of coding passes present.
    pub number_of_coding_passes: u32,
    /// JPEG 2000 sub-band tag: LL=0, HL=1, LH=2, HH=3.
    pub sub_band_type: u32,
    /// JPEG 2000 code-block style bits.
    pub style_flags: u32,
    /// Whether malformed entropy data is rejected.
    pub strict: bool,
    /// Fused coefficient dequantization multiplier.
    pub dequantization_step: f32,
}

/// One bounded classic Tier-1 pass segment.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaClassicSegment {
    /// Byte offset relative to the owning code-block payload.
    pub data_offset: u32,
    /// Segment byte length.
    pub data_length: u32,
    /// Inclusive first coding pass.
    pub start_coding_pass: u32,
    /// Exclusive final coding pass.
    pub end_coding_pass: u32,
    /// True for MQ arithmetic coding; false for raw bypass coding.
    pub use_arithmetic: bool,
}

/// One device coefficient target and its classic Tier-1 work.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct CudaClassicDecodeTarget<'a> {
    /// Device-resident f32 coefficient plane.
    pub coefficients: &'a CudaDeviceBuffer,
    /// Code-block jobs writing this plane.
    pub jobs: &'a [CudaClassicCodeBlockJob],
    /// Segment records referenced by the jobs.
    pub segments: &'a [CudaClassicSegment],
    /// Number of f32 words in the coefficient plane.
    pub output_words: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CudaClassicKernelJob {
    pub(crate) output_ptr: u64,
    pub(crate) coded_offset: u32,
    pub(crate) coded_len: u32,
    pub(crate) segment_offset: u32,
    pub(crate) segment_count: u32,
    pub(crate) scratch_offset: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) output_stride: u32,
    pub(crate) output_offset: u32,
    pub(crate) missing_msbs: u32,
    pub(crate) total_bitplanes: u32,
    pub(crate) number_of_coding_passes: u32,
    pub(crate) sub_band_type: u32,
    pub(crate) style_flags: u32,
    pub(crate) strict: u32,
    pub(crate) dequantization_step: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CudaClassicKernelSegment {
    pub(crate) data_offset: u32,
    pub(crate) data_length: u32,
    pub(crate) start_coding_pass: u32,
    pub(crate) end_coding_pass: u32,
    pub(crate) use_arithmetic: u32,
}

/// Status returned for one classic Tier-1 code-block.
#[doc(hidden)]
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaClassicStatus {
    /// Zero on success.
    pub code: u32,
    /// Kernel-defined failure detail.
    pub detail: u32,
    pub(crate) reserved0: u32,
    pub(crate) reserved1: u32,
}

/// Timings for one resident classic Tier-1 decode dispatch.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaClassicDecodeStageTimings {
    /// Host-observed jobs and segments upload time, in microseconds.
    pub job_upload_us: u128,
    /// Host-observed lookup-table upload time, in microseconds.
    pub table_upload_us: u128,
    /// Classic Tier-1 CUDA kernel time, in microseconds.
    pub kernel_us: u128,
    /// Host-observed status download time, in microseconds.
    pub status_d2h_us: u128,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CudaClassicKernelTables {
    pub(crate) mq_qe: [u32; 47],
    pub(crate) mq_transitions: [u32; 47],
    pub(crate) sign_contexts: [u16; 256],
    pub(crate) zero_contexts_ll_lh: [u8; 256],
    pub(crate) zero_contexts_hl: [u8; 256],
    pub(crate) zero_contexts_hh: [u8; 256],
}

const CLASSIC_KERNEL_TABLES: CudaClassicKernelTables = CudaClassicKernelTables {
    mq_qe: MQ_QE_VALUES,
    mq_transitions: PACKED_MQ_TRANSITION_VALUES,
    sign_contexts: PACKED_SIGN_CONTEXT_LOOKUP,
    zero_contexts_ll_lh: ZERO_CTX_LL_LH_LOOKUP,
    zero_contexts_hl: ZERO_CTX_HL_LOOKUP,
    zero_contexts_hh: ZERO_CTX_HH_LOOKUP,
};

struct PreparedClassicDecode {
    jobs: Vec<CudaClassicKernelJob>,
    segments: Vec<CudaClassicKernelSegment>,
    scratch_words: usize,
}

impl CudaContext {
    /// Allocate and clear one classic Tier-1 coefficient plane.
    #[doc(hidden)]
    pub fn allocate_classic_coefficients_with_pool(
        &self,
        output_words: usize,
        pool: &CudaBufferPool,
    ) -> Result<crate::memory::CudaPooledDeviceBuffer, CudaError> {
        if !pool.is_owned_by(self) {
            return Err(invalid(
                "classic coefficient pool must belong to the allocation context",
            ));
        }
        let bytes = checked_bytes::<f32>(output_words)?;
        let output = pool.take(bytes)?;
        self.memset_d32(pooled_device_buffer(&output)?, 0, output_words)?;
        self.synchronize()?;
        Ok(output)
    }

    /// Decode classic Tier-1 code-blocks into one or more device coefficient planes.
    #[doc(hidden)]
    pub fn decode_classic_codeblocks_multi_with_resources_and_pool(
        &self,
        resources: &CudaHtj2kDecodeResources,
        targets: &[CudaClassicDecodeTarget<'_>],
        pool: &CudaBufferPool,
        live_host_bytes: usize,
    ) -> Result<Vec<CudaClassicStatus>, CudaError> {
        self.decode_classic_codeblocks_multi_with_resources_and_pool_timed(
            resources,
            targets,
            pool,
            live_host_bytes,
            false,
        )
        .map(|(statuses, _)| statuses)
    }

    /// Decode classic Tier-1 code-blocks and return optional stage timings.
    #[doc(hidden)]
    pub fn decode_classic_codeblocks_multi_with_resources_and_pool_timed(
        &self,
        resources: &CudaHtj2kDecodeResources,
        targets: &[CudaClassicDecodeTarget<'_>],
        pool: &CudaBufferPool,
        live_host_bytes: usize,
        collect_stage_timings: bool,
    ) -> Result<(Vec<CudaClassicStatus>, CudaClassicDecodeStageTimings), CudaError> {
        validate_classic_launch_owners(self, resources, targets, pool)?;
        let mut host_budget =
            HostPhaseBudget::with_live_bytes("CUDA classic Tier-1 launch owners", live_host_bytes)?;
        let prepared = prepare_classic_decode(resources.payload_len, targets, &mut host_budget)?;
        if prepared.jobs.is_empty() {
            return Ok((Vec::new(), CudaClassicDecodeStageTimings::default()));
        }
        let payload = resources.payload.buffer()?;
        let job_upload_start = collect_stage_timings.then(Instant::now);
        let jobs = pool.upload_pinned(classic_jobs_as_bytes(&prepared.jobs))?;
        let segments = pool.upload_pinned(classic_segments_as_bytes(&prepared.segments))?;
        let job_upload_us = job_upload_start.map_or(0, |start| start.elapsed().as_micros());
        let table_upload_start = collect_stage_timings.then(Instant::now);
        let tables = pool.upload_pinned(classic_tables_as_bytes(&CLASSIC_KERNEL_TABLES))?;
        let table_upload_us = table_upload_start.map_or(0, |start| start.elapsed().as_micros());
        let statuses = pool.take(checked_bytes::<CudaClassicStatus>(prepared.jobs.len())?)?;
        let scratch = pool.take(checked_bytes::<u32>(prepared.scratch_words)?)?;

        let mut payload_ptr = payload.device_ptr();
        let mut jobs_ptr = pooled_device_buffer(&jobs)?.device_ptr();
        let mut segments_ptr = pooled_device_buffer(&segments)?.device_ptr();
        let mut tables_ptr = pooled_device_buffer(&tables)?.device_ptr();
        let mut statuses_ptr = pooled_device_buffer(&statuses)?.device_ptr();
        let mut scratch_ptr = pooled_device_buffer(&scratch)?.device_ptr();
        let mut params = cuda_kernel_params!(
            payload_ptr,
            jobs_ptr,
            segments_ptr,
            tables_ptr,
            statuses_ptr,
            scratch_ptr
        );
        let geometry = j2k_classic_codeblock_launch_geometry(prepared.jobs.len()).ok_or(
            CudaError::LengthTooLarge {
                len: prepared.jobs.len(),
            },
        )?;
        let function = self.inner.cuda_oxide_j2k_classic_decode_kernel_function(
            CudaKernel::J2kClassicDecodeCodeblocksMulti,
        )?;
        let pool_reuse_guard = pool.defer_reuse()?;
        let kernel_result = if collect_stage_timings {
            self.time_default_stream_named_us("j2k.classic.decode.tier1.batch", || {
                self.launch_kernel(function, geometry, &mut params)
            })
            .map(|((), elapsed_us)| elapsed_us)
        } else {
            self.with_nvtx_range("j2k.classic.decode.tier1.batch", || {
                self.launch_kernel(function, geometry, &mut params)
            })
            .map(|()| 0)
        };
        let kernel_us = match kernel_result {
            Ok(elapsed_us) => elapsed_us,
            Err(error) => return pool_reuse_guard.synchronize_then_error(error),
        };

        let mut host_statuses =
            host_budget.try_vec_filled(prepared.jobs.len(), CudaClassicStatus::default())?;
        let status_d2h_start = collect_stage_timings.then(Instant::now);
        if let Err(error) = statuses.copy_to_host(classic_statuses_as_bytes_mut(&mut host_statuses))
        {
            return pool_reuse_guard.release_after_recoverable_operation_error(error);
        }
        let status_d2h_us = status_d2h_start.map_or(0, |start| start.elapsed().as_micros());
        let release_result = pool_reuse_guard.release();
        let status_error = host_statuses
            .iter()
            .copied()
            .enumerate()
            .find(|(_, status)| status.code != 0)
            .map(|(index, status)| CudaError::KernelStatus {
                kernel: CLASSIC_KERNEL_NAME,
                code: status.code,
                detail: ((u32::try_from(index).unwrap_or(u32::MAX)) << 8) | (status.detail & 0xff),
            });
        match (status_error, release_result) {
            (Some(primary), Err(release)) => Err(select_resource_release_error(primary, release)),
            (Some(error), Ok(())) | (None, Err(error)) => Err(error),
            (None, Ok(())) => Ok((
                host_statuses,
                CudaClassicDecodeStageTimings {
                    job_upload_us,
                    table_upload_us,
                    kernel_us,
                    status_d2h_us,
                },
            )),
        }
    }
}

fn validate_classic_launch_owners(
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

fn prepare_classic_decode(
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

fn validate_classic_job(
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

fn checked_bytes<T>(count: usize) -> Result<usize, CudaError> {
    count
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(CudaError::LengthTooLarge { len: count })
}

fn invalid(message: &'static str) -> CudaError {
    CudaError::InvalidArgument {
        message: message.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_job() -> CudaClassicCodeBlockJob {
        CudaClassicCodeBlockJob {
            payload_offset: 0,
            payload_len: 7,
            segment_start: 0,
            segment_count: 1,
            width: 64,
            height: 64,
            output_stride: 64,
            output_offset: 0,
            missing_bitplanes: 0,
            total_bitplanes: 31,
            number_of_coding_passes: 91,
            sub_band_type: 3,
            style_flags: 0,
            strict: true,
            dequantization_step: 1.0,
        }
    }

    #[test]
    fn classic_preflight_accepts_maximum_contract() {
        let segments = [CudaClassicSegment {
            data_offset: 0,
            data_length: 7,
            start_coding_pass: 0,
            end_coding_pass: 91,
            use_arithmetic: true,
        }];
        validate_classic_job(7, &segments, 64 * 64, &max_job())
            .expect("maximum classic Tier-1 contract");
    }

    #[test]
    fn classic_preflight_accepts_zero_length_prefix_segment() {
        let mut job = max_job();
        job.number_of_coding_passes = 1;
        job.style_flags = STYLE_TERMALL;
        job.segment_count = 2;
        let segments = [
            CudaClassicSegment {
                data_offset: 0,
                data_length: 0,
                start_coding_pass: 0,
                end_coding_pass: 0,
                use_arithmetic: true,
            },
            CudaClassicSegment {
                data_offset: 0,
                data_length: 7,
                start_coding_pass: 0,
                end_coding_pass: 1,
                use_arithmetic: true,
            },
        ];
        validate_classic_job(7, &segments, 64 * 64, &job).expect("zero-length classic prefix");
    }

    #[test]
    fn classic_preflight_rejects_noncontiguous_segments_and_output_overrun() {
        let mut job = max_job();
        let segments = [CudaClassicSegment {
            data_offset: 1,
            data_length: 6,
            start_coding_pass: 0,
            end_coding_pass: 91,
            use_arithmetic: true,
        }];
        assert!(validate_classic_job(7, &segments, 64 * 64, &job).is_err());

        job.output_offset = 1;
        let segments = [CudaClassicSegment {
            data_offset: 0,
            data_length: 7,
            start_coding_pass: 0,
            end_coding_pass: 91,
            use_arithmetic: true,
        }];
        assert!(validate_classic_job(7, &segments, 64 * 64, &job).is_err());

        job.output_offset = 0;
        job.payload_len = 8;
        assert!(validate_classic_job(8, &segments, 64 * 64, &job).is_err());
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn classic_preflight_accepts_output_ranges_beyond_u32_words() {
        let mut job = max_job();
        job.width = 2;
        job.height = 2;
        job.output_stride = u32::MAX;
        job.number_of_coding_passes = 1;
        let segments = [CudaClassicSegment {
            data_offset: 0,
            data_length: 7,
            start_coding_pass: 0,
            end_coding_pass: 1,
            use_arithmetic: true,
        }];

        validate_classic_job(
            7,
            &segments,
            usize::try_from(u64::from(u32::MAX) + 2).expect("64-bit output words"),
            &job,
        )
        .expect("device output addressing must support the host-validated range");
    }

    #[test]
    fn classic_runtime_validates_empty_work_and_times_only_status_copy() {
        let source = include_str!("classic_decode.rs");
        let method = source
            .split("pub fn decode_classic_codeblocks_multi_with_resources_and_pool_timed")
            .nth(1)
            .expect("timed classic decode method");
        let owner_validation = method
            .find("validate_classic_launch_owners")
            .expect("owner validation");
        let prepare = method.find("prepare_classic_decode").expect("preparation");
        let empty_return = method
            .find("if prepared.jobs.is_empty()")
            .expect("validated empty fast path");
        assert!(owner_validation < empty_return && prepare < empty_return);

        let status_copy = method.find("statuses.copy_to_host").expect("status copy");
        let status_timing = method
            .find("let status_d2h_us")
            .expect("status timing result");
        let pool_release = method
            .find("let release_result = pool_reuse_guard.release()")
            .expect("pool release");
        assert!(status_copy < status_timing && status_timing < pool_release);
    }
}
