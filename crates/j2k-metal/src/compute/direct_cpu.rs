// SPDX-License-Identifier: MIT OR Apache-2.0

use std::time::Instant;

use j2k_native::{
    decode_ht_code_block_scalar_with_workspace,
    decode_ht_code_block_scalar_with_workspace_profiled,
    decode_j2k_code_block_scalar_with_workspace,
    decode_j2k_code_block_scalar_with_workspace_profiled, HtCodeBlockDecodeJob,
    HtCodeBlockDecodeProfile, HtCodeBlockDecodeWorkspace, J2kCodeBlockDecodeJob,
    J2kCodeBlockDecodeProfile, J2kCodeBlockDecodeWorkspace, J2kCodeBlockSegment, J2kCodeBlockStyle,
    J2kSubBandType,
};
use rayon::prelude::*;

use crate::{error::native_decode_error, Error};

use super::{
    checked_coefficient_len, hybrid_stage_signpost, packed_cpu_decode_coefficients,
    packed_cpu_decode_coefficients_in, packed_cpu_decode_output_len,
    record_hybrid_cpu_decode_inputs, record_hybrid_cpu_decode_worker_init,
    required_classic_output_len, required_ht_output_len, CpuTier1DecodeSubstageCounters,
    J2kClassicCleanupBatchJob, J2kClassicSegment, J2kHtCleanupBatchJob, PreparedClassicSubBand,
    PreparedClassicSubBandGroup, PreparedDirectGrayscalePlan, PreparedHtSubBand,
    PreparedHtSubBandGroup, HYBRID_CPU_DECODE_MIN_INPUTS_PER_TASK,
    J2K_CLASSIC_STYLE_RESET_CONTEXT_PROBABILITIES, J2K_CLASSIC_STYLE_SEGMENTATION_SYMBOLS,
    J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS,
    J2K_CLASSIC_STYLE_TERMINATION_ON_EACH_PASS, J2K_CLASSIC_STYLE_VERTICALLY_CAUSAL_CONTEXT,
    SIGNPOST_DECODE_HYBRID_CPU_TIER1,
};

#[cfg(test)]
pub(super) fn decode_prepared_classic_sub_band_on_cpu(
    sub_band: &PreparedClassicSubBand,
) -> Result<Vec<f32>, Error> {
    decode_prepared_classic_sub_band_on_cpu_profile(sub_band, None)
}

pub(super) fn decode_prepared_classic_sub_band_on_cpu_profile(
    sub_band: &PreparedClassicSubBand,
    profile_counters: Option<&CpuTier1DecodeSubstageCounters>,
) -> Result<Vec<f32>, Error> {
    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_CPU_TIER1);
    let len = checked_coefficient_len(
        sub_band.width,
        sub_band.height,
        "classic J2K MetalDirect hybrid sub-band size overflow",
    )?;
    let mut output = packed_cpu_decode_coefficients(1, len)?;
    if let Some(counters) = profile_counters {
        let mut scratch = ClassicCpuDecodeScratch::default();
        decode_prepared_classic_jobs_on_cpu_with_scratch_profiled(
            &sub_band.coded_data,
            &sub_band.segments,
            &sub_band.jobs,
            &mut output,
            &mut scratch,
            counters,
        )?;
    } else {
        decode_prepared_classic_jobs_on_cpu(
            &sub_band.coded_data,
            &sub_band.segments,
            &sub_band.jobs,
            &mut output,
        )?;
    }
    Ok(output)
}

pub(super) fn decode_prepared_classic_sub_band_group_on_cpu_profile(
    group: &PreparedClassicSubBandGroup,
    profile_counters: Option<&CpuTier1DecodeSubstageCounters>,
) -> Result<Vec<f32>, Error> {
    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_CPU_TIER1);
    let mut output = packed_cpu_decode_coefficients(1, group.total_coefficients)?;
    if let Some(counters) = profile_counters {
        let mut scratch = ClassicCpuDecodeScratch::default();
        decode_prepared_classic_jobs_on_cpu_with_scratch_profiled(
            &group.coded_data,
            &group.segments,
            &group.jobs,
            &mut output,
            &mut scratch,
            counters,
        )?;
    } else {
        decode_prepared_classic_jobs_on_cpu(
            &group.coded_data,
            &group.segments,
            &group.jobs,
            &mut output,
        )?;
    }
    Ok(output)
}

#[derive(Default)]
pub(super) struct ClassicCpuDecodeScratch {
    segments: Vec<J2kCodeBlockSegment>,
    decode: J2kCodeBlockDecodeWorkspace,
}

fn decode_prepared_classic_jobs_on_cpu(
    coded_data: &[u8],
    segments: &[J2kClassicSegment],
    jobs: &[J2kClassicCleanupBatchJob],
    output: &mut [f32],
) -> Result<(), Error> {
    let mut scratch = ClassicCpuDecodeScratch::default();
    decode_prepared_classic_jobs_on_cpu_with_scratch(
        coded_data,
        segments,
        jobs,
        output,
        &mut scratch,
    )
}

pub(super) fn decode_prepared_classic_jobs_on_cpu_with_scratch(
    coded_data: &[u8],
    segments: &[J2kClassicSegment],
    jobs: &[J2kClassicCleanupBatchJob],
    output: &mut [f32],
    scratch: &mut ClassicCpuDecodeScratch,
) -> Result<(), Error> {
    decode_prepared_classic_jobs_on_cpu_with_scratch_impl::<false>(
        coded_data, segments, jobs, output, scratch, None,
    )
}

pub(super) fn decode_prepared_classic_jobs_on_cpu_with_scratch_profiled(
    coded_data: &[u8],
    segments: &[J2kClassicSegment],
    jobs: &[J2kClassicCleanupBatchJob],
    output: &mut [f32],
    scratch: &mut ClassicCpuDecodeScratch,
    profile_counters: &CpuTier1DecodeSubstageCounters,
) -> Result<(), Error> {
    decode_prepared_classic_jobs_on_cpu_with_scratch_impl::<true>(
        coded_data,
        segments,
        jobs,
        output,
        scratch,
        Some(profile_counters),
    )
}

fn decode_prepared_classic_jobs_on_cpu_with_scratch_impl<const PROFILE: bool>(
    coded_data: &[u8],
    segments: &[J2kClassicSegment],
    jobs: &[J2kClassicCleanupBatchJob],
    output: &mut [f32],
    scratch: &mut ClassicCpuDecodeScratch,
    profile_counters: Option<&CpuTier1DecodeSubstageCounters>,
) -> Result<(), Error> {
    for job in jobs {
        let prep_started = PROFILE.then(Instant::now);
        let start = job.output_offset as usize;
        let segment_window = prepared_classic_segment_window(segments, job)?;
        scratch.segments.clear();
        scratch.segments.reserve(segment_window.len());
        for segment in segment_window {
            scratch.segments.push(prepared_classic_segment(segment)?);
        }
        let decode_job = prepared_classic_decode_job(coded_data, &scratch.segments, job)?;
        let required_len = required_classic_output_len(decode_job)?;
        let end = start
            .checked_add(required_len)
            .ok_or_else(|| Error::MetalKernel {
                message: "classic J2K MetalDirect hybrid output offset overflow".to_string(),
            })?;
        let Some(output_window) = output.get_mut(start..end) else {
            return Err(Error::MetalKernel {
                message: "classic J2K MetalDirect hybrid output slice is too small".to_string(),
            });
        };
        if let Some(started) = prep_started {
            profile_counters
                .expect("profile counters required for profiled classic decode")
                .record_classic_segment_prep(started);
        }
        if PROFILE {
            let decode_started = Instant::now();
            let mut profile = J2kCodeBlockDecodeProfile::default();
            decode_j2k_code_block_scalar_with_workspace_profiled(
                decode_job,
                output_window,
                &mut scratch.decode,
                &mut profile,
            )
            .map_err(native_decode_error)?;
            profile_counters
                .expect("profile counters required for profiled classic decode")
                .record_classic_block_decode(decode_started, &profile);
        } else {
            decode_j2k_code_block_scalar_with_workspace(
                decode_job,
                output_window,
                &mut scratch.decode,
            )
            .map_err(native_decode_error)?;
        }
    }
    Ok(())
}

fn prepared_classic_segment_window<'a>(
    segments: &'a [J2kClassicSegment],
    job: &J2kClassicCleanupBatchJob,
) -> Result<&'a [J2kClassicSegment], Error> {
    let segment_start = job.segment_offset as usize;
    let segment_end = segment_start
        .checked_add(job.segment_count as usize)
        .ok_or_else(|| Error::MetalKernel {
            message: "classic J2K MetalDirect hybrid segment span overflow".to_string(),
        })?;
    segments
        .get(segment_start..segment_end)
        .ok_or_else(|| Error::MetalKernel {
            message: "classic J2K MetalDirect hybrid segment span is invalid".to_string(),
        })
}

fn prepared_classic_decode_job<'a>(
    coded_data: &'a [u8],
    segments: &'a [J2kCodeBlockSegment],
    job: &J2kClassicCleanupBatchJob,
) -> Result<J2kCodeBlockDecodeJob<'a>, Error> {
    Ok(J2kCodeBlockDecodeJob {
        data: coded_data,
        segments,
        width: job.width,
        height: job.height,
        output_stride: job.output_stride as usize,
        missing_bit_planes: checked_u8(job.missing_msbs, "classic missing bit planes")?,
        number_of_coding_passes: checked_u8(job.number_of_coding_passes, "classic coding passes")?,
        total_bitplanes: checked_u8(job.total_bitplanes, "classic total bitplanes")?,
        roi_shift: checked_u8(job.roi_shift, "classic ROI shift")?,
        sub_band_type: prepared_classic_sub_band_type(job.sub_band_type)?,
        style: prepared_classic_style(job.style_flags),
        strict: job.strict != 0,
        dequantization_step: job.dequantization_step,
    })
}

fn prepared_classic_segment(segment: &J2kClassicSegment) -> Result<J2kCodeBlockSegment, Error> {
    Ok(J2kCodeBlockSegment {
        data_offset: segment.data_offset,
        data_length: segment.data_length,
        start_coding_pass: checked_u8(segment.start_coding_pass, "classic segment start pass")?,
        end_coding_pass: checked_u8(segment.end_coding_pass, "classic segment end pass")?,
        use_arithmetic: segment.use_arithmetic != 0,
    })
}

fn prepared_classic_sub_band_type(value: u32) -> Result<J2kSubBandType, Error> {
    match value {
        0 => Ok(J2kSubBandType::LowLow),
        1 => Ok(J2kSubBandType::HighLow),
        2 => Ok(J2kSubBandType::LowHigh),
        3 => Ok(J2kSubBandType::HighHigh),
        _ => Err(Error::MetalKernel {
            message: format!("classic J2K MetalDirect hybrid sub-band type {value} is invalid"),
        }),
    }
}

fn prepared_classic_style(flags: u32) -> J2kCodeBlockStyle {
    J2kCodeBlockStyle {
        selective_arithmetic_coding_bypass: (flags
            & J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS)
            != 0,
        reset_context_probabilities: (flags & J2K_CLASSIC_STYLE_RESET_CONTEXT_PROBABILITIES) != 0,
        termination_on_each_pass: (flags & J2K_CLASSIC_STYLE_TERMINATION_ON_EACH_PASS) != 0,
        vertically_causal_context: (flags & J2K_CLASSIC_STYLE_VERTICALLY_CAUSAL_CONTEXT) != 0,
        segmentation_symbols: (flags & J2K_CLASSIC_STYLE_SEGMENTATION_SYMBOLS) != 0,
    }
}

pub(super) fn decode_prepared_ht_sub_band_on_cpu_profile(
    sub_band: &PreparedHtSubBand,
    profile_counters: Option<&CpuTier1DecodeSubstageCounters>,
) -> Result<Vec<f32>, Error> {
    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_CPU_TIER1);
    let len = checked_coefficient_len(
        sub_band.width,
        sub_band.height,
        "HTJ2K MetalDirect hybrid sub-band size overflow",
    )?;
    let mut output = packed_cpu_decode_coefficients(1, len)?;
    if let Some(counters) = profile_counters {
        let mut workspace = HtCodeBlockDecodeWorkspace::default();
        decode_prepared_ht_jobs_on_cpu_with_workspace_profiled(
            &sub_band.coded_data,
            &sub_band.jobs,
            &mut output,
            &mut workspace,
            counters,
        )?;
    } else {
        decode_prepared_ht_jobs_on_cpu(&sub_band.coded_data, &sub_band.jobs, &mut output)?;
    }
    Ok(output)
}

pub(super) fn decode_prepared_ht_sub_band_group_on_cpu_profile(
    group: &PreparedHtSubBandGroup,
    profile_counters: Option<&CpuTier1DecodeSubstageCounters>,
) -> Result<Vec<f32>, Error> {
    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_CPU_TIER1);
    let mut output = packed_cpu_decode_coefficients(1, group.total_coefficients)?;
    if let Some(counters) = profile_counters {
        let mut workspace = HtCodeBlockDecodeWorkspace::default();
        decode_prepared_ht_jobs_on_cpu_with_workspace_profiled(
            &group.coded_arena.data,
            &group.jobs,
            &mut output,
            &mut workspace,
            counters,
        )?;
    } else {
        decode_prepared_ht_jobs_on_cpu(&group.coded_arena.data, &group.jobs, &mut output)?;
    }
    Ok(output)
}

fn decode_prepared_ht_jobs_on_cpu(
    coded_data: &[u8],
    jobs: &[J2kHtCleanupBatchJob],
    output: &mut [f32],
) -> Result<(), Error> {
    let mut workspace = HtCodeBlockDecodeWorkspace::default();
    decode_prepared_ht_jobs_on_cpu_with_workspace(coded_data, jobs, output, &mut workspace)
}

pub(super) fn decode_prepared_ht_jobs_on_cpu_with_workspace(
    coded_data: &[u8],
    jobs: &[J2kHtCleanupBatchJob],
    output: &mut [f32],
    workspace: &mut HtCodeBlockDecodeWorkspace,
) -> Result<(), Error> {
    decode_prepared_ht_jobs_on_cpu_with_workspace_impl::<false>(
        coded_data, jobs, output, workspace, None,
    )
}

pub(super) fn decode_prepared_ht_jobs_on_cpu_with_workspace_profiled(
    coded_data: &[u8],
    jobs: &[J2kHtCleanupBatchJob],
    output: &mut [f32],
    workspace: &mut HtCodeBlockDecodeWorkspace,
    profile_counters: &CpuTier1DecodeSubstageCounters,
) -> Result<(), Error> {
    decode_prepared_ht_jobs_on_cpu_with_workspace_impl::<true>(
        coded_data,
        jobs,
        output,
        workspace,
        Some(profile_counters),
    )
}

fn decode_prepared_ht_jobs_on_cpu_with_workspace_impl<const PROFILE: bool>(
    coded_data: &[u8],
    jobs: &[J2kHtCleanupBatchJob],
    output: &mut [f32],
    workspace: &mut HtCodeBlockDecodeWorkspace,
    profile_counters: Option<&CpuTier1DecodeSubstageCounters>,
) -> Result<(), Error> {
    for job in jobs {
        let start = job.output_offset as usize;
        let decode_job = prepared_ht_decode_job(coded_data, job)?;
        let required_len = required_ht_output_len(decode_job)?;
        let end = start
            .checked_add(required_len)
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K MetalDirect hybrid output offset overflow".to_string(),
            })?;
        let Some(output_window) = output.get_mut(start..end) else {
            return Err(Error::MetalKernel {
                message: "HTJ2K MetalDirect hybrid output slice is too small".to_string(),
            });
        };
        if PROFILE {
            let decode_started = Instant::now();
            let mut profile = HtCodeBlockDecodeProfile::default();
            decode_ht_code_block_scalar_with_workspace_profiled(
                decode_job,
                output_window,
                workspace,
                &mut profile,
            )
            .map_err(native_decode_error)?;
            profile_counters
                .expect("profile counters required for profiled HT decode")
                .record_ht_block_decode(decode_started, &profile);
        } else {
            decode_ht_code_block_scalar_with_workspace(decode_job, output_window, workspace)
                .map_err(native_decode_error)?;
        }
    }
    Ok(())
}

fn prepared_ht_decode_job<'a>(
    coded_data: &'a [u8],
    job: &J2kHtCleanupBatchJob,
) -> Result<HtCodeBlockDecodeJob<'a>, Error> {
    let start = job.coded_offset as usize;
    let len = job.coded_len as usize;
    let end = start.checked_add(len).ok_or_else(|| Error::MetalKernel {
        message: "HTJ2K MetalDirect hybrid coded span overflow".to_string(),
    })?;
    let Some(data) = coded_data.get(start..end) else {
        return Err(Error::MetalKernel {
            message: "HTJ2K MetalDirect hybrid coded span is invalid".to_string(),
        });
    };

    Ok(HtCodeBlockDecodeJob {
        data,
        cleanup_length: job.cleanup_length,
        refinement_length: job.refinement_length,
        width: job.width,
        height: job.height,
        output_stride: job.output_stride as usize,
        missing_bit_planes: checked_u8(job.missing_msbs, "HTJ2K missing bit planes")?,
        number_of_coding_passes: checked_u8(job.number_of_coding_passes, "HTJ2K coding passes")?,
        num_bitplanes: checked_u8(job.num_bitplanes, "HTJ2K total bitplanes")?,
        roi_shift: checked_u8(job.roi_shift, "HTJ2K ROI shift")?,
        stripe_causal: job.stripe_causal != 0,
        strict: true,
        dequantization_step: job.dequantization_step,
    })
}

fn checked_u8(value: u32, label: &str) -> Result<u8, Error> {
    u8::try_from(value).map_err(|_| Error::MetalKernel {
        message: format!("J2K MetalDirect hybrid {label} exceeds u8"),
    })
}

pub(super) struct ClassicCpuDecodeInput<'a> {
    pub(super) coded_data: &'a [u8],
    pub(super) segments: &'a [J2kClassicSegment],
    pub(super) jobs: &'a [J2kClassicCleanupBatchJob],
    pub(super) output_len: usize,
}

pub(super) struct HtCpuDecodeInput<'a> {
    pub(super) coded_data: &'a [u8],
    pub(super) jobs: &'a [J2kHtCleanupBatchJob],
    pub(super) output_len: usize,
}

fn decode_classic_inputs_on_cpu_parallel(
    inputs: &[ClassicCpuDecodeInput<'_>],
    profile_counters: Option<&CpuTier1DecodeSubstageCounters>,
    budget: &mut crate::batch_allocation::BatchMetadataBudget,
) -> Result<Vec<f32>, Error> {
    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_CPU_TIER1);
    record_hybrid_cpu_decode_inputs(inputs.len());
    let Some(output_len) = packed_cpu_decode_output_len(
        inputs.iter().map(|input| input.output_len),
        "classic J2K MetalDirect hybrid batch",
    )?
    else {
        return Ok(Vec::new());
    };
    let mut coefficients = packed_cpu_decode_coefficients_in(budget, inputs.len(), output_len)?;
    coefficients
        .par_chunks_mut(output_len)
        .zip(inputs.par_iter())
        .with_min_len(HYBRID_CPU_DECODE_MIN_INPUTS_PER_TASK)
        .try_for_each_init(
            || {
                record_hybrid_cpu_decode_worker_init();
                ClassicCpuDecodeScratch::default()
            },
            |scratch, (output, input)| {
                if let Some(counters) = profile_counters {
                    decode_prepared_classic_jobs_on_cpu_with_scratch_profiled(
                        input.coded_data,
                        input.segments,
                        input.jobs,
                        output,
                        scratch,
                        counters,
                    )
                } else {
                    decode_prepared_classic_jobs_on_cpu_with_scratch(
                        input.coded_data,
                        input.segments,
                        input.jobs,
                        output,
                        scratch,
                    )
                }
            },
        )?;
    Ok(coefficients)
}

fn decode_ht_inputs_on_cpu_parallel(
    inputs: &[HtCpuDecodeInput<'_>],
    profile_counters: Option<&CpuTier1DecodeSubstageCounters>,
    budget: &mut crate::batch_allocation::BatchMetadataBudget,
) -> Result<Vec<f32>, Error> {
    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_CPU_TIER1);
    record_hybrid_cpu_decode_inputs(inputs.len());
    let Some(output_len) = packed_cpu_decode_output_len(
        inputs.iter().map(|input| input.output_len),
        "HTJ2K MetalDirect hybrid batch",
    )?
    else {
        return Ok(Vec::new());
    };
    let mut coefficients = packed_cpu_decode_coefficients_in(budget, inputs.len(), output_len)?;
    coefficients
        .par_chunks_mut(output_len)
        .zip(inputs.par_iter())
        .with_min_len(HYBRID_CPU_DECODE_MIN_INPUTS_PER_TASK)
        .try_for_each_init(
            || {
                record_hybrid_cpu_decode_worker_init();
                HtCodeBlockDecodeWorkspace::default()
            },
            |workspace, (output, input)| {
                if let Some(counters) = profile_counters {
                    decode_prepared_ht_jobs_on_cpu_with_workspace_profiled(
                        input.coded_data,
                        input.jobs,
                        output,
                        workspace,
                        counters,
                    )
                } else {
                    decode_prepared_ht_jobs_on_cpu_with_workspace(
                        input.coded_data,
                        input.jobs,
                        output,
                        workspace,
                    )
                }
            },
        )?;
    Ok(coefficients)
}

pub(super) fn decode_classic_inputs_on_cpu_with_plan_cache(
    plan: &PreparedDirectGrayscalePlan,
    step_idx: usize,
    inputs: &[ClassicCpuDecodeInput<'_>],
    profile_counters: Option<&CpuTier1DecodeSubstageCounters>,
) -> Result<Vec<f32>, Error> {
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "classic J2K MetalDirect hybrid CPU Tier-1 coefficients",
    );
    if inputs.len() != 1 {
        return decode_classic_inputs_on_cpu_parallel(inputs, profile_counters, &mut budget);
    }

    let output_len = inputs[0].output_len;
    if let Some(coefficients) =
        plan.cached_cpu_tier1_coefficients(&mut budget, step_idx, output_len)?
    {
        return Ok(coefficients);
    }

    let coefficients =
        decode_classic_inputs_on_cpu_parallel(inputs, profile_counters, &mut budget)?;
    plan.store_cpu_tier1_coefficients(step_idx, output_len, coefficients)
}

pub(super) fn decode_ht_inputs_on_cpu_with_plan_cache(
    plan: &PreparedDirectGrayscalePlan,
    step_idx: usize,
    inputs: &[HtCpuDecodeInput<'_>],
    profile_counters: Option<&CpuTier1DecodeSubstageCounters>,
) -> Result<Vec<f32>, Error> {
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "HTJ2K MetalDirect hybrid CPU Tier-1 coefficients",
    );
    if inputs.len() != 1 {
        return decode_ht_inputs_on_cpu_parallel(inputs, profile_counters, &mut budget);
    }

    let output_len = inputs[0].output_len;
    if let Some(coefficients) =
        plan.cached_cpu_tier1_coefficients(&mut budget, step_idx, output_len)?
    {
        return Ok(coefficients);
    }

    let coefficients = decode_ht_inputs_on_cpu_parallel(inputs, profile_counters, &mut budget)?;
    plan.store_cpu_tier1_coefficients(step_idx, output_len, coefficients)
}
