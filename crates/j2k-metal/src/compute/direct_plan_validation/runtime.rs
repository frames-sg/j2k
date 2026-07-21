// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::PixelFormat;

use super::super::abi::{
    J2kClassicCleanupBatchJob, J2kClassicSegment, J2kHtCleanupBatchJob, J2K_CLASSIC_MAX_HEIGHT,
    J2K_CLASSIC_MAX_WIDTH, J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS,
};
use super::super::{
    DirectTier1Mode, PreparedDirectColorPlan, PreparedDirectGrayscalePlan,
    PreparedDirectGrayscaleStep,
};

pub(in crate::compute) fn prepared_direct_color_plan_supports_runtime(
    plan: &PreparedDirectColorPlan,
    fmt: PixelFormat,
) -> bool {
    let supported_format = matches!(
        fmt,
        PixelFormat::Rgb8
            | PixelFormat::Rgb16
            | PixelFormat::RgbI16
            | PixelFormat::Rgba8
            | PixelFormat::Rgba16
            | PixelFormat::RgbaI16
    );
    supported_format
        && plan.component_plans.len() == fmt.channels()
        && plan
            .component_plans
            .iter()
            .all(prepared_direct_component_plan_supports_runtime)
}

fn prepared_direct_component_plan_supports_runtime(plan: &PreparedDirectGrayscalePlan) -> bool {
    plan.tier1_prepare_mode == DirectTier1Mode::Metal
        && plan.steps.iter().all(|step| match step {
            PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => sub_band
                .jobs
                .iter()
                .all(|job| classic_prepared_job_supports_runtime(job, &sub_band.segments)),
            PreparedDirectGrayscaleStep::HtSubBand(sub_band) => {
                sub_band.jobs.iter().all(ht_prepared_job_supports_runtime)
            }
            PreparedDirectGrayscaleStep::Idwt(_) | PreparedDirectGrayscaleStep::Store(_) => true,
        })
        && plan.classic_groups.iter().all(|group| {
            group
                .jobs
                .iter()
                .all(|job| classic_prepared_job_supports_runtime(job, &group.segments))
        })
        && plan
            .ht_groups
            .iter()
            .all(|group| group.jobs.iter().all(ht_prepared_job_supports_runtime))
}

fn classic_prepared_job_supports_runtime(
    job: &J2kClassicCleanupBatchJob,
    segments: &[J2kClassicSegment],
) -> bool {
    if job.width == 0 || job.height == 0 {
        return true;
    }
    if job.width > J2K_CLASSIC_MAX_WIDTH || job.height > J2K_CLASSIC_MAX_HEIGHT {
        return false;
    }
    if job.output_stride < job.width || job.roi_shift != 0 {
        return false;
    }
    if job.total_bitplanes == 0 || job.total_bitplanes > 31 || job.missing_msbs >= 31 {
        return false;
    }
    let bitplanes = job.total_bitplanes.saturating_sub(job.missing_msbs);
    if bitplanes == 0 {
        return false;
    }

    let start = job.segment_offset as usize;
    let count = job.segment_count as usize;
    let Some(end) = start.checked_add(count) else {
        return false;
    };
    if end > segments.len() {
        return false;
    }
    let job_segments = &segments[start..end];
    if job.coded_len == 0 || job.number_of_coding_passes == 0 {
        return job.coded_len == 0
            && job.number_of_coding_passes == 0
            && job_segments.iter().all(|segment| {
                segment.data_offset == job.coded_offset
                    && segment.data_length == 0
                    && segment.start_coding_pass == 0
                    && segment.end_coding_pass == 0
            });
    }

    let max_coding_passes = 1 + 3 * (bitplanes - 1);
    if job.number_of_coding_passes > max_coding_passes || count == 0 {
        return false;
    }

    let uses_bypass = (job.style_flags & J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS) != 0;
    let mut expected_start = 0u32;
    let mut expected_offset = job.coded_offset;
    for segment in job_segments {
        if segment.start_coding_pass != expected_start
            || segment.start_coding_pass > segment.end_coding_pass
        {
            return false;
        }
        if uses_bypass {
            let expected_arithmetic =
                segment.start_coding_pass <= 9 || segment.start_coding_pass % 3 == 0;
            if (segment.use_arithmetic != 0) != expected_arithmetic {
                return false;
            }
            if segment.use_arithmetic == 0 {
                if segment.start_coding_pass % 3 != 1 {
                    return false;
                }
                if segment
                    .end_coding_pass
                    .saturating_sub(segment.start_coding_pass)
                    > 2
                {
                    return false;
                }
                if (segment.start_coding_pass..segment.end_coding_pass).any(|pass| pass % 3 == 0) {
                    return false;
                }
            }
        } else if segment.use_arithmetic == 0 {
            return false;
        }

        let Some(data_end) = segment.data_offset.checked_add(segment.data_length) else {
            return false;
        };
        if segment.data_offset != expected_offset
            || segment.data_offset < job.coded_offset
            || data_end > job.coded_offset.saturating_add(job.coded_len)
        {
            return false;
        }
        expected_offset = data_end;
        expected_start = segment.end_coding_pass;
    }

    expected_start == job.number_of_coding_passes
        && expected_offset == job.coded_offset.saturating_add(job.coded_len)
}

fn ht_prepared_job_supports_runtime(job: &J2kHtCleanupBatchJob) -> bool {
    if job.width == 0 || job.height == 0 {
        return true;
    }
    job.roi_shift == 0
        && job.output_stride >= job.width
        && crate::ht::supports_metal_ht_geometry(job.width, job.height)
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    fn valid_classic_job() -> J2kClassicCleanupBatchJob {
        J2kClassicCleanupBatchJob {
            coded_offset: 7,
            coded_len: 3,
            segment_offset: 0,
            segment_count: 1,
            width: 1,
            height: 1,
            output_stride: 1,
            output_offset: 0,
            missing_msbs: 0,
            total_bitplanes: 1,
            roi_shift: 0,
            number_of_coding_passes: 1,
            sub_band_type: 0,
            style_flags: 0,
            strict: 1,
            dequantization_step: 1.0,
        }
    }

    fn valid_classic_segment() -> J2kClassicSegment {
        J2kClassicSegment {
            data_offset: 7,
            data_length: 3,
            start_coding_pass: 0,
            end_coding_pass: 1,
            use_arithmetic: 1,
        }
    }

    #[test]
    fn classic_runtime_preflight_rejects_unimplemented_roi_shift_and_inconsistent_empty_job() {
        let segment = valid_classic_segment();
        let mut job = valid_classic_job();
        assert!(classic_prepared_job_supports_runtime(&job, &[segment]));

        job.roi_shift = 1;
        assert!(!classic_prepared_job_supports_runtime(&job, &[segment]));

        job.roi_shift = 0;
        job.number_of_coding_passes = 0;
        assert!(!classic_prepared_job_supports_runtime(&job, &[segment]));

        job.coded_len = 0;
        let empty_segment = J2kClassicSegment {
            data_offset: job.coded_offset,
            data_length: 0,
            start_coding_pass: 0,
            end_coding_pass: 0,
            use_arithmetic: 1,
        };
        assert!(classic_prepared_job_supports_runtime(
            &job,
            &[empty_segment]
        ));
    }
}
