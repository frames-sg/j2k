// SPDX-License-Identifier: Apache-2.0

use j2k_core::PixelFormat;
use j2k_native::J2kRect;

use super::{
    DirectTier1Mode, J2kClassicCleanupBatchJob, J2kClassicSegment, J2kDirectStoreStep,
    J2kHtCleanupBatchJob, PreparedClassicSubBand, PreparedClassicSubBandGroup,
    PreparedDirectColorPlan, PreparedDirectGrayscalePlan, PreparedDirectGrayscaleStep,
    PreparedDirectIdwt, PreparedHtSubBand, PreparedHtSubBandGroup, J2K_CLASSIC_MAX_HEIGHT,
    J2K_CLASSIC_MAX_WIDTH, J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS,
};
use crate::Error;

#[cfg(test)]
pub(super) fn prepared_direct_color_tier1_input_count(plan: &PreparedDirectColorPlan) -> usize {
    plan.component_plans
        .iter()
        .map(prepared_direct_component_tier1_input_count)
        .sum()
}

#[cfg(test)]
fn prepared_direct_component_tier1_input_count(plan: &PreparedDirectGrayscalePlan) -> usize {
    let mut count = 0;
    let mut step_idx = 0;
    while step_idx < plan.steps.len() {
        if let Some(group) = plan.classic_group_starting_at(step_idx) {
            count += 1;
            step_idx = group.end_step;
            continue;
        }
        if let Some(group) = plan.ht_group_starting_at(step_idx) {
            count += 1;
            step_idx = group.end_step;
            continue;
        }
        if matches!(
            &plan.steps[step_idx],
            PreparedDirectGrayscaleStep::ClassicSubBand(_)
                | PreparedDirectGrayscaleStep::HtSubBand(_)
        ) {
            count += 1;
        }
        step_idx += 1;
    }
    count
}

pub(super) fn prepared_direct_color_plan_supports_runtime(
    plan: &PreparedDirectColorPlan,
    fmt: PixelFormat,
) -> bool {
    matches!(
        fmt,
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16
    ) && plan.component_plans.len() == 3
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
    if job.output_stride < job.width {
        return false;
    }
    if job.roi_shift != 0 {
        return false;
    }
    if job.total_bitplanes == 0 || job.total_bitplanes > 31 || job.missing_msbs >= 31 {
        return false;
    }
    let bitplanes = job.total_bitplanes.saturating_sub(job.missing_msbs);
    if bitplanes == 0 {
        return false;
    }
    let max_coding_passes = 1 + 3 * (bitplanes - 1);
    if job.number_of_coding_passes == 0 || job.number_of_coding_passes > max_coding_passes {
        return false;
    }

    let start = job.segment_offset as usize;
    let count = job.segment_count as usize;
    let Some(end) = start.checked_add(count) else {
        return false;
    };
    if end > segments.len() || count == 0 {
        return false;
    }

    let uses_bypass = (job.style_flags & J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS) != 0;
    let mut expected_start = 0u32;
    let mut expected_offset = job.coded_offset;
    for segment in &segments[start..end] {
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

pub(super) fn classic_group_shapes_match(
    first: &PreparedClassicSubBandGroup,
    other: &PreparedClassicSubBandGroup,
) -> bool {
    first.end_step == other.end_step
        && first.total_coefficients == other.total_coefficients
        && first.members.len() == other.members.len()
        && first
            .members
            .iter()
            .zip(&other.members)
            .all(|(left, right)| left.offset_elements == right.offset_elements)
}

pub(super) fn ht_group_shapes_match(
    first: &PreparedHtSubBandGroup,
    other: &PreparedHtSubBandGroup,
) -> bool {
    first.end_step == other.end_step
        && first.total_coefficients == other.total_coefficients
        && first.members.len() == other.members.len()
        && first
            .members
            .iter()
            .zip(&other.members)
            .all(|(left, right)| left.offset_elements == right.offset_elements)
}

pub(super) fn classic_sub_band_shapes_match(
    first: &PreparedClassicSubBand,
    other: &PreparedClassicSubBand,
) -> bool {
    first.width == other.width && first.height == other.height
}

pub(super) fn ht_sub_band_shapes_match(
    first: &PreparedHtSubBand,
    other: &PreparedHtSubBand,
) -> bool {
    first.width == other.width && first.height == other.height
}

fn rect_shapes_match(first: J2kRect, other: J2kRect) -> bool {
    first.x0 == other.x0 && first.y0 == other.y0 && first.x1 == other.x1 && first.y1 == other.y1
}

pub(super) fn idwt_shapes_match(first: &PreparedDirectIdwt, other: &PreparedDirectIdwt) -> bool {
    first.step.transform == other.step.transform
        && rect_shapes_match(first.step.rect, other.step.rect)
        && first.output_window.x0 == other.output_window.x0
        && first.output_window.y0 == other.output_window.y0
        && first.output_window.x1 == other.output_window.x1
        && first.output_window.y1 == other.output_window.y1
        && rect_shapes_match(first.step.ll, other.step.ll)
        && rect_shapes_match(first.step.hl, other.step.hl)
        && rect_shapes_match(first.step.lh, other.step.lh)
        && rect_shapes_match(first.step.hh, other.step.hh)
}

pub(super) fn store_shapes_match(first: &J2kDirectStoreStep, other: &J2kDirectStoreStep) -> bool {
    rect_shapes_match(first.input_rect, other.input_rect)
        && first.source_x == other.source_x
        && first.source_y == other.source_y
        && first.copy_width == other.copy_width
        && first.copy_height == other.copy_height
        && first.output_width == other.output_width
        && first.output_height == other.output_height
        && first.output_x == other.output_x
        && first.output_y == other.output_y
        && first.addend.to_bits() == other.addend.to_bits()
}

pub(super) fn direct_preflight_invariant(message: &'static str) -> Error {
    Error::MetalKernel {
        message: format!("internal J2K Metal direct preflight error: {message}"),
    }
}

fn ht_prepared_job_supports_runtime(job: &J2kHtCleanupBatchJob) -> bool {
    if job.width == 0 || job.height == 0 {
        return true;
    }
    job.roi_shift == 0
        && job.output_stride >= job.width
        && crate::ht::supports_metal_ht_geometry(job.width, job.height)
}
