// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    load_i32, max_u32, unsigned_magnitude, J2kHtEncodeJob, J2kHtEncodeMultiInputJob,
    J2kHtEncodeParams,
};

pub(super) fn max_magnitude_serial(
    coefficients: *const i32,
    width: u32,
    height: u32,
    coefficient_stride: u32,
) -> u32 {
    let mut max_magnitude = 0;
    let mut y = 0;
    while y < height {
        let mut x = 0;
        while x < width {
            let magnitude = unsigned_magnitude(load_i32(coefficients, y * coefficient_stride + x));
            max_magnitude = max_u32(max_magnitude, magnitude);
            x += 1;
        }
        y += 1;
    }
    max_magnitude
}

fn coefficient_analysis_serial(
    coefficients: *const i32,
    width: u32,
    height: u32,
    coefficient_stride: u32,
) -> (u32, u32) {
    let mut max_magnitude = 0;
    let mut significant_count = 0;
    let mut y = 0;
    while y < height {
        let mut x = 0;
        while x < width {
            let magnitude = unsigned_magnitude(load_i32(coefficients, y * coefficient_stride + x));
            max_magnitude = max_u32(max_magnitude, magnitude);
            significant_count += u32::from(magnitude >= 4);
            x += 1;
        }
        y += 1;
    }
    (max_magnitude, significant_count)
}

pub(super) fn serial_analysis_for_passes(
    coefficients: *const i32,
    width: u32,
    height: u32,
    coefficient_stride: u32,
    target_coding_passes: u32,
) -> (u32, u32) {
    if target_coding_passes == 3 {
        coefficient_analysis_serial(coefficients, width, height, coefficient_stride)
    } else {
        (
            max_magnitude_serial(coefficients, width, height, coefficient_stride),
            0,
        )
    }
}

#[inline(always)]
pub(super) fn params_from_job(job: J2kHtEncodeJob) -> J2kHtEncodeParams {
    J2kHtEncodeParams {
        width: job.width,
        height: job.height,
        coefficient_stride: job.coefficient_stride,
        total_bitplanes: job.total_bitplanes,
        output_capacity: job.output_capacity,
        target_coding_passes: job.target_coding_passes,
    }
}

#[inline(always)]
pub(super) fn params_from_multi_job(job: J2kHtEncodeMultiInputJob) -> J2kHtEncodeParams {
    J2kHtEncodeParams {
        width: job.width,
        height: job.height,
        coefficient_stride: job.coefficient_stride,
        total_bitplanes: job.total_bitplanes,
        output_capacity: job.output_capacity,
        target_coding_passes: job.target_coding_passes,
    }
}
