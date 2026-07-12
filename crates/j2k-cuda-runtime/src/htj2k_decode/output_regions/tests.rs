// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

fn job(output_offset: u32, width: u32, height: u32, output_stride: u32) -> CudaHtj2kCodeBlockJob {
    CudaHtj2kCodeBlockJob {
        payload_offset: 0,
        width,
        height,
        payload_len: 0,
        cleanup_length: 0,
        refinement_length: 0,
        missing_bit_planes: 0,
        num_bitplanes: 1,
        number_of_coding_passes: 1,
        output_stride,
        output_offset,
        dequantization_step: 1.0,
        stripe_causal: false,
    }
}

#[test]
fn accepts_disjoint_rectangles_in_one_output_plane() {
    let jobs = [job(0, 4, 3, 8), job(4, 4, 3, 8), job(24, 8, 1, 8)];
    validate_disjoint_htj2k_job_outputs_with_live_bytes(&jobs, 32, 0)
        .expect("disjoint output rectangles");
}

#[test]
fn rejects_overlapping_rectangles_and_row_wrap() {
    let overlap = [job(0, 5, 2, 8), job(4, 4, 2, 8)];
    assert!(matches!(
        validate_disjoint_htj2k_job_outputs_with_live_bytes(&overlap, 16, 0),
        Err(CudaError::InvalidArgument { .. })
    ));

    assert!(matches!(
        validate_disjoint_htj2k_job_outputs_with_live_bytes(&[job(6, 4, 1, 8)], 16, 0),
        Err(CudaError::InvalidArgument { .. })
    ));
}

#[test]
fn accepts_disjoint_mixed_strides_and_rejects_overlapping_spans() {
    let disjoint = [job(0, 4, 1, 8), job(16, 4, 1, 16)];
    validate_disjoint_htj2k_job_outputs_with_live_bytes(&disjoint, 32, 0)
        .expect("disjoint mixed strides");

    let overlapping_spans = [job(0, 2, 2, 8), job(4, 2, 2, 4)];
    assert!(matches!(
        validate_disjoint_htj2k_job_outputs_with_live_bytes(&overlapping_spans, 16, 0),
        Err(CudaError::InvalidArgument { .. })
    ));
}

#[test]
fn accepts_large_disjoint_grid_without_quadratic_pair_scanning() {
    const STRIDE: u32 = 256;
    const ROWS: u32 = 128;
    let jobs = (0..STRIDE * ROWS)
        .map(|offset| job(offset, 1, 1, STRIDE))
        .collect::<Vec<_>>();

    validate_disjoint_htj2k_job_outputs_with_live_bytes(&jobs, jobs.len(), 0)
        .expect("large disjoint output grid");
}

#[test]
fn rejects_overlap_hidden_after_many_disjoint_rectangles() {
    const STRIDE: u32 = 128;
    const ROWS: u32 = 64;
    let mut jobs = (0..STRIDE * ROWS)
        .map(|offset| job(offset, 1, 1, STRIDE))
        .collect::<Vec<_>>();
    jobs.push(job(STRIDE * ROWS - 1, 1, 1, STRIDE));

    assert!(matches!(
        validate_disjoint_htj2k_job_outputs_with_live_bytes(&jobs, (STRIDE * ROWS) as usize, 0,),
        Err(CudaError::InvalidArgument { .. })
    ));
}

#[test]
fn accepts_rectangles_that_only_touch_on_row_or_column_boundaries() {
    let jobs = [job(0, 4, 1, 8), job(4, 4, 1, 8), job(8, 4, 1, 8)];
    validate_disjoint_htj2k_job_outputs_with_live_bytes(&jobs, 16, 0)
        .expect("touching output rectangles are disjoint");
}

#[test]
fn repeated_column_start_after_expiry_keeps_new_interval_indexed() {
    let jobs = [
        job(0, 2, 1, 8),
        job(8, 2, 2, 8),
        job(10, 2, 2, 8),
        job(17, 2, 1, 8),
    ];

    assert!(matches!(
        validate_disjoint_htj2k_job_outputs_with_live_bytes(&jobs, 32, 0),
        Err(CudaError::InvalidArgument { .. })
    ));
}

#[test]
fn coverage_planning_rejects_overlap_even_when_areas_sum_to_output() {
    let jobs = [job(0, 4, 1, 8), job(0, 4, 1, 8)];
    assert!(matches!(
        validate_htj2k_output_layout(&jobs, 8),
        Err(CudaError::InvalidArgument { .. })
    ));
}

#[test]
fn coverage_planning_distinguishes_full_output_from_gaps() {
    let full = [job(0, 4, 1, 8), job(4, 4, 1, 8)];
    let full_layout = validate_htj2k_output_layout(&full, 8).expect("full output layout");
    assert_eq!(full_layout.output_bytes, 8 * std::mem::size_of::<f32>());
    assert!(!full_layout.needs_zero_fill);

    let partial = [job(0, 3, 1, 8), job(4, 4, 1, 8)];
    let partial_layout = validate_htj2k_output_layout(&partial, 8).expect("partial output layout");
    assert!(partial_layout.needs_zero_fill);
}
