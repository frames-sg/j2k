// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::CudaError;

use super::super::{
    validate_forward_dwt_request, FORWARD_DWT_GEOMETRY_EXCEEDS_LAUNCH_LIMITS,
    FORWARD_DWT_LEVELS_EXCEED_GEOMETRY,
};

const MAX_DWT_HEIGHT: u32 = 65_535 * 16;

#[test]
fn forward_dwt_launch_geometry_accepts_the_exact_grid_y_boundary() {
    assert!(validate_forward_dwt_request(2, MAX_DWT_HEIGHT, 1).is_ok());
    assert!(validate_forward_dwt_request(2, MAX_DWT_HEIGHT + 1, 0).is_ok());
}

#[test]
fn forward_dwt_launch_geometry_rejects_one_grid_row_over() {
    let height = MAX_DWT_HEIGHT + 1;
    let error = validate_forward_dwt_request(2, height, 1)
        .expect_err("forward DWT grid y must fit CUDA limits");
    match error {
        CudaError::InvalidArgument { message } => assert_eq!(
            message,
            format!("{FORWARD_DWT_GEOMETRY_EXCEEDS_LAUNCH_LIMITS}: 2x{height}")
        ),
        other => panic!("expected invalid DWT launch geometry, got {other}"),
    }
}

#[test]
fn decomposition_level_error_precedes_launch_geometry_error() {
    let height = MAX_DWT_HEIGHT + 1;
    let error = validate_forward_dwt_request(2, height, 2)
        .expect_err("level ceiling must be checked before launch geometry");
    match error {
        CudaError::InvalidArgument { message } => assert_eq!(
            message,
            format!("{FORWARD_DWT_LEVELS_EXCEED_GEOMETRY}: requested 2, maximum 1 for 2x{height}")
        ),
        other => panic!("expected invalid DWT level geometry, got {other}"),
    }
}
