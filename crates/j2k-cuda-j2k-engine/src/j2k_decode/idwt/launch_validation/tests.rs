// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    error::CudaError, j2k_decode::CudaJ2kIdwtBatchKernelMode, kernels::CUDA_MAX_GRID_DIM_X,
};

use super::{
    validate_idwt_batch_launch, validate_idwt_single_launch, IDWT_LAUNCH_GEOMETRY_EXCEEDS_LIMITS,
};

const MAX_SINGLE_HEIGHT: u32 = 65_535 * 16;

#[test]
fn single_idwt_accepts_exact_grid_y_boundary_and_rejects_one_over() {
    assert!(validate_idwt_single_launch(2, MAX_SINGLE_HEIGHT).is_ok());
    let height = MAX_SINGLE_HEIGHT + 1;
    let error = validate_idwt_single_launch(2, height).expect_err("grid y one over");
    match error {
        CudaError::InvalidArgument { message } => assert_eq!(
            message,
            format!("{IDWT_LAUNCH_GEOMETRY_EXCEEDS_LIMITS}: single 2x{height}")
        ),
        other => panic!("expected invalid single IDWT launch geometry, got {other}"),
    }
}

#[test]
fn batch_idwt_accepts_exact_job_boundary_and_rejects_one_over() {
    let mode = CudaJ2kIdwtBatchKernelMode::Generic;
    assert!(validate_idwt_batch_launch(1, 1, 65_535, mode).is_ok());
    let error =
        validate_idwt_batch_launch(1, 1, 65_536, mode).expect_err("grid y job count one over");
    match error {
        CudaError::InvalidArgument { message } => assert_eq!(
            message,
            format!(
                "{IDWT_LAUNCH_GEOMETRY_EXCEEDS_LIMITS}: batch jobs=65536, maximum=1x1, mode={mode:?}"
            )
        ),
        other => panic!("expected invalid batch IDWT launch geometry, got {other}"),
    }
}

#[test]
fn cooperative_batch_idwt_enforces_the_grid_x_boundary() {
    for mode in [
        CudaJ2kIdwtBatchKernelMode::Cooperative53,
        CudaJ2kIdwtBatchKernelMode::Cooperative97,
    ] {
        assert!(validate_idwt_batch_launch(CUDA_MAX_GRID_DIM_X, 1, 1, mode).is_ok());
        let width = CUDA_MAX_GRID_DIM_X + 1;
        let error =
            validate_idwt_batch_launch(width, 1, 1, mode).expect_err("cooperative grid x one over");
        match error {
            CudaError::InvalidArgument { message } => assert_eq!(
                message,
                format!(
                    "{IDWT_LAUNCH_GEOMETRY_EXCEEDS_LIMITS}: batch jobs=1, maximum={width}x1, mode={mode:?}"
                )
            ),
            other => panic!("expected invalid cooperative IDWT launch geometry, got {other}"),
        }
    }
}
