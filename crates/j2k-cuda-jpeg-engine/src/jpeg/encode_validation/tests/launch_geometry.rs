// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{kernels::CUDA_MAX_GRID_DIM_X, CudaError};

use super::super::{validate_jpeg_encode_batch_launch, JPEG_BATCH_GEOMETRY_EXCEEDS_LAUNCH_LIMITS};

#[test]
fn jpeg_batch_grid_x_accepts_the_exact_boundary_and_rejects_one_over() {
    assert!(validate_jpeg_encode_batch_launch(CUDA_MAX_GRID_DIM_X).is_ok());
    let tile_count = CUDA_MAX_GRID_DIM_X + 1;
    let error = validate_jpeg_encode_batch_launch(tile_count).expect_err("grid x one over");
    match error {
        CudaError::InvalidArgument { message } => assert_eq!(
            message,
            format!("{JPEG_BATCH_GEOMETRY_EXCEEDS_LAUNCH_LIMITS}: tiles={tile_count}")
        ),
        other => panic!("expected invalid JPEG batch launch geometry, got {other}"),
    }
}

#[test]
fn jpeg_batch_grid_rejects_zero_tiles() {
    assert!(validate_jpeg_encode_batch_launch(0).is_err());
}
