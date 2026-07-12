// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    CudaLaunchGeometry, CUDA_MAX_BLOCK_DIM_X_Y, CUDA_MAX_BLOCK_DIM_Z, CUDA_MAX_GRID_DIM_X,
    CUDA_MAX_GRID_DIM_Y_Z, CUDA_MAX_THREADS_PER_BLOCK,
};

#[test]
fn exact_documented_grid_and_block_boundaries_are_valid() {
    assert!(CudaLaunchGeometry::new(
        (
            CUDA_MAX_GRID_DIM_X,
            CUDA_MAX_GRID_DIM_Y_Z,
            CUDA_MAX_GRID_DIM_Y_Z,
        ),
        (16, 1, CUDA_MAX_BLOCK_DIM_Z),
    )
    .is_some());
    assert!(CudaLaunchGeometry::new((1, 1, 1), (CUDA_MAX_BLOCK_DIM_X_Y, 1, 1)).is_some());
    assert!(CudaLaunchGeometry::new((1, 1, 1), (1, CUDA_MAX_BLOCK_DIM_X_Y, 1)).is_some());
    assert!(CudaLaunchGeometry::new((1, 1, 1), (32, 32, 1)).is_some());
}

#[test]
fn zero_or_one_over_grid_axes_are_rejected() {
    for grid in [
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
        (CUDA_MAX_GRID_DIM_X + 1, 1, 1),
        (1, CUDA_MAX_GRID_DIM_Y_Z + 1, 1),
        (1, 1, CUDA_MAX_GRID_DIM_Y_Z + 1),
    ] {
        assert_eq!(CudaLaunchGeometry::new(grid, (1, 1, 1)), None);
    }
}

#[test]
fn zero_one_over_or_oversubscribed_blocks_are_rejected() {
    for block in [
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
        (CUDA_MAX_BLOCK_DIM_X_Y + 1, 1, 1),
        (1, CUDA_MAX_BLOCK_DIM_X_Y + 1, 1),
        (1, 1, CUDA_MAX_BLOCK_DIM_Z + 1),
        (CUDA_MAX_THREADS_PER_BLOCK / 32 + 1, 32, 1),
    ] {
        assert_eq!(CudaLaunchGeometry::new((1, 1, 1), block), None);
    }
}
