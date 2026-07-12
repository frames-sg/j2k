// SPDX-License-Identifier: MIT OR Apache-2.0

use std::os::raw::c_uint;

/// Static CUDA launch limits for the compute capabilities supported by this
/// runtime. These are the cross-device maxima documented by NVIDIA; keeping
/// them here makes every launch deterministic before the Driver API is called.
pub(crate) const CUDA_MAX_GRID_DIM_X: c_uint = 2_147_483_647;
pub(crate) const CUDA_MAX_GRID_DIM_Y_Z: c_uint = 65_535;
pub(crate) const CUDA_MAX_BLOCK_DIM_X_Y: c_uint = 1_024;
pub(crate) const CUDA_MAX_BLOCK_DIM_Z: c_uint = 64;
pub(crate) const CUDA_MAX_THREADS_PER_BLOCK: c_uint = 1_024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CudaLaunchGeometry {
    grid: (c_uint, c_uint, c_uint),
    block: (c_uint, c_uint, c_uint),
}

impl CudaLaunchGeometry {
    pub(crate) const fn new(
        grid: (c_uint, c_uint, c_uint),
        block: (c_uint, c_uint, c_uint),
    ) -> Option<Self> {
        let geometry = Self { grid, block };
        if geometry.is_valid() {
            Some(geometry)
        } else {
            None
        }
    }

    pub(crate) const fn is_valid(self) -> bool {
        let (grid_x, grid_y, grid_z) = self.grid;
        let (block_x, block_y, block_z) = self.block;
        if grid_x == 0 || grid_y == 0 || grid_z == 0 || block_x == 0 || block_y == 0 || block_z == 0
        {
            return false;
        }
        if grid_x > CUDA_MAX_GRID_DIM_X
            || grid_y > CUDA_MAX_GRID_DIM_Y_Z
            || grid_z > CUDA_MAX_GRID_DIM_Y_Z
        {
            return false;
        }
        if block_x > CUDA_MAX_BLOCK_DIM_X_Y
            || block_y > CUDA_MAX_BLOCK_DIM_X_Y
            || block_z > CUDA_MAX_BLOCK_DIM_Z
        {
            return false;
        }
        block_x * block_y * block_z <= CUDA_MAX_THREADS_PER_BLOCK
    }

    pub(crate) const fn grid(self) -> (c_uint, c_uint, c_uint) {
        self.grid
    }

    pub(crate) const fn block(self) -> (c_uint, c_uint, c_uint) {
        self.block
    }
}

#[cfg(test)]
mod tests;
