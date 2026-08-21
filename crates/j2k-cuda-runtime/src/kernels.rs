// SPDX-License-Identifier: MIT OR Apache-2.0

mod geometry;
mod shared;
#[cfg(test)]
mod tests;

pub use geometry::CudaLaunchGeometry;
#[cfg(test)]
pub(crate) use geometry::{CUDA_MAX_GRID_DIM_X, CUDA_MAX_GRID_DIM_Y_Z};
pub(crate) use shared::copy_u8_launch_geometry;
#[cfg(feature = "cuda-oxide-copy-u8")]
pub(crate) use shared::cuda_oxide_copy_u8_ptx;
#[cfg(test)]
use shared::{x_blocks_launch_geometry, COPY_U8_THREADS, COPY_U8_THREADS_CUDA};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CudaKernel {
    #[cfg_attr(
        all(not(feature = "cuda-oxide-copy-u8"), not(test)),
        expect(
            dead_code,
            reason = "variant is used only by the CopyU8 kernel feature"
        )
    )]
    CopyU8,
}

impl CudaKernel {
    #[cfg_attr(
        all(not(j2k_cuda_oxide_enabled), not(test)),
        expect(
            dead_code,
            reason = "entrypoint lookup is used only when CUDA Oxide modules are built"
        )
    )]
    pub(crate) fn entrypoint(self) -> &'static [u8] {
        match self {
            Self::CopyU8 => b"j2k_copy_u8\0",
        }
    }
}
