// SPDX-License-Identifier: MIT OR Apache-2.0

use std::os::raw::c_uint;

use super::CudaLaunchGeometry;

pub(crate) fn copy_u8_launch_geometry(len: usize) -> Option<CudaLaunchGeometry> {
    x_blocks_launch_geometry(len, 1, COPY_U8_THREADS)
}

pub(super) const COPY_U8_THREADS: usize = 256;
pub(super) const COPY_U8_THREADS_CUDA: c_uint = 256;
#[cfg(feature = "cuda-oxide-copy-u8")]
const CUDA_OXIDE_COPY_U8_PTX: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/cuda_oxide_copy_u8.ptx"));

pub(super) fn x_blocks_launch_geometry(
    work_items: usize,
    grid_y: usize,
    threads_per_block: usize,
) -> Option<CudaLaunchGeometry> {
    if threads_per_block == 0 {
        return None;
    }
    let blocks = c_uint::try_from(work_items.div_ceil(threads_per_block)).ok()?;
    let grid_y = c_uint::try_from(grid_y).ok()?;
    let block_x = c_uint::try_from(threads_per_block).ok()?;
    CudaLaunchGeometry::new((blocks, grid_y, 1), (block_x, 1, 1))
}

pub(crate) fn with_grid_y(base: CudaLaunchGeometry, grid_y: c_uint) -> Option<CudaLaunchGeometry> {
    let (grid_x, _, grid_z) = base.grid();
    CudaLaunchGeometry::new((grid_x, grid_y, grid_z), base.block())
}

pub(crate) fn with_grid_z(base: CudaLaunchGeometry, grid_z: c_uint) -> Option<CudaLaunchGeometry> {
    let (grid_x, grid_y, _) = base.grid();
    CudaLaunchGeometry::new((grid_x, grid_y, grid_z), base.block())
}

#[cfg(feature = "cuda-oxide-copy-u8")]
pub(crate) fn cuda_oxide_copy_u8_ptx() -> &'static [u8] {
    CUDA_OXIDE_COPY_U8_PTX
}
