// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[cfg(all(feature = "cuda-oxide-copy-u8", j2k_cuda_oxide_copy_u8_built))]
#[test]
fn cuda_oxide_copy_u8_kernel_metadata_matches_generated_ptx() {
    let ptx = cuda_oxide_copy_u8_ptx();
    assert_eq!(ptx.last(), Some(&0));
    let source = std::str::from_utf8(&ptx[..ptx.len() - 1]).expect("ptx utf8");
    assert!(source.contains(".visible .entry j2k_copy_u8("));
    assert_eq!(CudaKernel::CopyU8.entrypoint(), b"j2k_copy_u8\0");
}

#[test]
fn copy_u8_launch_geometry_rounds_up_to_256_thread_blocks() {
    assert_eq!(copy_u8_launch_geometry(0), None);
    assert_eq!(copy_u8_launch_geometry(1).unwrap().grid(), (1, 1, 1));
    assert_eq!(copy_u8_launch_geometry(256).unwrap().grid(), (1, 1, 1));
    assert_eq!(copy_u8_launch_geometry(257).unwrap().grid(), (2, 1, 1));
}

#[test]
fn x_blocks_launch_geometry_rounds_work_items_and_preserves_y_grid() {
    let geometry = x_blocks_launch_geometry(513, 7, COPY_U8_THREADS).unwrap();

    assert_eq!(geometry.grid(), (3, 7, 1));
    assert_eq!(geometry.block(), (COPY_U8_THREADS_CUDA, 1, 1));
}

#[test]
fn x_blocks_launch_geometry_rejects_zero_threads() {
    assert_eq!(x_blocks_launch_geometry(513, 7, 0), None);
}

#[test]
#[cfg(target_pointer_width = "64")]
fn x_blocks_launch_geometry_enforces_static_grid_boundaries() {
    let max_work_items = CUDA_MAX_GRID_DIM_X as usize * COPY_U8_THREADS;
    assert!(copy_u8_launch_geometry(max_work_items).is_some());
    assert_eq!(copy_u8_launch_geometry(max_work_items + 1), None);
    assert!(x_blocks_launch_geometry(1, CUDA_MAX_GRID_DIM_Y_Z as usize, 1).is_some());
    assert_eq!(
        x_blocks_launch_geometry(1, CUDA_MAX_GRID_DIM_Y_Z as usize + 1, 1),
        None
    );
}
