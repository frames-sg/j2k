// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn kernel_inventory_forbids_test_only_orphan_entrypoints() {
    let kernel_source = include_str!("../kernels.rs");
    let production_kernel_source = kernel_source
        .split("\n#[cfg(test)]\nmod tests")
        .next()
        .expect("production kernel source");
    assert!(
        !production_kernel_source.contains("#[cfg_attr(not(test), allow(dead_code))]"),
        "production CUDA kernels must not use test-only dead-code exemptions"
    );

    let context_source = include_str!("../context.rs");
    for variant in [
        "J2kIdwtHorizontal",
        "J2kIdwtVertical",
        "Htj2kEncodeCodeblock",
        "J2kInverseDwtSingle",
        "J2kStoreRgb8Mct",
    ] {
        assert!(
            !production_kernel_source.contains(&format!("{variant},")),
            "orphan CUDA kernel variant returned: {variant}"
        );
        assert!(
            !context_source.contains(&format!("{variant},")),
            "test kernel inventory must not retain orphan variant: {variant}"
        );
    }
}

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

#[test]
fn cuda_launch_geometry_policy_is_centralized_and_defensively_enforced() {
    let geometry = include_str!("geometry.rs");
    let geometry_tests = include_str!("geometry/tests.rs");
    let execution = include_str!("../execution.rs");
    assert!(geometry.lines().count() < 100);
    assert!(geometry_tests.lines().count() < 100);
    for required in [
        "CUDA_MAX_GRID_DIM_X",
        "CUDA_MAX_GRID_DIM_Y_Z",
        "CUDA_MAX_BLOCK_DIM_X_Y",
        "CUDA_MAX_BLOCK_DIM_Z",
        "CUDA_MAX_THREADS_PER_BLOCK",
        "pub const fn is_valid",
        "pub const fn grid",
        "pub const fn block",
    ] {
        assert!(geometry.contains(required));
    }
    assert!(!geometry.contains("pub(crate) grid:"));
    assert!(!geometry.contains("pub(crate) block:"));
    let launch = execution
        .split("pub(crate) fn launch_kernel_async")
        .nth(1)
        .expect("launch_kernel_async source");
    let validation = launch
        .find("if !geometry.is_valid()")
        .expect("defensive geometry validation");
    let driver_scope = launch
        .find("with_current_resource_operation")
        .expect("CUDA driver operation scope");
    assert!(validation < driver_scope);
}
