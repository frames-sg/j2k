// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    error::CudaError,
    j2k_decode::{
        types::CudaJ2kIdwtMultiKernelJob, CudaJ2kIdwtBatchKernelMode, CudaJ2kIdwtJob, CudaJ2kRect,
    },
    kernels::j2k_idwt_tiled_horizontal_block_count,
};

use super::{
    plan_idwt_batch_launch, validate_idwt_batch_launch, validate_idwt_single_launch,
    IDWT_LAUNCH_GEOMETRY_EXCEEDS_LIMITS,
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
    assert!(validate_batch_launch(1, 1, 65_535, mode).is_ok());
    let error = validate_batch_launch(1, 1, 65_536, mode).expect_err("grid y job count one over");
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
fn cooperative_batch_idwt_enforces_shared_line_capacity() {
    for mode in [
        CudaJ2kIdwtBatchKernelMode::Cooperative53,
        CudaJ2kIdwtBatchKernelMode::Cooperative97,
    ] {
        assert!(validate_batch_launch(512, 512, 1, mode).is_ok());
        for (width, height) in [(513, 512), (512, 513)] {
            let error = validate_batch_launch(width, height, 1, mode)
                .expect_err("cooperative line exceeds shared memory");
            match error {
                CudaError::InvalidArgument { message } => assert_eq!(
                    message,
                    format!(
                        "{IDWT_LAUNCH_GEOMETRY_EXCEEDS_LIMITS}: batch jobs=1, maximum={width}x{height}, mode={mode:?}"
                    )
                ),
                other => panic!("expected invalid cooperative IDWT launch geometry, got {other}"),
            }
        }
    }
}

#[test]
fn mixed_orientation_tiled_batch_uses_largest_real_job_grid() {
    let jobs = [kernel_job(768, 512), kernel_job(512, 768)];
    let plan = plan_idwt_batch_launch(&jobs)
        .expect("valid mixed-orientation batch")
        .expect("nonempty launch plan");

    assert_eq!(plan.max_width, 768);
    assert_eq!(plan.max_height, 768);
    assert_eq!(
        plan.tiled_horizontal_blocks,
        5 * 48,
        "512x768 is the largest real tiled grid"
    );
}

fn kernel_job(width: u32, height: u32) -> CudaJ2kIdwtMultiKernelJob {
    CudaJ2kIdwtMultiKernelJob {
        ll_ptr: 0,
        hl_ptr: 0,
        lh_ptr: 0,
        hh_ptr: 0,
        output_ptr: 0,
        job: CudaJ2kIdwtJob {
            rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: width,
                y1: height,
            },
            ll_rect: CudaJ2kRect::default(),
            hl_rect: CudaJ2kRect::default(),
            lh_rect: CudaJ2kRect::default(),
            hh_rect: CudaJ2kRect::default(),
            irreversible97: 1,
        },
        reserved_tail: 0,
    }
}

fn validate_batch_launch(
    max_width: u32,
    max_height: u32,
    job_count: usize,
    mode: CudaJ2kIdwtBatchKernelMode,
) -> Result<(), CudaError> {
    let tiled_horizontal_blocks =
        j2k_idwt_tiled_horizontal_block_count(max_width as usize, max_height as usize)
            .expect("tile count fits usize");
    validate_idwt_batch_launch(
        max_width,
        max_height,
        tiled_horizontal_blocks,
        job_count,
        mode,
    )
}
