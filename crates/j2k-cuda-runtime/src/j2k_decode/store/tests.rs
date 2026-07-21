// SPDX-License-Identifier: MIT OR Apache-2.0

mod color_native;
mod zero_init;

use super::{
    batch::{validate_store_batch_launch, STORE_BATCH_GEOMETRY_EXCEEDS_LAUNCH_LIMITS},
    destination::validate_store_destination,
    validation::{
        validate_inverse_mct_overlap_flags, validate_store_context_matches,
        validate_store_plane_layout, INVERSE_MCT_PLANE_OVERLAP, STORE_CONTEXT_MISMATCH,
    },
};
use crate::{
    CudaContext, CudaError, CudaJ2kInverseMctJob, CudaJ2kStoreGray16Job, CudaJ2kStoreGray8Job,
    CudaJ2kStoreRgb16Job, CudaJ2kStoreRgb16MctJob, CudaJ2kStoreRgb8Job, CudaJ2kStoreRgb8MctJob,
    CudaJ2kStoreRgb8MctTarget,
};

macro_rules! zero_gray_store_job {
    ($job:ident) => {
        $job {
            input_width: 0,
            source_x: 0,
            source_y: 0,
            copy_width: 0,
            copy_height: 0,
            output_width: 0,
            output_height: 0,
            output_x: 0,
            output_y: 0,
            addend: 0.0,
            bit_depth: 8,
        }
    };
}

macro_rules! zero_rgb_store_job {
    ($job:ident) => {
        $job {
            input_width0: 0,
            input_width1: 0,
            input_width2: 0,
            source_x0: 0,
            source_y0: 0,
            source_x1: 0,
            source_y1: 0,
            source_x2: 0,
            source_y2: 0,
            copy_width: 0,
            copy_height: 0,
            output_width: 0,
            output_height: 0,
            output_x: 0,
            output_y: 0,
            addend0: 0.0,
            addend1: 0.0,
            addend2: 0.0,
            bit_depth0: 8,
            bit_depth1: 8,
            bit_depth2: 8,
            rgba: 0,
        }
    };
}

fn assert_context_mismatch<T>(result: Result<T, CudaError>) {
    match result {
        Err(CudaError::InvalidArgument { message }) => {
            assert_eq!(message, STORE_CONTEXT_MISMATCH);
        }
        Err(error) => panic!("expected a CUDA context mismatch, got {error}"),
        Ok(_) => panic!("expected a CUDA context mismatch"),
    }
}

fn assert_inverse_mct_overlap<T>(result: Result<T, CudaError>) {
    match result {
        Err(CudaError::InvalidArgument { message }) => {
            assert_eq!(message, INVERSE_MCT_PLANE_OVERLAP);
        }
        Err(error) => panic!("expected inverse MCT plane overlap, got {error}"),
        Ok(_) => panic!("expected inverse MCT plane overlap"),
    }
}

#[test]
fn store_context_validation_accepts_empty_and_matching_inputs() {
    assert!(validate_store_context_matches([]).is_ok());
    assert!(validate_store_context_matches([true, true, true]).is_ok());
}

#[test]
fn store_context_validation_rejects_any_mismatched_input() {
    for matches in [
        [false, true, true],
        [true, false, true],
        [true, true, false],
    ] {
        assert!(matches!(
            validate_store_context_matches(matches),
            Err(CudaError::InvalidArgument { .. })
        ));
    }
}

#[test]
fn inverse_mct_overlap_validation_rejects_each_plane_pair() {
    assert!(validate_inverse_mct_overlap_flags([false; 3]).is_ok());
    for overlaps in [
        [true, false, false],
        [false, true, false],
        [false, false, true],
    ] {
        assert_inverse_mct_overlap(validate_inverse_mct_overlap_flags(overlaps));
    }
}

fn assert_invalid_destination(result: Result<bool, CudaError>, expected_message_fragment: &str) {
    match result {
        Err(CudaError::InvalidArgument { message }) => {
            assert!(
                message.contains(expected_message_fragment),
                "unexpected destination error: {message}"
            );
        }
        Err(error) => panic!("expected an invalid destination error, got {error}"),
        Ok(full_coverage) => {
            panic!("expected an invalid destination error, got full_coverage={full_coverage}")
        }
    }
}

#[test]
fn store_destination_validation_rejects_out_of_bounds_extents() {
    assert_invalid_destination(
        validate_store_destination(8, 6, 7, 0, 2, 1, 1),
        "exceeds output bounds 8x6",
    );
    assert_invalid_destination(
        validate_store_destination(8, 6, 0, 5, 1, 2, 1),
        "exceeds output bounds 8x6",
    );
}

#[test]
fn store_destination_validation_rejects_coordinate_overflow() {
    assert_invalid_destination(
        validate_store_destination(u32::MAX, 1, u32::MAX, 0, 1, 1, 1),
        "x extent overflows u32",
    );
    assert_invalid_destination(
        validate_store_destination(1, u32::MAX, 0, u32::MAX, 1, 1, 1),
        "y extent overflows u32",
    );
}

#[test]
fn store_destination_validation_reports_exact_full_coverage() {
    assert!(validate_store_destination(8, 6, 0, 0, 8, 6, 1).expect("full output store"));
    assert!(validate_store_destination(0, 0, 0, 0, 0, 0, 1).expect("empty output store"));
}

#[test]
fn store_destination_validation_accepts_partial_and_zero_copy_rectangles() {
    assert!(!validate_store_destination(8, 6, 2, 1, 4, 3, 4).expect("partial output store"));
    assert!(!validate_store_destination(8, 6, 8, 6, 0, 0, 1).expect("empty edge store"));
    assert!(!validate_store_destination(8, 6, 3, 2, 0, 4, 1).expect("zero-width store"));
    assert!(!validate_store_destination(8, 6, 2, 4, 6, 0, 1).expect("zero-height store"));
}

#[test]
fn store_destination_validation_rejects_kernel_index_overflow() {
    assert_invalid_destination(
        validate_store_destination(u32::MAX, 2, 0, 0, u32::MAX, 2, 1),
        "copy geometry",
    );
    assert_invalid_destination(
        validate_store_destination(1_500_000_000, 1, 1_499_999_999, 0, 1, 1, 3),
        "destination element index",
    );
}

#[test]
fn store_source_validation_rejects_kernel_index_overflow() {
    let error = validate_store_plane_layout(usize::MAX, u32::MAX, 1, 1, 1, 1)
        .expect_err("source index exceeds CUDA ABI");
    assert!(matches!(error, CudaError::InvalidArgument { .. }));
}

#[test]
fn store_batch_launch_validation_enforces_the_active_job_grid_boundary() {
    assert!(validate_store_batch_launch(1, 65_535).is_ok());
    assert!(validate_store_batch_launch(0, 0).is_ok());
    let error = validate_store_batch_launch(1, 65_536).expect_err("grid y one over");
    match error {
        CudaError::InvalidArgument { message } => assert_eq!(
            message,
            format!(
                "{STORE_BATCH_GEOMETRY_EXCEEDS_LAUNCH_LIMITS}: active jobs=65536, maximum pixels=1"
            )
        ),
        other => panic!("expected invalid store batch launch geometry, got {other}"),
    }
}

#[test]
fn inverse_mct_rejects_aliasing_before_the_zero_length_fast_path() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let plane = context
        .allocate(std::mem::size_of::<f32>())
        .expect("resident plane");
    assert_inverse_mct_overlap(context.j2k_inverse_mct_device(
        &plane,
        &plane,
        &plane,
        CudaJ2kInverseMctJob {
            len: 0,
            irreversible97: 0,
            addend0: 0.0,
            addend1: 0.0,
            addend2: 0.0,
        },
    ));
}

#[test]
fn safe_store_apis_reject_foreign_context_buffers_and_keep_empty_batches_valid() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let context = CudaContext::system_default().expect("launch CUDA context");
    let foreign_context = CudaContext::system_default().expect("foreign CUDA context");
    let foreign_buffer = foreign_context.allocate(0).expect("foreign empty buffer");
    let gray8_job = zero_gray_store_job!(CudaJ2kStoreGray8Job);
    let gray16_job = zero_gray_store_job!(CudaJ2kStoreGray16Job);
    let rgb8_job = zero_rgb_store_job!(CudaJ2kStoreRgb8Job);
    let rgb16_job = zero_rgb_store_job!(CudaJ2kStoreRgb16Job);
    let inverse_mct_job = CudaJ2kInverseMctJob {
        len: 0,
        irreversible97: 0,
        addend0: 0.0,
        addend1: 0.0,
        addend2: 0.0,
    };
    let rgb8_mct_job = CudaJ2kStoreRgb8MctJob {
        store: rgb8_job,
        irreversible97: 0,
    };
    let target = CudaJ2kStoreRgb8MctTarget {
        plane0: &foreign_buffer,
        plane1: &foreign_buffer,
        plane2: &foreign_buffer,
        job: rgb8_mct_job,
    };

    assert_context_mismatch(context.j2k_store_gray8_device(&foreign_buffer, gray8_job));
    assert_context_mismatch(context.j2k_store_gray16_device(&foreign_buffer, gray16_job));
    assert_context_mismatch(context.j2k_inverse_mct_device(
        &foreign_buffer,
        &foreign_buffer,
        &foreign_buffer,
        inverse_mct_job,
    ));
    assert_context_mismatch(context.j2k_store_rgb8_device(
        &foreign_buffer,
        &foreign_buffer,
        &foreign_buffer,
        rgb8_job,
    ));
    assert_context_mismatch(context.j2k_store_rgb16_device(
        &foreign_buffer,
        &foreign_buffer,
        &foreign_buffer,
        rgb16_job,
    ));
    assert_context_mismatch(context.j2k_store_rgb8_mct_device(
        &foreign_buffer,
        &foreign_buffer,
        &foreign_buffer,
        rgb8_mct_job,
    ));
    assert_context_mismatch(context.j2k_store_rgb8_mct_batch_device(&[target]));
    assert_context_mismatch(context.j2k_store_rgb8_mct_batch_contiguous_device(&[target]));
    assert_context_mismatch(context.j2k_store_rgb16_mct_device(
        &foreign_buffer,
        &foreign_buffer,
        &foreign_buffer,
        CudaJ2kStoreRgb16MctJob {
            store: rgb16_job,
            irreversible97: 0,
        },
    ));

    let empty_batch = context
        .j2k_store_rgb8_mct_batch_device(&[])
        .expect("empty store batch");
    assert!(empty_batch.outputs().is_empty());
    assert_eq!(empty_batch.execution().kernel_dispatches(), 0);

    let empty_contiguous_batch = context
        .j2k_store_rgb8_mct_batch_contiguous_device(&[])
        .expect("empty contiguous store batch");
    assert!(empty_contiguous_batch.ranges().is_empty());
    assert_eq!(empty_contiguous_batch.output().byte_len(), 0);
    assert_eq!(empty_contiguous_batch.execution().kernel_dispatches(), 0);
}

#[test]
fn gray_store_rejects_out_of_bounds_destination_before_launch_when_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let input = context.upload_f32(&[17.0]).expect("gray input");
    let error = context
        .j2k_store_gray8_device(
            &input,
            CudaJ2kStoreGray8Job {
                input_width: 1,
                source_x: 0,
                source_y: 0,
                copy_width: 1,
                copy_height: 1,
                output_width: 1,
                output_height: 1,
                output_x: 1,
                output_y: 0,
                addend: 0.0,
                bit_depth: 8,
            },
        )
        .expect_err("out-of-bounds destination");
    assert!(matches!(error, CudaError::InvalidArgument { .. }));
}

#[test]
fn partial_and_zero_copy_gray_outputs_are_zero_initialized_when_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let input = context.upload_f32(&[17.0]).expect("gray input");
    let partial = context
        .j2k_store_gray8_device(
            &input,
            CudaJ2kStoreGray8Job {
                input_width: 1,
                source_x: 0,
                source_y: 0,
                copy_width: 1,
                copy_height: 1,
                output_width: 3,
                output_height: 2,
                output_x: 1,
                output_y: 1,
                addend: 0.0,
                bit_depth: 8,
            },
        )
        .expect("partial gray store");
    let mut partial_bytes = vec![255; 6];
    partial
        .buffer()
        .copy_to_host(&mut partial_bytes)
        .expect("download partial gray store");
    assert_eq!(partial_bytes, [0, 0, 0, 0, 17, 0]);

    let zero_copy = context
        .j2k_store_gray8_device(
            &input,
            CudaJ2kStoreGray8Job {
                input_width: 1,
                source_x: 0,
                source_y: 0,
                copy_width: 0,
                copy_height: 0,
                output_width: 3,
                output_height: 2,
                output_x: 3,
                output_y: 2,
                addend: 0.0,
                bit_depth: 8,
            },
        )
        .expect("zero-copy gray store");
    let mut zero_bytes = vec![255; 6];
    zero_copy
        .buffer()
        .copy_to_host(&mut zero_bytes)
        .expect("download zero-copy gray store");
    assert_eq!(zero_bytes, [0; 6]);
}

#[test]
fn partial_contiguous_rgb_batch_zeroes_unwritten_pixels_when_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let plane0 = context.upload_f32(&[10.0]).expect("plane 0");
    let plane1 = context.upload_f32(&[0.0]).expect("plane 1");
    let plane2 = context.upload_f32(&[0.0]).expect("plane 2");
    let output = context
        .j2k_store_rgb8_mct_batch_contiguous_device(&[CudaJ2kStoreRgb8MctTarget {
            plane0: &plane0,
            plane1: &plane1,
            plane2: &plane2,
            job: CudaJ2kStoreRgb8MctJob {
                store: CudaJ2kStoreRgb8Job {
                    input_width0: 1,
                    input_width1: 1,
                    input_width2: 1,
                    source_x0: 0,
                    source_y0: 0,
                    source_x1: 0,
                    source_y1: 0,
                    source_x2: 0,
                    source_y2: 0,
                    copy_width: 1,
                    copy_height: 1,
                    output_width: 2,
                    output_height: 1,
                    output_x: 1,
                    output_y: 0,
                    addend0: 0.0,
                    addend1: 0.0,
                    addend2: 0.0,
                    bit_depth0: 8,
                    bit_depth1: 8,
                    bit_depth2: 8,
                    rgba: 0,
                },
                irreversible97: 0,
            },
        }])
        .expect("partial contiguous RGB batch store");
    let mut bytes = vec![255; 6];
    output
        .output()
        .copy_to_host(&mut bytes)
        .expect("download partial contiguous RGB store");
    assert_eq!(bytes, [0, 0, 0, 10, 10, 10]);
}
