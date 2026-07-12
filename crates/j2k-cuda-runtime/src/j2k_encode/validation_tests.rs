// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    bytes::f32_slice_as_bytes,
    context::CudaContext,
    error::CudaError,
    execution::CudaExecutionStats,
    j2k_decode::CudaJ2kStridedInterleavedPixels,
    j2k_encode::{
        validate_encode_buffer_context, validate_quantize_region, CudaJ2kQuantizeJob,
        CudaJ2kQuantizeSubbandRegionJob, CudaJ2kResidentComponents,
    },
};

use super::validation::{validate_encode_context_matches, ENCODE_CONTEXT_MISMATCH};

const QUANTIZATION: CudaJ2kQuantizeJob = CudaJ2kQuantizeJob {
    step_exponent: 0,
    step_mantissa: 0,
    range_bits: 8,
    reversible: true,
};

fn assert_encode_context_mismatch<T>(result: Result<T, CudaError>) {
    match result {
        Err(CudaError::InvalidArgument { message }) => {
            assert_eq!(message, ENCODE_CONTEXT_MISMATCH);
        }
        Err(error) => panic!("expected a CUDA encode context mismatch, got {error}"),
        Ok(_) => panic!("expected a CUDA encode context mismatch"),
    }
}

#[test]
fn encode_context_validation_accepts_empty_and_matching_inputs() {
    assert!(validate_encode_context_matches([]).is_ok());
    assert!(validate_encode_context_matches([true, true, true]).is_ok());
}

#[test]
fn encode_context_validation_rejects_each_mismatched_input_position() {
    for matches in [
        [false, true, true],
        [true, false, true],
        [true, true, false],
    ] {
        assert_encode_context_mismatch(validate_encode_context_matches(matches));
    }
}

#[test]
fn safe_encode_resident_apis_reject_foreign_context_buffers() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let context = CudaContext::system_default().expect("launch CUDA context");
    let foreign_context = CudaContext::system_default().expect("foreign CUDA context");
    let foreign_pixels = foreign_context.allocate(0).expect("foreign pixel buffer");
    let mut foreign_components = CudaJ2kResidentComponents {
        buffer: foreign_context
            .allocate(0)
            .expect("foreign resident components"),
        num_pixels: 0,
        num_components: 3,
        execution: CudaExecutionStats::default(),
    };

    assert_encode_context_mismatch(context.j2k_deinterleave_strided_to_f32_resident(
        CudaJ2kStridedInterleavedPixels {
            buffer: &foreign_pixels,
            byte_offset: 0,
            width: 1,
            height: 1,
            pitch_bytes: 1,
            num_components: 1,
            bit_depth: 8,
            signed: false,
        },
    ));
    assert_encode_context_mismatch(context.j2k_forward_rct_resident(&mut foreign_components));
    assert_encode_context_mismatch(context.j2k_forward_ict_resident(&mut foreign_components));
    assert_encode_context_mismatch(context.j2k_forward_dwt53_resident_component(
        &foreign_components,
        0,
        0,
        0,
        0,
    ));
    assert_encode_context_mismatch(context.j2k_forward_dwt97_resident_component(
        &foreign_components,
        0,
        0,
        0,
        0,
    ));
    assert_encode_context_mismatch(context.j2k_quantize_subband_resident(
        &foreign_pixels,
        0,
        QUANTIZATION,
    ));
    assert_encode_context_mismatch(context.j2k_quantize_subband_region_resident(
        &foreign_pixels,
        CudaJ2kQuantizeSubbandRegionJob {
            x0: 0,
            y0: 0,
            width: 0,
            height: 0,
            stride: 0,
            quantization: QUANTIZATION,
        },
    ));

    let local_buffer = context.allocate(0).expect("local empty buffer");
    assert!(validate_encode_buffer_context(
        &context,
        [&local_buffer, &local_buffer, &local_buffer],
    )
    .is_ok());
    let mut local_components = CudaJ2kResidentComponents {
        buffer: local_buffer,
        num_pixels: 0,
        num_components: 3,
        execution: CudaExecutionStats::default(),
    };
    let execution = context
        .j2k_forward_rct_resident(&mut local_components)
        .expect("same-context empty resident RCT");
    assert_eq!(execution.kernel_dispatches(), 0);
}

#[test]
fn checked_device_copy_range_rejects_invalid_bounds_and_copies_valid_subranges() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let context = CudaContext::system_default().expect("CUDA context");
    let source = context.upload(&[1, 2, 3, 4]).expect("copy source");
    let reversed_start = source.byte_len().saturating_sub(1);
    let reversed_end = reversed_start.saturating_sub(1);

    assert!(matches!(
        context.copy_device_range_to_device_with_kernel(&source, reversed_start..reversed_end),
        Err(CudaError::InvalidArgument { .. })
    ));
    assert!(matches!(
        context.copy_device_range_to_device_with_kernel(&source, 0..5),
        Err(CudaError::OutputTooSmall {
            required: 5,
            have: 4,
        })
    ));

    let copied = context
        .copy_device_range_to_device_with_kernel(&source, 1..3)
        .expect("checked subrange copy");
    let mut bytes = [0u8; 2];
    copied.copy_to_host(&mut bytes).expect("copy readback");
    assert_eq!(bytes, [2, 3]);
}

#[test]
fn resident_dwt_copies_the_checked_component_plane_range() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let context = CudaContext::system_default().expect("CUDA context");
    let components = CudaJ2kResidentComponents {
        buffer: context
            .upload(f32_slice_as_bytes(&[1.0, 2.0]))
            .expect("resident component planes"),
        num_pixels: 1,
        num_components: 2,
        execution: CudaExecutionStats::default(),
    };

    assert!(matches!(
        context.j2k_forward_dwt53_resident_component(&components, 2, 1, 1, 0),
        Err(CudaError::InvalidArgument { .. })
    ));
    let output = context
        .j2k_forward_dwt53_resident_component(&components, 1, 1, 1, 0)
        .expect("second resident DWT component");
    assert_eq!(
        output
            .download_transformed()
            .expect("resident DWT readback"),
        vec![2.0]
    );
}

#[test]
fn empty_region_requires_no_source_storage() {
    let job = CudaJ2kQuantizeSubbandRegionJob {
        x0: u32::MAX,
        y0: u32::MAX,
        width: 0,
        height: 1,
        stride: 0,
        quantization: QUANTIZATION,
    };

    assert!(validate_quantize_region(job, 0).is_ok());
}

#[test]
fn region_validation_reports_exact_required_and_available_bytes() {
    let job = CudaJ2kQuantizeSubbandRegionJob {
        x0: 1,
        y0: 1,
        width: 2,
        height: 2,
        stride: 4,
        quantization: QUANTIZATION,
    };

    assert!(matches!(
        validate_quantize_region(job, 10),
        Err(CudaError::OutputTooSmall {
            required: 44,
            have: 40,
        })
    ));
}

#[test]
fn region_validation_rejects_rows_wider_than_the_stride() {
    let job = CudaJ2kQuantizeSubbandRegionJob {
        x0: 3,
        y0: 0,
        width: 2,
        height: 1,
        stride: 4,
        quantization: QUANTIZATION,
    };

    assert!(matches!(
        validate_quantize_region(job, 8),
        Err(CudaError::LengthTooLarge { len: 8 })
    ));
}
