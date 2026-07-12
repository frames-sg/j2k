// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    CudaContext, CudaDeviceBuffer, CudaJ2kStoreGray16Job, CudaJ2kStoreRgb16Job,
    CudaJ2kStoreRgb16MctJob, CudaJ2kStoreRgb8Job, CudaJ2kStoreRgb8MctJob,
    CudaJ2kStoreRgb8MctTarget,
};

fn test_context() -> Option<CudaContext> {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return None;
    }
    Some(CudaContext::system_default().expect("CUDA context"))
}

fn assert_u8_output(buffer: &CudaDeviceBuffer, expected: &[u8]) {
    let mut actual = vec![u8::MAX; buffer.byte_len()];
    buffer
        .copy_to_host(&mut actual)
        .expect("download store output");
    assert_eq!(actual, expected);
}

fn assert_u16_output(buffer: &CudaDeviceBuffer, expected: &[u16]) {
    let mut bytes = vec![u8::MAX; buffer.byte_len()];
    buffer
        .copy_to_host(&mut bytes)
        .expect("download 16-bit store output");
    let mut chunks = bytes.chunks_exact(2);
    let actual = chunks
        .by_ref()
        .map(|chunk| u16::from_ne_bytes([chunk[0], chunk[1]]))
        .collect::<Vec<_>>();
    assert!(chunks.remainder().is_empty());
    assert_eq!(actual, expected);
}

fn gray16_job(copy: bool) -> CudaJ2kStoreGray16Job {
    let copy_extent = u32::from(copy);
    CudaJ2kStoreGray16Job {
        input_width: 1,
        source_x: 0,
        source_y: 0,
        copy_width: copy_extent,
        copy_height: copy_extent,
        output_width: 2,
        output_height: 1,
        output_x: if copy { 1 } else { 2 },
        output_y: 0,
        addend: 0.0,
        bit_depth: 16,
    }
}

fn rgb8_job(copy: bool) -> CudaJ2kStoreRgb8Job {
    let copy_extent = u32::from(copy);
    CudaJ2kStoreRgb8Job {
        input_width0: 1,
        input_width1: 1,
        input_width2: 1,
        source_x0: 0,
        source_y0: 0,
        source_x1: 0,
        source_y1: 0,
        source_x2: 0,
        source_y2: 0,
        copy_width: copy_extent,
        copy_height: copy_extent,
        output_width: 2,
        output_height: 1,
        output_x: if copy { 1 } else { 2 },
        output_y: 0,
        addend0: 0.0,
        addend1: 0.0,
        addend2: 0.0,
        bit_depth0: 8,
        bit_depth1: 8,
        bit_depth2: 8,
        rgba: 0,
    }
}

fn rgb16_job(copy: bool) -> CudaJ2kStoreRgb16Job {
    let copy_extent = u32::from(copy);
    CudaJ2kStoreRgb16Job {
        input_width0: 1,
        input_width1: 1,
        input_width2: 1,
        source_x0: 0,
        source_y0: 0,
        source_x1: 0,
        source_y1: 0,
        source_x2: 0,
        source_y2: 0,
        copy_width: copy_extent,
        copy_height: copy_extent,
        output_width: 2,
        output_height: 1,
        output_x: if copy { 1 } else { 2 },
        output_y: 0,
        addend0: 0.0,
        addend1: 0.0,
        addend2: 0.0,
        bit_depth0: 16,
        bit_depth1: 16,
        bit_depth2: 16,
        rgba: 0,
    }
}

#[test]
fn gray16_partial_and_zero_copy_outputs_are_zero_initialized() {
    let Some(context) = test_context() else {
        return;
    };
    let input = context.upload_f32(&[17.0]).expect("gray16 input");

    let partial = context
        .j2k_store_gray16_device(&input, gray16_job(true))
        .expect("partial gray16 store");
    assert_u16_output(partial.buffer(), &[0, 17]);

    let zero_copy = context
        .j2k_store_gray16_device(&input, gray16_job(false))
        .expect("zero-copy gray16 store");
    assert_u16_output(zero_copy.buffer(), &[0, 0]);
}

#[test]
fn rgb8_partial_and_zero_copy_outputs_are_zero_initialized() {
    let Some(context) = test_context() else {
        return;
    };
    let plane0 = context.upload_f32(&[10.0]).expect("RGB8 plane 0");
    let plane1 = context.upload_f32(&[20.0]).expect("RGB8 plane 1");
    let plane2 = context.upload_f32(&[30.0]).expect("RGB8 plane 2");

    let partial = context
        .j2k_store_rgb8_device(&plane0, &plane1, &plane2, rgb8_job(true))
        .expect("partial RGB8 store");
    assert_u8_output(partial.buffer(), &[0, 0, 0, 10, 20, 30]);

    let zero_copy = context
        .j2k_store_rgb8_device(&plane0, &plane1, &plane2, rgb8_job(false))
        .expect("zero-copy RGB8 store");
    assert_u8_output(zero_copy.buffer(), &[0; 6]);
}

#[test]
fn rgb16_partial_and_zero_copy_outputs_are_zero_initialized() {
    let Some(context) = test_context() else {
        return;
    };
    let plane0 = context.upload_f32(&[10.0]).expect("RGB16 plane 0");
    let plane1 = context.upload_f32(&[20.0]).expect("RGB16 plane 1");
    let plane2 = context.upload_f32(&[30.0]).expect("RGB16 plane 2");

    let partial = context
        .j2k_store_rgb16_device(&plane0, &plane1, &plane2, rgb16_job(true))
        .expect("partial RGB16 store");
    assert_u16_output(partial.buffer(), &[0, 0, 0, 10, 20, 30]);

    let zero_copy = context
        .j2k_store_rgb16_device(&plane0, &plane1, &plane2, rgb16_job(false))
        .expect("zero-copy RGB16 store");
    assert_u16_output(zero_copy.buffer(), &[0; 6]);
}

#[test]
fn rgb16_mct_partial_and_zero_copy_outputs_are_zero_initialized() {
    let Some(context) = test_context() else {
        return;
    };
    let plane0 = context.upload_f32(&[10.0]).expect("RGB16 MCT plane 0");
    let plane1 = context.upload_f32(&[0.0]).expect("RGB16 MCT plane 1");
    let plane2 = context.upload_f32(&[0.0]).expect("RGB16 MCT plane 2");

    let partial = context
        .j2k_store_rgb16_mct_device(
            &plane0,
            &plane1,
            &plane2,
            CudaJ2kStoreRgb16MctJob {
                store: rgb16_job(true),
                irreversible97: 0,
            },
        )
        .expect("partial RGB16 MCT store");
    assert_u16_output(partial.buffer(), &[0, 0, 0, 10, 10, 10]);

    let zero_copy = context
        .j2k_store_rgb16_mct_device(
            &plane0,
            &plane1,
            &plane2,
            CudaJ2kStoreRgb16MctJob {
                store: rgb16_job(false),
                irreversible97: 0,
            },
        )
        .expect("zero-copy RGB16 MCT store");
    assert_u16_output(zero_copy.buffer(), &[0; 6]);
}

#[test]
fn noncontiguous_rgb8_mct_partial_and_zero_copy_outputs_are_zero_initialized() {
    let Some(context) = test_context() else {
        return;
    };
    let plane0 = context.upload_f32(&[10.0]).expect("RGB8 MCT plane 0");
    let plane1 = context.upload_f32(&[0.0]).expect("RGB8 MCT plane 1");
    let plane2 = context.upload_f32(&[0.0]).expect("RGB8 MCT plane 2");
    let target = |copy| CudaJ2kStoreRgb8MctTarget {
        plane0: &plane0,
        plane1: &plane1,
        plane2: &plane2,
        job: CudaJ2kStoreRgb8MctJob {
            store: rgb8_job(copy),
            irreversible97: 0,
        },
    };

    let partial = context
        .j2k_store_rgb8_mct_batch_device(&[target(true)])
        .expect("partial non-contiguous RGB8 MCT batch");
    assert_u8_output(&partial.outputs()[0], &[0, 0, 0, 10, 10, 10]);

    let zero_copy = context
        .j2k_store_rgb8_mct_batch_device(&[target(false)])
        .expect("zero-copy non-contiguous RGB8 MCT batch");
    assert_u8_output(&zero_copy.outputs()[0], &[0; 6]);
}
