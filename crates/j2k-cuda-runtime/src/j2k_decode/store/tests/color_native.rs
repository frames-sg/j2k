// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{CudaContext, CudaJ2kStoreRgbNativeJob, CudaJ2kStoreRgbNativeTarget};

#[test]
fn exact_native_rgb_batch_preserves_subnative_codes_and_layout_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let plane0 = context.upload_f32(&[1.0, 127.0]).expect("plane 0");
    let plane1 = context.upload_f32(&[3.0, 126.0]).expect("plane 1");
    let plane2 = context.upload_f32(&[5.0, 125.0]).expect("plane 2");
    for (layout, expected) in [
        (0, vec![1_u8, 3, 5, 127, 126, 125]),
        (1, vec![1_u8, 127, 3, 126, 5, 125]),
    ] {
        let output = context
            .j2k_store_rgb8_native_batch_contiguous_device(&[CudaJ2kStoreRgbNativeTarget {
                output_index: 0,
                plane0: &plane0,
                plane1: &plane1,
                plane2: &plane2,
                job: exact_native_rgb_job(7, layout),
            }])
            .expect("exact RGB U8 batch store");
        let mut actual = vec![0_u8; expected.len()];
        output
            .output()
            .copy_to_host(&mut actual)
            .expect("download exact RGB U8");
        assert_eq!(actual, expected);
    }

    let plane0 = context.upload_f32(&[1.0, 4095.0]).expect("plane 0");
    let plane1 = context.upload_f32(&[3.0, 4094.0]).expect("plane 1");
    let plane2 = context.upload_f32(&[5.0, 4093.0]).expect("plane 2");
    for (layout, expected) in [
        (0, vec![1_u16, 3, 5, 4095, 4094, 4093]),
        (1, vec![1_u16, 4095, 3, 4094, 5, 4093]),
    ] {
        let output = context
            .j2k_store_rgb16_native_batch_contiguous_device(&[CudaJ2kStoreRgbNativeTarget {
                output_index: 0,
                plane0: &plane0,
                plane1: &plane1,
                plane2: &plane2,
                job: exact_native_rgb_job(12, layout),
            }])
            .expect("exact RGB U16 batch store");
        let mut bytes = vec![0_u8; expected.len() * std::mem::size_of::<u16>()];
        output
            .output()
            .copy_to_host(&mut bytes)
            .expect("download exact RGB U16");
        let actual = bytes
            .chunks_exact(2)
            .map(|bytes| u16::from_ne_bytes([bytes[0], bytes[1]]))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    let plane0 = context.upload_f32(&[-2048.0, 2047.0]).expect("plane 0");
    let plane1 = context.upload_f32(&[-3.0, 2.0]).expect("plane 1");
    let plane2 = context.upload_f32(&[17.0, -19.0]).expect("plane 2");
    for (layout, expected) in [
        (0, vec![-2048_i16, -3, 17, 2047, 2, -19]),
        (1, vec![-2048_i16, 2047, -3, 2, 17, -19]),
    ] {
        let output = context
            .j2k_store_rgbi16_native_batch_contiguous_device(&[CudaJ2kStoreRgbNativeTarget {
                output_index: 0,
                plane0: &plane0,
                plane1: &plane1,
                plane2: &plane2,
                job: exact_native_rgb_job(12, layout),
            }])
            .expect("exact RGB I16 batch store");
        let mut bytes = vec![0_u8; expected.len() * std::mem::size_of::<i16>()];
        output
            .output()
            .copy_to_host(&mut bytes)
            .expect("download exact RGB I16");
        let actual = bytes
            .chunks_exact(2)
            .map(|bytes| i16::from_ne_bytes([bytes[0], bytes[1]]))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }
}

fn exact_native_rgb_job(bit_depth: u32, layout: u32) -> CudaJ2kStoreRgbNativeJob {
    CudaJ2kStoreRgbNativeJob {
        input_width0: 2,
        input_width1: 2,
        input_width2: 2,
        source_x0: 0,
        source_y0: 0,
        source_x1: 0,
        source_y1: 0,
        source_x2: 0,
        source_y2: 0,
        copy_width: 2,
        copy_height: 1,
        output_width: 2,
        output_height: 1,
        output_x: 0,
        output_y: 0,
        addend0: 0.0,
        addend1: 0.0,
        addend2: 0.0,
        bit_depth0: bit_depth,
        bit_depth1: bit_depth,
        bit_depth2: bit_depth,
        layout,
        transform: 0,
        reserved: 0,
    }
}
