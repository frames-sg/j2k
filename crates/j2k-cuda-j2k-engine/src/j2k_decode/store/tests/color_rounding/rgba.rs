// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{CudaContext, CudaJ2kStoreRgbaNativeJob, CudaJ2kStoreRgbaNativeTarget};

#[test]
fn irreversible_native_rgba_rounds_rgb_and_alpha_before_shift_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let plane0 = context.upload_f32(&[0.5]).expect("plane 0");
    let plane1 = context.upload_f32(&[0.0]).expect("plane 1");
    let plane2 = context.upload_f32(&[0.0]).expect("plane 2");
    let plane3 = context.upload_f32(&[2.5]).expect("plane 3");
    let output = crate::J2kCudaEngine::new(&context)
        .j2k_store_rgba8_native_batch_contiguous_device(&[CudaJ2kStoreRgbaNativeTarget {
            output_index: 0,
            plane0: &plane0,
            plane1: &plane1,
            plane2: &plane2,
            plane3: &plane3,
            job: irreversible_rgba_job(),
        }])
        .expect("irreversible RGBA U8 batch store");
    let mut actual = [0_u8; 4];
    output
        .output()
        .copy_to_host(&mut actual)
        .expect("download irreversible RGBA U8");
    assert_eq!(actual, [128, 128, 128, 130]);
}

fn irreversible_rgba_job() -> CudaJ2kStoreRgbaNativeJob {
    CudaJ2kStoreRgbaNativeJob {
        input_width0: 1,
        input_width1: 1,
        input_width2: 1,
        input_width3: 1,
        source_x0: 0,
        source_y0: 0,
        source_x1: 0,
        source_y1: 0,
        source_x2: 0,
        source_y2: 0,
        source_x3: 0,
        source_y3: 0,
        copy_width: 1,
        copy_height: 1,
        output_width: 1,
        output_height: 1,
        output_x: 0,
        output_y: 0,
        addend0: 128.0,
        addend1: 128.0,
        addend2: 128.0,
        addend3: 128.0,
        bit_depth0: 8,
        bit_depth1: 8,
        bit_depth2: 8,
        bit_depth3: 8,
        layout: 0,
        transform: 2,
        reserved: 0,
    }
}
