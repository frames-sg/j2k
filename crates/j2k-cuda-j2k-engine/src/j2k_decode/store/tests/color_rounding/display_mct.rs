// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{CudaContext, CudaJ2kStoreRgb8Job, CudaJ2kStoreRgb8MctJob, CudaJ2kStoreRgb8MctTarget};

mod rgb16;

#[test]
fn irreversible_rgb8_mct_batch_rounds_centered_ties_even_before_shift_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let plane0 = context.upload_f32(&[0.5, -1.5]).expect("plane 0");
    let plane1 = context.upload_f32(&[0.0, 0.0]).expect("plane 1");
    let plane2 = context.upload_f32(&[0.0, 0.0]).expect("plane 2");
    let output = crate::J2kCudaEngine::new(&context)
        .j2k_store_rgb8_mct_batch_contiguous_device(&[CudaJ2kStoreRgb8MctTarget {
            plane0: &plane0,
            plane1: &plane1,
            plane2: &plane2,
            job: CudaJ2kStoreRgb8MctJob {
                store: two_pixel_rgb8_job(),
                irreversible97: 1,
            },
        }])
        .expect("irreversible RGB8 MCT batch store");
    let mut actual = [0_u8; 6];
    output
        .output()
        .copy_to_host(&mut actual)
        .expect("download irreversible RGB8 MCT batch store");
    assert_eq!(actual, [128, 128, 128, 126, 126, 126]);
}

fn two_pixel_rgb8_job() -> CudaJ2kStoreRgb8Job {
    CudaJ2kStoreRgb8Job {
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
        addend0: 128.0,
        addend1: 128.0,
        addend2: 128.0,
        bit_depth0: 8,
        bit_depth1: 8,
        bit_depth2: 8,
        rgba: 0,
    }
}
