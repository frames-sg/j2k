// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{CudaContext, CudaJ2kStoreRgb16Job, CudaJ2kStoreRgb16MctJob};

#[test]
fn irreversible_rgb16_mct_rounds_centered_ties_even_before_shift_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let plane0 = context.upload_f32(&[0.5, -1.5]).expect("plane 0");
    let plane1 = context.upload_f32(&[0.0, 0.0]).expect("plane 1");
    let plane2 = context.upload_f32(&[0.0, 0.0]).expect("plane 2");
    let output = crate::J2kCudaEngine::new(&context)
        .j2k_store_rgb16_mct_device(
            &plane0,
            &plane1,
            &plane2,
            CudaJ2kStoreRgb16MctJob {
                store: two_pixel_rgb16_job(),
                irreversible97: 1,
            },
        )
        .expect("irreversible RGB16 MCT store");
    let mut bytes = [0_u8; 12];
    output
        .buffer()
        .copy_to_host(&mut bytes)
        .expect("download irreversible RGB16 MCT store");
    let actual = bytes
        .chunks_exact(2)
        .map(|sample| u16::from_ne_bytes([sample[0], sample[1]]))
        .collect::<Vec<_>>();
    assert_eq!(actual, [128, 128, 128, 126, 126, 126]);
}

fn two_pixel_rgb16_job() -> CudaJ2kStoreRgb16Job {
    CudaJ2kStoreRgb16Job {
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
        bit_depth0: 16,
        bit_depth1: 16,
        bit_depth2: 16,
        rgba: 0,
    }
}
