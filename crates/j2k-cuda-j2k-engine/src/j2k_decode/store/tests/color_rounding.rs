// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{CudaContext, CudaJ2kStoreRgbNativeJob, CudaJ2kStoreRgbNativeTarget};

use super::color_native::exact_native_rgb_job;

mod display_mct;
mod rgba;

#[test]
fn irreversible_native_rgb_rounds_centered_ties_even_before_shift_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let plane0 = context.upload_f32(&[0.5, -1.5]).expect("plane 0");
    let plane1 = context.upload_f32(&[0.0, 0.0]).expect("plane 1");
    let plane2 = context.upload_f32(&[0.0, 0.0]).expect("plane 2");
    let mut job: CudaJ2kStoreRgbNativeJob = exact_native_rgb_job(8, 0);
    job.input_width0 = 2;
    job.input_width1 = 2;
    job.input_width2 = 2;
    job.copy_width = 2;
    job.output_width = 2;
    job.addend0 = 128.0;
    job.addend1 = 128.0;
    job.addend2 = 128.0;
    job.transform = 2;

    let output = crate::J2kCudaEngine::new(&context)
        .j2k_store_rgb8_native_batch_contiguous_device(&[CudaJ2kStoreRgbNativeTarget {
            output_index: 0,
            plane0: &plane0,
            plane1: &plane1,
            plane2: &plane2,
            job,
        }])
        .expect("irreversible RGB U8 batch store");
    let mut actual = [0_u8; 6];
    output
        .output()
        .copy_to_host(&mut actual)
        .expect("download irreversible RGB U8");
    assert_eq!(&actual[..3], &[128, 128, 128]);
    assert_eq!(&actual[3..], &[126, 126, 126]);
}
