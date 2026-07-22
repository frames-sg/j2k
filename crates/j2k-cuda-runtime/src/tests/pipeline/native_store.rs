// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::cuda_runtime_gate;
use crate::{
    f32_slice_as_bytes, CudaContext, CudaExternalDeviceBufferViewMut, CudaJ2kStoreGray16Job,
    CudaJ2kStoreGray16Target, CudaJ2kStoreGrayI16Target, CudaJ2kStoreRgb8Job,
    CudaJ2kStoreRgb8MctJob, CudaJ2kStoreRgb8MctTarget,
};

#[test]
fn j2k_store_rgb8_mct_batch_writes_external_suballocation_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let plane0 = context
        .upload(f32_slice_as_bytes(&[16.0, 18.0, 21.0, 24.0]))
        .expect("upload external RGB plane 0");
    let plane1 = context
        .upload(f32_slice_as_bytes(&[-3.0, 4.0, 5.0, -6.0]))
        .expect("upload external RGB plane 1");
    let plane2 = context
        .upload(f32_slice_as_bytes(&[2.0, -1.0, 7.0, 3.0]))
        .expect("upload external RGB plane 2");
    let job = CudaJ2kStoreRgb8MctJob {
        store: CudaJ2kStoreRgb8Job {
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
            copy_height: 2,
            output_width: 2,
            output_height: 2,
            output_x: 0,
            output_y: 0,
            addend0: 128.0,
            addend1: 128.0,
            addend2: 128.0,
            bit_depth0: 8,
            bit_depth1: 8,
            bit_depth2: 8,
            rgba: 0,
        },
        irreversible97: 0,
    };
    let target = CudaJ2kStoreRgb8MctTarget {
        plane0: &plane0,
        plane1: &plane1,
        plane2: &plane2,
        job,
    };
    let expected = context
        .j2k_store_rgb8_mct_device(&plane0, &plane1, &plane2, job)
        .expect("owned RGB oracle");
    let mut expected_bytes = vec![0_u8; 12];
    expected
        .buffer()
        .copy_to_host(&mut expected_bytes)
        .expect("download owned RGB oracle");

    let mut allocation = context
        .upload(&[0xA5_u8; 16])
        .expect("external RGB destination");
    let pointer = allocation
        .device_ptr()
        .checked_add(2)
        .expect("external RGB suballocation pointer");
    // SAFETY: bytes 2..14 are a live disjoint subrange and the allocation is
    // retained until the asynchronous completion guard is finished.
    let mut destination = unsafe {
        CudaExternalDeviceBufferViewMut::from_raw_parts(
            &context,
            pointer,
            expected_bytes.len(),
            1,
            &mut allocation,
        )
    }
    .expect("external RGB view");
    // SAFETY: all three source planes and the external destination remain live
    // and unavailable for reuse until `queued.finish()` succeeds below.
    let (ranges, queued) = unsafe {
        context.j2k_store_rgb8_mct_batch_into_external_device_enqueue(&[target], &mut destination)
    }
    .expect("enqueue external RGB store");
    assert_eq!(
        ranges,
        [crate::CudaDeviceBufferRange { offset: 0, len: 12 }]
    );
    queued.finish().expect("finish external RGB store");
    drop(destination);

    let mut actual = [0_u8; 16];
    allocation
        .copy_to_host(&mut actual)
        .expect("download external RGB allocation");
    assert_eq!(&actual[..2], &[0xA5; 2]);
    assert_eq!(&actual[2..14], expected_bytes);
    assert_eq!(&actual[14..], &[0xA5; 2]);
}

#[test]
fn j2k_native_grayscale_batch_store_preserves_unsigned_and_signed_samples_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let unsigned = [0.0_f32, 1.0, 2047.0, 4095.0];
    let signed = [-2048.0_f32, -1.0, 0.0, 2047.0];
    let unsigned = context
        .upload(f32_slice_as_bytes(&unsigned))
        .expect("upload unsigned plane");
    let signed = context
        .upload(f32_slice_as_bytes(&signed))
        .expect("upload signed plane");
    let job = CudaJ2kStoreGray16Job {
        input_width: 2,
        source_x: 0,
        source_y: 0,
        copy_width: 2,
        copy_height: 2,
        output_width: 2,
        output_height: 2,
        output_x: 0,
        output_y: 0,
        addend: 0.0,
        bit_depth: 12,
    };

    let unsigned_output = context
        .j2k_store_gray16_batch_contiguous_device(&[CudaJ2kStoreGray16Target {
            output_index: 0,
            input: &unsigned,
            job,
        }])
        .expect("native Gray16 batch store");
    let signed_output = context
        .j2k_store_grayi16_batch_contiguous_device(&[CudaJ2kStoreGrayI16Target {
            output_index: 0,
            input: &signed,
            job,
        }])
        .expect("native GrayI16 batch store");

    let mut unsigned_bytes = vec![0_u8; 8];
    unsigned_output
        .output()
        .copy_to_host(&mut unsigned_bytes)
        .expect("download unsigned output");
    let unsigned_samples = unsigned_bytes
        .chunks_exact(2)
        .map(|sample| u16::from_le_bytes([sample[0], sample[1]]))
        .collect::<Vec<_>>();
    assert_eq!(unsigned_samples, [0, 1, 2047, 4095]);

    let mut signed_bytes = vec![0_u8; 8];
    signed_output
        .output()
        .copy_to_host(&mut signed_bytes)
        .expect("download signed output");
    let signed_samples = signed_bytes
        .chunks_exact(2)
        .map(|sample| i16::from_le_bytes([sample[0], sample[1]]))
        .collect::<Vec<_>>();
    assert_eq!(signed_samples, [-2048, -1, 0, 2047]);
}
