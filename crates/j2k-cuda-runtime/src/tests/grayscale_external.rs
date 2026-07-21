// SPDX-License-Identifier: MIT OR Apache-2.0

use super::cuda_runtime_gate;
use crate::{
    CudaContext, CudaError, CudaExternalDeviceBufferViewMut, CudaJ2kStoreGray16Job,
    CudaJ2kStoreGray16Target, CudaJ2kStoreGray8Job, CudaJ2kStoreGray8Target,
};

#[test]
fn external_grayscale_batch_store_rejects_bounds_and_device_identity_before_launch_when_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let foreign = CudaContext::system_default().expect("foreign CUDA context");
    let input = context.upload(&[0_u8; 4 * 4]).expect("f32 source bytes");
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
    let target = CudaJ2kStoreGray16Target {
        output_index: 0,
        input: &input,
        job,
    };

    let mut too_small = context.allocate(6).expect("undersized destination");
    let ptr = too_small.device_ptr();
    // SAFETY: `too_small` owns this live allocation and remains exclusively
    // borrowed until validation returns.
    let mut destination = unsafe {
        CudaExternalDeviceBufferViewMut::from_raw_parts(
            &context,
            ptr,
            6,
            std::mem::align_of::<u16>(),
            &mut too_small,
        )
    }
    .expect("undersized external view");
    // SAFETY: preflight rejects the undersized range before launch; both
    // referenced allocations remain live for the complete call.
    let error =
        unsafe { context.j2k_store_gray16_batch_into_external_device(&[target], &mut destination) }
            .expect_err("destination bounds must fail before launch");
    assert!(matches!(error, CudaError::OutputTooSmall { .. }));
    drop(destination);

    let mut foreign_allocation = foreign.allocate(8).expect("foreign destination");
    let foreign_ptr = foreign_allocation.device_ptr();
    // SAFETY: the foreign allocation is live and exclusively borrowed; store
    // validation must reject its distinct CUDA context.
    let mut foreign_destination = unsafe {
        CudaExternalDeviceBufferViewMut::from_raw_parts(
            &foreign,
            foreign_ptr,
            8,
            std::mem::align_of::<u16>(),
            &mut foreign_allocation,
        )
    }
    .expect("foreign external view");
    // SAFETY: context validation rejects the foreign destination before
    // launch; both referenced allocations remain live for the complete call.
    let error = unsafe {
        context.j2k_store_gray16_batch_into_external_device(&[target], &mut foreign_destination)
    }
    .expect_err("foreign destination context must fail before launch");
    assert!(matches!(error, CudaError::InvalidArgument { .. }));
}

#[test]
fn external_grayscale_batch_store_honors_suballocation_offsets_when_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let input = context
        .upload_f32(&[0.0, 1.0, 2.0, 3.0])
        .expect("grayscale source");
    let target = CudaJ2kStoreGray8Target {
        output_index: 0,
        input: &input,
        job: CudaJ2kStoreGray8Job {
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
            bit_depth: 8,
        },
    };
    let mut allocation = context
        .upload(&[0xA5; 12])
        .expect("initialized destination allocation");
    let suballocation_ptr = allocation
        .device_ptr()
        .checked_add(4)
        .expect("suballocation pointer");
    // SAFETY: bytes 4..8 are a live subrange of `allocation` and the owner is
    // exclusively borrowed for the lifetime of this destination view.
    let mut destination = unsafe {
        CudaExternalDeviceBufferViewMut::from_raw_parts(
            &context,
            suballocation_ptr,
            4,
            1,
            &mut allocation,
        )
    }
    .expect("external suballocation view");
    // SAFETY: source and destination allocations remain live and exclusively
    // borrowed through the synchronous completion boundary.
    let (ranges, _) =
        unsafe { context.j2k_store_gray8_batch_into_external_device(&[target], &mut destination) }
            .expect("direct store into suballocation");
    assert_eq!(ranges, [crate::CudaDeviceBufferRange { offset: 0, len: 4 }]);
    drop(destination);

    let mut actual = [0_u8; 12];
    allocation
        .copy_to_host(&mut actual)
        .expect("download full allocation");
    assert_eq!(&actual[..4], &[0xA5; 4]);
    assert_eq!(&actual[4..8], &[0, 1, 2, 3]);
    assert_eq!(&actual[8..], &[0xA5; 4]);
}

#[test]
fn external_batch_enqueue_api_requires_an_unsafe_lifetime_contract() {
    let source = include_str!("../j2k_decode/store/grayscale_batch/api.rs");
    for name in [
        "j2k_store_gray8_batch_into_external_device",
        "j2k_store_gray8_batch_into_external_device_enqueue",
        "j2k_store_gray16_batch_into_external_device",
        "j2k_store_gray16_batch_into_external_device_enqueue",
        "j2k_store_grayi16_batch_into_external_device",
        "j2k_store_grayi16_batch_into_external_device_enqueue",
    ] {
        assert!(
            source.contains(&format!("pub unsafe fn {name}")),
            "{name} must not let safe Rust detach borrowed CUDA allocations from pending work"
        );
    }
}
