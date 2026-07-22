// SPDX-License-Identifier: MIT OR Apache-2.0

use super::cuda_runtime_gate;
use crate::{CudaContext, CudaError, CudaExternalDeviceBufferViewMut};
#[cfg(feature = "cuda-oxide-j2k-ml")]
use crate::{CudaJ2kMlKernelConfig, CudaJ2kMlLayout, CudaJ2kMlNormalization, CudaJ2kMlSample};

#[test]
fn cuda_context_identity_distinguishes_clones_from_independent_contexts_when_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let cloned = context.clone();
    let independent = CudaContext::system_default().expect("independent CUDA context");

    assert!(context.is_same_context(&cloned));
    assert!(!context.is_same_context(&independent));
}

#[test]
fn retained_primary_context_identity_and_release_are_balanced_when_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let first = CudaContext::retain_primary(0).expect("retain primary context");
    let second = CudaContext::retain_primary(0).expect("retain primary context again");
    let owned = CudaContext::system_default().expect("independent owned context");

    assert!(first.is_same_context(&second));
    assert!(!first.is_same_context(&owned));
    assert_eq!(first.device_ordinal(), 0);

    drop(first);
    drop(second);
    let retained_again = CudaContext::retain_primary(0).expect("retain primary after release");
    assert_eq!(retained_again.device_ordinal(), 0);
}

#[test]
fn context_diagnostics_track_runtime_owned_allocations_and_h2d_when_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let before = context
        .diagnostics()
        .expect("diagnostics before allocations");
    let uploaded = context.upload(&[1_u8, 2, 3, 4]).expect("upload");
    let allocated = context.allocate(16).expect("allocation");
    let live = context
        .diagnostics()
        .expect("diagnostics with live buffers");

    assert_eq!(
        live.host_to_device_operations
            .saturating_sub(before.host_to_device_operations),
        1
    );
    assert_eq!(
        live.host_to_device_bytes
            .saturating_sub(before.host_to_device_bytes),
        4
    );
    assert_eq!(
        live.device_allocation_operations
            .saturating_sub(before.device_allocation_operations),
        2
    );
    assert_eq!(
        live.device_allocation_bytes
            .saturating_sub(before.device_allocation_bytes),
        20
    );
    assert_eq!(
        live.live_device_allocations
            .saturating_sub(before.live_device_allocations),
        2
    );
    assert_eq!(
        live.live_device_bytes
            .saturating_sub(before.live_device_bytes),
        20
    );
    assert!(live.peak_live_device_allocations >= live.live_device_allocations);
    assert!(live.peak_live_device_bytes >= live.live_device_bytes);

    drop(uploaded);
    drop(allocated);
    let released = context
        .diagnostics()
        .expect("diagnostics after releasing buffers");
    assert_eq!(
        released.live_device_allocations,
        before.live_device_allocations
    );
    assert_eq!(released.live_device_bytes, before.live_device_bytes);
}

#[cfg(all(feature = "cuda-oxide-copy-u8", j2k_cuda_oxide_copy_u8_built))]
#[test]
fn context_diagnostics_track_successful_kernel_submissions_when_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let before = context.diagnostics().expect("diagnostics before copy");
    let output = context
        .copy_with_kernel(&[1_u8, 2, 3, 4])
        .expect("kernel copy");
    let after = context.diagnostics().expect("diagnostics after copy");

    assert_eq!(
        after.kernel_launches.saturating_sub(before.kernel_launches),
        1
    );
    assert_eq!(
        after
            .context_host_synchronizations
            .saturating_sub(before.context_host_synchronizations),
        1
    );
    drop(output);
}

#[test]
fn external_cuda_view_rejects_foreign_context_and_never_owns_memory_when_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let foreign = CudaContext::system_default().expect("foreign CUDA context");
    let mut allocation = context.allocate(16).expect("device allocation");
    let ptr = allocation.device_ptr();
    let len = allocation.byte_len();

    // SAFETY: `allocation` owns the live range and is exclusively borrowed by
    // the view for the duration of this scope.
    let view = unsafe {
        CudaExternalDeviceBufferViewMut::from_raw_parts(&context, ptr, len, 4, &mut allocation)
    }
    .expect("external view");
    assert_eq!(view.device_ptr(), ptr);
    assert_eq!(view.byte_len(), 16);
    drop(view);

    // The external view has no ownership: the original allocation remains
    // live and is still responsible for freeing its memory.
    assert_eq!(allocation.device_ptr(), ptr);
    assert_eq!(allocation.byte_len(), 16);

    // SAFETY: the allocation remains live and exclusively borrowed, but the
    // deliberately foreign context must reject its pointer identity.
    let error = unsafe {
        CudaExternalDeviceBufferViewMut::from_raw_parts(&foreign, ptr, len, 4, &mut allocation)
    }
    .expect_err("foreign context must fail");
    assert!(matches!(error, CudaError::InvalidArgument { .. }));
}

#[cfg(feature = "cuda-oxide-j2k-ml")]
#[test]
fn j2k_ml_external_destination_checks_batch_offsets_before_launch_when_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let source = context.upload(&[1, 2, 3, 4]).expect("source upload");
    let mut allocation = context.allocate(4).expect("destination allocation");
    let ptr = allocation.device_ptr();
    let len = allocation.byte_len();
    // SAFETY: `allocation` owns this live four-byte range and the view holds
    // its exclusive borrow until validation returns.
    let mut destination = unsafe {
        CudaExternalDeviceBufferViewMut::from_raw_parts(&context, ptr, len, 1, &mut allocation)
    }
    .expect("external view");

    let error = context
        .j2k_ml_convert_into_external(
            source.device_ptr(),
            source.byte_len(),
            &mut destination,
            CudaJ2kMlKernelConfig {
                width: 2,
                height: 2,
                channels: 1,
                sample: CudaJ2kMlSample::U8,
                layout: CudaJ2kMlLayout::ChannelsFirst,
                destination_offset_elements: 1,
                normalization: CudaJ2kMlNormalization::Integer,
            },
        )
        .expect_err("offset must exceed destination bounds");
    assert!(matches!(error, CudaError::OutputTooSmall { .. }));
    drop(destination);
    assert_eq!(allocation.device_ptr(), ptr);
}
