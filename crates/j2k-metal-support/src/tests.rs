// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(target_os = "macos")]

use crate::{
    allocation::{checked_buffer_allocation_length, checked_texture_planned_bytes},
    buffer_access::checked_buffer_typed_range,
    checked_blit_command_encoder, checked_buffer_fill_bytes, checked_buffer_read_vec,
    checked_buffer_write, checked_command_buffer, checked_command_queue,
    checked_compute_command_encoder, checked_private_buffer, checked_shared_buffer_for_len,
    checked_shared_buffer_with_slice, checked_texture, commit_and_wait, one_d_threads_per_group,
    runtime::{classify_command_buffer_status, CommandBufferCompletion},
    system_default_device, two_d_threads_per_group, MetalSupportError,
};
use objc2::rc::autoreleasepool;
use objc2_metal::{
    MTLBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLTexture, MTLTextureDescriptor,
};
mod resident;

#[test]
fn command_buffer_status_classification_covers_every_metal_state() {
    use MTLCommandBufferStatus as Status;

    assert_eq!(
        classify_command_buffer_status(Status::Completed),
        CommandBufferCompletion::Completed
    );
    assert_eq!(
        classify_command_buffer_status(Status::Error),
        CommandBufferCompletion::Failed
    );
    for status in [
        Status::NotEnqueued,
        Status::Enqueued,
        Status::Committed,
        Status::Scheduled,
    ] {
        assert_eq!(
            classify_command_buffer_status(status),
            CommandBufferCompletion::Incomplete,
            "{status:?}"
        );
    }
}

#[derive(Clone, Copy)]
struct ZeroSizedAbi;

// SAFETY: This intentionally invalid zero-sized ABI implementation exists only
// to prove that the Metal validators reject zero-sized types.
unsafe impl j2k_core::accelerator::GpuAbi for ZeroSizedAbi {
    const NAME: &'static str = "ZeroSizedAbi";
}

#[test]
fn two_d_threads_per_group_clamps_empty_pipeline_limits() {
    let threads = two_d_threads_per_group(0, 0);
    assert_eq!((threads.width, threads.height, threads.depth), (1, 1, 1));
}

#[test]
fn one_d_threads_per_group_clamps_empty_pipeline_width() {
    let threads = one_d_threads_per_group(0);
    assert_eq!((threads.width, threads.height, threads.depth), (1, 1, 1));
}

#[test]
fn two_d_threads_per_group_preserves_simd_width_and_derives_height() {
    let threads = two_d_threads_per_group(32, 1024);
    assert_eq!((threads.width, threads.height, threads.depth), (32, 32, 1));
}

#[test]
fn buffer_allocation_length_enforces_device_and_process_caps() {
    assert_eq!(
        checked_buffer_allocation_length(64, 64).expect("exact device cap"),
        64
    );
    assert!(matches!(
        checked_buffer_allocation_length(65, 64),
        Err(MetalSupportError::BufferAllocationTooLarge {
            requested: 65,
            cap: 64,
        })
    ));

    let process_cap = j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;
    assert!(matches!(
        checked_buffer_allocation_length(process_cap + 1, usize::MAX),
        Err(MetalSupportError::BufferAllocationTooLarge { requested, cap })
            if requested == process_cap + 1 && cap == process_cap
    ));
}

#[test]
fn checked_allocators_reject_zero_sized_abi_and_keep_empty_buffers_real() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let Ok(device) = system_default_device() else {
        j2k_test_support::metal_device_unavailable_is_skip(module_path!());
        return;
    };

    assert!(matches!(
        checked_shared_buffer_for_len::<ZeroSizedAbi>(&device, 0),
        Err(MetalSupportError::BufferZeroSizedType {
            abi_name: "ZeroSizedAbi"
        })
    ));
    let private = checked_private_buffer(&device, 0).expect("empty private placeholder");
    assert_eq!(private.length(), 1);

    // SAFETY: The accessor checks the private storage mode before obtaining or
    // dereferencing the buffer's CPU contents pointer.
    let error = unsafe { checked_buffer_read_vec::<u8>(&private, 0, 1) }
        .expect_err("private storage is not CPU-visible");
    assert_eq!(error, MetalSupportError::BufferContentsUnavailable);
}

#[test]
fn checked_texture_returns_a_real_texture() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let Ok(device) = system_default_device() else {
        j2k_test_support::metal_device_unavailable_is_skip(module_path!());
        return;
    };
    let descriptor = MTLTextureDescriptor::new();
    // SAFETY: This fresh descriptor is confined to the current thread, and the
    // dimensions are valid nonzero texture geometry.
    unsafe {
        descriptor.setWidth(2);
        descriptor.setHeight(3);
        descriptor.setDepth(1);
    }
    let texture = checked_texture(&device, &descriptor).expect("bounded texture allocation");
    assert_eq!((texture.width(), texture.height()), (2, 3));
}

#[test]
fn texture_planned_bytes_enforce_exact_repository_cap() {
    let cap = j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;
    assert_eq!(checked_texture_planned_bytes(cap), Ok(cap));
    assert_eq!(
        checked_texture_planned_bytes(cap + 1),
        Err(MetalSupportError::TextureAllocationTooLarge {
            requested: cap + 1,
            cap,
        })
    );
    assert_eq!(
        checked_texture_planned_bytes(0),
        Err(MetalSupportError::TextureDescriptorInvalid {
            reason: "device reported a zero-byte texture allocation plan",
        })
    );
}

#[test]
fn checked_texture_rejects_zero_geometry_before_allocation() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let Ok(device) = system_default_device() else {
        j2k_test_support::metal_device_unavailable_is_skip(module_path!());
        return;
    };
    let descriptor = MTLTextureDescriptor::new();
    // SAFETY: This fresh descriptor is confined to the current thread. Zero is
    // deliberately set so repository validation can reject it before allocation.
    unsafe {
        descriptor.setWidth(0);
        descriptor.setHeight(3);
        descriptor.setDepth(1);
    }

    assert!(matches!(
        checked_texture(&device, &descriptor),
        Err(MetalSupportError::TextureDescriptorInvalid {
            reason: "width, height, depth, and array length must be nonzero",
        })
    ));
}

#[test]
fn commit_and_wait_accepts_unlabeled_command_buffer() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let Ok(device) = system_default_device() else {
        j2k_test_support::metal_device_unavailable_is_skip(module_path!());
        return;
    };
    let queue = checked_command_queue(&device).expect("Metal command queue");
    let command_buffer = checked_command_buffer(&queue).expect("Metal command buffer");
    commit_and_wait(&command_buffer).expect("unlabeled command buffer completion");
}

#[test]
fn checked_command_resources_create_real_encoders() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let Ok(device) = system_default_device() else {
        j2k_test_support::metal_device_unavailable_is_skip(module_path!());
        return;
    };
    let queue = checked_command_queue(&device).expect("Metal command queue");
    let command_buffer = checked_command_buffer(&queue).expect("Metal command buffer");
    let compute = checked_compute_command_encoder(&command_buffer).expect("Metal compute encoder");
    compute.endEncoding();
    let blit = checked_blit_command_encoder(&command_buffer).expect("Metal blit encoder");
    blit.endEncoding();
    commit_and_wait(&command_buffer).expect("empty encoder completion");
}

#[test]
fn checked_command_resources_survive_their_creation_autorelease_pool() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let Ok(device) = system_default_device() else {
        j2k_test_support::metal_device_unavailable_is_skip(module_path!());
        return;
    };
    let queue = checked_command_queue(&device).expect("Metal command queue");
    let command_buffer =
        autoreleasepool(|_| checked_command_buffer(&queue).expect("retained Metal command buffer"));
    let compute = autoreleasepool(|_| {
        checked_compute_command_encoder(&command_buffer).expect("retained Metal compute encoder")
    });
    compute.endEncoding();
    let blit = autoreleasepool(|_| {
        checked_blit_command_encoder(&command_buffer).expect("retained Metal blit encoder")
    });
    blit.endEncoding();

    commit_and_wait(&command_buffer).expect("retained command resources remain valid");
}

#[test]
fn buffer_readback_copies_typed_shared_buffer_values() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let Ok(device) = system_default_device() else {
        j2k_test_support::metal_device_unavailable_is_skip(module_path!());
        return;
    };
    let buffer = checked_shared_buffer_with_slice(&device, &[3_u32, 5, 8, 13])
        .expect("bounded shared upload");
    // SAFETY: CPU initialization is complete and no GPU work uses the buffer.
    let values =
        unsafe { checked_buffer_read_vec::<u32>(&buffer, 0, 4) }.expect("checked readback");
    assert_eq!(values, [3, 5, 8, 13]);
}

#[test]
fn buffer_write_and_fill_copy_into_shared_buffer() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let Ok(device) = system_default_device() else {
        j2k_test_support::metal_device_unavailable_is_skip(module_path!());
        return;
    };
    let buffer = checked_shared_buffer_for_len::<u32>(&device, 3).expect("bounded shared buffer");
    // SAFETY: The never-submitted buffer is accessed sequentially by the CPU.
    unsafe {
        checked_buffer_fill_bytes(&buffer, 0, 12, 0).expect("checked fill");
        checked_buffer_write::<u32>(&buffer, 0, &[21, 34, 55]).expect("checked write");
    }
    // SAFETY: CPU writes are complete and no GPU command can race the read.
    let values =
        unsafe { checked_buffer_read_vec::<u32>(&buffer, 0, 3) }.expect("checked readback");
    assert_eq!(values, [21, 34, 55]);
}

#[test]
fn checked_buffer_readback_rejects_out_of_bounds_range() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let Ok(device) = system_default_device() else {
        j2k_test_support::metal_device_unavailable_is_skip(module_path!());
        return;
    };
    let buffer =
        checked_shared_buffer_with_slice(&device, &[1_u32]).expect("bounded shared upload");
    // SAFETY: Validation rejects the range before any access.
    let err = unsafe { checked_buffer_read_vec::<u32>(&buffer, 0, 2) }.expect_err("bounds error");
    assert!(matches!(
        err,
        MetalSupportError::BufferBounds {
            offset_bytes: 0,
            byte_len: 8,
            buffer_len: 4,
        }
    ));
}

#[test]
fn checked_buffer_readback_rejects_unaligned_range() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let Ok(device) = system_default_device() else {
        j2k_test_support::metal_device_unavailable_is_skip(module_path!());
        return;
    };
    let buffer =
        checked_shared_buffer_with_slice(&device, &[1_u32, 2]).expect("bounded shared upload");
    // SAFETY: Validation rejects the range before any access.
    let err =
        unsafe { checked_buffer_read_vec::<u32>(&buffer, 1, 1) }.expect_err("alignment error");
    assert!(matches!(
        err,
        MetalSupportError::BufferAlignment {
            offset_bytes: 1,
            align: 4,
        }
    ));
}

#[test]
fn typed_range_rejects_overflow_and_zero_sized_abi() {
    let overflow = checked_buffer_typed_range::<u32>(usize::MAX, 0, usize::MAX)
        .expect_err("element byte length overflow");
    assert!(matches!(
        overflow,
        MetalSupportError::BufferBounds {
            offset_bytes: 0,
            byte_len: usize::MAX,
            buffer_len: usize::MAX,
        }
    ));
    let range_overflow = checked_buffer_typed_range::<u8>(usize::MAX, usize::MAX, 1)
        .expect_err("range end overflow");
    assert!(matches!(
        range_overflow,
        MetalSupportError::BufferBounds {
            offset_bytes: usize::MAX,
            byte_len: 1,
            buffer_len: usize::MAX,
        }
    ));
    assert!(matches!(
        checked_buffer_typed_range::<ZeroSizedAbi>(8, 0, 1),
        Err(MetalSupportError::BufferZeroSizedType {
            abi_name: "ZeroSizedAbi"
        })
    ));
}

#[test]
fn zero_length_readback_does_not_require_cpu_visible_contents() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let Ok(device) = system_default_device() else {
        j2k_test_support::metal_device_unavailable_is_skip(module_path!());
        return;
    };
    let buffer = checked_private_buffer(&device, 4).expect("bounded private buffer");
    // SAFETY: A zero-element request performs no memory access.
    let values = unsafe { checked_buffer_read_vec::<u32>(&buffer, 4, 0) }.expect("empty readback");
    assert!(values.is_empty());
}
