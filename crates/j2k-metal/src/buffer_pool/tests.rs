// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{MetalBufferPools, PoolLimits};
use j2k_metal_support::{
    checked_blit_command_encoder, checked_command_buffer, checked_command_queue,
    checked_private_buffer, checked_shared_buffer, checked_shared_buffer_with_bytes,
    commit_and_wait,
};
use metal::foreign_types::ForeignType;
use metal::Device;

fn device() -> Device {
    Device::system_default().expect("Metal device")
}

#[test]
fn completed_exact_size_reuse_updates_actual_byte_accounting() {
    let device = device();
    let pools =
        MetalBufferPools::with_limits_for_test(PoolLimits::new(64, 2), PoolLimits::new(64, 2));
    let mut buffer = checked_private_buffer(&device, 16).expect("private buffer");
    let upload = checked_shared_buffer_with_bytes(&device, &[0xa5; 16]).expect("upload buffer");
    let queue = checked_command_queue(&device).expect("command queue");
    let command_buffer = checked_command_buffer(&queue).expect("command buffer");
    let blit = checked_blit_command_encoder(&command_buffer).expect("blit encoder");
    blit.copy_from_buffer(&upload, 0, &buffer, 0, 16);
    blit.end_encoding();
    commit_and_wait(&command_buffer).expect("buffer work completion before recycle");
    let pointer = buffer.as_ptr();

    for _ in 0..16 {
        pools
            .recycle_private(16, buffer)
            .expect("recycle private buffer");
        assert_eq!(pools.private_diagnostics().unwrap().cached_bytes, 16);
        buffer = pools
            .take_private(&device, 16)
            .expect("take private buffer");
        assert_eq!(buffer.as_ptr(), pointer);
    }
    drop(buffer);
    let diagnostics = pools.private_diagnostics().unwrap();
    assert_eq!(diagnostics.cached_bytes, 0);
    assert_eq!(diagnostics.cached_buffers, 0);
    assert_eq!(diagnostics.peak_cached_bytes, 16);
}

#[test]
fn unique_sizes_evict_oldest_buffers_under_both_limits() {
    let device = device();
    let pools =
        MetalBufferPools::with_limits_for_test(PoolLimits::new(10, 2), PoolLimits::new(10, 2));
    for bytes in [4, 5, 6] {
        let buffer = checked_private_buffer(&device, bytes).expect("private buffer");
        pools
            .recycle_private(bytes, buffer)
            .expect("recycle private buffer");
    }

    let diagnostics = pools.private_diagnostics().unwrap();
    assert_eq!(diagnostics.cached_bytes, 6);
    assert_eq!(diagnostics.cached_buffers, 1);
    assert_eq!(diagnostics.evictions, 2);
}

#[test]
fn oversized_and_metadata_failed_recycles_drop_completed_buffers() {
    let device = device();
    let pools =
        MetalBufferPools::with_limits_for_test(PoolLimits::new(8, 2), PoolLimits::new(8, 2));
    pools
        .recycle_private(
            8,
            checked_private_buffer(&device, 8).expect("exact private buffer"),
        )
        .expect("exact-limit recycle");
    let oversized = checked_private_buffer(&device, 9).expect("oversized private buffer");
    pools
        .recycle_private(9, oversized)
        .expect("oversized recycle is a safe decline");
    pools.fail_next_private_metadata_reserve_for_test();
    let metadata_failure = checked_private_buffer(&device, 4).expect("private buffer");
    pools
        .recycle_private(4, metadata_failure)
        .expect("metadata failure is a safe decline");

    let diagnostics = pools.private_diagnostics().unwrap();
    assert_eq!(diagnostics.cached_bytes, 8);
    assert_eq!(diagnostics.cached_buffers, 1);
    assert_eq!(diagnostics.rejections, 2);
    assert_eq!(diagnostics.metadata_failures, 1);
}

#[test]
fn recorded_size_mismatch_is_a_typed_invariant_failure() {
    let device = device();
    let pools =
        MetalBufferPools::with_limits_for_test(PoolLimits::new(16, 2), PoolLimits::new(16, 2));
    let buffer = checked_private_buffer(&device, 8).expect("private buffer");
    assert!(matches!(
        pools.recycle_private(7, buffer),
        Err(crate::Error::MetalStateInvariant {
            state: "j2k metal private buffer pool",
            reason: "recorded buffer size differs from the Metal allocation length",
        })
    ));
    assert_eq!(pools.private_diagnostics().unwrap().size_mismatches, 1);
}

#[test]
fn private_and_shared_retention_are_isolated() {
    let device = device();
    let pools =
        MetalBufferPools::with_limits_for_test(PoolLimits::new(16, 1), PoolLimits::new(32, 1));
    pools
        .recycle_private(
            8,
            checked_private_buffer(&device, 8).expect("private buffer"),
        )
        .expect("recycle private");
    pools
        .recycle_shared(
            12,
            checked_shared_buffer(&device, 12).expect("shared buffer"),
        )
        .expect("recycle shared");

    assert_eq!(pools.private_diagnostics().unwrap().cached_bytes, 8);
    assert_eq!(pools.shared_diagnostics().unwrap().cached_bytes, 12);
}

#[test]
fn backend_session_exposes_typed_pool_high_water_diagnostics() {
    let session = crate::MetalBackendSession::system_default().expect("Metal backend session");
    let diagnostics = session
        .buffer_pool_diagnostics()
        .expect("buffer-pool diagnostics");
    assert!(diagnostics.private.cached_bytes <= diagnostics.private.peak_cached_bytes);
    assert!(diagnostics.shared.cached_bytes <= diagnostics.shared.peak_cached_bytes);
    assert!(diagnostics.private.cached_buffers <= diagnostics.private.peak_cached_buffers);
    assert!(diagnostics.shared.cached_buffers <= diagnostics.shared.peak_cached_buffers);
}
