// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{device, MetalBufferPools, PoolLimits};
use j2k_metal_support::{checked_private_buffer, checked_shared_buffer};

#[test]
fn byte_admitted_resident_working_set_is_fully_reused_after_warmup() {
    const RESIDENT_WORKING_SET_BUFFERS: usize = 103;
    const BUFFER_BYTES: usize = 4 * 1024;

    let device = device();
    let pools = MetalBufferPools::new(&device);
    let mut working_set = Vec::new();
    working_set
        .try_reserve_exact(RESIDENT_WORKING_SET_BUFFERS)
        .expect("resident working-set test metadata");
    for _ in 0..RESIDENT_WORKING_SET_BUFFERS {
        working_set.push(
            pools
                .take_private(&device, BUFFER_BYTES)
                .expect("cold resident private-buffer allocation"),
        );
    }
    for buffer in working_set.drain(..) {
        pools
            .recycle_private(buffer)
            .expect("resident private-buffer recycle");
    }

    let warm = pools.private_diagnostics().expect("warm pool diagnostics");
    assert_eq!(warm.cached_buffers, RESIDENT_WORKING_SET_BUFFERS);
    assert_eq!(
        warm.cached_bytes,
        RESIDENT_WORKING_SET_BUFFERS * BUFFER_BYTES
    );
    assert_eq!(warm.evictions, 0);

    crate::buffer_pool::reset_private_buffer_pool_misses_for_test();
    for _ in 0..RESIDENT_WORKING_SET_BUFFERS {
        working_set.push(
            pools
                .take_private(&device, BUFFER_BYTES)
                .expect("warm resident private-buffer reuse"),
        );
    }
    assert_eq!(crate::buffer_pool::private_buffer_pool_misses_for_test(), 0);
}

#[test]
fn private_and_shared_retention_are_isolated() {
    let device = device();
    let pools =
        MetalBufferPools::with_limits_for_test(PoolLimits::new(16, 1), PoolLimits::new(32, 1));
    pools
        .recycle_private_checked(
            8,
            checked_private_buffer(&device, 8).expect("private buffer"),
        )
        .expect("recycle private");
    pools
        .recycle_shared_checked(
            12,
            checked_shared_buffer(&device, 12).expect("shared buffer"),
        )
        .expect("recycle shared");

    assert_eq!(pools.private_diagnostics().unwrap().cached_bytes, 8);
    assert_eq!(pools.shared_diagnostics().unwrap().cached_bytes, 12);
}
