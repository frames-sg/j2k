// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{device, MetalBufferPools};
use j2k_metal_support::checked_private_buffer;

#[test]
fn ordered_warm_working_set_lookup_is_near_linear() {
    const WORKING_SET_BUFFERS: usize = 103;

    let device = device();
    let pools = MetalBufferPools::new(&device);
    for bytes in 1..=WORKING_SET_BUFFERS {
        pools
            .recycle_private_checked(
                bytes,
                checked_private_buffer(&device, bytes).expect("private working-set buffer"),
            )
            .expect("populate private working set");
    }

    crate::buffer_pool::reset_private_buffer_pool_take_probes_for_test();
    for bytes in 1..=WORKING_SET_BUFFERS {
        pools
            .take_private(&device, bytes)
            .expect("take warm private working-set buffer");
    }

    assert!(
        crate::buffer_pool::private_buffer_pool_take_probes_for_test() <= WORKING_SET_BUFFERS * 2,
        "ordered resident reuse must not search the warm pool quadratically"
    );
}

#[test]
fn fifo_take_preserves_deterministic_oldest_eviction() {
    let device = device();
    let pools = MetalBufferPools::with_limits_for_test(
        super::PoolLimits::new(64, 3),
        super::PoolLimits::new(64, 3),
    );
    for bytes in [4, 5, 6] {
        pools
            .recycle_private_checked(
                bytes,
                checked_private_buffer(&device, bytes).expect("private buffer"),
            )
            .expect("populate private pool");
    }
    drop(pools.take_private(&device, 4).expect("take oldest buffer"));
    for bytes in [4, 7] {
        pools
            .recycle_private_checked(
                bytes,
                checked_private_buffer(&device, bytes).expect("replacement private buffer"),
            )
            .expect("recycle replacement buffer");
    }

    crate::buffer_pool::reset_private_buffer_pool_misses_for_test();
    drop(
        pools
            .take_private(&device, 6)
            .expect("retained newer buffer"),
    );
    assert_eq!(crate::buffer_pool::private_buffer_pool_misses_for_test(), 0);
    drop(
        pools
            .take_private(&device, 5)
            .expect("reallocate evicted buffer"),
    );
    assert_eq!(crate::buffer_pool::private_buffer_pool_misses_for_test(), 1);
}
