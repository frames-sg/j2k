// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{device, MetalBufferPools};
use j2k_metal_support::{checked_private_buffer, checked_shared_buffer};

#[test]
fn production_private_and_shared_record_limits_are_independent() {
    const LEGACY_SHARED_RECORD_LIMIT: usize = 64;

    let device = device();
    let pools = MetalBufferPools::new(&device);
    for _ in 0..=LEGACY_SHARED_RECORD_LIMIT {
        pools
            .recycle_private_checked(
                1,
                checked_private_buffer(&device, 1).expect("private limit probe buffer"),
            )
            .expect("recycle private limit probe");
        pools
            .recycle_shared_checked(
                1,
                checked_shared_buffer(&device, 1).expect("shared limit probe buffer"),
            )
            .expect("recycle shared limit probe");
    }

    let private = pools.private_diagnostics().expect("private diagnostics");
    let shared = pools.shared_diagnostics().expect("shared diagnostics");
    assert_eq!(private.cached_buffers, LEGACY_SHARED_RECORD_LIMIT + 1);
    assert_eq!(private.evictions, 0);
    assert_eq!(shared.cached_buffers, LEGACY_SHARED_RECORD_LIMIT);
    assert_eq!(shared.evictions, 1);
}
