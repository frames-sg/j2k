// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use crate::metal_types::prelude::*;

use super::super::state::retained_bytes_for_device;
use super::{device, MetalBufferPools, PoolLimits};
use j2k_metal_support::{checked_private_buffer, checked_shared_buffer};

const MIB: usize = 1024 * 1024;
const GIB: u64 = 1024 * 1024 * 1024;

fn eighth(working_set: u64) -> usize {
    usize::try_from(working_set / 8).expect("test working set fits usize")
}

#[test]
fn retained_cap_is_an_eighth_of_the_working_set_between_256_mib_and_1_gib() {
    let any_buffer = usize::MAX;
    // An 8 GB Mac reports roughly two thirds of RAM as its working set.
    let eight_gb_class = 16 * GIB / 3;
    assert_eq!(
        retained_bytes_for_device(any_buffer, eight_gb_class),
        eighth(eight_gb_class)
    );
    // 16 GB and larger devices reach the 1 GiB ceiling.
    assert_eq!(
        retained_bytes_for_device(any_buffer, 32 * GIB / 3),
        1024 * MIB
    );
    assert_eq!(retained_bytes_for_device(any_buffer, 96 * GIB), 1024 * MIB);
    // Small or unreported working sets keep the previous 256 MiB cap.
    assert_eq!(retained_bytes_for_device(any_buffer, GIB), 256 * MIB);
    assert_eq!(retained_bytes_for_device(any_buffer, 0), 256 * MIB);
}

#[test]
fn retained_cap_never_exceeds_the_largest_device_buffer() {
    assert_eq!(retained_bytes_for_device(300 * MIB, 64 * GIB), 300 * MIB);
    assert_eq!(retained_bytes_for_device(100 * MIB, 0), 100 * MIB);
}

#[test]
fn production_retained_cap_matches_the_device_formula_for_both_pools() {
    let device = device();
    let expected = retained_bytes_for_device(
        device.maxBufferLength(),
        device.recommendedMaxWorkingSetSize(),
    );
    assert!(expected <= 1024 * MIB);
    assert_eq!(
        PoolLimits::private_for_device(&device).retained_bytes_for_test(),
        expected
    );
    assert_eq!(
        PoolLimits::shared_for_device(&device).retained_bytes_for_test(),
        expected
    );
}

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
