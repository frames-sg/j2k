// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bounded completed-buffer retention for the long-lived J2K Metal runtime.

use std::fs;

use super::{assert_pattern_checks, repo_root, PatternCheck};

fn read(relative: &str) -> String {
    fs::read_to_string(repo_root().join(relative))
        .unwrap_or_else(|error| panic!("read {relative}: {error}"))
}

#[test]
fn metal_buffer_pools_use_flat_fallible_actual_byte_ledgers() {
    let facade = read("crates/j2k-metal/src/buffer_pool.rs");
    let state = read("crates/j2k-metal/src/buffer_pool/state.rs");
    let session = read("crates/j2k-metal/src/session.rs");
    let runtime = read("crates/j2k-metal/src/compute/runtime.rs");
    let error = read("crates/j2k-metal/src/error.rs");

    assert_pattern_checks(&[
        PatternCheck::new("Metal buffer-pool facade", &facade)
            .required(&[
                "private: Mutex<PoolState>",
                "shared: Mutex<PoolState>",
                "PoolLimits::for_device(device)",
                "checked_private_buffer(device, bytes)",
                "checked_shared_buffer(device, bytes)",
                "MetalBufferPoolsDiagnostics",
            ])
            .forbidden(&["HashMap", ".entry(", ".or_default()", "Vec::with_capacity"]),
        PatternCheck::new("Metal buffer-pool state ledger", &state)
            .required(&[
                "DEFAULT_RETAINED_BYTES_PER_POOL",
                "DEFAULT_RETAINED_BUFFERS_PER_POOL",
                "device.max_buffer_length()",
                "entries: Vec<PooledBuffer>",
                "usize::try_from(buffer.length())",
                "try_reserve_exact(1)",
                "fn evict_oldest",
                "checked_add(actual_bytes)",
                "checked_sub(evicted.bytes)",
                "metadata_capacity: self.entries.capacity()",
            ])
            .forbidden(&["HashMap", ".entry(", ".or_default()", ".saturating_add("]),
        PatternCheck::new("Metal buffer-pool runtime wiring", &runtime).required(&[
            "buffer_pools: MetalBufferPools::new(device)",
            "fn buffer_pool_diagnostics(",
            "self.buffer_pools.diagnostics()",
        ]),
        PatternCheck::new("Metal buffer-pool public diagnostics", &session).required(&[
            "pub fn buffer_pool_diagnostics(&self)",
            "self.runtime()?.buffer_pool_diagnostics()",
        ]),
        PatternCheck::new("Metal buffer-pool typed invariant", &error).required(&[
            "MetalStateInvariant",
            "state: &'static str",
            "reason: &'static str",
        ]),
    ]);
}

#[test]
fn metal_buffer_pool_regressions_cover_limits_failures_and_isolation() {
    let tests = read("crates/j2k-metal/src/buffer_pool/tests.rs");
    for required in [
        "completed_exact_size_reuse_updates_actual_byte_accounting",
        "unique_sizes_evict_oldest_buffers_under_both_limits",
        "oversized_and_metadata_failed_recycles_drop_completed_buffers",
        "recorded_size_mismatch_is_a_typed_invariant_failure",
        "private_and_shared_retention_are_isolated",
        "backend_session_exposes_typed_pool_high_water_diagnostics",
    ] {
        assert!(
            tests.contains(required),
            "missing pool regression {required}"
        );
    }
}

#[test]
fn metal_buffer_pool_modules_stay_focused() {
    for (relative, limit) in [
        ("crates/j2k-metal/src/buffer_pool.rs", 225),
        ("crates/j2k-metal/src/buffer_pool/state.rs", 250),
        ("crates/j2k-metal/src/buffer_pool/tests.rs", 150),
    ] {
        let source = read(relative);
        assert!(
            source.lines().count() < limit,
            "{relative} must stay below {limit} lines"
        );
        assert!(!source.contains("include!(") && !source.contains("use super::*"));
    }
}
