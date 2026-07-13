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
    let resident_limits = read("crates/j2k-metal/src/resident_limits.rs");
    let encode_config = read("crates/j2k-metal/src/encode/config.rs");
    let session = read("crates/j2k-metal/src/session.rs");
    let runtime = read("crates/j2k-metal/src/compute/runtime.rs");
    let scratch = read("crates/j2k-metal/src/compute/direct_scratch.rs");
    let error = read("crates/j2k-metal/src/error.rs");

    assert_pattern_checks(&[
        PatternCheck::new("Metal buffer-pool facade", &facade)
            .required(&[
                "private: Mutex<PoolState>",
                "shared: Mutex<PoolState>",
                "PoolLimits::private_for_device(device)",
                "PoolLimits::shared_for_device(device)",
                "Result<PooledBuffer, Error>",
                "recycle_private(&self, buffer: PooledBuffer)",
                "checked_private_buffer(device, bytes)",
                "checked_shared_buffer(device, bytes)",
                "MetalBufferPoolsDiagnostics",
            ])
            .forbidden(&["HashMap", ".entry(", ".or_default()", "Vec::with_capacity"]),
        PatternCheck::new("Metal buffer-pool state ledger", &state)
            .required(&[
                "DEFAULT_RETAINED_BYTES_PER_POOL",
                "DEFAULT_PRIVATE_RETAINED_BUFFERS_PER_POOL",
                "DEFAULT_SHARED_RETAINED_BUFFERS_PER_POOL",
                "device.max_buffer_length()",
                "entries: VecDeque<PooledBuffer>",
                "pub(crate) struct PooledBuffer",
                "fn new_checked(expected_bytes: usize, buffer: Buffer)",
                "usize::try_from(buffer.length())",
                ".iter().position(",
                ".remove(index)",
                ".push_back(",
                ".pop_front()",
                "try_reserve_exact(1)",
                "fn evict_oldest",
                "checked_add(actual_bytes)",
                "checked_sub(evicted.bytes)",
                "metadata_capacity: self.entries.capacity()",
            ])
            .forbidden(&[
                "HashMap",
                ".entry(",
                ".or_default()",
                ".saturating_add(",
                ".rposition(",
                ".swap_remove(",
                "entries: Vec<PooledBuffer>",
                "fn recycle(&mut self, expected_bytes: usize",
            ]),
        PatternCheck::new("Metal resident working-set limits", &resident_limits)
            .required(&[
                "DEFAULT_RESIDENT_CHUNK_TILES: usize = 512",
                "MAX_RESIDENT_COMPONENTS: usize = 3",
                "PRIVATE_DWT_SCRATCH_PER_COMPONENT: usize = 1",
                "BASE_PRIVATE_BUFFERS_PER_RESIDENT_BATCH: usize = 7",
                "CLASSIC_SPLIT_TOKEN_PRIVATE_BUFFERS_PER_BATCH: usize = 4",
                "DEFAULT_RESIDENT_PRIVATE_WORKING_SET_BUFFERS",
                "RESIDENT_PRIVATE_POOL_BUFFER_LIMIT: usize = 4_096",
            ])
            .forbidden(&["Vec", "HashMap", "unsafe"]),
        PatternCheck::new("Metal default resident retention shape", &encode_config).required(&[
            "GPU_ENCODE_DEFAULT_INFLIGHT_TILES: usize",
            "crate::resident_limits::DEFAULT_RESIDENT_CHUNK_TILES",
        ]),
        PatternCheck::new("Metal buffer-pool runtime wiring", &runtime).required(&[
            "buffer_pools: MetalBufferPools::new(device)",
            "fn buffer_pool_diagnostics(",
            "self.buffer_pools.diagnostics()",
        ]),
        PatternCheck::new("typed recyclable Metal buffer ownership", &scratch)
            .required(&[
                "Vec<PooledBuffer>",
                "runtime.recycle_private_buffer(buffer)",
                "runtime.recycle_shared_buffer(buffer)",
            ])
            .forbidden(&["Vec<(usize, Buffer)>"]),
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
    let lookup = read("crates/j2k-metal/src/buffer_pool/tests/lookup.rs");
    let warm_reuse = read("crates/j2k-metal/src/buffer_pool/tests/warm_reuse.rs");
    let production_limits = read("crates/j2k-metal/src/buffer_pool/tests/production_limits.rs");
    for required in [
        "completed_exact_size_reuse_updates_actual_byte_accounting",
        "unique_sizes_evict_oldest_buffers_under_both_limits",
        "mod warm_reuse;",
        "mod lookup;",
        "mod production_limits;",
        "oversized_and_metadata_failed_recycles_drop_completed_buffers",
        "recorded_size_mismatch_is_a_typed_invariant_failure",
        "backend_session_exposes_typed_pool_high_water_diagnostics",
    ] {
        assert!(
            tests.contains(required),
            "missing pool regression {required}"
        );
    }
    for required in [
        "ordered_warm_working_set_lookup_is_near_linear",
        "fifo_take_preserves_deterministic_oldest_eviction",
    ] {
        assert!(
            lookup.contains(required),
            "missing pool lookup regression {required}"
        );
    }
    for required in [
        "byte_admitted_resident_working_set_is_fully_reused_after_warmup",
        "private_and_shared_retention_are_isolated",
    ] {
        assert!(
            warm_reuse.contains(required),
            "missing warm-retention regression {required}"
        );
    }
    assert!(
        production_limits.contains("production_private_and_shared_record_limits_are_independent"),
        "missing production private/shared pool-limit regression"
    );
}

#[test]
fn metal_buffer_pool_modules_stay_focused() {
    for (relative, limit) in [
        ("crates/j2k-metal/src/buffer_pool.rs", 225),
        ("crates/j2k-metal/src/buffer_pool/state.rs", 250),
        ("crates/j2k-metal/src/buffer_pool/diagnostics.rs", 100),
        ("crates/j2k-metal/src/buffer_pool/test_support.rs", 100),
        ("crates/j2k-metal/src/buffer_pool/tests.rs", 150),
        ("crates/j2k-metal/src/buffer_pool/tests/lookup.rs", 75),
        (
            "crates/j2k-metal/src/buffer_pool/tests/production_limits.rs",
            60,
        ),
        ("crates/j2k-metal/src/buffer_pool/tests/warm_reuse.rs", 85),
        ("crates/j2k-metal/src/resident_limits.rs", 75),
    ] {
        let source = read(relative);
        assert!(
            source.lines().count() < limit,
            "{relative} must stay below {limit} lines"
        );
        assert!(!source.contains("include!(") && !source.contains("use super::*"));
    }
}
