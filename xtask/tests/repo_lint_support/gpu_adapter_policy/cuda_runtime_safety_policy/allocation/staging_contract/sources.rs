// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{fs, path::Path};

pub(super) struct StagingSources {
    pub(super) readback: String,
    pub(super) pinned: String,
    pub(super) pinned_operations: String,
    pub(super) pinned_api: String,
    pub(super) pinned_checkout: String,
    pub(super) pinned_checkout_tests: String,
    pub(super) pinned_gate: String,
    pub(super) pinned_policy: String,
    pub(super) pinned_policy_tests: String,
    pub(super) pinned_recycle: String,
    pub(super) pinned_pool_state: String,
    pub(super) pinned_pool_diagnostics: String,
    pub(super) pinned_pool_tests: String,
    pub(super) pinned_pool_active_tests: String,
    pub(super) device_pool: String,
    pub(super) pinned_token: String,
    pub(super) kernel_cache: String,
    pub(super) surface: String,
}

impl StagingSources {
    pub(super) fn read(root: &Path) -> Self {
        let read = |relative: &str| {
            fs::read_to_string(root.join(relative))
                .unwrap_or_else(|error| panic!("read {relative}: {error}"))
        };
        let pinned_pool = read("crates/j2k-cuda-runtime/src/memory/pinned_staging/pool.rs");
        let pinned_pool_active =
            read("crates/j2k-cuda-runtime/src/memory/pinned_staging/pool/active.rs");
        Self {
            readback: read("crates/j2k-cuda-runtime/src/memory/pool/readback.rs"),
            pinned: read("crates/j2k-cuda-runtime/src/memory/pinned_staging.rs"),
            pinned_operations: [
                read("crates/j2k-cuda-runtime/src/memory/pinned_staging/operations.rs"),
                read("crates/j2k-cuda-runtime/src/memory/pinned_staging/operations/growth.rs"),
            ]
            .concat(),
            pinned_api: read("crates/j2k-cuda-runtime/src/memory/pinned_staging/operations/api.rs"),
            pinned_checkout: read(
                "crates/j2k-cuda-runtime/src/memory/pinned_staging/operations/checkout.rs",
            ),
            pinned_checkout_tests: read(
                "crates/j2k-cuda-runtime/src/memory/pinned_staging/operations/checkout/tests.rs",
            ),
            pinned_gate: read(
                "crates/j2k-cuda-runtime/src/memory/pinned_staging/operations/gate.rs",
            ),
            pinned_policy: read(
                "crates/j2k-cuda-runtime/src/memory/pinned_staging/operations/policy.rs",
            ),
            pinned_policy_tests: read(
                "crates/j2k-cuda-runtime/src/memory/pinned_staging/operations/policy/tests.rs",
            ),
            pinned_recycle: read(
                "crates/j2k-cuda-runtime/src/memory/pinned_staging/operations/recycle.rs",
            ),
            pinned_pool_state: [pinned_pool, pinned_pool_active].concat(),
            pinned_pool_diagnostics: read(
                "crates/j2k-cuda-runtime/src/memory/pinned_staging/pool/diagnostics.rs",
            ),
            pinned_pool_tests: read(
                "crates/j2k-cuda-runtime/src/memory/pinned_staging/pool/tests.rs",
            ),
            pinned_pool_active_tests: read(
                "crates/j2k-cuda-runtime/src/memory/pinned_staging/pool/active/tests.rs",
            ),
            device_pool: read("crates/j2k-cuda-runtime/src/memory/pool.rs"),
            pinned_token: [
                read("crates/j2k-cuda-runtime/src/context/pinned_host.rs"),
                read("crates/j2k-cuda-runtime/src/context/pinned_host/tests.rs"),
            ]
            .concat(),
            kernel_cache: read("crates/j2k-cuda-runtime/src/context/kernel_cache.rs"),
            surface: read("crates/j2k-cuda/src/surface.rs"),
        }
    }
}
