// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::super::repo_root;

#[test]
fn accelerator_boundary_tests_keep_focused_owners() {
    let root = repo_root();
    for (owner, module_decl, child, symbol, max_lines) in [
        (
            "crates/j2k-cuda-runtime/src/tests.rs",
            "mod context_diagnostics;",
            "crates/j2k-cuda-runtime/src/tests/context_diagnostics.rs",
            "fn runtime_diagnostics_count_device_to_host_transfers_when_required",
            100usize,
        ),
        (
            "crates/j2k-cuda-runtime/src/tests/pipeline.rs",
            "mod native_store;",
            "crates/j2k-cuda-runtime/src/tests/pipeline/native_store.rs",
            "fn j2k_native_grayscale_batch_store_preserves_unsigned_and_signed_samples_when_runtime_required",
            220usize,
        ),
        (
            "crates/j2k-cuda/src/session.rs",
            "mod tests;",
            "crates/j2k-cuda/src/session/tests.rs",
            "fn uninitialized_decode_pool_diagnostics_are_empty",
            120usize,
        ),
    ] {
        let owner_source = fs::read_to_string(root.join(owner))
            .unwrap_or_else(|error| panic!("read {owner}: {error}"));
        let child_source = fs::read_to_string(root.join(child))
            .unwrap_or_else(|error| panic!("read {child}: {error}"));
        assert!(owner_source.contains(module_decl));
        assert!(!owner_source.contains(symbol));
        assert!(child_source.contains(symbol));
        assert!(child_source.lines().count() < max_lines);
        assert!(!child_source.lines().any(|line| line.trim() == "use super::*;"));
    }
}
