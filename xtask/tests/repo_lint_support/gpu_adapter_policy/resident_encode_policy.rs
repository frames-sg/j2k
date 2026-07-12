// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::repo_root;

mod allocation_checks;
mod batch_allocation_policy;
mod contract_policy;
mod error_preservation_policy;
mod image_allocation_policy;
mod input_error_policy;
mod packetization_error_policy;
mod session_resource_policy;

#[test]
fn resident_encode_policy_children_stay_focused() {
    let root = repo_root();
    for relative in [
        "xtask/tests/repo_lint_support/gpu_adapter_policy/resident_encode_policy/allocation_checks.rs",
        "xtask/tests/repo_lint_support/gpu_adapter_policy/resident_encode_policy/allocation_checks/adapter_contract.rs",
        "xtask/tests/repo_lint_support/gpu_adapter_policy/resident_encode_policy/batch_allocation_policy.rs",
        "xtask/tests/repo_lint_support/gpu_adapter_policy/resident_encode_policy/contract_policy.rs",
        "xtask/tests/repo_lint_support/gpu_adapter_policy/resident_encode_policy/contract_policy/ownership.rs",
        "xtask/tests/repo_lint_support/gpu_adapter_policy/resident_encode_policy/error_preservation_policy.rs",
        "xtask/tests/repo_lint_support/gpu_adapter_policy/resident_encode_policy/image_allocation_policy.rs",
        "xtask/tests/repo_lint_support/gpu_adapter_policy/resident_encode_policy/input_error_policy.rs",
        "xtask/tests/repo_lint_support/gpu_adapter_policy/resident_encode_policy/packetization_error_policy.rs",
        "xtask/tests/repo_lint_support/gpu_adapter_policy/resident_encode_policy/session_resource_policy.rs",
    ] {
        let source = fs::read_to_string(root.join(relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"));
        assert!(
            source.lines().count() < 150,
            "{relative} must stay below the resident-encode policy line-count ratchet"
        );
        assert!(
            !source
                .lines()
                .any(|line| line.trim() == "use super::*;")
                && !source
                    .lines()
                    .any(|line| line.trim_start().starts_with("include!(")),
            "{relative} must use explicit real-Rust module boundaries"
        );
    }
}
