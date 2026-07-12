// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::super::super::{assert_pattern_checks, repo_root, PatternCheck};

mod cache_contract;
mod phase_contract;
mod resource_creation_contract;
mod staging_contract;

#[test]
fn cuda_host_allocation_failures_remain_fallible_and_ownership_safe() {
    let root = repo_root();
    let read = |relative: &str| {
        let path = root.join(relative);
        fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
    };
    let core_allocation = read("crates/j2k-core/src/host_allocation.rs");
    let runtime_allocation = [
        read("crates/j2k-cuda-runtime/src/allocation.rs"),
        read("crates/j2k-cuda-runtime/src/allocation/phase.rs"),
    ]
    .concat();

    assert_pattern_checks(&[
        PatternCheck::new("shared no-std host allocation primitive", &core_allocation).required(&[
            "pub struct HostAllocationError",
            "pub struct HostAllocationBudget",
            "pub struct HostAllocationLimitError",
            "pub fn check_capacity<T>(",
            "pub fn account_capacity<T>(",
            "pub fn account_bytes(",
            "pub fn account_vec<T>(",
            "capacity.saturating_mul(core::mem::size_of::<T>())",
            ".try_reserve_exact(capacity)",
            "pub fn try_host_vec_filled<T: Clone>(",
            "pub fn try_host_vec_from_slice<T: Copy>(",
            "pub fn try_host_vec_resize<T: Clone>(",
        ]),
        PatternCheck::new("CUDA allocation error mapping", &runtime_allocation).required(&[
            "pub(crate) struct HostPhaseBudget",
            "pub(crate) fn with_live_bytes(",
            "pub(crate) const fn live_bytes(",
            "pub(crate) fn account_bytes(",
            "pub(crate) fn account_capacity<T>(",
            ".check_capacity::<T>(capacity)",
            ".account_vec(values)",
            "CudaError::HostAllocationTooLarge",
            "CudaError::HostAllocationFailed",
        ]),
    ]);
    cache_contract::assert_policy(root);
    phase_contract::assert_policy(root);
    resource_creation_contract::assert_policy(root);
    staging_contract::assert_policy(root);
}
