// SPDX-License-Identifier: MIT OR Apache-2.0

use super::allocation_checks::assert_cuda_image_derived_encode_allocation_contract;
use crate::repo_lint_support::repo_root;

#[test]
fn cuda_image_derived_encode_allocations_are_fallible_and_owned_results_move() {
    assert_cuda_image_derived_encode_allocation_contract(repo_root());
}
