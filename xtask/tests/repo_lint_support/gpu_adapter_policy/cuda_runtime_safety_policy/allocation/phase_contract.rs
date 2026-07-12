// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cross-boundary aggregate host-allocation ownership for CUDA phases.

use std::path::Path;

mod checks;
mod ordering;
mod sources;

pub(super) fn assert_policy(root: &Path) {
    assert_policy_module_focused();
    let sources = sources::PhaseSources::read(root);

    checks::assert_decode_contracts(&sources);
    checks::assert_encode_and_transfer_contracts(&sources);

    assert_phase_ordering(&sources);
}

fn assert_policy_module_focused() {
    assert!(
        include_str!("phase_contract.rs").lines().count() < 175,
        "CUDA aggregate phase-allocation policy must remain focused"
    );
}

fn assert_phase_ordering(sources: &sources::PhaseSources) {
    ordering::assert_policy(
        &sources.decode_completion,
        &sources.encode_completion,
        &sources.packetize,
    );
}
