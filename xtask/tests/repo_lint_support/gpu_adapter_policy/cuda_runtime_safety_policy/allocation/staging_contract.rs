// SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::Path;

mod checks;
mod sources;

pub(super) fn assert_policy(root: &Path) {
    let sources = sources::StagingSources::read(root);
    checks::assert_pinned_operation_contracts(&sources);
    checks::assert_pinned_pool_contracts(&sources);
    checks::assert_other_allocation_contracts(&sources);

    let cap_check = sources
        .pinned_operations
        .find("validate_pinned_upload_staging_len(len, DEFAULT_MAX_HOST_ALLOCATION_BYTES)?;")
        .expect("pinned staging cap check");
    let host_allocation = sources
        .pinned_operations
        .find("cuMemHostAlloc")
        .expect("pinned host allocation");
    assert!(
        cap_check < host_allocation,
        "pinned staging must reject over-cap allocations before CUDA driver work"
    );

    for (reserve, push) in [
        ("buffers.try_reserve(1)", "buffers.push(staging)"),
        ("uncertain.try_reserve(1)", "uncertain.push(staging)"),
    ] {
        assert!(
            sources
                .pinned_pool_state
                .find(reserve)
                .expect("pinned metadata reserve")
                < sources
                    .pinned_pool_state
                    .find(push)
                    .expect("pinned token retention"),
            "pinned staging must reserve cache metadata before retaining raw tokens"
        );
    }

    let reserve = sources
        .kernel_cache
        .find("modules.try_reserve(1)")
        .expect("kernel cache reserve");
    let load = sources
        .kernel_cache
        .find("let compiled = CompiledKernel::load(self, key)?;")
        .expect("kernel module load");
    assert!(
        reserve < load,
        "kernel cache must reserve before module creation"
    );
}
