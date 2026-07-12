// SPDX-License-Identifier: MIT OR Apache-2.0

//! Allocation and diagnostic integrity ratchets for CUDA transcode adapters.

mod allocation;
mod call_arguments;
mod collection_scan;
mod phase_budget;
mod resident_metadata;
mod runtime_diagnostics;
mod sources;
mod stage_error_mapping;
mod validator_mapping;

use self::sources::CudaTranscodeSources;

#[test]
fn cuda_transcode_host_staging_remains_bounded_and_fallible() {
    let sources = CudaTranscodeSources::read();
    allocation::assert_policy(&sources);
    phase_budget::assert_policy(&sources);
    resident_metadata::assert_policy(&sources);
}

#[test]
fn cuda_transcode_runtime_failures_retain_complete_diagnostics() {
    runtime_diagnostics::assert_policy(&CudaTranscodeSources::read());
}

#[test]
fn cuda_transcode_errors_retain_typed_stage_classification() {
    let sources = CudaTranscodeSources::read();
    stage_error_mapping::assert_policy(&sources);
    validator_mapping::assert_policy(&sources);
}

#[test]
fn cuda_transcode_policy_stays_focused() {
    assert!(
        include_str!("transcode_cuda_policy.rs").lines().count() < 100,
        "CUDA transcode policy shell must stay below its focused-module ratchet"
    );
    assert!(
        include_str!("transcode_cuda_policy/sources.rs")
            .lines()
            .count()
            < 100,
        "CUDA transcode source inventory must stay focused"
    );
}
