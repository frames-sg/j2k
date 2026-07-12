// SPDX-License-Identifier: MIT OR Apache-2.0

//! Aggregate, fallible ownership and structure contract for J2K CPU batches.

use std::fs;

use super::{assert_pattern_checks, read_source_files, repo_root, PatternCheck};

#[test]
fn j2k_batch_uses_one_authoritative_aggregate_plan() {
    let root = repo_root();
    let facade =
        fs::read_to_string(root.join("crates/j2k/src/batch.rs")).expect("read J2K batch facade");
    let allocation = fs::read_to_string(root.join("crates/j2k/src/batch/allocation.rs"))
        .expect("read J2K batch allocation policy");
    let planning = read_source_files(
        root,
        &[
            "crates/j2k/src/batch/planning.rs",
            "crates/j2k/src/batch/planning/tests.rs",
        ],
    );
    let decode =
        fs::read_to_string(root.join("crates/j2k/src/decode.rs")).expect("read J2K warning owner");

    assert_pattern_checks(&[
        PatternCheck::new("J2K public typed batch boundary", &facade).required(&[
            "pub type TileBatchError = j2k_core::BatchDecodeError<J2kError>;",
            "TileBatchError::Tile(j2k_core::TileBatchError { index: 0, source })",
            "drop(shared_direct_plan);",
            "scheduler::collect_results(results)",
        ]),
        PatternCheck::new("J2K fixed aggregate allocation policy", &allocation)
            .required(&[
                "MAX_GENERIC_BATCH_WORKERS: usize = 4",
                "J2K_BATCH_METADATA_ALLOWANCE_BYTES: usize = 64 * 1024 * 1024",
                "j2k_native::DEFAULT_MAX_DECODE_BYTES",
                ".checked_mul(MAX_GENERIC_BATCH_WORKERS)",
                ".checked_add(J2K_BATCH_METADATA_ALLOWANCE_BYTES)",
                "try_host_vec_with_capacity(capacity)",
                "actual_warning_owner_bytes",
            ])
            .forbidden(&["512 * 1024 * 1024", "Vec::with_capacity(", "vec!["]),
        PatternCheck::new("J2K worker and metadata planner", &planning)
            .required(&[
                "fn select_batch_plan_with_limits(",
                "size_of::<J2kBatchResultSlot>()",
                "size_of::<BatchOutcome>()",
                "size_of::<J2kDecodeWarning>()",
                "size_of::<BatchWorker>()",
                "size_of::<ScopedWorkerHandle<'static>>()",
                "exact_aggregate_cap_is_accepted_and_one_over_is_rejected",
                "requested_worker_count_reduces_to_fit_aggregate_claims",
                "one_worker_rejects_when_claim_and_metadata_cannot_fit",
                "metadata_one_over_is_public_infrastructure_error",
                "aggregate_overflow_is_a_typed_cap_failure",
                "planning_claims_do_not_allocate_worker_workspace",
                "empty_batch_plan_is_a_typed_infrastructure_error",
            ])
            .forbidden(&[
                "Vec::with_capacity(",
                "vec![",
                "512 * 1024 * 1024",
                "debug_assert!",
            ]),
        PatternCheck::new("J2K allocation-free warning owner", &decode)
            .required(&[
                "const _: [(); 0] = [(); core::mem::size_of::<J2kDecodeWarning>()];",
                "warnings.push(J2kDecodeWarning::LenientDecodeMode)",
                "decode_warning_owner_is_statically_allocation_free",
            ])
            .forbidden(&["Vec::from([J2kDecodeWarning::LenientDecodeMode])"]),
    ]);
}

#[test]
fn j2k_batch_scheduler_has_typed_workers_and_one_disjoint_result_owner() {
    let root = repo_root();
    let production = read_source_files(
        root,
        &[
            "crates/j2k/src/batch.rs",
            "crates/j2k/src/batch/allocation.rs",
            "crates/j2k/src/batch/direct.rs",
            "crates/j2k/src/batch/planning.rs",
            "crates/j2k/src/batch/scheduler.rs",
            "crates/j2k/src/batch/worker.rs",
        ],
    );
    let error =
        fs::read_to_string(root.join("crates/j2k/src/error.rs")).expect("read J2K error boundary");
    let native_source = fs::read_to_string(root.join("crates/j2k/src/error/native_source.rs"))
        .expect("read facade-owned native source boundary");
    let decode_error_routes = read_source_files(
        root,
        &[
            "crates/j2k/src/backend.rs",
            "crates/j2k/src/decode.rs",
            "crates/j2k/src/decode/output/u8.rs",
            "crates/j2k/src/view.rs",
            "crates/j2k/src/parse/codestream.rs",
            "crates/j2k/src/batch/direct.rs",
            "crates/j2k/src/batch/worker.rs",
        ],
    );

    assert_pattern_checks(&[
        PatternCheck::new("J2K fallible scoped scheduler", &production)
            .required(&[
                "try_vec_with_capacity(job_count, \"J2K ordered worker result slots\")",
                "std::thread::Builder::new().spawn_scoped(scope",
                "BatchInfrastructureError::WorkerSpawnFailed",
                "BatchInfrastructureError::WorkerPanicked",
                ".zip(results.chunks_mut(plan.chunk_size))",
                "drop(worker);",
                "try_collect_ordered_batch_results_with_limits(",
                "J2K_BATCH_METADATA_ALLOWANCE_BYTES",
                "ensure_disjoint_result_slots",
                "jobs.iter_mut().zip(results)",
            ])
            .forbidden(&[
                "Vec::with_capacity(",
                "collect_indexed_batch_results(",
                "IndexedBatchResult",
                "std::panic::resume_unwind",
                "scope.spawn(",
                "unreachable!",
                ".collect::<Vec",
            ]),
        PatternCheck::new("J2K direct-owner lifecycle", &production).required(&[
            "BatchAllocationBudget::with_baseline",
            "prepare_execution_allocation_bytes",
            ".reconcile(worker_bytes)",
            ".reconcile(execution_bytes)",
            "self.release();",
            "self.direct.release();",
            "drop(shared_direct_plan);",
        ]),
        PatternCheck::new("J2K heap-free decode errors", &error).required(&[
            "NativeDecode {",
            "CodestreamHeader {",
            "BackendComponentPlaneTooShort {",
            "InternalInvariant {",
            "source: NativeBackendError",
            "into_heap_free_batch_decode_error",
            "is_heap_free_batch_decode_error",
            "native_backend_errors_remain_typed_and_heap_free",
            "batch_error_normalization_drops_heap_owning_legacy_details",
        ]),
        PatternCheck::new("J2K heap-free opaque native source", &native_source)
            .required(&[
                "source: NativeBackendErrorSource",
                "Decode(DecodeError)",
                "CodestreamHeader(J2kCodestreamHeaderError)",
                "impl core::error::Error for NativeBackendError",
            ])
            .forbidden(&["Box<dyn", "message: String", "pub source:"]),
        PatternCheck::new(
            "J2K decode routes avoid heap formatting",
            &decode_error_routes,
        )
        .required(&[
            "J2kError::BackendComponentPlaneTooShort",
            "J2kError::CodestreamHeader",
            "J2kError::into_heap_free_batch_decode_error",
        ])
        .forbidden(&["J2kError::backend(format!", "error.to_string()"]),
    ]);
}

#[test]
fn shared_batch_collection_exposes_only_fallible_typed_integrity_paths() {
    let root = repo_root();
    let collection = fs::read_to_string(root.join("crates/j2k-core/src/batch/collection.rs"))
        .expect("read shared batch collection");
    let exports = fs::read_to_string(root.join("crates/j2k-core/src/batch.rs"))
        .expect("read shared batch exports");

    assert_pattern_checks(&[
        PatternCheck::new("fallible shared batch collectors", &collection)
            .required(&[
                "pub fn try_collect_indexed_batch_results",
                "pub fn try_collect_ordered_batch_results",
                "BatchInfrastructureError::ResultIndexOutOfBounds",
                "BatchInfrastructureError::DuplicateResult",
                "BatchInfrastructureError::MissingResult",
                "try_host_vec_with_capacity",
            ])
            .forbidden(&[
                "pub fn collect_indexed_batch_results",
                "Vec::with_capacity(",
                "assert!(",
                ".expect(",
            ]),
        PatternCheck::new("fallible shared batch exports", &exports)
            .required(&["try_collect_indexed_batch_results"])
            .forbidden(&["\n    collect_indexed_batch_results,"]),
    ]);
}

#[test]
fn j2k_batch_root_and_responsibility_modules_stay_focused() {
    let root = repo_root();
    for (relative, max_lines) in [
        ("crates/j2k/src/batch.rs", 225),
        ("crates/j2k/src/batch/allocation.rs", 240),
        ("crates/j2k/src/batch/planning.rs", 220),
        ("crates/j2k/src/batch/planning/tests.rs", 130),
        ("crates/j2k/src/batch/scheduler.rs", 260),
        ("crates/j2k/src/batch/worker.rs", 210),
        ("crates/j2k/src/batch/worker/tests.rs", 80),
        ("crates/j2k/src/batch/direct.rs", 350),
        ("crates/j2k/src/batch/direct/planning.rs", 140),
        ("crates/j2k/src/batch/admission.rs", 230),
        ("crates/j2k/src/batch/admission/tests.rs", 150),
    ] {
        let source = fs::read_to_string(root.join(relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"));
        assert!(
            source.lines().count() < max_lines,
            "{relative} must stay below {max_lines} lines"
        );
    }
}
