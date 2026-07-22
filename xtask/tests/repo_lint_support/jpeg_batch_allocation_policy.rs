// SPDX-License-Identifier: MIT OR Apache-2.0

//! Aggregate, fallible ownership contract for JPEG batch/session decode.

use std::fs;

use super::{assert_pattern_checks, read_source_files, repo_root, PatternCheck};

#[test]
fn jpeg_batch_session_keeps_scheduling_and_results_fallible() {
    let root = repo_root();
    let production = read_source_files(
        root,
        &[
            "crates/j2k-jpeg/src/batch_session.rs",
            "crates/j2k-jpeg/src/batch_session/runtime.rs",
            "crates/j2k-jpeg/src/batch_session/scheduler.rs",
            "crates/j2k-jpeg/src/batch_session/collection.rs",
            "crates/j2k-jpeg/src/batch_session/planning.rs",
            "crates/j2k-jpeg/src/batch_session/worker.rs",
        ],
    );
    let allocation =
        fs::read_to_string(root.join("crates/j2k-jpeg/src/batch_session/allocation.rs"))
            .expect("read JPEG batch allocation planner");

    assert_pattern_checks(&[
        PatternCheck::new("JPEG fallible batch scheduler", &production)
            .required(&[
                "fn prepare_batch<R, O>(",
                "fn prepare_job_planning(",
                "select_batch_plan(",
                "try_vec_with_capacity(",
                "spawn_scoped(scope",
                "BatchInfrastructureError::WorkerSpawnFailed",
                "BatchInfrastructureError::ParallelWorkerPanicked",
                "try_collect_ordered_batch_results_with_limits(",
                "retain_within_planned_warning_claim(",
                "zip(results.par_chunks_mut(plan.chunk_size))",
                "zip(results.chunks_mut(plan.chunk_size))",
                "ensure_plan_capacity(plan, capacity_extra)",
            ])
            .forbidden(&[
                "Vec::with_capacity(",
                "scope.spawn(",
                "std::panic::resume_unwind",
                "collect_indexed_batch_results(job_count",
                "expect(\"JPEG batch worker slot poisoned\")",
            ]),
        PatternCheck::new("JPEG aggregate batch allocation plan", &allocation)
            .required(&[
                "JPEG_BATCH_HOST_CAP_BYTES",
                "JPEG_BATCH_METADATA_ALLOWANCE_BYTES",
                "JPEG_CODEC_HOST_CAP_BYTES",
                "struct BatchMetadataLayout",
                "struct BatchPlan",
                "fn select_batch_plan_with_limits(",
                "fn ensure_planning_phase(",
                "fn try_vec_with_retained_metadata<T>(",
                "job.retained_result_bytes()",
                "retained_worker_bytes(worker_index)",
                "HostAllocationFailed",
            ])
            .forbidden(&["Vec::with_capacity(", "vec![", "panic!("]),
    ]);
}

#[test]
fn prepared_batch_and_shared_collector_expose_typed_outer_failures() {
    let root = repo_root();
    let tile = fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/tile.rs"))
        .expect("read JPEG tile facade");
    let decoder = fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder.rs"))
        .expect("read JPEG decoder facade");
    let core = read_source_files(
        root,
        &[
            "crates/j2k-core/src/batch.rs",
            "crates/j2k-core/src/batch/collection.rs",
        ],
    );

    assert_pattern_checks(&[
        PatternCheck::new("prepared JPEG outer failure API", &tile).required(&[
            "pub fn decode_prepared_jpeg_tiles_rgb8(",
            "crate::PreparedTileBatchError",
            "planned_jpeg_tile_decode_live_bytes(",
        ]),
        PatternCheck::new("JPEG batch error aliases", &decoder).required(&[
            "pub type TileBatchError = j2k_core::BatchDecodeError<JpegError>;",
            "pub type PreparedTileBatchError = j2k_core::BatchInfrastructureError;",
        ]),
        PatternCheck::new("shared typed fallible batch collection", &core).required(&[
            "pub enum BatchInfrastructureError",
            "pub enum BatchDecodeError<E>",
            "pub fn try_collect_indexed_batch_results<T, E>(",
            "pub fn try_collect_ordered_batch_results_with_limits<T, E>(",
            "retained_live_bytes.max(retained_collection_bytes)",
            "ResultIndexOutOfBounds",
            "DuplicateResult",
            "MissingResult",
            "ResultKindMismatch",
            "allocation_bytes::<T>(ordered.capacity())",
        ]),
    ]);
}

#[test]
fn jpeg_batch_boundaries_reuse_and_allocator_categories_remain_covered() {
    let root = repo_root();
    let coverage = read_source_files(
        root,
        &[
            "crates/j2k-jpeg/src/batch_session/allocation.rs",
            "crates/j2k-jpeg/src/batch_session/allocation/tests.rs",
            "crates/j2k-jpeg/src/batch_session/collection.rs",
            "crates/j2k-jpeg/src/batch_session/scheduler.rs",
            "crates/j2k-jpeg/src/batch_session/worker.rs",
        ],
    );

    assert_pattern_checks(&[
        PatternCheck::new("JPEG batch allocation regressions", &coverage).required(&[
            "exact_cap_is_accepted_and_one_over_is_rejected",
            "planner_reduces_concurrency_to_fit_aggregate_worker_claims",
            "stale_retained_worker_bytes_participate_in_the_next_plan",
            "overflow_is_a_cap_error_not_an_allocator_error",
            "overflowing_high_concurrency_candidate_can_reduce_to_a_fitting_worker",
            "allocator_failure_category_is_not_flattened_into_a_cap_error",
            "fallible_vector_rejects_over_cap_before_allocator_entry",
            "planning_phase_accepts_maximum_codec_plus_exact_metadata_and_rejects_one_over",
            "metadata_vectors_share_one_allowance_instead_of_individual_caps",
            "retained_metadata_and_actual_summary_share_one_exact_boundary",
            "completed_warning_owner_cannot_exceed_planned_metadata_claim",
            "allocator_capacity_delta_has_an_exact_plan_boundary",
            "missing_worker_slot_is_typed_instead_of_index_panicking",
            "prepared_outer_collection_reports_typed_cap_failure",
            "prepared_outer_collection_distinguishes_excess_from_missing_slots",
        ]),
    ]);
}
