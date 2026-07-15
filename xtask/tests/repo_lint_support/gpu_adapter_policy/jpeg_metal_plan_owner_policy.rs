// SPDX-License-Identifier: MIT OR Apache-2.0

//! Retained owner-graph bounds for every JPEG Metal queued-request collection.

use super::*;

struct PlanOwnerSources {
    ledger_main: String,
    ledger: String,
    session_main: String,
    session: String,
    raw_builders: String,
    shared_input: String,
    tests: String,
    execution: String,
}

impl PlanOwnerSources {
    fn read(root: &Path) -> Self {
        Self {
            ledger_main: read_source_files(
                root,
                &["crates/j2k-jpeg-metal/src/plan_owner_ledger.rs"],
            ),
            ledger: read_source_files(
                root,
                &[
                    "crates/j2k-jpeg-metal/src/plan_owner_ledger.rs",
                    "crates/j2k-jpeg-metal/src/plan_owner_ledger/execution.rs",
                    "crates/j2k-jpeg-metal/src/plan_owner_ledger/request_count.rs",
                ],
            ),
            session_main: read_source_files(root, &["crates/j2k-jpeg-metal/src/session.rs"]),
            session: read_source_files(
                root,
                &[
                    "crates/j2k-jpeg-metal/src/session.rs",
                    "crates/j2k-jpeg-metal/src/session/allocation.rs",
                    "crates/j2k-jpeg-metal/src/session/completions.rs",
                ],
            ),
            raw_builders: read_source_files(
                root,
                &[
                    "crates/j2k-jpeg-metal/src/codec_batch.rs",
                    "crates/j2k-jpeg-metal/src/lib.rs",
                ],
            ),
            shared_input: read_source_files(
                root,
                &[
                    "crates/j2k-jpeg-metal/src/session.rs",
                    "crates/j2k-jpeg-metal/src/tile_batch.rs",
                ],
            ),
            tests: read_source_files(
                root,
                &[
                    "crates/j2k-jpeg-metal/src/session/tests/ledger.rs",
                    "crates/j2k-jpeg-metal/src/session/tests/ledger/completions.rs",
                    "crates/j2k-jpeg-metal/src/session/tests/ledger/owners.rs",
                    "crates/j2k-jpeg-metal/src/session/tests/ledger/transactions.rs",
                    "crates/j2k-jpeg-metal/src/batch.rs",
                    "crates/j2k-jpeg-metal/src/tile_batch.rs",
                ],
            ),
            execution: read_source_files(
                root,
                &[
                    "crates/j2k-jpeg-metal/src/batch.rs",
                    "crates/j2k-jpeg-metal/src/batch/flush.rs",
                    "crates/j2k-jpeg-metal/src/batch/grouping.rs",
                    "crates/j2k-jpeg-metal/src/lib.rs",
                    "crates/j2k-jpeg-metal/src/surface.rs",
                    "crates/j2k-jpeg-metal/src/surface/batch_buffer.rs",
                    "crates/j2k-jpeg-metal/src/surface/batch_texture.rs",
                    "crates/j2k-jpeg-metal/src/surface/resident_tile.rs",
                    "crates/j2k-jpeg-metal/src/surface/texture_tile.rs",
                ],
            ),
        }
    }
}

#[test]
fn jpeg_metal_request_collections_share_one_identity_aware_owner_ledger() {
    let root = repo_root();
    let sources = PlanOwnerSources::read(root);

    assert_pattern_checks(&[
        PatternCheck::new(
            "JPEG Metal independent owner identity ledger",
            &sources.ledger,
        )
        .required(&[
            "DEFAULT_MAX_HOST_ALLOCATION_BYTES",
            "SharedJpegInput::ptr_eq(&queued.input, &request.input)",
            "SharedJpegFastPacket::ptr_eq(owner, packet)",
            "request.retained_input_bytes()?",
            "request.retained_packet_bytes()?",
            "cache_retained_bytes.checked_add(retained_bytes)",
            "what: \"JPEG Metal queued and cached plan owner graphs\"",
            "pub(crate) const fn commit",
            "pub(crate) const fn reset",
            "MAX_QUEUED_JPEG_REQUESTS: usize = 4096",
            "preflight_request_count(retained.len())?",
            "identity_scan_work_bound_accepts_exact_and_rejects_one_more",
        ])
        .forbidden(&["repeated_owner", "retained_plan_bytes"]),
        PatternCheck::new("JPEG Metal session owner admission", &sources.session).required(&[
            "queued_plan_ledger.preflight(",
            "self.queued_plan_ledger.commit(owner_admission)",
            "self.queued_plan_ledger.reset()",
            "pub(crate) fn take_queued_requests",
            "pub(crate) fn queue_request_with_retained_metadata",
            "JPEG Metal transactional queue growth",
            "completed_host_bytes",
            "resolve_jpeg_plan_with_external_live(",
            ".resolve_with_external_live(input, adapter_live_bytes)",
            "try_copy_from_slice_with_external_live(",
        ]),
        PatternCheck::new(
            "JPEG Metal adapter-local request owner admission",
            &sources.raw_builders,
        )
        .required(&[
            "let mut plan_owners = PlanOwnerLedger::default()",
            "plan_owners.preflight(",
            "plan_owners.commit(admission)",
            "preflight_collective_metadata(",
            "state.jpeg_plan_cache_diagnostics().retained_bytes",
        ]),
        PatternCheck::new("JPEG Metal shared Arc input path", &sources.shared_input).required(&[
            "TileRequestInput::Shared(input)",
            "resolve_arc_jpeg_plan_with_external_live(",
            "SharedJpegInput::try_from_arc_with_external_live(",
            "resolve_shared_jpeg_plan_with_external_live(",
            "shared_tile_request_reuses_caller_arc_payload_without_copying",
        ]),
        PatternCheck::new("JPEG Metal owner-ledger regressions", &sources.tests).required(&[
            "queued_and_cached_plan_limit_accepts_exact_and_rejects_one_over_before_mutation",
            "plan_build_baseline_accepts_exact_hit_and_rejects_one_over_before_admission",
            "repeated_fully_shared_plan_is_charged_once_in_the_queue",
            "same_input_adds_a_later_distinct_packet_owner",
            "disabled_cache_charges_each_new_packet_arc_for_the_same_input",
            "more_than_eight_distinct_queued_plans_remain_accounted_after_cache_eviction",
            "transactional_queue_growth_rejects_live_old_plus_new_without_mutating_session",
            "completed_host_outputs_compose_at_exact_cap_and_reject_one_over_transactionally",
            "grouped_execution_persists_all_group_metadata_and_owner_baselines",
        ]),
        PatternCheck::new(
            "JPEG Metal execution live-owner propagation",
            &sources.execution,
        )
        .required(&[
            "batch_execution_budget(",
            "execution_collective_owner_bytes",
            "grouped_request_metadata_bytes",
            "sort_unstable_by_key",
            "decode_surface_from_shared_input",
            "decoder_with_external_live",
            "retained_host_capacity_bytes",
            "store_completed_result",
            "queue_request_with_retained_metadata",
        ]),
    ]);
    assert_owner_structure(root, &sources);
}

fn assert_owner_structure(root: &Path, sources: &PlanOwnerSources) {
    assert_eq!(
        sources
            .raw_builders
            .matches("plan_owners.preflight(")
            .count(),
        3,
        "both RGB8 builders and the direct device batch must preflight owner graphs"
    );
    assert!(
        sources.ledger_main.lines().count() < 140 && sources.session_main.lines().count() < 650,
        "JPEG Metal session/owner-ledger production must remain focused"
    );
    for source in [
        "crates/j2k-jpeg-metal/src/batch.rs",
        "crates/j2k-jpeg-metal/src/batch/flush.rs",
        "crates/j2k-jpeg-metal/src/batch/grouping.rs",
    ] {
        let lines = fs::read_to_string(root.join(source))
            .unwrap_or_else(|error| panic!("read {source}: {error}"))
            .lines()
            .count();
        assert!(lines < 550, "{source} must remain a focused batch module");
    }
}
