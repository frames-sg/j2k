// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::*;

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one table-driven policy verifies the planning, cache, and status test ownership split"
)]
fn metal_ht_chunk_tests_are_split_by_planning_cache_and_status_behavior() {
    let root = repo_root();
    let ht_chunks_root = root.join("crates/j2k-metal/src/compute/decode_dispatch/ht_chunks");
    let tests_root = root.join("crates/j2k-metal/src/compute/decode_dispatch/ht_chunks/tests");
    let ht_chunks_shell =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/decode_dispatch/ht_chunks.rs"))
            .expect("read Metal HT chunk module shell");
    let shell = fs::read_to_string(
        root.join("crates/j2k-metal/src/compute/decode_dispatch/ht_chunks/tests.rs"),
    )
    .expect("read Metal HT chunk test shell");

    assert_pattern_checks(&[PatternCheck::new("Metal HT chunk test shell", &shell)
        .required(&["mod cache;", "mod parity;", "mod planning;", "mod status;"])
        .forbidden(&[
            "fn tiny_caps_split_jobs_in_pass_order_and_keep_source_mapping(",
            "fn second_prepared_ht_submission_reuses_immutable_gpu_arenas(",
            "fn reused_ht_status_storage_is_overwritten_by_every_dispatched_job(",
        ])]);
    assert!(
        shell.lines().count() < 25,
        "HT chunk test shell must only own behavior-module wiring"
    );

    for (name, required, max_lines) in [
        (
            "planning.rs",
            "tiny_caps_split_jobs_in_pass_order_and_keep_source_mapping",
            225,
        ),
        (
            "parity.rs",
            "forced_multi_chunk_metal_output_matches_cpu_coefficients",
            150,
        ),
        (
            "cache.rs",
            "second_prepared_ht_submission_reuses_immutable_gpu_arenas",
            150,
        ),
        (
            "status.rs",
            "reused_ht_status_storage_is_overwritten_by_every_dispatched_job",
            425,
        ),
    ] {
        let path = tests_root.join(name);
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        assert!(
            source.contains(required),
            "{} must own {required}",
            path.display()
        );
        assert!(
            source.lines().count() < max_lines,
            "{} must stay below its focused-test line-count ratchet",
            path.display()
        );
    }

    let planning = fs::read_to_string(tests_root.join("planning.rs"))
        .expect("read Metal HT chunk planning test owner");
    let cache = fs::read_to_string(tests_root.join("cache.rs"))
        .expect("read Metal HT chunk cache test owner");
    assert_pattern_checks(&[
        PatternCheck::new("Metal HT chunk module shell", &ht_chunks_shell)
            .forbidden(&["mod referenced_tests;"]),
        PatternCheck::new("Metal HT chunk planning test owner", &planning)
            .required(&["mod fixtures;", "mod referenced;"]),
        PatternCheck::new("Metal HT chunk cache test owner", &cache).required(&["mod referenced;"]),
    ]);
    for (path, required, max_lines) in [
        (
            tests_root.join("planning/fixtures.rs"),
            "prepared_fixtures_select_expected_dedicated_metal_ht_pipelines",
            150,
        ),
        (
            tests_root.join("planning/referenced.rs"),
            "referenced_payload_pack_preserves_job_order_and_rebases_offsets",
            150,
        ),
        (
            tests_root.join("cache/referenced.rs"),
            "second_referenced_prepared_submission_uploads_no_immutable_arenas",
            150,
        ),
    ] {
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        assert!(
            source.contains(required),
            "{} must own {required}",
            path.display()
        );
        assert!(
            source.lines().count() < max_lines,
            "{} must stay below its focused-test line-count ratchet",
            path.display()
        );
    }
    assert!(
        !ht_chunks_root.join("referenced_tests.rs").exists(),
        "referenced HT planning and cache tests must live with their behavior owners"
    );
}

#[test]
fn metal_stacked_execution_is_split_by_codec_stage() {
    let root = repo_root().join("crates/j2k-metal/src/compute/direct_stacked_batch");
    let staged_modules = [
        ("command_submission.rs", 250),
        ("command_submission/classic_tier1.rs", 400),
        ("command_submission/ht_tier1.rs", 400),
        ("command_submission/reconstruction.rs", 300),
        ("command_submission/final_store.rs", 200),
        ("repeated_grayscale/execution.rs", 200),
        ("repeated_grayscale/execution/tier1.rs", 300),
        ("repeated_grayscale/execution/reconstruction.rs", 250),
        ("repeated_grayscale/execution/final_store.rs", 250),
    ];

    for (relative, max_lines) in staged_modules {
        let path = root.join(relative);
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        assert!(
            source.lines().count() < max_lines,
            "{} must stay below its codec-stage line-count ratchet",
            path.display()
        );
    }

    let submission = fs::read_to_string(root.join("command_submission.rs"))
        .expect("read stacked command-submission shell");
    let repeated = fs::read_to_string(root.join("repeated_grayscale/execution.rs"))
        .expect("read repeated grayscale execution shell");
    assert_pattern_checks(&[
        PatternCheck::new("stacked command-submission stage shell", &submission)
            .required(&[
                "mod classic_tier1;",
                "mod final_store;",
                "mod ht_tier1;",
                "mod reconstruction;",
            ])
            .forbidden(&[
                "fn submit_classic_group(",
                "fn submit_ht_group(",
                "fn submit_idwt(",
                "fn submit_store(",
            ]),
        PatternCheck::new("repeated grayscale execution stage shell", &repeated)
            .required(&["mod final_store;", "mod reconstruction;", "mod tier1;"])
            .forbidden(&[
                "fn encode_classic_group(",
                "fn encode_ht_group(",
                "fn encode_idwt(",
                "fn encode_store(",
            ]),
    ]);
}

#[test]
fn metal_direct_destination_is_split_by_submission_and_group_encoding() {
    let root = repo_root().join("crates/j2k-metal/src/compute/direct_grayscale_execute");
    let facade = fs::read_to_string(root.join("destination.rs"))
        .expect("read direct grayscale destination facade");
    let submission = fs::read_to_string(root.join("destination/submission.rs"))
        .expect("read direct destination submission lifecycle module");
    let group_encode = fs::read_to_string(root.join("destination/group_encode.rs"))
        .expect("read direct destination group encoding module");

    assert_pattern_checks(&[
        PatternCheck::new("direct grayscale destination facade", &facade)
            .required(&[
                "mod group_encode;",
                "mod submission;",
                "submit_prepared_direct_grayscale_plan_batch_into_group",
            ])
            .forbidden(&[
                "struct SubmittedDirectDestination",
                "struct GrayscaleGroupEncoder",
                "fn commit_direct_destination(",
                "fn encode_stacked_grayscale_destination(",
            ]),
        PatternCheck::new("direct destination submission lifecycle", &submission).required(&[
            "enum DirectDestinationConsumerOrdering",
            "struct SubmittedDirectDestination",
            "fn commit_direct_destination(",
            "impl Drop for SubmittedDirectDestination",
        ]),
        PatternCheck::new("direct grayscale group encoding", &group_encode).required(&[
            "struct GrayscaleGroupEncoder",
            "fn encode_stacked(",
            "fn encode_individually(",
            "fn encode_stacked_grayscale_destination(",
        ]),
    ]);
    for (relative, source, max_lines) in [
        ("destination.rs", &facade, 150),
        ("destination/submission.rs", &submission, 325),
        ("destination/group_encode.rs", &group_encode, 275),
    ] {
        assert!(
            source.lines().count() < max_lines,
            "{relative} must stay below its focused-responsibility line-count ratchet"
        );
    }
}

#[test]
fn metal_distinct_classic_batch_execution_is_split_from_cleanup_dispatch() {
    let root = repo_root().join("crates/j2k-metal/src/compute/decode_dispatch");
    let cleanup = fs::read_to_string(root.join("classic_cleanup.rs"))
        .expect("read Metal classic cleanup module");
    let distinct = fs::read_to_string(root.join("classic_cleanup/distinct_batch.rs"))
        .expect("read Metal distinct classic batch module");

    assert_pattern_checks(&[
        PatternCheck::new("Metal classic cleanup dispatch ownership", &cleanup)
            .required(&["mod distinct_batch;"])
            .forbidden(&["fn append_distinct_classic_batch("]),
        PatternCheck::new("Metal distinct classic batch ownership", &distinct).required(&[
            "fn append_distinct_classic_batch(",
            "fn encode_distinct_classic_batches_to_buffer_in_encoder",
            "dispatch_zero_u32_buffer_in_encoder",
        ]),
    ]);
}
