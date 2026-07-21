// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::{assert_pattern_checks, repo_root, PatternCheck};

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
fn metal_prepared_ht_execution_cache_is_split_from_immutable_execution() {
    let root = repo_root().join("crates/j2k-metal/src/compute/decode_dispatch/ht_chunks");
    let prepared =
        fs::read_to_string(root.join("prepared.rs")).expect("read prepared HT execution owner");
    let cache = fs::read_to_string(root.join("prepared/cache.rs"))
        .expect("read prepared HT execution cache owner");
    let entry = fs::read_to_string(root.join("prepared/cache/entry.rs"))
        .expect("read prepared HT execution cache-entry owner");

    assert_pattern_checks(&[
        PatternCheck::new("prepared HT immutable execution ownership", &prepared)
            .required(&[
                "pub(super) mod cache;",
                "struct PreparedMetalHtExecution",
                "struct PreparedMetalHtChunk",
                "fn prepare(",
                "let packed = plan.pack_chunk(chunk_index)?;",
            ])
            .forbidden(&[
                "struct PreparedMetalHtInputKey",
                "enum PreparedMetalHtPayloadKey",
                "struct PreparedMetalHtExecutionCacheEntry",
                "struct PreparedMetalHtExecutionCache",
                "fn get_or_prepare(",
                "fn find(",
                "fn evict_until_fits(",
                "fn retained_bytes(",
                "fn host_bytes(",
                "fn device_bytes(",
            ]),
        PatternCheck::new("prepared HT execution cache ownership", &cache)
            .required(&[
                "mod entry;",
                "struct PreparedMetalHtExecutionCache",
                "fn get_or_prepare(",
                "fn prepare_entry(",
                "fn insert_entry(",
                "fn find(",
                "fn evict_until_fits(",
                "fn prepared_metal_ht_execution(",
            ])
            .forbidden(&[
                "struct PreparedMetalHtInputKey",
                "enum PreparedMetalHtPayloadKey",
                "fn retained_bytes(",
                "fn host_bytes(",
                "fn device_bytes(",
                "let packed = plan.pack_chunk(chunk_index)?;",
                "copied_slice_buffer(device, &packed.coded_data)",
                "copied_slice_buffer(device, &packed.jobs)",
            ]),
        PatternCheck::new("prepared HT cache-entry ownership", &entry)
            .required(&[
                "struct PreparedMetalHtInputKey",
                "enum PreparedMetalHtPayloadKey",
                "struct PreparedMetalHtExecutionCacheEntry",
                "fn prepare(",
                "fn matches(",
                "fn retained_bytes(",
                "fn host_bytes(",
                "fn device_bytes(",
            ])
            .forbidden(&[
                "fn get_or_prepare(",
                "fn insert_entry(",
                "fn evict_until_fits(",
            ]),
    ]);
    assert!(
        prepared.lines().count() < 175,
        "prepared.rs must remain the focused immutable execution/chunk owner"
    );
    assert!(
        cache.lines().count() < 275,
        "prepared/cache.rs must remain the focused cache-policy owner"
    );
    assert!(
        entry.lines().count() < 275,
        "prepared/cache/entry.rs must remain the focused identity/accounting owner"
    );

    let (_, lookup_tail) = cache
        .split_once("fn get_or_prepare(")
        .expect("find prepared HT cache lookup orchestrator");
    let (lookup, _) = lookup_tail
        .split_once("\n    fn prepare_entry(")
        .expect("isolate prepared HT cache lookup orchestrator");
    for phase in ["self.prepare_entry(", "self.insert_entry("] {
        assert!(lookup.contains(phase), "cache lookup must delegate {phase}");
    }
    for implementation_detail in [
        "PreparedMetalHtInputKey {",
        "self.entries.push(",
        "input_key_host_bytes(",
    ] {
        assert!(
            !lookup.contains(implementation_detail),
            "cache lookup must not own {implementation_detail}"
        );
    }
}
