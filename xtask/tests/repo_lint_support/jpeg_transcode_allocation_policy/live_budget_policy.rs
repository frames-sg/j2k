// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::{assert_pattern_checks, repo_root, PatternCheck};

fn read(relative: &str) -> String {
    fs::read_to_string(repo_root().join(relative))
        .unwrap_or_else(|error| panic!("failed to read {relative}: {error}"))
}

#[test]
fn transcode_phases_share_actual_capacity_with_lowered_native_caps() {
    let allocation = read("crates/j2k-transcode/src/allocation.rs");
    let live = read("crates/j2k-transcode/src/jpeg_to_htj2k/live_budget.rs");
    let scratch = read("crates/j2k-transcode/src/jpeg_to_htj2k/scratch.rs");
    let single = read("crates/j2k-transcode/src/jpeg_to_htj2k/single_tile_encode.rs");
    let batch = read("crates/j2k-transcode/src/jpeg_to_htj2k/batch/encode.rs");
    let batch_actual = read("crates/j2k-transcode/src/jpeg_to_htj2k/batch/actual_live.rs");
    let batch_individual = read("crates/j2k-transcode/src/jpeg_to_htj2k/batch/individual.rs");
    let batch_live = read("crates/j2k-transcode/src/jpeg_to_htj2k/batch/encode/live.rs");
    let batch_precomputed =
        read("crates/j2k-transcode/src/jpeg_to_htj2k/batch/encode/precomputed.rs");

    assert_pattern_checks(&[
        PatternCheck::new("transcode allocator actual capacity", &allocation).required(&[
            "checked_capacity_bytes::<T>(values.capacity())?",
            "try_host_vec_with_capacity(capacity)",
            "try_reserve_exact",
        ]),
        PatternCheck::new("transcode retained-owner budget", &live).required(&[
            "struct HostLiveBudget",
            "allocator_capacity",
            "remaining_bytes",
            "live_budget_accepts_exact_cap_and_rejects_one_byte_over",
        ]),
        PatternCheck::new("transcode reusable scratch accounting", &scratch).required(&[
            "self.dct53_grid.retained_bytes()?",
            "self.dct97_grid.retained_bytes()?",
            "self.integer_idct_blocks.capacity()",
        ]),
        PatternCheck::new("single-tile lowered native cap", &single).required(&[
            "encode_precomputed_htj2k_53_with_accelerator_and_max_host_bytes(",
            "encode_precomputed_htj2k_97_with_accelerator_and_max_host_bytes(",
            "max_host_bytes",
        ]),
        PatternCheck::new("sequential aggregate batch admission", &batch)
            .required(&[
                "checked_batch_live_bytes(",
                "completed_outputs",
                "drop(jpeg)",
                "encode_precomputed_htj2k_53_with_accelerator_and_max_host_bytes(",
                "encode_preencoded_htj2k_97_owned_with_accelerator_and_max_host_bytes(",
            ])
            .forbidden(&["prepared_tiles.into_par_iter()"]),
        PatternCheck::new("batch retained owner graph", &batch_live).required(&[
            "jpeg.retained_bytes()?",
            "block.encoded.data.capacity()",
            "block.coefficients.capacity()",
            "accumulated_batch_outputs_accept_exact_cap_and_reject_one_over",
        ]),
        PatternCheck::new(
            "parallel preparation actual-capacity boundary",
            &batch_actual,
        )
        .required(&[
            "validate_integer_prepared_collection(",
            "validate_float97_prepared_collection(",
            "outer_capacity",
            "scratch.retained_bytes()?",
            "budget.add_bytes(bytes?)?",
        ]),
        PatternCheck::new("individual batch accumulated outputs", &batch_individual).required(&[
            "completed_outputs",
            "encoded_transcode_retained_bytes(encoded)?",
            "external.live_bytes()",
        ]),
        PatternCheck::new("owned 9/7 batch lowered native cap", &batch_precomputed).required(&[
            "encode_precomputed_htj2k_97_batch_owned_with_accelerator_and_max_host_bytes(",
            "native_external.remaining_bytes()?",
            "remaining_codestreams",
        ]),
    ]);
}

#[test]
fn transcode_live_budget_policy_stays_focused() {
    assert!(include_str!("live_budget_policy.rs").lines().count() < 90);
}
