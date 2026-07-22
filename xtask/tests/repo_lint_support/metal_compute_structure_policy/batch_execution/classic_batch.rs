// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::{assert_pattern_checks, repo_root, PatternCheck};

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

#[test]
fn repeated_classic_subband_variants_share_one_execution_lifecycle() {
    let source = fs::read_to_string(
        repo_root().join("crates/j2k-metal/src/compute/decode_dispatch/classic_subband.rs"),
    )
    .expect("read Metal classic sub-band execution module");

    assert_pattern_checks(&[PatternCheck::new(
        "Metal classic sub-band shared execution ownership",
        &source,
    )
    .required(&[
        "struct ClassicBatchView",
        "fn encode_repeated_classic_batch_to_buffer_in_command_buffer(",
    ])
    .forbidden(&["clippy::too_many_lines"])]);
    assert_eq!(
        source
            .matches("encode_repeated_classic_batch_to_buffer_in_command_buffer(")
            .count(),
        3,
        "the repeated execution owner must have one definition and two typed entry-point calls"
    );
}
