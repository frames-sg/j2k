// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::{assert_pattern_checks, repo_root, PatternCheck};

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
