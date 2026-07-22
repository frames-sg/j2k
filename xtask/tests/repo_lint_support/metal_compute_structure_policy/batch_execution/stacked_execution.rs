// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::{assert_pattern_checks, repo_root, PatternCheck};

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
