// SPDX-License-Identifier: MIT OR Apache-2.0

//! Structural ownership and size ratchets for changed-line coverage tooling.

use std::fs;

use super::repo_root;

mod coordination_parsing;
mod evaluation_exclusion;
mod line_ratchets;
mod source_analysis_tests;

fn read(relative_path: &str) -> String {
    fs::read_to_string(repo_root().join(relative_path))
        .unwrap_or_else(|error| panic!("read {relative_path}: {error}"))
}
