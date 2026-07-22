// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::repo_root;

mod adapter;
mod benchmark_evidence;
mod benchmark_prepare_policy;
mod benchmark_support_structure;
mod features;
mod interop;

fn read(relative: &str) -> String {
    fs::read_to_string(repo_root().join(relative)).unwrap_or_else(|error| {
        panic!("read {relative}: {error}");
    })
}
