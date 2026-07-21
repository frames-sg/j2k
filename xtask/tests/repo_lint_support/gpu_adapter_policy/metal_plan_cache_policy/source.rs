// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::super::super::{repo_root, rust_sources};

pub(super) fn read(relative: &str) -> String {
    fs::read_to_string(repo_root().join(relative))
        .unwrap_or_else(|error| panic!("read {relative}: {error}"))
}

pub(super) fn cache_family_sources() -> Vec<(String, String)> {
    let root = repo_root();
    let mut paths = rust_sources(&root.join("crates/j2k-metal/src/session"));
    paths.push(root.join("crates/j2k-metal/src/session.rs"));
    paths.push(root.join("crates/j2k-metal/src/hybrid.rs"));
    paths.sort();
    paths.dedup();
    paths
        .into_iter()
        .map(|path| {
            let relative = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .into_owned();
            let source = fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("read {relative}: {error}"));
            (relative, source)
        })
        .collect()
}

pub(super) fn owner_graph_sources() -> Vec<(String, String)> {
    [
        "crates/j2k-native/src/direct_plan.rs",
        "crates/j2k-native/src/direct_plan/allocation.rs",
        "crates/j2k-metal/src/compute/direct_cache.rs",
        "crates/j2k-metal/src/compute/direct_plan_types.rs",
        "crates/j2k-metal/src/compute/direct_plan_types/allocation.rs",
        "crates/j2k-metal/src/decoder/core.rs",
        "crates/j2k-metal/src/decoder/direct_paths.rs",
        "crates/j2k-metal/src/decoder/tests.rs",
        "crates/j2k-metal/src/session.rs",
        "crates/j2k-metal/src/session/direct_plan_cache.rs",
        "crates/j2k-metal/src/hybrid.rs",
    ]
    .into_iter()
    .map(|relative| (relative.to_string(), read(relative)))
    .collect()
}
