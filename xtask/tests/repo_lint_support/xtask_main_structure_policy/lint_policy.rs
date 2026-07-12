// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::super::{assert_pattern_checks, repo_root, rust_sources, PatternCheck};
use super::read;

#[test]
fn xtask_manifest_keeps_pedantic_clippy_enabled() {
    let manifest = read("xtask/Cargo.toml");
    assert_pattern_checks(&[PatternCheck::new("xtask manifest", &manifest)
        .normalized_required(&["pedantic = { level = \"warn\", priority = -1 }"])
        .normalized_forbidden(&["pedantic = \"allow\"", "pedantic = { level = \"allow\""])]);
    for relative_dir in ["xtask/src", "xtask/tests"] {
        for path in rust_sources(&repo_root().join(relative_dir)) {
            let source = fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
            assert!(
                source
                    .lines()
                    .all(|line| !line.trim_start().starts_with(concat!("#!", "[allow"))),
                "{} must not use crate- or module-wide lint allows",
                path.strip_prefix(repo_root()).unwrap_or(&path).display()
            );
        }
    }
}
