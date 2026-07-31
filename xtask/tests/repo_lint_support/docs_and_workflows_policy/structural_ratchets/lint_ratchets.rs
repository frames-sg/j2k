// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::repo_root;

fn source_before_cfg_test_module<'a>(source: &'a str, relative: &str) -> &'a str {
    source.split_once("#[cfg(test)]\nmod tests").map_or_else(
        || {
            assert!(
                !relative.ends_with("/tests.rs"),
                "{relative} is test-only and must not enter the production panic scan"
            );
            source
        },
        |(production, _)| production,
    )
}

#[test]
fn panic_hotspot_production_paths_do_not_use_unwrap_or_expect() {
    for relative in [
        "crates/j2k-cuda/src/encode.rs",
        "crates/j2k-jpeg/src/entropy/block.rs",
        "crates/j2k-jpeg/src/entropy/huffman.rs",
        "crates/j2k-jpeg/src/entropy/progressive.rs",
        "crates/j2k-jpeg/src/entropy/progressive/model.rs",
        "crates/j2k-jpeg/src/entropy/progressive/allocation.rs",
        "crates/j2k-jpeg/src/entropy/progressive/scan.rs",
        "crates/j2k-jpeg/src/entropy/progressive/terminal.rs",
        "crates/j2k-jpeg/src/entropy/progressive/render.rs",
        "crates/j2k-jpeg/src/entropy/sequential.rs",
    ] {
        let source = fs::read_to_string(repo_root().join(relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"));
        let production = source_before_cfg_test_module(&source, relative);
        for forbidden in [".unwrap(", ".expect("] {
            assert!(
                !production.contains(forbidden),
                "{relative} production path must not use panic-on-error `{forbidden}`"
            );
        }
    }
}
