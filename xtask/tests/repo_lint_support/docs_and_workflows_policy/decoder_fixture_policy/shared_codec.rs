// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::repo_root;

#[test]
fn shared_codec_fixtures_are_split_by_format_and_provenance() {
    let root = repo_root();
    let source_root = root.join("crates/j2k-test-support/src/fixtures");
    let shell = fs::read_to_string(root.join("crates/j2k-test-support/src/fixtures.rs"))
        .expect("read shared fixture shell");
    let modules = [
        ("jpeg", 175),
        ("jp2", 175),
        ("generated_htj2k", 300),
        ("openjph", 350),
    ];

    assert!(
        shell.lines().count() < 75,
        "shared fixture root must remain a constants and re-export shell"
    );
    for (module, max_lines) in modules {
        assert!(shell.contains(&format!("mod {module};")));
        assert!(shell.contains(&format!("pub use {module}::{{")));
        let source = fs::read_to_string(source_root.join(format!("{module}.rs")))
            .unwrap_or_else(|error| panic!("read shared fixture module {module}: {error}"));
        assert!(
            source.lines().count() < max_lines,
            "shared fixture module {module} exceeded its focused line-count ratchet"
        );
    }
}
