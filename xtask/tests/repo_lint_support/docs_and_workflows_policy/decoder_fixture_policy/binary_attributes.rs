// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::repo_root;

#[test]
fn binary_codec_fixtures_disable_git_text_conversion() {
    let attributes = fs::read_to_string(repo_root().join(".gitattributes"))
        .expect("read repository Git attributes");

    for extension in ["ppm", "pgm", "raw", "gray", "j2k", "j2c", "jp2", "jph"] {
        let rule = format!("*.{extension} binary");
        assert!(
            attributes.lines().any(|line| line.trim() == rule),
            "{rule} must disable text conversion for binary codec fixtures"
        );
    }
}
