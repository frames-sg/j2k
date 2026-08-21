// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{collections::BTreeSet, fs};

use crate::repo_lint_support::{repo_root, rust_sources};

#[test]
fn metal_raw_buffer_contents_access_stays_confined_to_checked_helpers() {
    let root = repo_root();
    let allowed = BTreeSet::from([
        "crates/j2k-metal-support/src/buffer_access.rs",
        "crates/j2k-metal/src/engine/direct_buffers.rs",
        "crates/j2k-jpeg-metal/src/buffers.rs",
    ]);

    for source_root in [
        "crates/j2k-metal-support/src",
        "crates/j2k-metal/src",
        "crates/j2k-jpeg-metal/src",
        "crates/j2k-transcode-metal/src",
    ] {
        for path in rust_sources(&root.join(source_root)) {
            let relative = path
                .strip_prefix(root)
                .expect("source path under repo root")
                .to_string_lossy()
                .replace('\\', "/");
            if allowed.contains(relative.as_str()) {
                continue;
            }
            let source = fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("read {relative}: {error}"));
            assert!(
                !source.contains(".contents()"),
                "raw Metal buffer contents access must stay inside checked helpers; found in {relative}"
            );
        }
    }
}
