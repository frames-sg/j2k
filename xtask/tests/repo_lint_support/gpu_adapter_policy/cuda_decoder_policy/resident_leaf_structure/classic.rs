// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::super::super::super::repo_root;

#[test]
fn classic_and_ht_component_leaves_stay_focused() {
    let root = repo_root();
    for (name, relative, limit) in [
        (
            "classic cleanup submission",
            "crates/j2k-cuda/src/decoder/resident/cleanup_dequant/classic.rs",
            100,
        ),
        (
            "classic component planning",
            "crates/j2k-cuda/src/decoder/resident/component/classic.rs",
            175,
        ),
        (
            "HT component job conversion",
            "crates/j2k-cuda/src/decoder/resident/component/ht.rs",
            50,
        ),
    ] {
        let source = fs::read_to_string(root.join(relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"));
        assert!(
            source.lines().count() < limit,
            "CUDA resident {name} leaf must stay focused"
        );
    }
}
