// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{fs, path::Path};

pub(super) fn assert_direct_executor_line_ratchets(root: &Path) {
    let source_root = root.join("crates/j2k-metal/src");
    for (relative, max_lines) in [
        ("compute/direct_commands.rs", 160),
        ("compute/direct_grayscale_execute.rs", 500),
        ("compute/direct_grayscale_execute/allocation.rs", 150),
        ("compute/direct_grayscale_execute/single.rs", 350),
        ("compute/direct_grayscale_execute/component_plane.rs", 100),
        (
            "compute/direct_grayscale_execute/component_plane/execution.rs",
            450,
        ),
        (
            "compute/direct_grayscale_execute/component_plane/execution/final_plane.rs",
            150,
        ),
        ("compute/direct_stacked_batch/repeated_grayscale.rs", 100),
        (
            "compute/direct_stacked_batch/repeated_grayscale/execution.rs",
            600,
        ),
    ] {
        let path = source_root.join(relative);
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        assert!(
            source.lines().count() < max_lines,
            "{relative} must stay below its direct-executor line-count ratchet"
        );
    }
}
