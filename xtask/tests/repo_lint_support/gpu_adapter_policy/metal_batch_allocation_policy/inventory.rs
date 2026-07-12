// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{collections::BTreeMap, fs, path::Path};

use super::super::super::{repo_root, rust_sources};

#[test]
fn every_remaining_raw_metal_capacity_has_an_explicit_non_batch_disposition() {
    let root = repo_root();
    let mut actual = BTreeMap::<String, usize>::new();
    for crate_dir in ["crates/j2k-metal/src", "crates/j2k-jpeg-metal/src"] {
        for path in rust_sources(&root.join(crate_dir)) {
            let relative = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            if is_test_or_tool_source(&relative) {
                continue;
            }
            let source = fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
            let production = source
                .split("#[cfg(test)]\nmod tests")
                .next()
                .unwrap_or(&source);
            let count = production.lines().filter(|line| raw_capacity(line)).count();
            if count != 0 {
                actual.insert(relative, count);
            }
        }
    }

    // Fixed J2K syntax/component bounds, fixed three-plane owners, profiling
    // bookkeeping, and coefficient payload transforms are intentionally outside
    // ALLOC-018. The latter remain owned by ALLOC-001/ALLOC-013; every
    // caller-count-derived adapter batch metadata owner must stay absent here.
    let expected = BTreeMap::from([
        ("crates/j2k-jpeg-metal/src/viewport/model.rs".to_string(), 1),
        (
            "crates/j2k-metal/src/compute/direct_stacked_batch.rs".to_string(),
            2,
        ),
        (
            "crates/j2k-metal/src/compute/forward_transform.rs".to_string(),
            3,
        ),
        ("crates/j2k-metal/src/compute/gpu_timing.rs".to_string(), 1),
        (
            "crates/j2k-metal/src/compute/lossless_prepare/batch_item.rs".to_string(),
            3,
        ),
        (
            "crates/j2k-metal/src/compute/lossless_prepare/single.rs".to_string(),
            3,
        ),
        ("crates/j2k-metal/src/encode/plan.rs".to_string(), 2),
    ]);
    assert_eq!(actual, expected, "raw Metal capacity inventory changed");
}

fn raw_capacity(line: &str) -> bool {
    line.contains("Vec::with_capacity")
        || (line.contains("Vec::<") && line.contains(">::with_capacity"))
}

fn is_test_or_tool_source(relative: &str) -> bool {
    let path = Path::new(relative);
    relative.contains("/tests/")
        || relative.contains("/test_support/")
        || relative.contains("/bin/")
        || path.file_name().is_some_and(|name| {
            let name = name.to_string_lossy();
            name == "tests.rs" || name == "test_helpers.rs" || name.ends_with("_tests.rs")
        })
}
