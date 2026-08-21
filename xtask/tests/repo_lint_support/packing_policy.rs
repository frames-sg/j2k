// SPDX-License-Identifier: MIT OR Apache-2.0

//! Keep full-image and region output packing on one sample-conversion policy.

use std::fs;

use super::{repo_root, rust_sources};

#[test]
fn native_byte_packing_has_one_sample_conversion_owner() {
    let root = repo_root();
    let native_sources = rust_sources(&root.join("crates/j2k-native/src"));
    let mut policy_owners = Vec::new();
    let mut full_entrypoints = Vec::new();
    let mut region_entrypoints = Vec::new();

    for path in native_sources {
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        let relative = path
            .strip_prefix(root)
            .expect("native source is inside repository")
            .to_string_lossy()
            .into_owned();
        if source.contains("struct SampleConversionPolicy") {
            policy_owners.push(relative.clone());
        }
        if source.contains("fn interleave_and_convert(") {
            full_entrypoints.push(relative.clone());
        }
        if source.contains("fn interleave_and_convert_region(") {
            region_entrypoints.push(relative);
        }
    }

    let expected = vec!["crates/j2k-native/src/color/packing.rs".to_owned()];
    assert_eq!(
        policy_owners.as_slice(),
        expected.as_slice(),
        "sample conversion policy owner drifted"
    );
    assert_eq!(
        full_entrypoints.as_slice(),
        expected.as_slice(),
        "full-image packer owner drifted"
    );
    assert_eq!(
        region_entrypoints.as_slice(),
        expected.as_slice(),
        "region packer owner drifted"
    );
}

#[test]
fn shared_packing_policy_keeps_fast_paths_and_checked_window() {
    let source = fs::read_to_string(repo_root().join("crates/j2k-native/src/color/packing.rs"))
        .expect("read shared packing policy");
    for required in [
        "fn quantize(",
        "struct SampleWindow",
        "fn interleave_window(",
        "match num_components",
        "1 =>",
        "2 =>",
        "3 =>",
        "4 =>",
        "checked_mul",
        "OutputBufferTooSmall",
    ] {
        assert!(
            source.contains(required),
            "packing invariant missing: {required}"
        );
    }
}
