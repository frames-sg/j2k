// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{collections::BTreeSet, fs};

use super::{assert_pattern_checks, repo_root, PatternCheck};

fn read(relative_path: &str) -> String {
    fs::read_to_string(repo_root().join(relative_path))
        .unwrap_or_else(|error| panic!("read {relative_path}: {error}"))
}

fn braced_export_names<'a>(source: &'a str, prefix: &str) -> BTreeSet<&'a str> {
    let exports = source
        .split_once(prefix)
        .unwrap_or_else(|| panic!("missing export prefix `{prefix}`"))
        .1
        .split_once("};")
        .unwrap_or_else(|| panic!("unterminated export after `{prefix}`"))
        .0;
    exports
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .collect()
}

#[test]
fn transcode_accelerator_contracts_preserve_root_and_compatibility_paths() {
    let root = read("crates/j2k-transcode/src/lib.rs");
    let contracts = read("crates/j2k-transcode/src/accelerator_contracts.rs");
    assert!(
        root.lines().count() < 250,
        "j2k-transcode/src/lib.rs must stay below its public-contract boundary ratchet"
    );
    assert_pattern_checks(&[
        PatternCheck::new("transcode crate root", &root)
            .required(&[
                "mod accelerator_contracts;",
                "pub use self::accelerator_contracts::{",
                "#[doc(hidden)]\npub mod accelerator {",
                "pub use crate::{",
                "#[cfg(feature = \"dev-support\")]\n#[doc(hidden)]\npub mod dev_support;",
                "pub struct DctGridToReversibleDwt53Job",
                "pub struct ReversibleDwt53FirstLevel",
                "pub struct Dwt97BatchStageTimings",
                "pub enum TranscodeStageError",
            ])
            .forbidden(&[
                "include!(\"accelerator.rs\")",
                "#[path = \"accelerator.rs\"]",
                "#[doc(hidden)]\npub mod accelerator;",
                "pub use self::accelerator::{",
            ]),
        PatternCheck::new("transcode accelerator contracts", &contracts)
            .required(&[
                "pub trait DctToWaveletStageAccelerator",
                "pub struct DctToWaveletStageCounters",
                "impl fmt::Display for TranscodeStageError",
            ])
            .forbidden(&[
                "allow(dead_code)",
                "expect(dead_code)",
                "feature = \"dev-support\"",
                "include!(",
            ]),
    ]);

    let mut root_exports = braced_export_names(&root, "pub use self::accelerator_contracts::{");
    root_exports.extend([
        "DctGridToReversibleDwt53Job",
        "Dwt97BatchStageTimings",
        "ReversibleDwt53FirstLevel",
        "TranscodeStageError",
    ]);
    assert_eq!(
        root_exports,
        braced_export_names(&root, "pub use crate::{"),
        "root and compatibility accelerator exports drifted"
    );
}
