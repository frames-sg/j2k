// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{fs, path::Path};

const MODULES: &[(&str, usize)] = &[
    ("entropy/sequential/emit/four_component.rs", 150),
    ("entropy/sequential/emit/output.rs", 275),
    ("entropy/sequential/emit/region420.rs", 200),
    ("entropy/sequential/emit/rgb.rs", 290),
    ("entropy/sequential/emit/rgb444.rs", 90),
    ("entropy/sequential/emit/types.rs", 60),
    ("entropy/sequential/emit/upsample.rs", 175),
];

fn source(path: &str) -> String {
    fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src").join(path))
        .unwrap_or_else(|error| panic!("read {path}: {error}"))
}

#[test]
fn sequential_emit_uses_focused_real_modules() {
    let allow_attribute = ["#[", "allow"].concat();
    let include_macro = ["include", "!("].concat();
    let wildcard_import = ["use super::", "*"].concat();
    let wildcard_reexport = ["pub(super) use self::", "*"].concat();
    let root = source("entropy/sequential/emit.rs");

    assert!(
        root.lines().count() < 50,
        "sequential/emit.rs must remain a focused module shell"
    );
    for declaration in [
        "mod four_component;",
        "mod output;",
        "mod region420;",
        "mod rgb;",
        "mod rgb444;",
        "mod types;",
        "mod upsample;",
    ] {
        assert!(
            root.contains(declaration),
            "sequential/emit.rs must contain {declaration}"
        );
    }
    for required_item in [
        "emit_stripe",
        "emit_stripe_rgb",
        "emit_stripe_rgb_420_region",
        "emit_stripe_rgb_444",
        "Fast420RegionStripe",
        "StripeEmit",
        "StripeNeighbors",
    ] {
        assert!(
            root.contains(required_item),
            "sequential/emit.rs must reexport {required_item}"
        );
    }
    assert!(!root.contains(&allow_attribute));
    assert!(!root.contains(&include_macro));
    assert!(!root.contains(&wildcard_reexport));

    for (path, max_lines) in MODULES {
        let module = source(path);
        assert!(
            module.lines().count() < *max_lines,
            "{path} must stay below its focused-module line-count ratchet"
        );
        assert!(
            !module.contains(&allow_attribute),
            "{path} must avoid broad allows"
        );
        assert!(
            !module.contains(&include_macro),
            "{path} must be a real module"
        );
        assert!(
            !module.contains(&wildcard_import),
            "{path} must use explicit imports"
        );
    }
}
