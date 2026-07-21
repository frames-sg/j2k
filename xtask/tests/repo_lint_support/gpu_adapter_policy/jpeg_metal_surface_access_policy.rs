// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::{assert_pattern_checks, repo_root, PatternCheck};

#[test]
fn jpeg_metal_surface_byte_access_remains_fallible_and_typed() {
    let root = repo_root();
    let surface = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/surface.rs"))
        .expect("read JPEG Metal surface source");
    let regressions =
        fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/tests/reusable_output.rs"))
            .expect("read JPEG Metal reusable-output regressions");

    assert_pattern_checks(&[
        PatternCheck::new("fallible JPEG Metal surface byte access", &surface)
            .required(&[
                "pub fn as_bytes(&self) -> Result<Cow<'_, [u8]>, Error>",
                "Error::MetalStatePoisoned {",
                "host_backed_byte_access_remains_borrowed_and_fallible",
            ])
            .forbidden(&[
                "pub fn as_bytes(&self) -> Cow<'_, [u8]>",
                "Metal surface storage must be synchronized, CPU-visible, and bounded",
            ]),
        PatternCheck::new("JPEG Metal surface failure regressions", &regressions).required(&[
            "reusable_output_surface_as_bytes_reports_poisoned_access_gate",
            "reusable_output_surface_as_bytes_retains_typed_range_source",
            "MetalSupportError::BufferBounds",
        ]),
    ]);
}

#[test]
fn jpeg_metal_surface_access_policy_stays_focused() {
    assert!(
        include_str!("jpeg_metal_surface_access_policy.rs")
            .lines()
            .count()
            < 45
    );
}
