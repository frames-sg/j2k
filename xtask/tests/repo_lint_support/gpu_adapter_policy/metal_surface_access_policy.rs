// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::{assert_pattern_checks, repo_root, PatternCheck};

#[test]
fn j2k_metal_surface_byte_access_remains_fallible() {
    let surface = fs::read_to_string(repo_root().join("crates/j2k-metal/src/surface.rs"))
        .expect("read J2K Metal surface source");

    assert_pattern_checks(&[
        PatternCheck::new("fallible J2K Metal surface byte access", &surface)
            .required(&[
                "pub fn as_bytes(&self) -> Result<Cow<'_, [u8]>, Error>",
                "host_backed_byte_access_borrows_the_validated_range",
                "invalid_host_backed_range_returns_an_error_without_panicking",
                "cloning_a_host_surface_shares_the_pixel_owner",
                "Host(Arc<Vec<u8>>)",
            ])
            .forbidden(&[
                "pub fn as_bytes(&self) -> Cow<'_, [u8]>",
                ".expect(\"validated J2K Metal surface byte range\")",
            ]),
    ]);
}

#[test]
fn metal_surface_access_policy_stays_focused() {
    assert!(
        include_str!("metal_surface_access_policy.rs")
            .lines()
            .count()
            < 40,
        "Metal surface access policy must stay focused"
    );
}
