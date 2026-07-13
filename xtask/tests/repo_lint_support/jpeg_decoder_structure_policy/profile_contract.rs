// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{fs, path::Path};

use super::{assert_pattern_checks, PatternCheck};

pub(super) fn assert_policy(root: &Path) {
    let production = read(root, "crates/j2k-jpeg/src/decoder/routing/profile.rs");
    let support = read(
        root,
        "crates/j2k-jpeg/src/decoder/routing/profile/test_support.rs",
    );
    let tests = read(root, "crates/j2k-jpeg/src/decoder/routing/profile/tests.rs");
    let policy = read(
        root,
        "xtask/tests/repo_lint_support/jpeg_decoder_structure_policy/profile_contract.rs",
    );

    for (label, source, max_lines) in [
        ("JPEG routing profile source", &production, 140),
        ("JPEG routing profile test support", &support, 125),
        ("JPEG routing profile tests", &tests, 300),
        ("JPEG routing profile policy", &policy, 100),
    ] {
        assert!(
            source.lines().count() < max_lines,
            "{label} must stay below its focused {max_lines}-line ratchet"
        );
    }
    for (label, source) in [
        ("JPEG routing profile source", &production),
        ("JPEG routing profile test support", &support),
        ("JPEG routing profile tests", &tests),
    ] {
        assert!(
            !source.contains("use super::*"),
            "{label} must use explicit imports"
        );
    }

    assert_pattern_checks(&[
        PatternCheck::new("JPEG routing profile ownership", &production).required(&[
            "struct DecodeProfileRecord",
            "fn emit_decode_profile_fields<const N: usize>(",
            "emit_jpeg_profile_fields(operation, op, path, build)",
            "#[cfg(test)]\nmod test_support;",
            "#[cfg(test)]\nmod tests;",
        ]),
        PatternCheck::new("JPEG routing profile capture contract", &support).required(&[
            "thread_local!",
            "PhantomData<Rc<()>>",
            "impl Drop for TestProfileSinkGuard",
        ]),
        PatternCheck::new("JPEG routing profile regressions", &tests).required(&[
            "emit_dispatches_full_profile_with_exact_bounded_fields",
            "emit_dispatches_region_and_scaled_region_profiles_with_exact_fields",
            "profile_capture_is_thread_local_nested_and_restored",
        ]),
    ]);
}

fn read(root: &Path, relative: &str) -> String {
    fs::read_to_string(root.join(relative))
        .unwrap_or_else(|error| panic!("read {relative}: {error}"))
}
