// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{fs, path::Path};

use super::{assert_pattern_checks, PatternCheck};

pub(super) fn assert_policy(root: &Path) {
    let production =
        fs::read_to_string(root.join("crates/j2k-jpeg/src/adapter/fast_packet/error.rs"))
            .expect("read JPEG fast-packet error source");
    let tests =
        fs::read_to_string(root.join("crates/j2k-jpeg/src/adapter/fast_packet/error/tests.rs"))
            .expect("read JPEG fast-packet error tests");
    let policy = fs::read_to_string(root.join(
        "xtask/tests/repo_lint_support/gpu_adapter_policy/jpeg_fast_packet_routing_policy/error_contract.rs",
    ))
    .expect("read JPEG fast-packet error policy");

    for (label, source, max_lines) in [
        ("JPEG fast-packet error source", &production, 125),
        ("JPEG fast-packet error tests", &tests, 175),
        ("JPEG fast-packet error policy", &policy, 75),
    ] {
        assert!(
            source.lines().count() < max_lines,
            "{label} must stay below its focused {max_lines}-line ratchet"
        );
    }

    assert_pattern_checks(&[
        PatternCheck::new("typed JPEG fast-packet failure", &production).required(&[
            "#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]",
            "Decode(#[from] JpegError)",
            "pub const fn is_capability_mismatch(&self) -> bool",
            "Self::UnsupportedSampling",
            "Self::EntropyMarkerUnsupported { .. }",
            "impl CodecError for FastPacketError",
            "#[cfg(test)]\nmod tests;",
        ]),
        PatternCheck::new("typed JPEG fast-packet regressions", &tests).required(&[
            "direct_classification_distinguishes_capability_truncation_and_input_errors",
            "decode_classification_delegates_to_each_typed_jpeg_category",
            "decode_conversion_preserves_display_and_typed_source",
        ]),
    ]);
}
