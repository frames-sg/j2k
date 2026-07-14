// SPDX-License-Identifier: MIT OR Apache-2.0

use super::CriticalPathClass;

pub(in crate::coverage) fn classify_path(path: &str) -> Option<CriticalPathClass> {
    if is_parser_path(path) {
        return Some(CriticalPathClass::Parser);
    }
    if is_ownership_path(path) {
        return Some(CriticalPathClass::Ownership);
    }
    if is_public_api_path(path) {
        return Some(CriticalPathClass::PublicApi);
    }
    if is_security_path(path) {
        return Some(CriticalPathClass::Security);
    }
    if is_safety_path(path) {
        return Some(CriticalPathClass::Safety);
    }
    if is_codec_correctness_path(path) || is_release_correctness_tool(path) {
        return Some(CriticalPathClass::Correctness);
    }
    None
}

pub(super) fn is_codec_behavior_path(path: &str) -> bool {
    path.starts_with("crates/")
        && (path.contains("/src/") || path.ends_with("/build.rs"))
        && !path.starts_with("crates/j2k-compare/")
        && !path.starts_with("crates/j2k-test-support/")
        && !path.starts_with("crates/j2k-transcode-test-support/")
}

pub(super) fn is_hardware_only_path(path: &str) -> bool {
    path.contains("cuda")
        || path.contains("metal")
        || path.contains("/adapter/baseline_encode/")
        || path.contains("/adapter/fast_packet/")
}

fn is_parser_path(path: &str) -> bool {
    [
        "/parse/",
        "/parser/",
        "/header/",
        "/jp2/",
        "/segment/",
        "packet_header",
    ]
    .iter()
    .any(|marker| path.contains(marker))
        || ["/parse.rs", "/parser.rs", "/header.rs", "/segment.rs"]
            .iter()
            .any(|suffix| path.ends_with(suffix))
        || path.ends_with("/codestream.rs")
}

fn is_ownership_path(path: &str) -> bool {
    [
        "/allocation",
        "/session/",
        "/surface",
        "/buffer",
        "/workspace",
        "/cache/",
        "/pool/",
        "/lease",
        "retention",
        "ownership",
        "resource",
    ]
    .iter()
    .any(|marker| path.contains(marker))
        || [
            "/batch.rs",
            "/session.rs",
            "/surface.rs",
            "/buffer.rs",
            "/workspace.rs",
            "/cache.rs",
        ]
        .iter()
        .any(|suffix| path.ends_with(suffix))
}

fn is_public_api_path(path: &str) -> bool {
    path.ends_with("/src/lib.rs")
        || path.contains("/api/")
        || path.ends_with("/api.rs")
        || path.ends_with("/error.rs")
        || path.contains("/traits/")
        || path.ends_with("/traits.rs")
        || path == "xtask/src/stable_api.rs"
}

fn is_security_path(path: &str) -> bool {
    path.contains("security")
        || path.contains("release_integrity")
        || path.contains("unsafe_audit")
        || path.contains("provenance")
}

fn is_safety_path(path: &str) -> bool {
    !path.contains("counter_validation")
        && (path.contains("unsafe")
            || path.contains("/validation/")
            || path.ends_with("/validation.rs")
            || path.contains("/bounds"))
}

fn is_codec_correctness_path(path: &str) -> bool {
    path.starts_with("crates/j2k-codec-math/src/")
        || [
            "/routing/",
            "/planning/",
            "checkpoint",
            "packet_plan",
            "packetization",
            "contract",
        ]
        .iter()
        .any(|marker| path.contains(marker))
        || ["/route.rs", "/routing.rs", "/plan.rs", "/planning.rs"]
            .iter()
            .any(|suffix| path.ends_with(suffix))
}

fn is_release_correctness_tool(path: &str) -> bool {
    path.starts_with("xtask/src/coverage")
        || path.starts_with("xtask/src/release")
        || path.starts_with("xtask/src/semver")
        || path.starts_with("xtask/src/stable_api")
        || path.starts_with("xtask/src/publication_gate")
}
