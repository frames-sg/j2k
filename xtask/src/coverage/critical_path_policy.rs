// SPDX-License-Identifier: MIT OR Apache-2.0

use super::model::{ChangedCoverageResult, CoverageLane};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum CriticalPathClass {
    Safety,
    Correctness,
    Ownership,
    PublicApi,
    Parser,
    Security,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ResidualDisposition {
    Unreachable,
    HardwareOnly,
    Trivial,
    LowRiskTooling,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ZeroBodyAudit {
    Critical(CriticalPathClass),
    Residual(ResidualDisposition),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ZeroBodyKind {
    Function,
    ExecutableBody,
    DeferredBody,
    OpaqueMacro,
}

impl ZeroBodyKind {
    pub(super) const fn id(self) -> &'static str {
        match self {
            Self::Function => "function",
            Self::ExecutableBody => "executable-body",
            Self::DeferredBody => "deferred-body",
            Self::OpaqueMacro => "opaque-macro",
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub(super) struct AuditedZeroBodyFinding<'a> {
    pub(super) kind: ZeroBodyKind,
    pub(super) finding: &'a str,
    pub(super) audit: ZeroBodyAudit,
}

impl CriticalPathClass {
    pub(super) const fn id(self) -> &'static str {
        match self {
            Self::Safety => "safety",
            Self::Correctness => "correctness",
            Self::Ownership => "ownership",
            Self::PublicApi => "public-api",
            Self::Parser => "parser",
            Self::Security => "security",
        }
    }

    pub(super) const fn reason(self) -> &'static str {
        match self {
            Self::Safety => "measurable lines are included in the safety-critical coverage gate",
            Self::Correctness => {
                "measurable lines are included in the codec-correctness coverage gate"
            }
            Self::Ownership => {
                "measurable lines are included in the ownership and resource-lifecycle coverage gate"
            }
            Self::PublicApi => {
                "measurable lines are included in the public-API coverage gate"
            }
            Self::Parser => {
                "measurable lines are included in the untrusted-input parser coverage gate"
            }
            Self::Security => {
                "measurable lines are included in the security and release-integrity coverage gate"
            }
        }
    }
}

impl ResidualDisposition {
    pub(super) const fn id(self) -> &'static str {
        match self {
            Self::Unreachable => "unreachable",
            Self::HardwareOnly => "hardware-only",
            Self::Trivial => "trivial",
            Self::LowRiskTooling => "low-risk-tooling",
        }
    }

    pub(super) const fn reason(self) -> &'static str {
        match self {
            Self::Unreachable => {
                "validated construction makes this defensive branch structurally unreachable"
            }
            Self::HardwareOnly => {
                "the host lane cannot execute this hardware-owned path; its accelerator lane owns runtime evidence"
            }
            Self::Trivial => {
                "the finding is a formatting, accessor, or reporting shim with no independent state transition"
            }
            Self::LowRiskTooling => {
                "the path is non-release tooling or test support outside codec, ownership, parser, public API, and security boundaries"
            }
        }
    }
}

pub(super) fn classify_path(path: &str) -> Option<CriticalPathClass> {
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
    if is_codec_production_path(path) || is_release_correctness_tool(path) {
        return Some(CriticalPathClass::Correctness);
    }
    None
}

pub(super) fn audit_zero_body(
    lane: CoverageLane,
    kind: ZeroBodyKind,
    finding: &str,
) -> ZeroBodyAudit {
    let path = finding_path(finding);
    if is_explicitly_unreachable(finding) {
        return ZeroBodyAudit::Residual(ResidualDisposition::Unreachable);
    }
    if lane == CoverageLane::Host && is_hardware_only_path(path) {
        return ZeroBodyAudit::Residual(ResidualDisposition::HardwareOnly);
    }
    if is_trivial_finding(kind, finding) {
        return ZeroBodyAudit::Residual(ResidualDisposition::Trivial);
    }
    classify_path(path).map_or(
        ZeroBodyAudit::Residual(ResidualDisposition::LowRiskTooling),
        ZeroBodyAudit::Critical,
    )
}

pub(super) fn audited_zero_body_findings(
    lane: CoverageLane,
    result: &ChangedCoverageResult,
) -> Vec<AuditedZeroBodyFinding<'_>> {
    let groups = [
        (
            ZeroBodyKind::Function,
            result.changed_functions_without_covered_body.as_slice(),
        ),
        (
            ZeroBodyKind::ExecutableBody,
            result
                .changed_executable_bodies_without_covered_body
                .as_slice(),
        ),
        (
            ZeroBodyKind::DeferredBody,
            result
                .changed_deferred_bodies_without_covered_compiler_region
                .as_slice(),
        ),
        (
            ZeroBodyKind::OpaqueMacro,
            result.changed_opaque_macros.as_slice(),
        ),
    ];
    groups
        .into_iter()
        .flat_map(|(kind, findings)| {
            findings.iter().map(move |finding| AuditedZeroBodyFinding {
                kind,
                finding,
                audit: audit_zero_body(lane, kind, finding),
            })
        })
        .collect()
}

fn finding_path(finding: &str) -> &str {
    finding.split_once("::").map_or(finding, |(path, _)| path)
}

fn is_parser_path(path: &str) -> bool {
    [
        "/parse/",
        "/parser/",
        "/header/",
        "/jp2/",
        "/segment/",
        "codestream",
        "packet_header",
    ]
    .iter()
    .any(|marker| path.contains(marker))
        || ["/parse.rs", "/parser.rs", "/header.rs", "/segment.rs"]
            .iter()
            .any(|suffix| path.ends_with(suffix))
}

fn is_ownership_path(path: &str) -> bool {
    [
        "/allocation",
        "/batch/",
        "/session/",
        "/surface",
        "/buffer",
        "/workspace",
        "/cache/",
        "ownership",
        "resource",
        "resident",
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
    path.contains("unsafe") || path.contains("validation") || path.contains("/bounds")
}

fn is_codec_production_path(path: &str) -> bool {
    path.starts_with("crates/")
        && (path.contains("/src/") || path.ends_with("/build.rs"))
        && !path.starts_with("crates/j2k-compare/")
        && !path.starts_with("crates/j2k-test-support/")
        && !path.starts_with("crates/j2k-transcode-test-support/")
}

fn is_release_correctness_tool(path: &str) -> bool {
    path.starts_with("xtask/src/coverage")
        || path.starts_with("xtask/src/release")
        || path.starts_with("xtask/src/semver")
        || path.starts_with("xtask/src/stable_api")
        || path.starts_with("xtask/src/publication_gate")
}

fn is_hardware_only_path(path: &str) -> bool {
    path.contains("cuda")
        || path.contains("metal")
        || path.contains("/adapter/baseline_encode/")
        || path.contains("/adapter/fast_packet/")
}

fn is_trivial_finding(kind: ZeroBodyKind, finding: &str) -> bool {
    let trivial_symbol = [
        "::fmt@",
        "::source@",
        "::is_empty@",
        "::len@",
        "::output_len@",
        "::bytes_allocated@",
        "::level@",
        "::level_count@",
        "::ll@",
        "::reset@",
        "::to_string@",
    ]
    .iter()
    .any(|marker| finding.contains(marker));
    let trivial_macro = kind == ZeroBodyKind::OpaqueMacro
        && [
            "opaque-macro-invocation:format@",
            "opaque-macro-invocation:eprintln@",
            "opaque-macro-invocation:println@",
            "opaque-macro-invocation:matches@",
            "opaque-macro-invocation:json@",
        ]
        .iter()
        .any(|marker| finding.contains(marker));
    trivial_symbol || trivial_macro
}

fn is_explicitly_unreachable(finding: &str) -> bool {
    matches!(
        finding,
        "xtask/src/release_commands/package_gate.rs::closure@72"
            | "xtask/src/release_commands/package_gate.rs::closure@85"
    )
}
