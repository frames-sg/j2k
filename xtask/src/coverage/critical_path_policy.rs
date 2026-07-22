// SPDX-License-Identifier: MIT OR Apache-2.0

use super::model::{ChangedCoverageResult, CoverageLane};

mod classification;

pub(super) use classification::classify_path;
use classification::{is_codec_behavior_path, is_hardware_only_path};

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
                "measurable lines are included in the codec contract, planning, routing, and shared-math coverage gate"
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
                "exact backend behavior and parity evidence owns this accelerator implementation path"
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

pub(super) fn audit_zero_body(
    _lane: CoverageLane,
    kind: ZeroBodyKind,
    finding: &str,
) -> ZeroBodyAudit {
    let path = finding_path(finding);
    if is_explicitly_unreachable(finding) {
        return ZeroBodyAudit::Residual(ResidualDisposition::Unreachable);
    }
    if is_trivial_finding(kind, finding) {
        return ZeroBodyAudit::Residual(ResidualDisposition::Trivial);
    }
    if let Some(class) = classify_path(path) {
        return ZeroBodyAudit::Critical(class);
    }
    if is_hardware_only_path(path) {
        return ZeroBodyAudit::Residual(ResidualDisposition::HardwareOnly);
    }
    if is_codec_behavior_path(path) {
        return ZeroBodyAudit::Critical(CriticalPathClass::Correctness);
    }
    ZeroBodyAudit::Residual(ResidualDisposition::LowRiskTooling)
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
