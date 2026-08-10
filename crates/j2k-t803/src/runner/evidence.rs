// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{collections::BTreeSet, fs, path::Path};

use crate::{ReportStatus, T803Manifest, T803Report};

use super::cache;

mod decoder;
mod encoder;

use decoder::verify_decoder_evidence;
use encoder::verify_encoder_evidence;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum EvidenceScope {
    Cpu,
    Cuda,
    Metal,
    All,
}

impl EvidenceScope {
    pub(super) fn parse(value: &str) -> Result<Self, String> {
        match value {
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda),
            "metal" => Ok(Self::Metal),
            "all" => Ok(Self::All),
            _ => Err(format!(
                "unknown T.803 evidence scope {value:?}; expected cpu|cuda|metal|all"
            )),
        }
    }

    const fn argument(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Metal => "metal",
            Self::All => "all",
        }
    }

    const fn description(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Cuda => "CUDA adapter",
            Self::Metal => "Metal adapter",
            Self::All => "aggregate CPU/CUDA/Metal",
        }
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ReportLane {
    iut: String,
    os: String,
    arch: String,
}

impl ReportLane {
    fn new(iut: &str, os: &str, arch: &str) -> Self {
        Self {
            iut: iut.to_string(),
            os: os.to_string(),
            arch: arch.to_string(),
        }
    }
}

pub(super) fn verify_reports(
    _cache_dir: &Path,
    report_paths: &[impl AsRef<Path>],
    candidate_sha: Option<&str>,
    scope: EvidenceScope,
) -> Result<(), String> {
    let candidate_sha = candidate_sha.filter(|sha| is_git_sha(sha)).ok_or_else(|| {
        "t803 verify requires --candidate-sha with 40 or 64 lowercase hex digits".to_string()
    })?;
    let manifest = cache::load_manifest()?;
    let expected_lanes = required_lanes(scope);
    if report_paths.len() != expected_lanes.len() {
        return Err(format!(
            "t803 verify --scope {} requires exactly {} report(s): {}",
            scope.argument(),
            expected_lanes.len(),
            expected_lanes
                .iter()
                .map(lane_name)
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }

    let mut observed_lanes = BTreeSet::new();
    for report_path in report_paths {
        let report_path = report_path.as_ref();
        let json = fs::read_to_string(report_path)
            .map_err(|error| format!("read {}: {error}", report_path.display()))?;
        let report = T803Report::from_json(&json).map_err(|error| error.to_string())?;
        if report.to_json().map_err(|error| error.to_string())? != json {
            return Err(format!(
                "{} is not canonical deterministic report JSON",
                report_path.display()
            ));
        }
        verify_report(&report, &manifest, candidate_sha)?;
        let lane = ReportLane::new(&report.iut.name, &report.platform.os, &report.platform.arch);
        if !observed_lanes.insert(lane.clone()) {
            return Err(format!(
                "duplicate T.803 report for {} {} {}",
                lane.iut, lane.os, lane.arch
            ));
        }
        let markdown_path = report_path.with_extension("md");
        let markdown = fs::read_to_string(&markdown_path)
            .map_err(|error| format!("read {}: {error}", markdown_path.display()))?;
        if report.to_markdown().map_err(|error| error.to_string())? != markdown {
            return Err(format!(
                "{} does not match its canonical JSON evidence",
                markdown_path.display()
            ));
        }
    }
    verify_required_lanes(scope, &observed_lanes)
}

fn verify_required_lanes(
    scope: EvidenceScope,
    observed: &BTreeSet<ReportLane>,
) -> Result<(), String> {
    let expected = required_lanes(scope);
    if observed == &expected {
        return Ok(());
    }
    let missing = expected
        .difference(observed)
        .map(lane_name)
        .collect::<Vec<_>>();
    let unexpected = observed
        .difference(&expected)
        .map(lane_name)
        .collect::<Vec<_>>();
    Err(format!(
        "T.803 {} report lanes are incomplete; missing: {}; unexpected: {}",
        scope.description(),
        if missing.is_empty() {
            "none".to_string()
        } else {
            missing.join(", ")
        },
        if unexpected.is_empty() {
            "none".to_string()
        } else {
            unexpected.join(", ")
        }
    ))
}

fn required_lanes(scope: EvidenceScope) -> BTreeSet<ReportLane> {
    let cpu = [
        ReportLane::new("j2k", "linux", "x86_64"),
        ReportLane::new("j2k", "macos", "aarch64"),
        ReportLane::new("j2k", "windows", "x86_64"),
    ];
    let cuda = ReportLane::new("j2k-cuda", "linux", "x86_64");
    let metal = ReportLane::new("j2k-metal", "macos", "aarch64");
    match scope {
        EvidenceScope::Cpu => cpu.into_iter().collect(),
        EvidenceScope::Cuda => BTreeSet::from([cuda]),
        EvidenceScope::Metal => BTreeSet::from([metal]),
        EvidenceScope::All => cpu.into_iter().chain([cuda, metal]).collect(),
    }
}

fn lane_name(lane: &ReportLane) -> String {
    match (lane.iut.as_str(), lane.os.as_str(), lane.arch.as_str()) {
        ("j2k", "linux", "x86_64") => "Linux x64 CPU".to_string(),
        ("j2k", "macos", "aarch64") => "macOS arm64 CPU".to_string(),
        ("j2k", "windows", "x86_64") => "Windows x64 CPU".to_string(),
        ("j2k-cuda", "linux", "x86_64") => "Linux x64 CUDA adapter".to_string(),
        ("j2k-metal", "macos", "aarch64") => "macOS arm64 Metal adapter".to_string(),
        _ => format!("{} {} {}", lane.iut, lane.os, lane.arch),
    }
}

fn verify_report(
    report: &T803Report,
    manifest: &T803Manifest,
    candidate_sha: &str,
) -> Result<(), String> {
    if report.status != ReportStatus::Pass {
        return Err(format!("{} T.803 report did not pass", report.iut.name));
    }
    if report.iut.candidate_sha != candidate_sha {
        return Err(format!(
            "{} report candidate SHA is {}, expected {candidate_sha}",
            report.iut.name, report.iut.candidate_sha
        ));
    }
    if report
        .features
        .iter()
        .any(|feature| feature.contains("development") || feature.contains("dirty"))
    {
        return Err(format!(
            "{} report contains development-only feature evidence",
            report.iut.name
        ));
    }
    if report.source_archive_sha256 != manifest.source.archive_sha256
        || report.corpus != manifest.files
    {
        return Err(format!(
            "{} report corpus provenance differs from the pinned manifest",
            report.iut.name
        ));
    }
    verify_decoder_evidence(
        &report.iut.name,
        report.suite,
        &report.iut.claim,
        &report.cases,
        manifest,
    )?;
    verify_encoder_evidence(&report.iut.name, &report.encoder)?;
    Ok(())
}

fn is_git_sha(value: &str) -> bool {
    matches!(value.len(), 40 | 64)
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_evidence_scopes_cpu_and_each_adapter_independently() {
        let mut cpu_lanes = BTreeSet::from([
            ReportLane::new("j2k", "linux", "x86_64"),
            ReportLane::new("j2k", "macos", "aarch64"),
            ReportLane::new("j2k", "windows", "x86_64"),
        ]);
        verify_required_lanes(EvidenceScope::Cpu, &cpu_lanes).expect("complete CPU evidence lanes");

        cpu_lanes.remove(&ReportLane::new("j2k", "windows", "x86_64"));
        let error = verify_required_lanes(EvidenceScope::Cpu, &cpu_lanes)
            .expect_err("Windows CPU evidence is mandatory for the CPU claim");
        assert!(error.contains("Windows x64 CPU"), "{error}");

        verify_required_lanes(
            EvidenceScope::Cuda,
            &BTreeSet::from([ReportLane::new("j2k-cuda", "linux", "x86_64")]),
        )
        .expect("CUDA evidence is independent of Metal evidence");
        verify_required_lanes(
            EvidenceScope::Metal,
            &BTreeSet::from([ReportLane::new("j2k-metal", "macos", "aarch64")]),
        )
        .expect("Metal evidence is independent of CUDA evidence");

        let all_lanes = BTreeSet::from([
            ReportLane::new("j2k", "linux", "x86_64"),
            ReportLane::new("j2k", "macos", "aarch64"),
            ReportLane::new("j2k", "windows", "x86_64"),
            ReportLane::new("j2k-cuda", "linux", "x86_64"),
            ReportLane::new("j2k-metal", "macos", "aarch64"),
        ]);
        verify_required_lanes(EvidenceScope::All, &all_lanes)
            .expect("aggregate evidence remains available as a convenience");
    }
}
