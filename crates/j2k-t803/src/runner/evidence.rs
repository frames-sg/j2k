// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{collections::BTreeSet, fs, path::Path};

use sha2::{Digest, Sha256};

use crate::encoder::{ics_path, matrix_path, reference_decoder_identity};
use crate::{
    CaseStatus, EncoderEvidence, EncoderIcs, EncoderIut, EncoderMatrix, EncoderQualityStatus,
    ReportStatus, T803Manifest, T803Report,
};

use super::cache;

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
    if !report.iut.claim.contains("Profile-1 Cclass-1")
        || !report.iut.claim.contains("Profile-1 Cclass-1HF")
        || !report.iut.claim.contains("Annex G JP2 reader")
        || report.iut.claim.contains("full Part 1")
    {
        return Err(format!(
            "{} report uses an invalid claim label",
            report.iut.name
        ));
    }
    verify_encoder_evidence(&report.iut.name, &report.encoder)?;

    let expected_count = manifest.decoder_cases.len() + manifest.jp2_cases.len();
    if report.cases.len() != expected_count {
        return Err(format!(
            "{} report contains {} cases, expected {expected_count}",
            report.iut.name,
            report.cases.len()
        ));
    }
    for (observed, expected) in report.cases.iter().zip(&manifest.decoder_cases) {
        if observed.id != expected.id
            || observed.table != expected.table
            || observed.allowed_peak != expected.peak
            || observed.allowed_mse != Some(expected.mse)
        {
            return Err(format!(
                "{} report case {} differs from the pinned decoder matrix",
                report.iut.name, observed.id
            ));
        }
    }
    for (observed, expected) in report.cases[manifest.decoder_cases.len()..]
        .iter()
        .zip(&manifest.jp2_cases)
    {
        if observed.id != expected.id
            || observed.table != "G.1"
            || observed.allowed_peak != expected.peak
            || observed.allowed_mse.is_some()
        {
            return Err(format!(
                "{} report case {} differs from the pinned Annex G matrix",
                report.iut.name, observed.id
            ));
        }
    }
    Ok(())
}

fn verify_encoder_evidence(iut_name: &str, evidence: &EncoderEvidence) -> Result<(), String> {
    let iut = match iut_name {
        "j2k" => EncoderIut::Cpu,
        "j2k-cuda" => EncoderIut::Cuda,
        "j2k-metal" => EncoderIut::Metal,
        other => return Err(format!("unknown encoder IUT {other:?}")),
    };
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| "resolve j2k workspace root".to_string())?;
    let expected_matrix_path = matrix_path();
    let expected_ics_path = ics_path(iut);
    let matrix_text = fs::read_to_string(root.join(expected_matrix_path))
        .map_err(|error| format!("read {expected_matrix_path}: {error}"))?;
    let matrix = EncoderMatrix::parse(&matrix_text).map_err(|error| error.to_string())?;
    let ics_bytes = fs::read(root.join(expected_ics_path))
        .map_err(|error| format!("read {expected_ics_path}: {error}"))?;
    let ics_text = std::str::from_utf8(&ics_bytes)
        .map_err(|error| format!("read {expected_ics_path} as UTF-8: {error}"))?;
    let ics = EncoderIcs::parse(ics_text).map_err(|error| error.to_string())?;
    if ics.iut != iut {
        return Err(format!(
            "{expected_ics_path} identifies the wrong encoder IUT"
        ));
    }
    ics.validate_against(&matrix)
        .map_err(|error| error.to_string())?;

    let actual_ics_sha256 = format!("{:x}", Sha256::digest(&ics_bytes));
    if evidence.ics_path != expected_ics_path || evidence.ics_sha256 != actual_ics_sha256 {
        return Err(format!(
            "{iut_name} encoder ICS SHA-256 or path differs from {expected_ics_path}"
        ));
    }
    if evidence.matrix_path != expected_matrix_path
        || evidence.matrix_case_count != ics.matrix_case_count()
        || evidence.matrix_case_sha256 != ics.matrix_case_sha256()
    {
        return Err(format!(
            "{iut_name} encoder matrix identity differs from the committed ICS"
        ));
    }
    let (standard, implementation, version) = reference_decoder_identity();
    if evidence.reference_decoder.standard != standard
        || evidence.reference_decoder.implementation != implementation
        || evidence.reference_decoder.version != version
    {
        return Err(format!(
            "{iut_name} encoder reference decoder is not the pinned T.804 OpenJPEG build"
        ));
    }
    if evidence.standards_status != ReportStatus::Pass
        || evidence.quality_status != ReportStatus::Pass
        || evidence.status != ReportStatus::Pass
    {
        return Err(format!("{iut_name} encoder evidence did not pass"));
    }

    let expected_cases = matrix.selected_cases(iut).collect::<Vec<_>>();
    if evidence.cases.len() != expected_cases.len() {
        return Err(format!(
            "{iut_name} encoder evidence contains {} cases, expected {}",
            evidence.cases.len(),
            expected_cases.len()
        ));
    }
    for (observed, expected) in evidence.cases.iter().zip(expected_cases) {
        if observed.id != expected.id
            || observed.mode != expected.mode
            || observed.status != CaseStatus::Pass
            || observed.quality_status == EncoderQualityStatus::Fail
        {
            return Err(format!(
                "{iut_name} encoder case {} differs from the committed encoder matrix",
                observed.id
            ));
        }
    }
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
    use sha2::{Digest, Sha256};

    use super::*;
    use crate::{
        CaseStatus, EncodeRouteStage, EncodeRouteStageName, EncoderCaseReport, EncoderEvidence,
        EncoderIcs, EncoderIut, EncoderMatrix, EncoderMode, EncoderQualityStatus,
        EncoderReferenceIdentity, ExecutionLocation, RouteKind,
    };

    #[test]
    fn encoder_verification_rejects_ics_or_case_inventory_tampering() {
        let mut evidence = committed_cpu_evidence();
        verify_encoder_evidence("j2k", &evidence).expect("committed CPU encoder evidence");

        evidence.ics_sha256 = "0".repeat(64);
        let error = verify_encoder_evidence("j2k", &evidence)
            .expect_err("the report must pin the exact committed ICS bytes");
        assert!(error.contains("ICS SHA-256"), "{error}");

        let mut evidence = committed_cpu_evidence();
        evidence.cases[0].mode = match evidence.cases[0].mode {
            EncoderMode::Lossless => EncoderMode::Lossy,
            EncoderMode::Lossy => EncoderMode::Lossless,
        };
        let error = verify_encoder_evidence("j2k", &evidence)
            .expect_err("the report must retain the selected matrix case modes");
        assert!(error.contains("encoder matrix"), "{error}");
    }

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

    fn committed_cpu_evidence() -> EncoderEvidence {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("workspace root");
        let matrix_path = "corpus/j2k-conformance/encoder-matrix-v1.toml";
        let ics_path = "corpus/j2k-conformance/encoder-ics-cpu.toml";
        let matrix = EncoderMatrix::parse(
            &fs::read_to_string(root.join(matrix_path)).expect("read encoder matrix"),
        )
        .expect("valid encoder matrix");
        let ics_bytes = fs::read(root.join(ics_path)).expect("read CPU ICS");
        let ics = EncoderIcs::parse(std::str::from_utf8(&ics_bytes).expect("UTF-8 ICS"))
            .expect("valid CPU ICS");
        ics.validate_against(&matrix).expect("ICS matches matrix");
        let cases = matrix
            .selected_cases(EncoderIut::Cpu)
            .map(|case| passing_case(&case.id, case.mode))
            .collect();
        let (standard, implementation, version) = crate::encoder::reference_decoder_identity();
        EncoderEvidence::new(
            ics_path.to_string(),
            format!("{:x}", Sha256::digest(ics_bytes)),
            matrix_path.to_string(),
            ics.matrix_case_count(),
            ics.matrix_case_sha256().to_string(),
            EncoderReferenceIdentity {
                standard: standard.to_string(),
                implementation: implementation.to_string(),
                version: version.to_string(),
            },
            cases,
        )
        .expect("valid CPU encoder evidence")
    }

    fn passing_case(id: &str, mode: EncoderMode) -> EncoderCaseReport {
        let (lossless_exact, psnr_db, quality_status, quality_requirement) = match mode {
            EncoderMode::Lossless => (Some(true), None, EncoderQualityStatus::NotApplicable, None),
            EncoderMode::Lossy => (
                None,
                Some(40.0),
                EncoderQualityStatus::Pass,
                Some("test quality gate".to_string()),
            ),
        };
        EncoderCaseReport {
            id: id.to_string(),
            mode,
            status: CaseStatus::Pass,
            route: RouteKind::Cpu,
            reference_decode_success: true,
            lossless_exact,
            encoded_bytes: Some(1),
            actual_bits_per_pixel: Some(1.0),
            psnr_db,
            psnr_infinite: false,
            quality_status,
            quality_requirement,
            quality_error: None,
            error: None,
            stages: cpu_stages(),
        }
    }

    fn cpu_stages() -> Vec<EncodeRouteStage> {
        [
            EncodeRouteStageName::InputPreparation,
            EncodeRouteStageName::ForwardRct,
            EncodeRouteStageName::ForwardIct,
            EncodeRouteStageName::ForwardDwt53,
            EncodeRouteStageName::ForwardDwt97,
            EncodeRouteStageName::Quantization,
            EncodeRouteStageName::Tier1,
            EncodeRouteStageName::Packetization,
        ]
        .into_iter()
        .map(|stage| EncodeRouteStage {
            stage,
            location: ExecutionLocation::Cpu,
        })
        .chain(
            [
                EncodeRouteStageName::HostToDevice,
                EncodeRouteStageName::DeviceToHost,
            ]
            .into_iter()
            .map(|stage| EncodeRouteStage {
                stage,
                location: ExecutionLocation::NotUsed,
            }),
        )
        .collect()
    }
}
