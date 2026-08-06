use std::{collections::BTreeSet, fmt::Write as _};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::manifest::{validate_path, validate_sha256, CorpusFile, SOURCE_URL, STANDARD};
use crate::EncoderMode;

/// Overall report result.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ReportStatus {
    /// Every selected case passed.
    Pass,
    /// At least one selected case failed or errored.
    Fail,
}

/// Result of one selected conformance case.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum CaseStatus {
    /// Measured errors are within the inclusive bounds.
    Pass,
    /// Decode completed but exceeded at least one bound.
    Fail,
    /// Decode or comparison did not complete.
    Error,
}

/// Auditable classification of the complete execution route.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RouteKind {
    /// All used stages ran on the CPU.
    Cpu,
    /// Used stages include both CPU and one accelerator.
    Hybrid,
    /// Every used stage ran on one accelerator.
    DeviceNative,
}

/// Location used by one decoder stage.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ExecutionLocation {
    /// Host CPU.
    Cpu,
    /// CUDA device.
    Cuda,
    /// Metal device.
    Metal,
    /// Stage was not needed for this route.
    NotUsed,
}

/// Decoder stages disclosed for every case.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RouteStageName {
    /// Container and codestream parsing.
    Parsing,
    /// Tier-1 entropy decoding.
    Tier1,
    /// Coefficient dequantization.
    Dequantization,
    /// Inverse discrete wavelet transform.
    Idwt,
    /// Multiple-component transform.
    Mct,
    /// Colour conversion and output normalization.
    ColorOutput,
    /// Host-to-device transfer.
    HostToDevice,
    /// Device-to-host transfer.
    DeviceToHost,
}

/// Encoder stages disclosed for every Annex D matrix case.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum EncodeRouteStageName {
    /// Input sample unpacking, level shift, or component-plane preparation.
    InputPreparation,
    /// Forward reversible colour transform.
    ForwardRct,
    /// Forward irreversible colour transform.
    ForwardIct,
    /// Forward reversible 5/3 wavelet transform.
    ForwardDwt53,
    /// Forward irreversible 9/7 wavelet transform.
    ForwardDwt97,
    /// Irreversible sub-band quantization.
    Quantization,
    /// Part 1 Tier-1 code-block coding.
    Tier1,
    /// Tier-2 packet formation and codestream writing.
    Packetization,
    /// Host-to-device transfer.
    HostToDevice,
    /// Device-to-host transfer.
    DeviceToHost,
}

/// Execution location for one decoder stage.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RouteStage {
    /// Stage being disclosed.
    pub stage: RouteStageName,
    /// Where the stage ran, or that it was not used.
    pub location: ExecutionLocation,
}

/// Execution location for one encoder stage.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncodeRouteStage {
    /// Stage being disclosed.
    pub stage: EncodeRouteStageName,
    /// Where the stage ran, or that it was not used.
    pub location: ExecutionLocation,
}

/// Identity and claim text for the implementation under test.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct IutIdentity {
    /// Crate or adapter name.
    pub name: String,
    /// Candidate version.
    pub version: String,
    /// Exact source revision under test.
    pub candidate_sha: String,
    /// Precise candidate claim; never a generic Part 1 claim.
    pub claim: String,
}

/// Operating-system and hardware identity for one run.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PlatformIdentity {
    /// Operating system.
    pub os: String,
    /// Processor architecture.
    pub arch: String,
    /// CPU or accelerator hardware description.
    pub hardware: String,
    /// Accelerator driver, or `not-applicable` for CPU runs.
    pub driver: String,
}

/// Aggregate route counts for the complete selected decoder matrix.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecoderRouteSummary {
    /// Total selected decoder and Annex G cases.
    pub total: usize,
    /// Cases whose used stages all ran on one accelerator.
    pub device_native: usize,
    /// Cases whose used stages ran across CPU and one accelerator.
    pub hybrid: usize,
    /// Cases whose used stages all ran on CPU.
    pub cpu: usize,
}

/// Independent native-component comparison against `OpenJPEG` before T.803 normalization.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeComponentOracleEvidence {
    /// Repository-relative official codestream path.
    pub codestream_path: String,
    /// SHA-256 of the official codestream bytes.
    pub codestream_sha256: String,
    /// Semantic rule that selected this codestream for the independent audit.
    pub selection: String,
    /// Independent decoder implementation.
    pub implementation: String,
    /// Independent decoder version.
    pub version: String,
    /// Exact library or executable identity.
    pub library: String,
    /// Number of components compared in codestream order.
    pub component_count: usize,
    /// Total native samples compared across all components.
    pub compared_sample_count: u64,
    /// Canonical SHA-256 of production-decoder component metadata and samples.
    pub production_components_sha256: String,
    /// Canonical SHA-256 of independent-decoder component metadata and samples.
    pub openjpeg_components_sha256: String,
    /// Whether every component's metadata and samples matched exactly.
    pub exact: bool,
}

/// Metrics, bounds, and route evidence for one selected case.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CaseReport {
    /// Stable case identifier from the manifest.
    pub id: String,
    /// T.803 table containing the case.
    pub table: String,
    /// Case result.
    pub status: CaseStatus,
    /// Auditable route classification for this case.
    pub route: RouteKind,
    /// Measured peak error, when comparison completed.
    pub peak: Option<u64>,
    /// Measured MSE, when required and comparison completed.
    pub mse: Option<f64>,
    /// Inclusive peak bound.
    pub allowed_peak: u64,
    /// Inclusive MSE bound, absent for Annex G peak-only comparisons.
    pub allowed_mse: Option<f64>,
    /// Diagnostic for an errored case.
    pub error: Option<String>,
    /// Complete per-stage route disclosure.
    pub stages: Vec<RouteStage>,
}

/// Exact T.804 reference implementation used for Annex D testing.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderReferenceIdentity {
    /// Reference-software standard.
    pub standard: String,
    /// Reference decoder implementation.
    pub implementation: String,
    /// Exact implementation version.
    pub version: String,
}

/// Result of the project-defined lossy quality gate.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum EncoderQualityStatus {
    /// The separately declared project quality gate passed.
    Pass,
    /// The separately declared project quality gate failed.
    Fail,
    /// No lossy quality gate applies, as for lossless cases.
    NotApplicable,
}

/// Result of one informative Annex D/F encoder matrix case.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderCaseReport {
    /// Stable case identifier from the encoder matrix.
    pub id: String,
    /// Lossless or lossy encoder mode.
    pub mode: EncoderMode,
    /// Case result.
    pub status: CaseStatus,
    /// Auditable route classification.
    pub route: RouteKind,
    /// Whether the T.804 reference implementation fully decoded the codestream.
    pub reference_decode_success: bool,
    /// Exact input equality for a lossless case; absent for lossy cases.
    pub lossless_exact: Option<bool>,
    /// Produced codestream size when encoding completed.
    pub encoded_bytes: Option<u64>,
    /// Actual codestream bits per reference-grid pixel.
    pub actual_bits_per_pixel: Option<f64>,
    /// Project quality metric for lossy output; not an Annex D acceptance rule.
    pub psnr_db: Option<f64>,
    /// Whether lossy output was exact, giving mathematically infinite PSNR.
    pub psnr_infinite: bool,
    /// Result of the separately declared project quality gate.
    pub quality_status: EncoderQualityStatus,
    /// Human-readable, auditable quality-gate requirement.
    pub quality_requirement: Option<String>,
    /// Diagnostic when the quality gate fails.
    pub quality_error: Option<String>,
    /// Diagnostic for a failed or errored case.
    pub error: Option<String>,
    /// Complete per-stage route disclosure.
    pub stages: Vec<EncodeRouteStage>,
}

/// Informative encoder evidence attached to one T.803 run.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderEvidence {
    /// Annex F ICS repository path.
    pub ics_path: String,
    /// SHA-256 of the exact ICS bytes.
    pub ics_sha256: String,
    /// Encoder matrix repository path.
    pub matrix_path: String,
    /// Expected cases for this IUT.
    pub matrix_case_count: usize,
    /// Canonical SHA-256 of this IUT's selected matrix cases.
    pub matrix_case_sha256: String,
    /// T.804 decoder implementation identity.
    pub reference_decoder: EncoderReferenceIdentity,
    /// Per-case evidence in matrix order.
    pub cases: Vec<EncoderCaseReport>,
    /// Encoder result derived from `cases`.
    pub standards_status: ReportStatus,
    /// Project quality-gate result derived from `cases`.
    pub quality_status: ReportStatus,
    /// Combined encoder evidence result.
    pub status: ReportStatus,
}

impl EncoderEvidence {
    /// Build validated encoder evidence and derive its final status.
    pub fn new(
        ics_path: String,
        ics_sha256: String,
        matrix_path: String,
        matrix_case_count: usize,
        matrix_case_sha256: String,
        reference_decoder: EncoderReferenceIdentity,
        cases: Vec<EncoderCaseReport>,
    ) -> Result<Self, ReportError> {
        let standards_status = derive_status(cases.iter().map(|case| case.status));
        let quality_status = derive_quality_status(&cases);
        let status = combine_status(standards_status, quality_status);
        let evidence = Self {
            ics_path,
            ics_sha256,
            matrix_path,
            matrix_case_count,
            matrix_case_sha256,
            reference_decoder,
            cases,
            standards_status,
            quality_status,
            status,
        };
        evidence.validate()?;
        Ok(evidence)
    }

    fn validate(&self) -> Result<(), ReportError> {
        validate_path(&self.ics_path)
            .map_err(|error| ReportError::Validation(error.to_string()))?;
        validate_path(&self.matrix_path)
            .map_err(|error| ReportError::Validation(error.to_string()))?;
        validate_sha256(&self.ics_sha256, "encoder ICS")
            .map_err(|error| ReportError::Validation(error.to_string()))?;
        validate_sha256(&self.matrix_case_sha256, "encoder matrix")
            .map_err(|error| ReportError::Validation(error.to_string()))?;
        if self.matrix_case_count == 0 || self.cases.len() != self.matrix_case_count {
            return report_error("encoder case count does not match the pinned matrix");
        }
        if [
            &self.reference_decoder.standard,
            &self.reference_decoder.implementation,
            &self.reference_decoder.version,
        ]
        .into_iter()
        .any(String::is_empty)
        {
            return report_error("encoder reference decoder identity must not be empty");
        }
        let mut previous_id = None;
        for case in &self.cases {
            if case.id.is_empty()
                || previous_id.is_some_and(|previous| previous >= case.id.as_str())
            {
                return report_error("encoder case ids must be non-empty, sorted, and unique");
            }
            previous_id = Some(case.id.as_str());
            validate_encoder_case(case)?;
            validate_encode_route(case)?;
        }
        let standards_status = derive_status(self.cases.iter().map(|case| case.status));
        let quality_status = derive_quality_status(&self.cases);
        if self.standards_status != standards_status
            || self.quality_status != quality_status
            || self.status != combine_status(standards_status, quality_status)
        {
            return report_error("encoder statuses do not match case results");
        }
        Ok(())
    }
}

/// Versioned JSON and Markdown evidence for one T.803 run.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct T803Report {
    /// Report schema version.
    pub schema_version: u32,
    /// Standard edition used by the run.
    pub standard: String,
    /// Official attachment handle.
    pub source_url: String,
    /// Observed attachment SHA-256.
    pub source_archive_sha256: String,
    /// Implementation under test.
    pub iut: IutIdentity,
    /// Platform and hardware identity.
    pub platform: PlatformIdentity,
    /// Enabled build features in sorted order.
    pub features: Vec<String>,
    /// Selected corpus hashes in path order.
    pub corpus: Vec<CorpusFile>,
    /// Independent component-level decoder evidence before T.803 normalization.
    pub native_component_oracles: Vec<NativeComponentOracleEvidence>,
    /// Aggregate decoder route counts derived from `cases`.
    pub decoder_routes: DecoderRouteSummary,
    /// Per-case evidence in manifest order.
    pub cases: Vec<CaseReport>,
    /// Informative Annex D/F encoder evidence.
    pub encoder: EncoderEvidence,
    /// Overall result derived from `cases`.
    pub status: ReportStatus,
}

/// Error returned for invalid evidence or report serialization.
#[derive(Debug, Error)]
pub enum ReportError {
    /// Report evidence is incomplete or inconsistent.
    #[error("invalid T.803 report: {0}")]
    Validation(String),
    /// JSON serialization or parsing failed.
    #[error("invalid T.803 report JSON: {0}")]
    Json(#[from] serde_json::Error),
}

impl T803Report {
    /// Build a report and derive its final status from the case results.
    #[expect(
        clippy::too_many_arguments,
        reason = "the constructor mirrors the validated top-level evidence sections"
    )]
    pub fn new(
        iut: IutIdentity,
        platform: PlatformIdentity,
        source_archive_sha256: String,
        features: Vec<String>,
        corpus: Vec<CorpusFile>,
        native_component_oracles: Vec<NativeComponentOracleEvidence>,
        cases: Vec<CaseReport>,
        encoder: EncoderEvidence,
    ) -> Result<Self, ReportError> {
        let decoder_routes = summarize_routes(&cases);
        let decoder_status = derive_status(cases.iter().map(|case| case.status));
        let status = if decoder_status == ReportStatus::Pass && encoder.status == ReportStatus::Pass
        {
            ReportStatus::Pass
        } else {
            ReportStatus::Fail
        };
        let report = Self {
            schema_version: 3,
            standard: STANDARD.to_string(),
            source_url: SOURCE_URL.to_string(),
            source_archive_sha256,
            iut,
            platform,
            features,
            corpus,
            native_component_oracles,
            decoder_routes,
            cases,
            encoder,
            status,
        };
        report.validate()?;
        Ok(report)
    }

    /// Serialize a validated report as stable pretty-printed JSON.
    pub fn to_json(&self) -> Result<String, ReportError> {
        self.validate()?;
        let mut json = serde_json::to_string_pretty(self)?;
        json.push('\n');
        Ok(json)
    }

    /// Parse and validate report JSON.
    pub fn from_json(json: &str) -> Result<Self, ReportError> {
        let report = serde_json::from_str::<Self>(json)?;
        report.validate()?;
        Ok(report)
    }

    /// Render a validated report as deterministic Markdown.
    pub fn to_markdown(&self) -> Result<String, ReportError> {
        self.validate()?;
        let mut markdown = format!(
            "# T.803 conformance evidence\n\n- Standard: {}\n- IUT: {} {}\n- Candidate SHA: {}\n- Claim: {}\n- Platform: {} {}\n- Hardware: {}\n- Driver: {}\n- Device-native: {} / {}\n- Hybrid: {} / {}\n- CPU-routed: {} / {}\n- Final status: {}\n",
            self.standard,
            markdown_cell(&self.iut.name),
            markdown_cell(&self.iut.version),
            markdown_cell(&self.iut.candidate_sha),
            markdown_cell(&self.iut.claim),
            markdown_cell(&self.platform.os),
            markdown_cell(&self.platform.arch),
            markdown_cell(&self.platform.hardware),
            markdown_cell(&self.platform.driver),
            self.decoder_routes.device_native,
            self.decoder_routes.total,
            self.decoder_routes.hybrid,
            self.decoder_routes.total,
            self.decoder_routes.cpu,
            self.decoder_routes.total,
            report_status_name(self.status),
        );
        self.push_native_component_oracles(&mut markdown);
        markdown.push_str("\n| Case | Table | Status | Route | Peak (measured / allowed) | MSE (measured / allowed) | Route stages |\n|---|---|---:|---|---:|---:|---|\n");
        self.push_decoder_cases(&mut markdown);
        self.push_encoder_evidence(&mut markdown);
        Ok(markdown)
    }

    fn push_native_component_oracles(&self, markdown: &mut String) {
        markdown.push_str("\n## Native component oracle\n");
        for oracle in &self.native_component_oracles {
            let _ = write!(
                markdown,
                "\n- {} {} (`{}`) decoded `{}` (`{}`) before T.803 normalization.\n- Selection: {}.\n- {} components / {} samples: {}.\n- Production SHA-256: `{}`.\n- OpenJPEG SHA-256: `{}`.\n",
                markdown_cell(&oracle.implementation),
                markdown_cell(&oracle.version),
                markdown_cell(&oracle.library),
                markdown_cell(&oracle.codestream_path),
                oracle.codestream_sha256,
                markdown_cell(&oracle.selection),
                oracle.component_count,
                oracle.compared_sample_count,
                if oracle.exact { "exact" } else { "mismatch" },
                oracle.production_components_sha256,
                oracle.openjpeg_components_sha256,
            );
        }
    }

    fn push_decoder_cases(&self, markdown: &mut String) {
        for case in &self.cases {
            let peak = case
                .peak
                .map_or_else(|| "error".to_string(), |value| value.to_string());
            let mse = match (case.mse, case.allowed_mse) {
                (Some(measured), Some(allowed)) => format!("{measured:.6} / {allowed:.6}"),
                (None, None) => "not-applicable".to_string(),
                _ => "error".to_string(),
            };
            let stages = case
                .stages
                .iter()
                .map(|stage| {
                    format!(
                        "{}={}",
                        stage_name(stage.stage),
                        location_name(stage.location)
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            let _ = writeln!(
                markdown,
                "| {} | {} | {} | {} | {} / {} | {} | {} |",
                markdown_cell(&case.id),
                markdown_cell(&case.table),
                case_status_name(case.status),
                route_kind_name(case.route),
                peak,
                case.allowed_peak,
                mse,
                stages,
            );
        }
    }

    fn push_encoder_evidence(&self, markdown: &mut String) {
        let _ = write!(
            markdown,
            "\n## Informative Annex D/F encoder evidence\n\n- Procedure: T.803 Annex D/F (informative)\n- ICS: {} (`{}`)\n- Matrix: {} ({} cases, `{}`)\n- Reference decoder: {} {} ({})\n- Standards status: {}\n- Quality-gate status: {}\n- Combined encoder status: {}\n\n| Case | Mode | Standards status | Quality status | Route | Reference decode | Lossless exact | Bytes | Bits/pixel | PSNR | Quality requirement | Route stages |\n|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|---|\n",
            markdown_cell(&self.encoder.ics_path),
            self.encoder.ics_sha256,
            markdown_cell(&self.encoder.matrix_path),
            self.encoder.matrix_case_count,
            self.encoder.matrix_case_sha256,
            markdown_cell(&self.encoder.reference_decoder.implementation),
            markdown_cell(&self.encoder.reference_decoder.version),
            markdown_cell(&self.encoder.reference_decoder.standard),
            report_status_name(self.encoder.standards_status),
            report_status_name(self.encoder.quality_status),
            report_status_name(self.encoder.status),
        );
        for case in &self.encoder.cases {
            let stages = case
                .stages
                .iter()
                .map(|stage| {
                    format!(
                        "{}={}",
                        encode_stage_name(stage.stage),
                        location_name(stage.location)
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            let lossless_exact = case
                .lossless_exact
                .map_or_else(|| "not-applicable".to_string(), |exact| exact.to_string());
            let encoded_bytes = case
                .encoded_bytes
                .map_or_else(|| "error".to_string(), |bytes| bytes.to_string());
            let bits_per_pixel = case
                .actual_bits_per_pixel
                .map_or_else(|| "error".to_string(), |value| format!("{value:.6}"));
            let psnr = if case.psnr_infinite {
                "infinite".to_string()
            } else {
                case.psnr_db.map_or_else(
                    || "not-applicable".to_string(),
                    |value| format!("{value:.6}"),
                )
            };
            let quality_requirement = case
                .quality_requirement
                .as_deref()
                .map_or_else(|| "not-applicable".to_string(), markdown_cell);
            let _ = writeln!(
                markdown,
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                markdown_cell(&case.id),
                encoder_mode_name(case.mode),
                case_status_name(case.status),
                quality_status_name(case.quality_status),
                route_kind_name(case.route),
                case.reference_decode_success,
                lossless_exact,
                encoded_bytes,
                bits_per_pixel,
                psnr,
                quality_requirement,
                stages,
            );
        }
        markdown.push_str(
            "\nEncoder results above are informative under T.803 and are not decoder compliance claims. Conformance does not establish robustness, security, adoption, or performance.\n",
        );
    }

    fn validate(&self) -> Result<(), ReportError> {
        if self.schema_version != 3 || self.standard != STANDARD || self.source_url != SOURCE_URL {
            return report_error("schema, standard, or source URL does not match T.803 v3");
        }
        validate_sha256(&self.source_archive_sha256, "report archive")
            .map_err(|error| ReportError::Validation(error.to_string()))?;
        if [
            &self.iut.name,
            &self.iut.version,
            &self.iut.candidate_sha,
            &self.iut.claim,
        ]
        .into_iter()
        .any(String::is_empty)
        {
            return report_error("IUT identity fields must not be empty");
        }
        if [
            &self.platform.os,
            &self.platform.arch,
            &self.platform.hardware,
            &self.platform.driver,
        ]
        .into_iter()
        .any(String::is_empty)
        {
            return report_error("platform identity fields must not be empty");
        }
        if !is_strictly_sorted(&self.features) {
            return report_error("features must be sorted and unique");
        }
        if self.corpus.is_empty() || !is_sorted_by(&self.corpus, |entry| entry.path.as_str()) {
            return report_error("corpus hashes must be non-empty, sorted, and unique");
        }
        for entry in &self.corpus {
            validate_path(&entry.path)
                .map_err(|error| ReportError::Validation(error.to_string()))?;
            validate_sha256(&entry.sha256, &entry.path)
                .map_err(|error| ReportError::Validation(error.to_string()))?;
        }
        if self.native_component_oracles.is_empty()
            || !is_sorted_by(&self.native_component_oracles, |oracle| {
                oracle.codestream_path.as_str()
            })
        {
            return report_error(
                "native component oracle evidence must be non-empty, sorted, and unique",
            );
        }
        for oracle in &self.native_component_oracles {
            validate_native_component_oracle(oracle)?;
            if !self.corpus.iter().any(|entry| {
                entry.path == oracle.codestream_path && entry.sha256 == oracle.codestream_sha256
            }) {
                return report_error(format!(
                    "native component oracle {} is not pinned by the report corpus",
                    oracle.codestream_path
                ));
            }
        }
        if self.cases.is_empty() {
            return report_error("report must contain at least one case");
        }
        if self.decoder_routes != summarize_routes(&self.cases) {
            return report_error("decoder route summary does not match per-case routes");
        }
        let mut ids = BTreeSet::new();
        for case in &self.cases {
            if case.id.is_empty() || !ids.insert(case.id.as_str()) {
                return report_error("case ids must be non-empty and unique");
            }
            validate_case(case)?;
            validate_route(case, case.route)?;
        }
        self.encoder.validate()?;
        let decoder_status = derive_status(self.cases.iter().map(|case| case.status));
        let derived =
            if decoder_status == ReportStatus::Pass && self.encoder.status == ReportStatus::Pass {
                ReportStatus::Pass
            } else {
                ReportStatus::Fail
            };
        if self.status != derived {
            return report_error("final status does not match case results");
        }
        Ok(())
    }
}

fn validate_case(case: &CaseReport) -> Result<(), ReportError> {
    let valid_metrics = case
        .mse
        .into_iter()
        .chain(case.allowed_mse)
        .all(|value| value.is_finite() && value >= 0.0);
    if !valid_metrics {
        return report_error(format!("{} has invalid MSE data", case.id));
    }
    match case.status {
        CaseStatus::Pass | CaseStatus::Fail => {
            if case.peak.is_none()
                || case.error.is_some()
                || (case.allowed_mse.is_some() != case.mse.is_some())
            {
                return report_error(format!("{} has incomplete comparison metrics", case.id));
            }
        }
        CaseStatus::Error => {
            if case.error.as_deref().is_none_or(str::is_empty)
                || case.peak.is_some()
                || case.mse.is_some()
            {
                return report_error(format!("{} has invalid error evidence", case.id));
            }
        }
    }
    Ok(())
}

fn validate_native_component_oracle(
    oracle: &NativeComponentOracleEvidence,
) -> Result<(), ReportError> {
    validate_path(&oracle.codestream_path)
        .map_err(|error| ReportError::Validation(error.to_string()))?;
    validate_sha256(&oracle.codestream_sha256, &oracle.codestream_path)
        .map_err(|error| ReportError::Validation(error.to_string()))?;
    validate_sha256(
        &oracle.production_components_sha256,
        "production native components",
    )
    .map_err(|error| ReportError::Validation(error.to_string()))?;
    validate_sha256(
        &oracle.openjpeg_components_sha256,
        "OpenJPEG native components",
    )
    .map_err(|error| ReportError::Validation(error.to_string()))?;
    if oracle.selection.is_empty()
        || oracle.implementation != "OpenJPEG"
        || oracle.version.is_empty()
        || oracle.library.is_empty()
        || oracle.component_count <= 4
        || oracle.compared_sample_count < oracle.component_count as u64
    {
        return report_error(format!(
            "{} has incomplete native component oracle identity or coverage",
            oracle.codestream_path
        ));
    }
    if !oracle.exact || oracle.production_components_sha256 != oracle.openjpeg_components_sha256 {
        return report_error(format!(
            "{} did not match OpenJPEG component-for-component",
            oracle.codestream_path
        ));
    }
    Ok(())
}

fn validate_encoder_case(case: &EncoderCaseReport) -> Result<(), ReportError> {
    for (name, value) in [
        ("bits per pixel", case.actual_bits_per_pixel),
        ("PSNR", case.psnr_db),
    ] {
        if value.is_some_and(|value| !value.is_finite() || value < 0.0) {
            return report_error(format!("{} has invalid {name}", case.id));
        }
    }
    if case.mode == EncoderMode::Lossless && case.psnr_db.is_some() {
        return report_error(format!("{} reports PSNR for a lossless case", case.id));
    }
    if case.psnr_infinite && case.psnr_db.is_some() {
        return report_error(format!("{} reports finite and infinite PSNR", case.id));
    }
    if case.mode == EncoderMode::Lossless && case.psnr_infinite {
        return report_error(format!("{} reports PSNR for a lossless case", case.id));
    }
    if case.mode == EncoderMode::Lossy && case.lossless_exact.is_some() {
        return report_error(format!(
            "{} reports lossless equality for a lossy case",
            case.id
        ));
    }
    match (case.mode, case.quality_status) {
        (EncoderMode::Lossless, EncoderQualityStatus::NotApplicable) => {
            if case.quality_requirement.is_some() || case.quality_error.is_some() {
                return report_error(format!("{} has a lossless quality gate", case.id));
            }
        }
        (EncoderMode::Lossy, EncoderQualityStatus::Pass) => {
            if case
                .quality_requirement
                .as_deref()
                .is_none_or(str::is_empty)
                || case.quality_error.is_some()
                || (!case.psnr_infinite && case.psnr_db.is_none())
            {
                return report_error(format!(
                    "{} has incomplete passing quality evidence",
                    case.id
                ));
            }
        }
        (EncoderMode::Lossy, EncoderQualityStatus::Fail) => {
            if case
                .quality_requirement
                .as_deref()
                .is_none_or(str::is_empty)
                || case.quality_error.as_deref().is_none_or(str::is_empty)
            {
                return report_error(format!(
                    "{} has incomplete failed quality evidence",
                    case.id
                ));
            }
        }
        _ => return report_error(format!("{} has an invalid quality status", case.id)),
    }
    match case.status {
        CaseStatus::Pass => {
            let exact = case.mode == EncoderMode::Lossy || case.lossless_exact == Some(true);
            if !case.reference_decode_success
                || !exact
                || case.encoded_bytes.is_none()
                || case.actual_bits_per_pixel.is_none()
                || case.error.is_some()
            {
                return report_error(format!(
                    "{} has incomplete passing encoder evidence",
                    case.id
                ));
            }
        }
        CaseStatus::Fail => {
            if case.encoded_bytes.is_none()
                || case.actual_bits_per_pixel.is_none()
                || case.error.as_deref().is_none_or(str::is_empty)
            {
                return report_error(format!("{} has invalid failed encoder evidence", case.id));
            }
        }
        CaseStatus::Error => {
            if case.reference_decode_success
                || case.lossless_exact == Some(true)
                || case.error.as_deref().is_none_or(str::is_empty)
            {
                return report_error(format!("{} has invalid encoder error evidence", case.id));
            }
        }
    }
    Ok(())
}

fn validate_route(case: &CaseReport, route: RouteKind) -> Result<(), ReportError> {
    let required = [
        RouteStageName::Parsing,
        RouteStageName::Tier1,
        RouteStageName::Dequantization,
        RouteStageName::Idwt,
        RouteStageName::Mct,
        RouteStageName::ColorOutput,
        RouteStageName::HostToDevice,
        RouteStageName::DeviceToHost,
    ];
    let stages = case
        .stages
        .iter()
        .map(|stage| stage.stage)
        .collect::<BTreeSet<_>>();
    if case.stages.len() != required.len() || stages != required.into_iter().collect() {
        return report_error(format!("{} does not disclose every route stage", case.id));
    }
    validate_route_locations(
        &case.id,
        route,
        case.stages.iter().map(|stage| stage.location),
    )
}

fn validate_encode_route(case: &EncoderCaseReport) -> Result<(), ReportError> {
    let required = [
        EncodeRouteStageName::InputPreparation,
        EncodeRouteStageName::ForwardRct,
        EncodeRouteStageName::ForwardIct,
        EncodeRouteStageName::ForwardDwt53,
        EncodeRouteStageName::ForwardDwt97,
        EncodeRouteStageName::Quantization,
        EncodeRouteStageName::Tier1,
        EncodeRouteStageName::Packetization,
        EncodeRouteStageName::HostToDevice,
        EncodeRouteStageName::DeviceToHost,
    ];
    let stages = case
        .stages
        .iter()
        .map(|stage| stage.stage)
        .collect::<BTreeSet<_>>();
    if case.stages.len() != required.len() || stages != required.into_iter().collect() {
        return report_error(format!(
            "{} does not disclose every encoder route stage",
            case.id
        ));
    }
    validate_route_locations(
        &case.id,
        case.route,
        case.stages.iter().map(|stage| stage.location),
    )
}

fn validate_route_locations(
    id: &str,
    route: RouteKind,
    locations: impl Iterator<Item = ExecutionLocation>,
) -> Result<(), ReportError> {
    let locations = locations
        .filter(|location| *location != ExecutionLocation::NotUsed)
        .collect::<BTreeSet<_>>();
    let uses_cpu = locations.contains(&ExecutionLocation::Cpu);
    let device_count = usize::from(locations.contains(&ExecutionLocation::Cuda))
        + usize::from(locations.contains(&ExecutionLocation::Metal));
    let valid = match route {
        RouteKind::Cpu => uses_cpu && device_count == 0,
        RouteKind::Hybrid => uses_cpu && device_count == 1,
        RouteKind::DeviceNative => !uses_cpu && device_count == 1,
    };
    if valid {
        Ok(())
    } else {
        report_error(format!(
            "{id} route stages contradict the {} label",
            route_kind_name(route)
        ))
    }
}

fn is_strictly_sorted(values: &[String]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

fn is_sorted_by<T>(values: &[T], key: impl Fn(&T) -> &str) -> bool {
    values.windows(2).all(|pair| key(&pair[0]) < key(&pair[1]))
}

fn summarize_routes(cases: &[CaseReport]) -> DecoderRouteSummary {
    let mut summary = DecoderRouteSummary {
        total: cases.len(),
        device_native: 0,
        hybrid: 0,
        cpu: 0,
    };
    for case in cases {
        match case.route {
            RouteKind::Cpu => summary.cpu += 1,
            RouteKind::Hybrid => summary.hybrid += 1,
            RouteKind::DeviceNative => summary.device_native += 1,
        }
    }
    summary
}

fn derive_status(statuses: impl Iterator<Item = CaseStatus>) -> ReportStatus {
    if statuses
        .into_iter()
        .all(|status| status == CaseStatus::Pass)
    {
        ReportStatus::Pass
    } else {
        ReportStatus::Fail
    }
}

fn derive_quality_status(cases: &[EncoderCaseReport]) -> ReportStatus {
    if cases
        .iter()
        .any(|case| case.quality_status == EncoderQualityStatus::Fail)
    {
        ReportStatus::Fail
    } else {
        ReportStatus::Pass
    }
}

fn combine_status(left: ReportStatus, right: ReportStatus) -> ReportStatus {
    if left == ReportStatus::Pass && right == ReportStatus::Pass {
        ReportStatus::Pass
    } else {
        ReportStatus::Fail
    }
}

fn markdown_cell(value: &str) -> String {
    value.replace('|', "\\|").replace('\n', " ")
}

fn report_status_name(status: ReportStatus) -> &'static str {
    match status {
        ReportStatus::Pass => "pass",
        ReportStatus::Fail => "fail",
    }
}

fn case_status_name(status: CaseStatus) -> &'static str {
    match status {
        CaseStatus::Pass => "pass",
        CaseStatus::Fail => "fail",
        CaseStatus::Error => "error",
    }
}

fn quality_status_name(status: EncoderQualityStatus) -> &'static str {
    match status {
        EncoderQualityStatus::Pass => "pass",
        EncoderQualityStatus::Fail => "fail",
        EncoderQualityStatus::NotApplicable => "not-applicable",
    }
}

fn route_kind_name(route: RouteKind) -> &'static str {
    match route {
        RouteKind::Cpu => "cpu",
        RouteKind::Hybrid => "hybrid",
        RouteKind::DeviceNative => "device-native",
    }
}

fn stage_name(stage: RouteStageName) -> &'static str {
    match stage {
        RouteStageName::Parsing => "parsing",
        RouteStageName::Tier1 => "tier1",
        RouteStageName::Dequantization => "dequantization",
        RouteStageName::Idwt => "idwt",
        RouteStageName::Mct => "mct",
        RouteStageName::ColorOutput => "color-output",
        RouteStageName::HostToDevice => "host-to-device",
        RouteStageName::DeviceToHost => "device-to-host",
    }
}

fn encode_stage_name(stage: EncodeRouteStageName) -> &'static str {
    match stage {
        EncodeRouteStageName::InputPreparation => "input-preparation",
        EncodeRouteStageName::ForwardRct => "forward-rct",
        EncodeRouteStageName::ForwardIct => "forward-ict",
        EncodeRouteStageName::ForwardDwt53 => "forward-dwt53",
        EncodeRouteStageName::ForwardDwt97 => "forward-dwt97",
        EncodeRouteStageName::Quantization => "quantization",
        EncodeRouteStageName::Tier1 => "tier1",
        EncodeRouteStageName::Packetization => "packetization",
        EncodeRouteStageName::HostToDevice => "host-to-device",
        EncodeRouteStageName::DeviceToHost => "device-to-host",
    }
}

fn encoder_mode_name(mode: EncoderMode) -> &'static str {
    match mode {
        EncoderMode::Lossless => "lossless",
        EncoderMode::Lossy => "lossy",
    }
}

fn location_name(location: ExecutionLocation) -> &'static str {
    match location {
        ExecutionLocation::Cpu => "cpu",
        ExecutionLocation::Cuda => "cuda",
        ExecutionLocation::Metal => "metal",
        ExecutionLocation::NotUsed => "not-used",
    }
}

fn report_error<T>(message: impl Into<String>) -> Result<T, ReportError> {
    Err(ReportError::Validation(message.into()))
}
