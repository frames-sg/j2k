use std::{collections::BTreeSet, fmt::Write as _};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::manifest::{
    validate_path, validate_sha256, CorpusFile, T803Suite, SOURCE_URL, STANDARD,
};
mod encoder;
mod execution;
mod part15;

use encoder::push_encoder_markdown;
pub use encoder::{
    EncodeRouteStage, EncodeRouteStageName, EncoderCaseReport, EncoderDispatchEvidence,
    EncoderEvidence, EncoderQualityStatus, EncoderReferenceIdentity,
    EncoderSupplementalReferenceIdentity,
};
use execution::{
    accelerator_execution_name, location_name, route_kind_name, stage_name, summarize_routes,
    validate_decoder_route,
};
pub use execution::{
    AcceleratorExecutionEvidence, DecoderRouteSummary, ExecutionLocation, RouteKind, RouteStage,
    RouteStageName,
};
use part15::{
    derive_native_ht_coverage, push_native_ht_coverage_markdown, validate_native_ht_coverage,
    validate_part15_case,
};
pub use part15::{
    HtCodeBlockSetMode, NativeHtCoverageAxis, NativeHtCoverageCase, NativeHtCoverageEvidence,
    Part15CaseEvidence, Part15CodestreamEvidence, Part15EvidenceClassification,
};

/// Overall report result.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ReportStatus {
    /// Every selected case passed.
    Pass,
    /// At least one selected case failed or errored.
    Fail,
}

const CURRENT_REPORT_SCHEMA_VERSION: u32 = 7;

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
    /// Completed accelerator counters behind a hybrid or device-native route.
    #[serde(default)]
    pub accelerator_execution: Option<AcceleratorExecutionEvidence>,
    /// Formal Part 15 selection and parsed codestream facts, when applicable.
    #[serde(default)]
    pub part15: Option<Part15CaseEvidence>,
}

/// Versioned JSON and Markdown evidence for one T.803 run.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct T803Report {
    /// Report schema version.
    pub schema_version: u32,
    /// Formal suite selected for this run.
    #[serde(default = "historical_suite")]
    pub suite: T803Suite,
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
    /// Representative native HT Tier-1 coverage for a Part 15 adapter IUT.
    #[serde(default)]
    pub native_ht_coverage: Option<NativeHtCoverageEvidence>,
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
        suite: T803Suite,
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
        let native_ht_coverage = if features.iter().any(|feature| feature == "adapter-iut")
            && matches!(suite, T803Suite::Part15 | T803Suite::All)
        {
            derive_native_ht_coverage(&cases)
        } else {
            None
        };
        let decoder_status = derive_status(cases.iter().map(|case| case.status));
        let coverage_status = native_ht_coverage
            .as_ref()
            .map_or(ReportStatus::Pass, |coverage| coverage.status);
        let status = if decoder_status == ReportStatus::Pass
            && encoder.status == ReportStatus::Pass
            && coverage_status == ReportStatus::Pass
        {
            ReportStatus::Pass
        } else {
            ReportStatus::Fail
        };
        let report = Self {
            schema_version: CURRENT_REPORT_SCHEMA_VERSION,
            suite,
            standard: STANDARD.to_string(),
            source_url: SOURCE_URL.to_string(),
            source_archive_sha256,
            iut,
            platform,
            features,
            corpus,
            native_component_oracles,
            decoder_routes,
            native_ht_coverage,
            cases,
            encoder,
            status,
        };
        report.validate()?;
        Ok(report)
    }

    /// Serialize a validated report as stable pretty-printed JSON.
    pub fn to_json(&self) -> Result<String, ReportError> {
        if self.schema_version != CURRENT_REPORT_SCHEMA_VERSION {
            return report_error("historical report schemas are read-only");
        }
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
            "# T.803 conformance evidence\n\n- Standard: {}\n- Suite: {}\n- IUT: {} {}\n- Candidate SHA: {}\n- Claim: {}\n- Platform: {} {}\n- Hardware: {}\n- Driver: {}\n- Device-native: {} / {}\n- Hybrid: {} / {}\n- CPU-routed: {} / {}\n- Final status: {}\n",
            self.standard,
            suite_name(self.suite),
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
        push_native_ht_coverage_markdown(self.native_ht_coverage.as_ref(), &mut markdown);
        markdown.push_str("\n| Case | Table | Status | Route | Peak (measured / allowed) | MSE (measured / allowed) | Route stages | Completed accelerator observations |\n|---|---|---:|---|---:|---:|---|---|\n");
        self.push_decoder_cases(&mut markdown);
        push_encoder_markdown(&self.encoder, &mut markdown);
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
            let accelerator_execution =
                accelerator_execution_name(case.accelerator_execution.as_ref());
            let _ = writeln!(
                markdown,
                "| {} | {} | {} | {} | {} / {} | {} | {} | {} |",
                markdown_cell(&case.id),
                markdown_cell(&case.table),
                case_status_name(case.status),
                route_kind_name(case.route),
                peak,
                case.allowed_peak,
                mse,
                stages,
                accelerator_execution,
            );
        }
    }

    #[expect(
        clippy::too_many_lines,
        reason = "the report root validates one deterministic cross-section evidence contract"
    )]
    fn validate(&self) -> Result<(), ReportError> {
        if !matches!(self.schema_version, 3..=CURRENT_REPORT_SCHEMA_VERSION)
            || self.standard != STANDARD
            || self.source_url != SOURCE_URL
        {
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
            if let Some(part15) = &case.part15 {
                validate_part15_case(&case.id, part15)?;
            }
            validate_decoder_route(case, self.schema_version)?;
        }
        let requires_native_ht_coverage = self.schema_version >= 5
            && self.features.iter().any(|feature| feature == "adapter-iut")
            && matches!(self.suite, T803Suite::Part15 | T803Suite::All);
        if requires_native_ht_coverage != self.native_ht_coverage.is_some() {
            return report_error(
                "native HT coverage presence does not match the selected IUT and suite",
            );
        }
        validate_native_ht_coverage(&self.cases, self.native_ht_coverage.as_ref())?;
        self.encoder.validate(self.schema_version)?;
        let decoder_status = derive_status(self.cases.iter().map(|case| case.status));
        let coverage_status = self
            .native_ht_coverage
            .as_ref()
            .map_or(ReportStatus::Pass, |coverage| coverage.status);
        let derived = if decoder_status == ReportStatus::Pass
            && self.encoder.status == ReportStatus::Pass
            && coverage_status == ReportStatus::Pass
        {
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

fn is_strictly_sorted(values: &[String]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

fn is_sorted_by<T>(values: &[T], key: impl Fn(&T) -> &str) -> bool {
    values.windows(2).all(|pair| key(&pair[0]) < key(&pair[1]))
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

fn report_error<T>(message: impl Into<String>) -> Result<T, ReportError> {
    Err(ReportError::Validation(message.into()))
}

const fn historical_suite() -> T803Suite {
    T803Suite::Part1
}

const fn suite_name(suite: T803Suite) -> &'static str {
    match suite {
        T803Suite::Part1 => "part1",
        T803Suite::Part15 => "part15",
        T803Suite::All => "all",
    }
}
