// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{collections::BTreeSet, fmt::Write as _};

use serde::{Deserialize, Serialize};

use crate::{
    manifest::{validate_path, validate_sha256},
    EncoderMode, EncoderReferenceDecoder,
};

mod dispatch;
mod reference;

pub use dispatch::EncoderDispatchEvidence;
use dispatch::{encoder_dispatches_name, validate_encoder_dispatches};
use reference::{
    push_supplemental_reference_markdown, reference_decoder_name, validate_reference_decoders,
};
pub use reference::{EncoderReferenceIdentity, EncoderSupplementalReferenceIdentity};

use super::{
    case_status_name, derive_status,
    execution::{location_name, route_kind_name, validate_route_locations},
    markdown_cell, report_error, report_status_name, CaseStatus, ExecutionLocation, ReportError,
    ReportStatus, RouteKind,
};

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

/// Execution location for one encoder stage.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncodeRouteStage {
    /// Stage being disclosed.
    pub stage: EncodeRouteStageName,
    /// Where the stage ran, or that it was not used.
    pub location: ExecutionLocation,
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
    /// Independent decoder selected by the committed matrix.
    #[serde(default)]
    pub reference_decoder: EncoderReferenceDecoder,
    /// Whether the selected reference implementation fully decoded the codestream.
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
    /// Completed production accelerator dispatch counters.
    #[serde(default)]
    pub accelerator_dispatches: Option<EncoderDispatchEvidence>,
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
    /// Executable identities for selected supplemental interoperability decoders.
    #[serde(default)]
    pub supplemental_reference_decoders: Vec<EncoderSupplementalReferenceIdentity>,
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
    #[expect(
        clippy::too_many_arguments,
        reason = "the constructor mirrors the validated encoder evidence sections"
    )]
    pub fn new(
        ics_path: String,
        ics_sha256: String,
        matrix_path: String,
        matrix_case_count: usize,
        matrix_case_sha256: String,
        reference_decoder: EncoderReferenceIdentity,
        supplemental_reference_decoders: Vec<EncoderSupplementalReferenceIdentity>,
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
            supplemental_reference_decoders,
            cases,
            standards_status,
            quality_status,
            status,
        };
        evidence.validate(super::CURRENT_REPORT_SCHEMA_VERSION)?;
        Ok(evidence)
    }

    pub(super) fn validate(&self, schema_version: u32) -> Result<(), ReportError> {
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
        validate_reference_decoders(
            &self.reference_decoder,
            &self.supplemental_reference_decoders,
            &self.cases,
        )?;
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
            validate_encoder_dispatches(case, schema_version >= 7)?;
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

pub(super) fn push_encoder_markdown(evidence: &EncoderEvidence, markdown: &mut String) {
    let _ = write!(
        markdown,
        "\n## Informative Annex D/F encoder evidence\n\n- Procedure: T.803 Annex D/F (informative)\n- ICS: {} (`{}`)\n- Matrix: {} ({} cases, `{}`)\n- Primary reference decoder: {} {} ({})\n- Standards status: {}\n- Quality-gate status: {}\n- Combined encoder status: {}\n",
        markdown_cell(&evidence.ics_path),
        evidence.ics_sha256,
        markdown_cell(&evidence.matrix_path),
        evidence.matrix_case_count,
        evidence.matrix_case_sha256,
        markdown_cell(&evidence.reference_decoder.implementation),
        markdown_cell(&evidence.reference_decoder.version),
        markdown_cell(&evidence.reference_decoder.standard),
        report_status_name(evidence.standards_status),
        report_status_name(evidence.quality_status),
        report_status_name(evidence.status),
    );
    push_supplemental_reference_markdown(&evidence.supplemental_reference_decoders, markdown);
    markdown.push_str("\n| Case | Mode | Standards status | Quality status | Route | Reference decoder | Reference decode | Lossless exact | Bytes | Bits/pixel | PSNR | Quality requirement | Route stages | Accelerator dispatches |\n|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---|---|---|\n");
    for case in &evidence.cases {
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
        let dispatches = encoder_dispatches_name(case.accelerator_dispatches.as_ref());
        let _ = writeln!(
            markdown,
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            markdown_cell(&case.id),
            encoder_mode_name(case.mode),
            case_status_name(case.status),
            quality_status_name(case.quality_status),
            route_kind_name(case.route),
            reference_decoder_name(case.reference_decoder),
            case.reference_decode_success,
            lossless_exact,
            encoded_bytes,
            bits_per_pixel,
            psnr,
            quality_requirement,
            stages,
            dispatches,
        );
    }
    markdown.push_str(
        "\nEncoder results above are informative under T.803 and are not decoder compliance claims. Conformance does not establish robustness, security, adoption, or performance.\n",
    );
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

fn quality_status_name(status: EncoderQualityStatus) -> &'static str {
    match status {
        EncoderQualityStatus::Pass => "pass",
        EncoderQualityStatus::Fail => "fail",
        EncoderQualityStatus::NotApplicable => "not-applicable",
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
