// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{collections::BTreeSet, fmt::Write as _};

use serde::{Deserialize, Serialize};

use crate::Part15CaseMetadata;

use super::{
    markdown_cell, report_error, report_status_name, CaseReport, CaseStatus, ReportError,
    ReportStatus,
};

/// Part 15 code-block set declared by the selected codestream.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum HtCodeBlockSetMode {
    /// Every code block uses HT coding.
    HtOnly,
    /// HT coding is declared, but classic coding remains permitted.
    HtDeclared,
    /// Classic and HT code blocks may be mixed.
    Mixed,
}

/// Whether Part 15 evidence is a formal ETS result or extended project evidence.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Part15EvidenceClassification {
    /// Selected from an official T.803 Part 15 ETS/BSET.
    Formal,
    /// Project-defined non-ETS coverage.
    Extended,
}

/// CAP, CPF, COD, and SIZ facts parsed from a selected HTJ2K codestream.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "the booleans preserve independently auditable CAP, COD, and transform facts"
)]
pub struct Part15CodestreamEvidence {
    /// Raw `Pcap` value.
    pub pcap: u32,
    /// Raw `Ccap15` value.
    pub ccap15: u16,
    /// Declared HT code-block set.
    pub mode: HtCodeBlockSetMode,
    /// Whether multiple HT sets are advertised.
    pub multiple_ht_sets: bool,
    /// Whether region-of-interest coding is advertised.
    pub roi: bool,
    /// Whether heterogeneous HT sets are advertised.
    pub heterogeneous: bool,
    /// Whether irreversible HT coding is advertised.
    pub ht_irreversible: bool,
    /// Magnitude bound parsed from `Ccap15`.
    pub bmagb: u8,
    /// Number of quality layers declared by COD.
    pub quality_layers: u8,
    /// Whether COD selects HT block coding by default.
    pub default_ht_block_coding: bool,
    /// Whether COD permits mixed classic/HT block coding.
    pub default_mixed_block_coding: bool,
    /// CPF profile words in codestream order.
    pub corresponding_profile_words: Vec<u16>,
    /// Whether COD selects the reversible 5/3 wavelet transform.
    pub reversible: bool,
    /// SIZ component count.
    pub component_count: u16,
}

/// Formal selection and parsed codestream facts for one Part 15 comparison.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Part15CaseEvidence {
    /// Formal or extended evidence classification.
    pub classification: Part15EvidenceClassification,
    /// BSET, compliance-class, MMAGB, BMAGB, and tolerance selection.
    pub selection: Part15CaseMetadata,
    /// Shared production-parser CAP/CPF/COD/SIZ facts.
    pub codestream: Part15CodestreamEvidence,
}

/// Representative native HT Tier-1 coverage axis.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum NativeHtCoverageAxis {
    /// HT cleanup without refinement.
    CleanupOnly,
    /// HT SigProp/MagRef refinement.
    Refinement,
    /// HTONLY codestream capability.
    HtOnly,
    /// MIXED codestream capability.
    Mixed,
    /// Reversible 5/3 transform.
    Reversible,
    /// Irreversible 9/7 transform.
    Irreversible,
    /// One-component grayscale output.
    Grayscale,
    /// Multi-component colour output.
    Color,
    /// More than one quality layer.
    Multilayer,
    /// Smallest BMAGB in the selected formal matrix.
    MinimumBmagb,
    /// Largest BMAGB in the selected formal matrix.
    MaximumBmagb,
}

const REQUIRED_NATIVE_HT_AXES: [NativeHtCoverageAxis; 11] = [
    NativeHtCoverageAxis::CleanupOnly,
    NativeHtCoverageAxis::Refinement,
    NativeHtCoverageAxis::HtOnly,
    NativeHtCoverageAxis::Mixed,
    NativeHtCoverageAxis::Reversible,
    NativeHtCoverageAxis::Irreversible,
    NativeHtCoverageAxis::Grayscale,
    NativeHtCoverageAxis::Color,
    NativeHtCoverageAxis::Multilayer,
    NativeHtCoverageAxis::MinimumBmagb,
    NativeHtCoverageAxis::MaximumBmagb,
];

/// One official passing case proving one native HT coverage axis.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeHtCoverageCase {
    /// Axis proved by this case.
    pub axis: NativeHtCoverageAxis,
    /// Stable official comparison identifier.
    pub case_id: String,
}

/// Derived representative native HT Tier-1 coverage for one adapter IUT.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeHtCoverageEvidence {
    /// Smallest BMAGB selected by the formal matrix.
    pub selected_bmagb_min: u8,
    /// Largest BMAGB selected by the formal matrix.
    pub selected_bmagb_max: u8,
    /// BMAGB values reached by passing accelerator-assisted HT cases.
    pub observed_bmagb: Vec<u8>,
    /// First passing official case proving each covered axis, sorted by axis.
    pub coverage: Vec<NativeHtCoverageCase>,
    /// Required axes not proved by any passing accelerator-assisted case.
    pub missing_axes: Vec<NativeHtCoverageAxis>,
    /// Result derived from `missing_axes`.
    pub status: ReportStatus,
}

pub(super) fn derive_native_ht_coverage(cases: &[CaseReport]) -> Option<NativeHtCoverageEvidence> {
    let selected_bmagb = cases
        .iter()
        .filter_map(|case| case.part15.as_ref().map(|part15| part15.selection.bmagb))
        .collect::<BTreeSet<_>>();
    let (&selected_bmagb_min, &selected_bmagb_max) =
        (selected_bmagb.first()?, selected_bmagb.last()?);
    let mut observed_bmagb = BTreeSet::new();
    let mut coverage = Vec::new();

    for case in cases {
        let (Some(part15), Some(execution)) = (&case.part15, case.accelerator_execution.as_ref())
        else {
            continue;
        };
        if case.status != CaseStatus::Pass || execution.ht_tier1_dispatches == 0 {
            continue;
        }
        let bmagb = part15.selection.bmagb;
        observed_bmagb.insert(bmagb);
        let mut record = |axis, applies| {
            if applies
                && !coverage
                    .iter()
                    .any(|covered: &NativeHtCoverageCase| covered.axis == axis)
            {
                coverage.push(NativeHtCoverageCase {
                    axis,
                    case_id: case.id.clone(),
                });
            }
        };
        record(
            NativeHtCoverageAxis::CleanupOnly,
            execution.ht_refinement_dispatches == 0,
        );
        record(
            NativeHtCoverageAxis::Refinement,
            execution.ht_refinement_dispatches > 0,
        );
        record(
            NativeHtCoverageAxis::HtOnly,
            part15.codestream.mode == HtCodeBlockSetMode::HtOnly,
        );
        record(
            NativeHtCoverageAxis::Mixed,
            part15.codestream.mode == HtCodeBlockSetMode::Mixed,
        );
        record(
            NativeHtCoverageAxis::Reversible,
            part15.codestream.reversible,
        );
        record(
            NativeHtCoverageAxis::Irreversible,
            !part15.codestream.reversible,
        );
        record(
            NativeHtCoverageAxis::Grayscale,
            part15.codestream.component_count == 1,
        );
        record(
            NativeHtCoverageAxis::Color,
            part15.codestream.component_count >= 3,
        );
        record(
            NativeHtCoverageAxis::Multilayer,
            part15.codestream.quality_layers > 1,
        );
        record(
            NativeHtCoverageAxis::MinimumBmagb,
            bmagb == selected_bmagb_min,
        );
        record(
            NativeHtCoverageAxis::MaximumBmagb,
            bmagb == selected_bmagb_max,
        );
    }
    coverage.sort_by_key(|covered| covered.axis);
    let covered = coverage
        .iter()
        .map(|covered| covered.axis)
        .collect::<BTreeSet<_>>();
    let missing_axes = REQUIRED_NATIVE_HT_AXES
        .into_iter()
        .filter(|axis| !covered.contains(axis))
        .collect::<Vec<_>>();
    Some(NativeHtCoverageEvidence {
        selected_bmagb_min,
        selected_bmagb_max,
        observed_bmagb: observed_bmagb.into_iter().collect(),
        coverage,
        status: if missing_axes.is_empty() {
            ReportStatus::Pass
        } else {
            ReportStatus::Fail
        },
        missing_axes,
    })
}

pub(super) fn validate_part15_case(
    id: &str,
    evidence: &Part15CaseEvidence,
) -> Result<(), ReportError> {
    if evidence.classification != Part15EvidenceClassification::Formal
        || evidence.selection.bmagb != evidence.codestream.bmagb
        || evidence.selection.bmagb > evidence.selection.mmagb
        || !(8..=74).contains(&evidence.codestream.bmagb)
        || evidence.codestream.quality_layers == 0
        || evidence.codestream.component_count == 0
    {
        return report_error(format!(
            "{id} has invalid Part 15 selection or codestream evidence"
        ));
    }
    Ok(())
}

pub(super) fn validate_native_ht_coverage(
    cases: &[CaseReport],
    actual: Option<&NativeHtCoverageEvidence>,
) -> Result<(), ReportError> {
    let Some(actual) = actual else {
        return Ok(());
    };
    let expected = derive_native_ht_coverage(cases);
    if Some(actual) != expected.as_ref() {
        return report_error("native HT coverage does not match per-case evidence");
    }
    Ok(())
}

pub(super) fn push_native_ht_coverage_markdown(
    coverage: Option<&NativeHtCoverageEvidence>,
    markdown: &mut String,
) {
    let Some(coverage) = coverage else {
        return;
    };
    let covered = coverage
        .coverage
        .iter()
        .map(|covered| {
            format!(
                "{}={}",
                coverage_axis_name(covered.axis),
                markdown_cell(&covered.case_id)
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let missing = coverage
        .missing_axes
        .iter()
        .map(|axis| coverage_axis_name(*axis))
        .collect::<Vec<_>>()
        .join(", ");
    let _ = write!(
        markdown,
        "\n## Native HT Tier-1 coverage\n\n- Selected BMAGB range: {} through {}.\n- Accelerator-observed BMAGB values: {:?}.\n- Covered axes: {}.\n- Missing axes: {}.\n- Coverage status: {}.\n",
        coverage.selected_bmagb_min,
        coverage.selected_bmagb_max,
        coverage.observed_bmagb,
        covered,
        if missing.is_empty() { "none" } else { &missing },
        report_status_name(coverage.status),
    );
}

const fn coverage_axis_name(axis: NativeHtCoverageAxis) -> &'static str {
    match axis {
        NativeHtCoverageAxis::CleanupOnly => "cleanup-only",
        NativeHtCoverageAxis::Refinement => "refinement",
        NativeHtCoverageAxis::HtOnly => "ht-only",
        NativeHtCoverageAxis::Mixed => "mixed",
        NativeHtCoverageAxis::Reversible => "reversible",
        NativeHtCoverageAxis::Irreversible => "irreversible",
        NativeHtCoverageAxis::Grayscale => "grayscale",
        NativeHtCoverageAxis::Color => "color",
        NativeHtCoverageAxis::Multilayer => "multilayer",
        NativeHtCoverageAxis::MinimumBmagb => "minimum-bmagb",
        NativeHtCoverageAxis::MaximumBmagb => "maximum-bmagb",
    }
}

#[cfg(test)]
mod tests;
