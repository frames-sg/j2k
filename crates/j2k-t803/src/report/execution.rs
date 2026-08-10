// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use super::{report_error, CaseReport, ReportError};

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

/// Execution location for one decoder stage.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RouteStage {
    /// Stage being disclosed.
    pub stage: RouteStageName,
    /// Where the stage ran, or that it was not used.
    pub location: ExecutionLocation,
}

/// Completed accelerator observations behind a decoder route classification.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AcceleratorExecutionEvidence {
    /// Accelerator that produced these observations.
    pub backend: ExecutionLocation,
    /// Completed HT Tier-1 dispatches.
    pub ht_tier1_dispatches: usize,
    /// Completed HT dispatches containing at least one refinement job.
    pub ht_refinement_dispatches: usize,
    /// Completed classic Tier-1 dispatches.
    pub classic_tier1_dispatches: usize,
    /// Completed dequantization dispatches, including fused work.
    pub dequantization_dispatches: usize,
    /// Completed inverse-DWT dispatches.
    pub idwt_dispatches: usize,
    /// Completed inverse multi-component transform dispatches.
    pub mct_dispatches: usize,
    /// Completed final output dispatches.
    pub color_output_dispatches: usize,
    /// CUDA compressed-payload bytes uploaded for the completed decode.
    pub uploaded_payload_bytes: Option<usize>,
    /// Metal host inputs made available for completed Tier-1 dispatches.
    pub metal_host_inputs: Option<usize>,
    /// Whether the final device output was successfully transferred to the host.
    pub device_to_host_completed: bool,
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

#[expect(
    clippy::too_many_lines,
    reason = "route validation checks one observed stage/counter contract without duplicating cross-field invariants"
)]
pub(super) fn validate_decoder_route(
    case: &CaseReport,
    schema_version: u32,
) -> Result<(), ReportError> {
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
        case.route,
        case.stages.iter().map(|stage| stage.location),
    )?;

    if schema_version == 3 {
        if case.accelerator_execution.is_some() {
            return report_error(format!(
                "{} historical schema cannot contain accelerator execution evidence",
                case.id
            ));
        }
        return Ok(());
    }

    let Some(execution) = &case.accelerator_execution else {
        return if case.route == RouteKind::Cpu {
            Ok(())
        } else {
            report_error(format!(
                "{} accelerator execution evidence is missing",
                case.id
            ))
        };
    };
    if case.route == RouteKind::Cpu {
        return report_error(format!(
            "{} CPU route must not contain accelerator execution evidence",
            case.id
        ));
    }
    if !matches!(
        execution.backend,
        ExecutionLocation::Cuda | ExecutionLocation::Metal
    ) {
        return report_error(format!(
            "{} accelerator execution backend is not CUDA or Metal",
            case.id
        ));
    }
    if execution.ht_tier1_dispatches + execution.classic_tier1_dispatches == 0
        || execution.ht_refinement_dispatches > execution.ht_tier1_dispatches
        || execution.dequantization_dispatches == 0
        || execution.color_output_dispatches == 0
    {
        return report_error(format!(
            "{} accelerator execution counters are incomplete",
            case.id
        ));
    }
    match execution.backend {
        ExecutionLocation::Cuda
            if execution
                .uploaded_payload_bytes
                .is_none_or(|bytes| bytes == 0)
                || execution.metal_host_inputs.is_some() =>
        {
            return report_error(format!(
                "{} CUDA execution transfer observations are invalid",
                case.id
            ));
        }
        ExecutionLocation::Metal
            if execution.metal_host_inputs.is_none_or(|inputs| inputs == 0)
                || execution.uploaded_payload_bytes.is_some() =>
        {
            return report_error(format!(
                "{} Metal execution transfer observations are invalid",
                case.id
            ));
        }
        _ => {}
    }

    let location = |stage| {
        case.stages
            .iter()
            .find(|candidate| candidate.stage == stage)
            .map(|candidate| candidate.location)
            .expect("route stage set was validated above")
    };
    for stage in [
        RouteStageName::Tier1,
        RouteStageName::Dequantization,
        RouteStageName::ColorOutput,
        RouteStageName::HostToDevice,
    ] {
        if location(stage) != execution.backend {
            return report_error(format!(
                "{} accelerator execution contradicts the {} stage",
                case.id,
                stage_name(stage)
            ));
        }
    }
    for (stage, dispatches) in [
        (RouteStageName::Idwt, execution.idwt_dispatches),
        (RouteStageName::Mct, execution.mct_dispatches),
    ] {
        let stage_location = location(stage);
        if (dispatches > 0 && stage_location != execution.backend)
            || (dispatches == 0 && stage_location == execution.backend)
        {
            return report_error(format!(
                "{} accelerator execution contradicts the {} stage",
                case.id,
                stage_name(stage)
            ));
        }
    }
    let download_location = location(RouteStageName::DeviceToHost);
    if (execution.device_to_host_completed && download_location != execution.backend)
        || (!execution.device_to_host_completed && download_location == execution.backend)
    {
        return report_error(format!(
            "{} accelerator execution contradicts the device-to-host stage",
            case.id
        ));
    }
    Ok(())
}

pub(super) fn validate_route_locations(
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

pub(super) fn summarize_routes(cases: &[CaseReport]) -> DecoderRouteSummary {
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

pub(super) fn accelerator_execution_name(
    execution: Option<&AcceleratorExecutionEvidence>,
) -> String {
    let Some(execution) = execution else {
        return "not-applicable".to_string();
    };
    let transfer = match execution.backend {
        ExecutionLocation::Cuda => format!(
            "uploaded-payload-bytes={}",
            execution.uploaded_payload_bytes.unwrap_or_default()
        ),
        ExecutionLocation::Metal => format!(
            "metal-host-inputs={}",
            execution.metal_host_inputs.unwrap_or_default()
        ),
        _ => "invalid-backend".to_string(),
    };
    format!(
        "backend={}, ht-tier1={}, ht-refinement={}, classic-tier1={}, dequantization={}, idwt={}, mct={}, color-output={}, {}, device-to-host={}",
        location_name(execution.backend),
        execution.ht_tier1_dispatches,
        execution.ht_refinement_dispatches,
        execution.classic_tier1_dispatches,
        execution.dequantization_dispatches,
        execution.idwt_dispatches,
        execution.mct_dispatches,
        execution.color_output_dispatches,
        transfer,
        if execution.device_to_host_completed {
            "completed"
        } else {
            "not-completed"
        },
    )
}

pub(super) const fn route_kind_name(route: RouteKind) -> &'static str {
    match route {
        RouteKind::Cpu => "cpu",
        RouteKind::Hybrid => "hybrid",
        RouteKind::DeviceNative => "device-native",
    }
}

pub(super) const fn stage_name(stage: RouteStageName) -> &'static str {
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

pub(super) const fn location_name(location: ExecutionLocation) -> &'static str {
    match location {
        ExecutionLocation::Cpu => "cpu",
        ExecutionLocation::Cuda => "cuda",
        ExecutionLocation::Metal => "metal",
        ExecutionLocation::NotUsed => "not-used",
    }
}
