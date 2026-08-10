// SPDX-License-Identifier: MIT OR Apache-2.0

use serde::{Deserialize, Serialize};

use super::{EncodeRouteStageName, EncoderCaseReport};
use crate::report::{report_error, ExecutionLocation, ReportError};

/// Completed accelerator dispatch counters for one encoder case.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderDispatchEvidence {
    /// Pixel deinterleave and level-shift dispatches.
    pub deinterleave: usize,
    /// Forward reversible colour-transform dispatches.
    pub forward_rct: usize,
    /// Forward irreversible colour-transform dispatches.
    pub forward_ict: usize,
    /// Forward reversible 5/3 DWT dispatches.
    pub forward_dwt53: usize,
    /// Forward irreversible 9/7 DWT dispatches.
    pub forward_dwt97: usize,
    /// Sub-band quantization dispatches.
    pub quantize_subband: usize,
    /// Classic Tier-1 code-block dispatches.
    pub tier1_code_block: usize,
    /// HTJ2K code-block dispatches.
    pub ht_code_block: usize,
    /// Tier-2 packetization dispatches.
    pub packetization: usize,
}

pub(super) fn validate_encoder_dispatches(
    case: &EncoderCaseReport,
    required: bool,
) -> Result<(), ReportError> {
    let Some(dispatches) = case.accelerator_dispatches else {
        return if required {
            report_error(format!(
                "{} does not publish encoder accelerator dispatch counters",
                case.id
            ))
        } else {
            Ok(())
        };
    };
    let location = |stage| {
        case.stages
            .iter()
            .find(|candidate| candidate.stage == stage)
            .map(|candidate| candidate.location)
            .expect("encoder route stage set was validated before dispatch counters")
    };
    for (stage, count) in [
        (
            EncodeRouteStageName::InputPreparation,
            dispatches.deinterleave,
        ),
        (EncodeRouteStageName::ForwardRct, dispatches.forward_rct),
        (EncodeRouteStageName::ForwardIct, dispatches.forward_ict),
        (EncodeRouteStageName::ForwardDwt53, dispatches.forward_dwt53),
        (EncodeRouteStageName::ForwardDwt97, dispatches.forward_dwt97),
        (
            EncodeRouteStageName::Quantization,
            dispatches.quantize_subband,
        ),
        (
            EncodeRouteStageName::Tier1,
            dispatches
                .tier1_code_block
                .saturating_add(dispatches.ht_code_block),
        ),
        (
            EncodeRouteStageName::Packetization,
            dispatches.packetization,
        ),
    ] {
        let stage_location = location(stage);
        let ran_on_device = matches!(
            stage_location,
            ExecutionLocation::Cuda | ExecutionLocation::Metal
        );
        if (count > 0) != ran_on_device {
            return report_error(format!(
                "{} encoder dispatch counters contradict the {stage:?} stage",
                case.id
            ));
        }
    }

    let any_dispatch = [
        dispatches.deinterleave,
        dispatches.forward_rct,
        dispatches.forward_ict,
        dispatches.forward_dwt53,
        dispatches.forward_dwt97,
        dispatches.quantize_subband,
        dispatches.tier1_code_block,
        dispatches.ht_code_block,
        dispatches.packetization,
    ]
    .into_iter()
    .any(|count| count > 0);
    for stage in [
        EncodeRouteStageName::HostToDevice,
        EncodeRouteStageName::DeviceToHost,
    ] {
        let transfer_on_device = matches!(
            location(stage),
            ExecutionLocation::Cuda | ExecutionLocation::Metal
        );
        if transfer_on_device != any_dispatch {
            return report_error(format!(
                "{} encoder dispatch counters contradict the {stage:?} stage",
                case.id
            ));
        }
    }
    Ok(())
}

pub(super) fn encoder_dispatches_name(dispatches: Option<&EncoderDispatchEvidence>) -> String {
    let Some(dispatches) = dispatches else {
        return "not-recorded".to_string();
    };
    format!(
        "deinterleave={}, forward-rct={}, forward-ict={}, forward-dwt53={}, forward-dwt97={}, quantize={}, classic-tier1={}, ht-tier1={}, packetization={}",
        dispatches.deinterleave,
        dispatches.forward_rct,
        dispatches.forward_ict,
        dispatches.forward_dwt53,
        dispatches.forward_dwt97,
        dispatches.quantize_subband,
        dispatches.tier1_code_block,
        dispatches.ht_code_block,
        dispatches.packetization,
    )
}
