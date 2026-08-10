// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::J2kEncodeDispatchReport;

use crate::{
    EncodeRouteStage, EncodeRouteStageName, EncoderDispatchEvidence, EncoderMode, EncoderOperation,
    ExecutionLocation, RouteKind,
};

use crate::encoder::{EncoderCase, EncoderInputKind};

pub(super) struct EvaluatedRoute {
    pub(super) route: RouteKind,
    pub(super) stages: Vec<EncodeRouteStage>,
    pub(super) dispatches: EncoderDispatchEvidence,
}

pub(super) fn route_evidence(
    case: &EncoderCase,
    dispatch: J2kEncodeDispatchReport,
    device: Option<ExecutionLocation>,
) -> EvaluatedRoute {
    let device = device
        .filter(|location| matches!(location, ExecutionLocation::Cuda | ExecutionLocation::Metal));
    let location = |required: bool, count: usize| {
        if count > 0 {
            device.unwrap_or(ExecutionLocation::Cpu)
        } else if required {
            ExecutionLocation::Cpu
        } else {
            ExecutionLocation::NotUsed
        }
    };
    let encode = case.operation == EncoderOperation::Encode;
    let interleaved_colour =
        encode && case.input == EncoderInputKind::Interleaved && matches!(case.components, 3 | 4);
    let transfer_location = if dispatch.any() {
        device.unwrap_or(ExecutionLocation::Cpu)
    } else {
        ExecutionLocation::NotUsed
    };
    let stages = Vec::from([
        EncodeRouteStage {
            stage: EncodeRouteStageName::InputPreparation,
            location: location(true, dispatch.deinterleave),
        },
        EncodeRouteStage {
            stage: EncodeRouteStageName::ForwardRct,
            location: location(
                encode && case.mode == EncoderMode::Lossless && interleaved_colour,
                dispatch.forward_rct,
            ),
        },
        EncodeRouteStage {
            stage: EncodeRouteStageName::ForwardIct,
            location: location(
                encode && case.mode == EncoderMode::Lossy && interleaved_colour,
                dispatch.forward_ict,
            ),
        },
        EncodeRouteStage {
            stage: EncodeRouteStageName::ForwardDwt53,
            location: location(
                encode && case.mode == EncoderMode::Lossless && case.decomposition_levels > 0,
                dispatch.forward_dwt53,
            ),
        },
        EncodeRouteStage {
            stage: EncodeRouteStageName::ForwardDwt97,
            location: location(
                encode && case.mode == EncoderMode::Lossy && case.decomposition_levels > 0,
                dispatch.forward_dwt97,
            ),
        },
        EncodeRouteStage {
            stage: EncodeRouteStageName::Quantization,
            location: location(encode, dispatch.quantize_subband),
        },
        EncodeRouteStage {
            stage: EncodeRouteStageName::Tier1,
            location: location(
                true,
                dispatch
                    .tier1_code_block
                    .saturating_add(dispatch.ht_code_block),
            ),
        },
        EncodeRouteStage {
            stage: EncodeRouteStageName::Packetization,
            location: location(true, dispatch.packetization),
        },
        EncodeRouteStage {
            stage: EncodeRouteStageName::HostToDevice,
            location: transfer_location,
        },
        EncodeRouteStage {
            stage: EncodeRouteStageName::DeviceToHost,
            location: transfer_location,
        },
    ]);
    let uses_cpu = stages
        .iter()
        .any(|stage| stage.location == ExecutionLocation::Cpu);
    let uses_device = stages.iter().any(|stage| {
        matches!(
            stage.location,
            ExecutionLocation::Cuda | ExecutionLocation::Metal
        )
    });
    let route = match (uses_cpu, uses_device) {
        (true, true) => RouteKind::Hybrid,
        (false, true) => RouteKind::DeviceNative,
        _ => RouteKind::Cpu,
    };
    EvaluatedRoute {
        route,
        stages,
        dispatches: dispatch.into(),
    }
}

impl From<J2kEncodeDispatchReport> for EncoderDispatchEvidence {
    fn from(dispatch: J2kEncodeDispatchReport) -> Self {
        Self {
            deinterleave: dispatch.deinterleave,
            forward_rct: dispatch.forward_rct,
            forward_ict: dispatch.forward_ict,
            forward_dwt53: dispatch.forward_dwt53,
            forward_dwt97: dispatch.forward_dwt97,
            quantize_subband: dispatch.quantize_subband,
            tier1_code_block: dispatch.tier1_code_block,
            ht_code_block: dispatch.ht_code_block,
            packetization: dispatch.packetization,
        }
    }
}
