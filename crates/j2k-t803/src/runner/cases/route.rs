// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    AcceleratorExecutionEvidence, ExecutionLocation, RouteKind, RouteStage, RouteStageName,
};

#[derive(Clone, Debug)]
pub(in crate::runner) struct RouteEvidence {
    pub(in crate::runner) kind: RouteKind,
    pub(in crate::runner) stages: Vec<RouteStage>,
    pub(in crate::runner) accelerator_execution: Option<AcceleratorExecutionEvidence>,
}

pub(in crate::runner) fn cpu_route(mct: bool) -> RouteEvidence {
    route_evidence(ExecutionLocation::Cpu, None, mct)
}

#[cfg(any(feature = "cuda-runner", feature = "metal-runner"))]
pub(in crate::runner) fn parse_only_route() -> RouteEvidence {
    RouteEvidence {
        kind: RouteKind::Cpu,
        stages: Vec::from([
            RouteStage {
                stage: RouteStageName::Parsing,
                location: ExecutionLocation::Cpu,
            },
            RouteStage {
                stage: RouteStageName::Tier1,
                location: ExecutionLocation::NotUsed,
            },
            RouteStage {
                stage: RouteStageName::Dequantization,
                location: ExecutionLocation::NotUsed,
            },
            RouteStage {
                stage: RouteStageName::Idwt,
                location: ExecutionLocation::NotUsed,
            },
            RouteStage {
                stage: RouteStageName::Mct,
                location: ExecutionLocation::NotUsed,
            },
            RouteStage {
                stage: RouteStageName::ColorOutput,
                location: ExecutionLocation::NotUsed,
            },
            RouteStage {
                stage: RouteStageName::HostToDevice,
                location: ExecutionLocation::NotUsed,
            },
            RouteStage {
                stage: RouteStageName::DeviceToHost,
                location: ExecutionLocation::NotUsed,
            },
        ]),
        accelerator_execution: None,
    }
}

fn route_evidence(
    parsing: ExecutionLocation,
    device: Option<ExecutionLocation>,
    mct: bool,
) -> RouteEvidence {
    let execution = device.unwrap_or(ExecutionLocation::Cpu);
    RouteEvidence {
        kind: if device.is_some() {
            RouteKind::Hybrid
        } else {
            RouteKind::Cpu
        },
        stages: Vec::from([
            RouteStage {
                stage: RouteStageName::Parsing,
                location: parsing,
            },
            RouteStage {
                stage: RouteStageName::Tier1,
                location: execution,
            },
            RouteStage {
                stage: RouteStageName::Dequantization,
                location: execution,
            },
            RouteStage {
                stage: RouteStageName::Idwt,
                location: execution,
            },
            RouteStage {
                stage: RouteStageName::Mct,
                location: if mct {
                    execution
                } else {
                    ExecutionLocation::NotUsed
                },
            },
            RouteStage {
                stage: RouteStageName::ColorOutput,
                location: execution,
            },
            RouteStage {
                stage: RouteStageName::HostToDevice,
                location: device.unwrap_or(ExecutionLocation::NotUsed),
            },
            RouteStage {
                stage: RouteStageName::DeviceToHost,
                location: device.unwrap_or(ExecutionLocation::NotUsed),
            },
        ]),
        accelerator_execution: None,
    }
}
