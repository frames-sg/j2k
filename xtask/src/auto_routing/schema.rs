// SPDX-License-Identifier: MIT OR Apache-2.0

pub(super) use j2k_test_support::{
    AutoRoutingBackend as Backend, AutoRoutingCell as Cell, AutoRoutingCodec as Codec,
    AutoRoutingContainer as Container, AutoRoutingEvidence as Evidence,
    AutoRoutingExecution as Execution, AutoRoutingManifest as ExternalManifest,
    AutoRoutingManifestCase as ExternalCase, AutoRoutingOperation as Operation,
    AutoRoutingRoute as Route, AutoRoutingWorkloadKind as WorkloadKind,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct WorkloadIdentity {
    pub(super) kind: WorkloadKind,
    pub(super) codec: Codec,
    pub(super) container: Container,
}

#[derive(Debug)]
pub(super) struct ValidatedManifest {
    pub(super) schema_version: u32,
    pub(super) cases: std::collections::BTreeMap<String, WorkloadIdentity>,
}
