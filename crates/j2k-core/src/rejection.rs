// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed capability and adapter-contract rejection reasons.

use core::fmt;

/// Stable category for a backend request rejected before execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CapabilityRejectionKind {
    /// The requested pixel or sample format is unsupported.
    UnsupportedFormat,
    /// Component sampling is unsupported.
    UnsupportedSampling,
    /// Sample precision or coded bitplane count is unsupported.
    UnsupportedBitDepth,
    /// The requested decode or encode operation is unsupported.
    UnsupportedOperation,
    /// A required prepared execution plan is absent or incompatible.
    MissingPreparedPlan,
    /// The compressed container or transfer shape is unsupported.
    UnsupportedContainer,
    /// Validated dimensions, ranges, or component geometry are inconsistent.
    GeometryMismatch,
    /// A bounded backend resource or address range cannot represent the request.
    ResourceLimit,
    /// A request belongs to a different device, context, or session.
    ContextMismatch,
    /// Checked internal ownership or execution state violated its contract.
    ContractViolation,
}

/// Typed internal rejection rendered to stable text at an adapter boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CapabilityRejection {
    kind: CapabilityRejectionKind,
    reason: &'static str,
}

impl CapabilityRejection {
    const fn new(kind: CapabilityRejectionKind, reason: &'static str) -> Self {
        Self { kind, reason }
    }

    /// Reject an unsupported pixel or sample format.
    #[must_use]
    pub const fn unsupported_format(reason: &'static str) -> Self {
        Self::new(CapabilityRejectionKind::UnsupportedFormat, reason)
    }

    /// Reject unsupported component sampling.
    #[must_use]
    pub const fn unsupported_sampling(reason: &'static str) -> Self {
        Self::new(CapabilityRejectionKind::UnsupportedSampling, reason)
    }

    /// Reject unsupported sample precision or coded bitplanes.
    #[must_use]
    pub const fn unsupported_bit_depth(reason: &'static str) -> Self {
        Self::new(CapabilityRejectionKind::UnsupportedBitDepth, reason)
    }

    /// Reject an unsupported operation.
    #[must_use]
    pub const fn unsupported_operation(reason: &'static str) -> Self {
        Self::new(CapabilityRejectionKind::UnsupportedOperation, reason)
    }

    /// Reject an absent or incompatible prepared plan.
    #[must_use]
    pub const fn missing_prepared_plan(reason: &'static str) -> Self {
        Self::new(CapabilityRejectionKind::MissingPreparedPlan, reason)
    }

    /// Reject an unsupported compressed container or transfer shape.
    #[must_use]
    pub const fn unsupported_container(reason: &'static str) -> Self {
        Self::new(CapabilityRejectionKind::UnsupportedContainer, reason)
    }

    /// Reject inconsistent dimensions, ranges, or component geometry.
    #[must_use]
    pub const fn geometry_mismatch(reason: &'static str) -> Self {
        Self::new(CapabilityRejectionKind::GeometryMismatch, reason)
    }

    /// Reject a request outside a bounded backend resource or address limit.
    #[must_use]
    pub const fn resource_limit(reason: &'static str) -> Self {
        Self::new(CapabilityRejectionKind::ResourceLimit, reason)
    }

    /// Reject a request bound to a different device, context, or session.
    #[must_use]
    pub const fn context_mismatch(reason: &'static str) -> Self {
        Self::new(CapabilityRejectionKind::ContextMismatch, reason)
    }

    /// Reject checked internal ownership or execution state.
    #[must_use]
    pub const fn contract_violation(reason: &'static str) -> Self {
        Self::new(CapabilityRejectionKind::ContractViolation, reason)
    }

    /// Typed rejection category.
    #[must_use]
    pub const fn kind(self) -> CapabilityRejectionKind {
        self.kind
    }

    /// Stable diagnostic rendered by the public adapter error.
    #[must_use]
    pub const fn reason(self) -> &'static str {
        self.reason
    }
}

impl fmt::Display for CapabilityRejection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.reason)
    }
}
