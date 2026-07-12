// SPDX-License-Identifier: MIT OR Apache-2.0

/// Stable profile labels for a Metal backend route decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct MetalRouteProfileLabels {
    /// Route decision label emitted in GPU route profiles.
    pub decision: &'static str,
    /// Route reason label emitted in GPU route profiles.
    pub reason: &'static str,
}

impl MetalRouteProfileLabels {
    /// Construct route profile labels from stable string values.
    #[must_use]
    pub const fn new(decision: &'static str, reason: &'static str) -> Self {
        Self { decision, reason }
    }
}

/// Route profile labels for CPU host execution.
#[must_use]
pub const fn cpu_host_route() -> MetalRouteProfileLabels {
    MetalRouteProfileLabels::new("cpu_host", "none")
}

/// Route profile labels for Metal kernel execution.
#[must_use]
pub const fn metal_kernel_route() -> MetalRouteProfileLabels {
    MetalRouteProfileLabels::new("metal_kernel", "none")
}

/// Route profile labels for an explicit Metal request rejected by codec policy.
#[must_use]
pub const fn reject_explicit_metal_route(reason: &'static str) -> MetalRouteProfileLabels {
    MetalRouteProfileLabels::new("reject_explicit_metal", reason)
}

/// Route profile labels for backend requests unsupported by the Metal adapter.
#[must_use]
pub const fn reject_unsupported_backend_route() -> MetalRouteProfileLabels {
    MetalRouteProfileLabels::new("reject_unsupported_backend", "unsupported_backend")
}

/// Route profile labels for hosts without an available Metal runtime.
#[must_use]
pub const fn metal_unavailable_route() -> MetalRouteProfileLabels {
    MetalRouteProfileLabels::new("metal_unavailable", "metal_unavailable")
}
