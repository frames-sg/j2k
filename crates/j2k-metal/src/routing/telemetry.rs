// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{BackendRequest, PixelFormat};
use j2k_metal_support::{
    cpu_host_route, reject_explicit_metal_route, reject_unsupported_backend_route,
    MetalRouteProfileLabels,
};

use super::decision::RouteDecision;

pub(super) fn observe(backend: BackendRequest, fmt: PixelFormat, decision: RouteDecision) {
    if !j2k_profile::gpu_route_profile_enabled() {
        return;
    }
    let labels = labels(decision);
    match fields(backend, fmt, labels) {
        Ok(fields) => j2k_profile::emit_gpu_route_fields("j2k", "metal", &fields),
        Err(error) => j2k_profile::emit_profile_error("metal_gpu_route_fields", &error),
    }
}

fn fields(
    backend: BackendRequest,
    fmt: PixelFormat,
    labels: MetalRouteProfileLabels,
) -> j2k_profile::ProfileResult<[j2k_profile::ProfileField; 5]> {
    Ok([
        j2k_profile::ProfileField::label("request", format_args!("{backend:?}"))?,
        j2k_profile::ProfileField::label("fmt", format_args!("{fmt:?}"))?,
        j2k_profile::ProfileField::label("op", "full")?,
        j2k_profile::ProfileField::label("decision", labels.decision)?,
        j2k_profile::ProfileField::label("reason", labels.reason)?,
    ])
}

fn labels(decision: RouteDecision) -> MetalRouteProfileLabels {
    match decision {
        RouteDecision::CpuHost => cpu_host_route(),
        #[cfg(target_os = "macos")]
        RouteDecision::MetalKernel => j2k_metal_support::metal_kernel_route(),
        RouteDecision::RejectExplicitMetal { reason } => {
            reject_explicit_metal_route(reason.profile_reason())
        }
        RouteDecision::RejectUnsupportedBackend { .. } => reject_unsupported_backend_route(),
        #[cfg(not(target_os = "macos"))]
        RouteDecision::MetalUnavailable => j2k_metal_support::metal_unavailable_route(),
    }
}

#[cfg(test)]
mod tests {
    use j2k_core::{BackendRequest, PixelFormat};

    use super::{fields, labels};
    use crate::routing::{rejection::unsupported_metal_format, RouteDecision};

    #[test]
    fn route_fields_preserve_the_stable_profile_schema() {
        let fields = fields(
            BackendRequest::Metal,
            PixelFormat::Rgb8,
            labels(RouteDecision::CpuHost),
        )
        .expect("bounded route fields");

        assert_eq!(
            fields.map(|field| (field.key().to_string(), field.into_value())),
            [
                ("request".to_string(), "Metal".to_string()),
                ("fmt".to_string(), "Rgb8".to_string()),
                ("op".to_string(), "full".to_string()),
                ("decision".to_string(), "cpu_host".to_string()),
                ("reason".to_string(), "none".to_string()),
            ]
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn route_labels_cover_every_macos_decision() {
        for (decision, expected) in [
            (RouteDecision::CpuHost, ("cpu_host", "none")),
            (RouteDecision::MetalKernel, ("metal_kernel", "none")),
            (
                RouteDecision::RejectExplicitMetal {
                    reason: unsupported_metal_format(PixelFormat::Rgba16),
                },
                ("reject_explicit_metal", "unsupported_format"),
            ),
            (
                RouteDecision::RejectUnsupportedBackend {
                    request: BackendRequest::Cuda,
                },
                ("reject_unsupported_backend", "unsupported_backend"),
            ),
        ] {
            let actual = labels(decision);
            assert_eq!((actual.decision, actual.reason), expected);
        }
    }
}
