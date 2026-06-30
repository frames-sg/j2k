#[cfg(any(feature = "std", test))]
use alloc::{vec, vec::Vec};

#[cfg(any(feature = "std", test))]
use core::fmt;

#[cfg(feature = "std")]
use crate::{profile_stage_mode_from_env, ProfileStageMode, ProfileSummary};
#[cfg(any(feature = "std", test))]
use crate::{ProfileField, SummaryLabel};

/// Environment variable that controls GPU route profiling rows or summaries.
#[cfg(feature = "std")]
const GPU_ROUTE_PROFILE_ENV: &str = "J2K_GPU_ROUTE_PROFILE";

#[cfg(feature = "std")]
/// Returns the cached GPU route profiling mode from the process environment.
pub(crate) fn gpu_route_profile_stage_mode() -> ProfileStageMode {
    static MODE: std::sync::OnceLock<ProfileStageMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| profile_stage_mode_from_env(GPU_ROUTE_PROFILE_ENV))
}

#[cfg(feature = "std")]
/// Returns whether GPU route profiling is enabled for this process.
pub fn gpu_route_profile_enabled() -> bool {
    gpu_route_profile_stage_mode() != ProfileStageMode::Disabled
}

#[cfg(any(feature = "std", test))]
pub(crate) fn gpu_route_summary_labels() -> Vec<SummaryLabel> {
    vec![
        SummaryLabel::new("op", "route_op"),
        SummaryLabel::same("request"),
        SummaryLabel::same("fmt"),
        SummaryLabel::same("decision"),
        SummaryLabel::same("reason"),
        SummaryLabel::same("has_fast_packet"),
        SummaryLabel::same("supports_output_format"),
        SummaryLabel::same("hardware_decode"),
    ]
}

#[cfg(any(feature = "std", test))]
fn gpu_route_decision_fields<R, F>(
    route: (&str, R, F, &str),
    extra_fields: impl IntoIterator<Item = ProfileField>,
) -> Vec<ProfileField>
where
    R: fmt::Display,
    F: fmt::Display,
{
    let (op, request, pixel_format, decision) = route;
    let mut fields = vec![
        ProfileField::label("op", op),
        ProfileField::label("request", request),
        ProfileField::label("fmt", pixel_format),
        ProfileField::label("decision", decision),
    ];
    fields.extend(extra_fields);
    fields
}

#[cfg(any(feature = "std", test))]
fn gpu_route_surface_fields<R, F>(
    route: (&str, R, F, &str),
    dimensions: (u32, u32),
    extra_fields: impl IntoIterator<Item = ProfileField>,
) -> Vec<ProfileField>
where
    R: fmt::Display,
    F: fmt::Display,
{
    let (op, request, pixel_format, decision) = route;
    let mut fields = vec![
        ProfileField::label("op", op),
        ProfileField::label("request", request),
        ProfileField::label("fmt", pixel_format),
        ProfileField::label("width", dimensions.0),
        ProfileField::label("height", dimensions.1),
        ProfileField::label("decision", decision),
    ];
    fields.extend(extra_fields);
    fields
}

#[cfg(feature = "std")]
fn gpu_route_profile_summary() -> ProfileSummary {
    ProfileSummary::counts_only(gpu_route_summary_labels())
}

#[cfg(feature = "std")]
thread_local! {
    static GPU_ROUTE_PROFILE_SUMMARY: std::cell::RefCell<ProfileSummary> =
        std::cell::RefCell::new(gpu_route_profile_summary().emit_on_drop());
}

#[cfg(feature = "std")]
/// Emits or records a GPU route profile row.
pub fn emit_gpu_route_profile<K, V>(codec: &str, path: &str, fields: &[(K, V)])
where
    K: AsRef<str>,
    V: AsRef<str>,
{
    crate::emit_profile_row(
        gpu_route_profile_stage_mode(),
        &GPU_ROUTE_PROFILE_SUMMARY,
        codec,
        "gpu_route",
        path,
        fields,
    );
}

#[cfg(feature = "std")]
/// Emits or records a typed GPU route profile row with caller-defined field order.
pub fn emit_gpu_route_fields(codec: &str, path: &str, fields: &[ProfileField]) {
    crate::emit_profile_fields(
        gpu_route_profile_stage_mode(),
        &GPU_ROUTE_PROFILE_SUMMARY,
        codec,
        "gpu_route",
        path,
        fields,
    );
}

#[cfg(feature = "std")]
/// Emits or records a typed GPU route decision row.
pub fn emit_gpu_route_decision_profile<R, F>(
    codec_path: (&str, &str),
    route: (&str, R, F, &str),
    extra_fields: impl IntoIterator<Item = ProfileField>,
) where
    R: fmt::Display,
    F: fmt::Display,
{
    if !gpu_route_profile_enabled() {
        return;
    }

    let fields = gpu_route_decision_fields(route, extra_fields);
    crate::emit_profile_fields(
        gpu_route_profile_stage_mode(),
        &GPU_ROUTE_PROFILE_SUMMARY,
        codec_path.0,
        "gpu_route",
        codec_path.1,
        &fields,
    );
}

#[cfg(feature = "std")]
/// Emits or records a typed GPU route decision row that includes output dimensions.
pub fn emit_gpu_route_surface_profile<R, F>(
    codec_path: (&str, &str),
    route: (&str, R, F, &str),
    dimensions: (u32, u32),
    extra_fields: impl IntoIterator<Item = ProfileField>,
) where
    R: fmt::Display,
    F: fmt::Display,
{
    if !gpu_route_profile_enabled() {
        return;
    }

    let fields = gpu_route_surface_fields(route, dimensions, extra_fields);
    crate::emit_profile_fields(
        gpu_route_profile_stage_mode(),
        &GPU_ROUTE_PROFILE_SUMMARY,
        codec_path.0,
        "gpu_route",
        codec_path.1,
        &fields,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{format::format_profile_fields, MetricUnit};

    #[test]
    fn surface_route_fields_preserve_compat_order() {
        let fields = gpu_route_surface_fields(
            ("wrap_surface", "Cuda", "Rgb8", "cuda_upload"),
            (16, 8),
            [ProfileField::metric(
                "kernel_dispatches",
                2_u32,
                MetricUnit::Count,
            )],
        );

        assert_eq!(
            format_profile_fields("j2k", "gpu_route", "cuda", &fields),
            "j2k_profile codec=j2k op=gpu_route path=cuda op=wrap_surface request=Cuda fmt=Rgb8 width=16 height=8 decision=cuda_upload kernel_dispatches=2"
        );
    }

    #[test]
    fn decision_route_fields_preserve_compat_order() {
        let fields = gpu_route_decision_fields(
            ("full", "Cuda", "Rgb8", "owned_cuda_unavailable"),
            [ProfileField::label(
                "reason",
                "cuda_runtime_feature_disabled",
            )],
        );

        assert_eq!(
            format_profile_fields("jpeg", "gpu_route", "cuda", &fields),
            "j2k_profile codec=jpeg op=gpu_route path=cuda op=full request=Cuda fmt=Rgb8 decision=owned_cuda_unavailable reason=cuda_runtime_feature_disabled"
        );
    }

    #[test]
    fn custom_route_fields_preserve_caller_order() {
        let fields = [
            ProfileField::label("request", "Metal"),
            ProfileField::label("fmt", "Rgb8"),
            ProfileField::label("op", "full"),
            ProfileField::label("decision", "metal_kernel"),
            ProfileField::label("reason", "none"),
        ];

        assert_eq!(
            format_profile_fields("j2k", "gpu_route", "metal", &fields),
            "j2k_profile codec=j2k op=gpu_route path=metal request=Metal fmt=Rgb8 op=full decision=metal_kernel reason=none"
        );
    }
}
