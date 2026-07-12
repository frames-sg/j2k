// j2k-coverage: shared-accelerator-host
#[cfg(any(feature = "std", test))]
use alloc::vec::Vec;

#[cfg(any(feature = "std", test))]
use core::fmt;

#[cfg(any(feature = "std", test))]
use crate::allocation::{ensure_limit, try_vec, HeapBudget};
#[cfg(feature = "std")]
use crate::{profile_stage_mode_from_env, ProfileStageMode, ProfileSummary};
#[cfg(any(feature = "std", test))]
use crate::{ProfileField, ProfileLimits, ProfileResult, SummaryLabel};

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
pub(crate) fn gpu_route_summary_labels() -> ProfileResult<Vec<SummaryLabel>> {
    const LABELS: [(&str, &str); 8] = [
        ("op", "route_op"),
        ("request", "request"),
        ("fmt", "fmt"),
        ("decision", "decision"),
        ("reason", "reason"),
        ("has_fast_packet", "has_fast_packet"),
        ("supports_output_format", "supports_output_format"),
        ("hardware_decode", "hardware_decode"),
    ];
    let limits = ProfileLimits::default();
    let mut budget = HeapBudget::new(0, limits.max_retained_bytes());
    let mut labels = try_vec(LABELS.len(), &mut budget, "GPU route summary labels")?;
    for (input_key, summary_key) in LABELS {
        let label = SummaryLabel::new(input_key, summary_key)?;
        budget.include(label.retained_bytes()?, "GPU route summary labels")?;
        labels.push(label);
    }
    Ok(labels)
}

#[cfg(any(feature = "std", test))]
fn gpu_route_decision_fields<R, F>(
    route: (&str, R, F, &str),
    extra_fields: impl IntoIterator<Item = ProfileField>,
) -> ProfileResult<Vec<ProfileField>>
where
    R: fmt::Display,
    F: fmt::Display,
{
    let (op, request, pixel_format, decision) = route;
    let (mut fields, mut budget) = bounded_field_vec()?;
    push_field(&mut fields, &mut budget, ProfileField::label("op", op)?)?;
    push_field(
        &mut fields,
        &mut budget,
        ProfileField::label("request", request)?,
    )?;
    push_field(
        &mut fields,
        &mut budget,
        ProfileField::label("fmt", pixel_format)?,
    )?;
    push_field(
        &mut fields,
        &mut budget,
        ProfileField::label("decision", decision)?,
    )?;
    for field in extra_fields {
        push_field(&mut fields, &mut budget, field)?;
    }
    Ok(fields)
}

#[cfg(any(feature = "std", test))]
fn gpu_route_surface_fields<R, F>(
    route: (&str, R, F, &str),
    dimensions: (u32, u32),
    extra_fields: impl IntoIterator<Item = ProfileField>,
) -> ProfileResult<Vec<ProfileField>>
where
    R: fmt::Display,
    F: fmt::Display,
{
    let (op, request, pixel_format, decision) = route;
    let (mut fields, mut budget) = bounded_field_vec()?;
    push_field(&mut fields, &mut budget, ProfileField::label("op", op)?)?;
    push_field(
        &mut fields,
        &mut budget,
        ProfileField::label("request", request)?,
    )?;
    push_field(
        &mut fields,
        &mut budget,
        ProfileField::label("fmt", pixel_format)?,
    )?;
    push_field(
        &mut fields,
        &mut budget,
        ProfileField::label("width", dimensions.0)?,
    )?;
    push_field(
        &mut fields,
        &mut budget,
        ProfileField::label("height", dimensions.1)?,
    )?;
    push_field(
        &mut fields,
        &mut budget,
        ProfileField::label("decision", decision)?,
    )?;
    for field in extra_fields {
        push_field(&mut fields, &mut budget, field)?;
    }
    Ok(fields)
}

#[cfg(any(feature = "std", test))]
fn bounded_field_vec() -> ProfileResult<(Vec<ProfileField>, HeapBudget)> {
    let limits = ProfileLimits::default();
    let mut budget = HeapBudget::new(0, limits.max_retained_bytes());
    let fields = try_vec(limits.max_fields(), &mut budget, "GPU route profile fields")?;
    Ok((fields, budget))
}

#[cfg(any(feature = "std", test))]
fn push_field(
    fields: &mut Vec<ProfileField>,
    budget: &mut HeapBudget,
    field: ProfileField,
) -> ProfileResult<()> {
    let limits = ProfileLimits::default();
    let count = fields
        .len()
        .checked_add(1)
        .ok_or(crate::ProfileError::SizeOverflow {
            what: "GPU route profile field count",
        })?;
    ensure_limit(count, limits.max_fields(), "GPU route profile field count")?;
    budget.include(field.retained_bytes()?, "GPU route profile fields")?;
    fields.push(field);
    Ok(())
}

#[cfg(feature = "std")]
fn gpu_route_profile_summary() -> ProfileSummary {
    match gpu_route_summary_labels().and_then(ProfileSummary::counts_only) {
        Ok(summary) => summary,
        Err(error) => {
            crate::emit::emit_profile_error("gpu_route_summary_init", &error);
            ProfileSummary::empty_counts_only()
        }
    }
}

#[cfg(feature = "std")]
thread_local! {
    static GPU_ROUTE_PROFILE_SUMMARY: std::cell::RefCell<ProfileSummary> =
        std::cell::RefCell::new(gpu_route_profile_summary().emit_on_drop());
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

    let fields = match gpu_route_decision_fields(route, extra_fields) {
        Ok(fields) => fields,
        Err(error) => {
            crate::emit::emit_profile_error("gpu_route_decision_fields", &error);
            return;
        }
    };
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

    let fields = match gpu_route_surface_fields(route, dimensions, extra_fields) {
        Ok(fields) => fields,
        Err(error) => {
            crate::emit::emit_profile_error("gpu_route_surface_fields", &error);
            return;
        }
    };
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
    use crate::format::format_profile_fields;

    #[test]
    fn surface_route_fields_preserve_compat_order() {
        let fields = gpu_route_surface_fields(
            ("wrap_surface", "Cuda", "Rgb8", "cuda_upload"),
            (16, 8),
            [ProfileField::metric("kernel_dispatches", 2_u32).expect("valid metric field")],
        )
        .expect("valid surface fields");

        assert_eq!(
            format_profile_fields("j2k", "gpu_route", "cuda", &fields)
                .expect("route row should format"),
            "j2k_profile codec=j2k op=gpu_route path=cuda op=wrap_surface request=Cuda fmt=Rgb8 width=16 height=8 decision=cuda_upload kernel_dispatches=2",
        );
    }

    #[test]
    fn decision_route_fields_preserve_compat_order() {
        let fields = gpu_route_decision_fields(
            ("full", "Cuda", "Rgb8", "owned_cuda_unavailable"),
            [
                ProfileField::label("reason", "cuda_runtime_feature_disabled")
                    .expect("valid reason field"),
            ],
        )
        .expect("valid decision fields");

        assert_eq!(
            format_profile_fields("jpeg", "gpu_route", "cuda", &fields)
                .expect("route row should format"),
            "j2k_profile codec=jpeg op=gpu_route path=cuda op=full request=Cuda fmt=Rgb8 decision=owned_cuda_unavailable reason=cuda_runtime_feature_disabled",
        );
    }

    #[test]
    fn custom_route_fields_preserve_caller_order() {
        let fields = [
            ProfileField::label("request", "Metal").expect("valid request field"),
            ProfileField::label("fmt", "Rgb8").expect("valid format field"),
            ProfileField::label("op", "full").expect("valid operation field"),
            ProfileField::label("decision", "metal_kernel").expect("valid decision field"),
            ProfileField::label("reason", "none").expect("valid reason field"),
        ];

        assert_eq!(
            format_profile_fields("j2k", "gpu_route", "metal", &fields)
                .expect("route row should format"),
            "j2k_profile codec=j2k op=gpu_route path=metal request=Metal fmt=Rgb8 op=full decision=metal_kernel reason=none",
        );
    }
}
