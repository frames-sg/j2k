// SPDX-License-Identifier: MIT OR Apache-2.0

use std::cell::RefCell;
use std::fmt;

use j2k_core::{BackendRequest, PixelFormat};
use j2k_profile::{same_summary_labels, ProfileField, ProfileResult};

thread_local! {
    static METAL_BATCH_PROFILE_SUMMARY: RefCell<j2k_profile::ProfileSummary> =
        RefCell::new(new_metal_batch_profile_summary().emit_on_drop());
}

fn new_metal_batch_profile_summary() -> j2k_profile::ProfileSummary {
    match same_summary_labels(&[
        "slice",
        "stage",
        "pipeline",
        "processor",
        "metric_kind",
        "aggregation",
        "route",
        "backend",
        "fmt",
        "outcome",
    ])
    .and_then(j2k_profile::ProfileSummary::new)
    {
        Ok(summary) => summary,
        Err(error) => {
            j2k_profile::emit_profile_error("metal_batch_summary_init", &error);
            j2k_profile::ProfileSummary::default()
        }
    }
}

pub(crate) use j2k_profile::{elapsed_us, profile_now};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MetalBatchProfileValue<T> {
    None,
    Mixed,
    Uniform(T),
}

impl<T: fmt::Debug> fmt::Display for MetalBatchProfileValue<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => formatter.write_str("none"),
            Self::Mixed => formatter.write_str("mixed"),
            Self::Uniform(value) => write!(formatter, "{value:?}"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MetalBatchProfileRow<'a> {
    pub(crate) slice: &'a str,
    pub(crate) stage: &'a str,
    pub(crate) pipeline: &'a str,
    pub(crate) processor: &'a str,
    pub(crate) route: &'a str,
    pub(crate) backend: MetalBatchProfileValue<BackendRequest>,
    pub(crate) fmt: MetalBatchProfileValue<PixelFormat>,
    pub(crate) request_count: usize,
    pub(crate) output_count: usize,
    pub(crate) elapsed_us: u128,
    pub(crate) outcome: &'a str,
}

#[cfg(target_os = "macos")]
pub(crate) fn metal_profile_stage_mode() -> j2k_profile::ProfileStageMode {
    crate::profile_env::metal_profile_stage_mode()
}

#[cfg(not(target_os = "macos"))]
pub(crate) const fn metal_profile_stage_mode() -> j2k_profile::ProfileStageMode {
    j2k_profile::ProfileStageMode::Disabled
}

pub(crate) fn metal_profile_stages_enabled() -> bool {
    metal_profile_stage_mode() != j2k_profile::ProfileStageMode::Disabled
}

pub(crate) fn emit_metal_batch_profile_row(path: &str, row: &MetalBatchProfileRow<'_>) {
    let fields = match format_metal_batch_profile_fields(row) {
        Ok(fields) => fields,
        Err(error) => {
            j2k_profile::emit_profile_error("metal_batch_fields", &error);
            return;
        }
    };
    j2k_profile::emit_profile_fields(
        metal_profile_stage_mode(),
        &METAL_BATCH_PROFILE_SUMMARY,
        "j2k",
        "metal_batch",
        path,
        fields.as_slice(),
    );
}

pub(crate) fn format_metal_batch_profile_fields(
    row: &MetalBatchProfileRow<'_>,
) -> ProfileResult<[ProfileField; 14]> {
    Ok([
        ProfileField::label("slice", row.slice)?,
        ProfileField::label("stage", row.stage)?,
        ProfileField::label("pipeline", row.pipeline)?,
        ProfileField::label("processor", row.processor)?,
        ProfileField::label("metric", "wall_us")?,
        ProfileField::label("metric_kind", "wall_elapsed")?,
        ProfileField::label("aggregation", "exclusive")?,
        ProfileField::label("route", row.route)?,
        ProfileField::label("backend", row.backend)?,
        ProfileField::label("fmt", row.fmt)?,
        ProfileField::metric("request_count", row.request_count)?,
        ProfileField::metric("output_count", row.output_count)?,
        ProfileField::metric("elapsed_us", row.elapsed_us)?,
        ProfileField::label("outcome", row.outcome)?,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metal_batch_profile_fields_include_processor_and_timing_context() {
        let fields = format_metal_batch_profile_fields(&MetalBatchProfileRow {
            slice: "decode_batch",
            stage: "execute",
            pipeline: "metal_cpu_hybrid",
            processor: "hybrid",
            route: "auto_repeated_region_scaled_direct_metal",
            backend: MetalBatchProfileValue::Uniform(BackendRequest::Auto),
            fmt: MetalBatchProfileValue::Uniform(PixelFormat::Rgb8),
            request_count: 16,
            output_count: 16,
            elapsed_us: 42,
            outcome: "metal_surface",
        })
        .expect("bounded Metal batch profile fields");

        assert!(
            fields
                .iter()
                .any(|field| field.key() == "pipeline" && field.value() == "metal_cpu_hybrid"),
            "hybrid batch profile rows must identify the Metal/CPU hybrid pipeline"
        );
        assert!(
            fields
                .iter()
                .any(|field| field.key() == "processor" && field.value() == "hybrid"),
            "hybrid batch profile rows must identify whether time is CPU, Metal, transfer, wait, or scheduler work"
        );
        assert!(
            fields
                .iter()
                .any(|field| field.key() == "metric_kind" && field.value() == "wall_elapsed"),
            "hybrid batch profile rows must identify wall-time semantics"
        );
        assert!(
            fields
                .iter()
                .any(|field| field.key() == "aggregation" && field.value() == "exclusive"),
            "hybrid batch profile rows must identify whether elapsed time is exclusive or aggregated"
        );
        assert_eq!(
            fields
                .iter()
                .map(|field| (field.key(), field.value()))
                .collect::<Vec<_>>(),
            vec![
                ("slice", "decode_batch"),
                ("stage", "execute"),
                ("pipeline", "metal_cpu_hybrid"),
                ("processor", "hybrid"),
                ("metric", "wall_us"),
                ("metric_kind", "wall_elapsed"),
                ("aggregation", "exclusive"),
                ("route", "auto_repeated_region_scaled_direct_metal"),
                ("backend", "Auto"),
                ("fmt", "Rgb8"),
                ("request_count", "16"),
                ("output_count", "16"),
                ("elapsed_us", "42"),
                ("outcome", "metal_surface"),
            ]
        );
    }

    #[test]
    fn metal_batch_profile_fields_reject_oversized_owned_values() {
        let oversized = "x".repeat(j2k_profile::ProfileLimits::default().max_token_bytes() + 1);
        let error = format_metal_batch_profile_fields(&MetalBatchProfileRow {
            slice: &oversized,
            stage: "execute",
            pipeline: "metal_cpu_hybrid",
            processor: "hybrid",
            route: "auto",
            backend: MetalBatchProfileValue::Uniform(BackendRequest::Auto),
            fmt: MetalBatchProfileValue::Uniform(PixelFormat::Rgb8),
            request_count: 1,
            output_count: 1,
            elapsed_us: 1,
            outcome: "metal_surface",
        })
        .expect_err("oversized profile token must fail before emission");

        assert!(matches!(
            error,
            j2k_profile::ProfileError::LimitExceeded {
                what: "field value",
                ..
            }
        ));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_batch_profile_uses_shared_summary_stage_mode() {
        let _guard = crate::profile_env::force_metal_profile_stage_mode_for_test(
            j2k_profile::ProfileStageMode::Summary,
        );
        METAL_BATCH_PROFILE_SUMMARY.with(|summary| {
            summary
                .borrow_mut()
                .take_formatted_rows()
                .expect("clear Metal batch summary rows");
        });

        emit_metal_batch_profile_row(
            "decode",
            &MetalBatchProfileRow {
                slice: "decode_batch",
                stage: "execute",
                pipeline: "metal_cpu_hybrid",
                processor: "hybrid",
                route: "auto",
                backend: MetalBatchProfileValue::Uniform(BackendRequest::Auto),
                fmt: MetalBatchProfileValue::Uniform(PixelFormat::Rgb8),
                request_count: 2,
                output_count: 2,
                elapsed_us: 42,
                outcome: "metal_surface",
            },
        );

        let rows = METAL_BATCH_PROFILE_SUMMARY.with(|summary| {
            summary
                .borrow_mut()
                .take_formatted_rows()
                .expect("format Metal batch summary rows")
        });
        assert_eq!(
            rows,
            vec![
                "j2k_profile_summary codec=j2k op=metal_batch path=decode slice=decode_batch stage=execute pipeline=metal_cpu_hybrid processor=hybrid metric_kind=wall_elapsed aggregation=exclusive route=auto backend=Auto fmt=Rgb8 outcome=metal_surface count=1 elapsed_us_sum=42 elapsed_us_avg=42 output_count_sum=2 request_count_sum=2"
            ]
        );
    }
}
