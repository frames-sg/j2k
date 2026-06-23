// SPDX-License-Identifier: MIT OR Apache-2.0

use std::cell::RefCell;
use std::sync::OnceLock;
use std::time::Instant;

use j2k_profile::{profile_stage_mode_from_env, same_summary_labels, ProfileStageMode};

const METAL_PROFILE_STAGES_ENV: &str = "J2K_METAL_PROFILE_STAGES";

thread_local! {
    static METAL_BATCH_PROFILE_SUMMARY: RefCell<j2k_profile::ProfileSummary> =
        RefCell::new(j2k_profile::ProfileSummary::new(same_summary_labels(&[
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
        ])).emit_on_drop());
}

pub(crate) type ProfileInstant = Instant;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MetalBatchProfileRow<'a> {
    pub(crate) slice: &'a str,
    pub(crate) stage: &'a str,
    pub(crate) pipeline: &'a str,
    pub(crate) processor: &'a str,
    pub(crate) route: &'a str,
    pub(crate) backend: &'a str,
    pub(crate) fmt: &'a str,
    pub(crate) request_count: usize,
    pub(crate) output_count: usize,
    pub(crate) elapsed_us: u128,
    pub(crate) outcome: &'a str,
}

pub(crate) fn metal_profile_stage_mode() -> ProfileStageMode {
    static MODE: OnceLock<ProfileStageMode> = OnceLock::new();
    *MODE.get_or_init(|| profile_stage_mode_from_env(METAL_PROFILE_STAGES_ENV))
}

pub(crate) fn metal_profile_stages_enabled() -> bool {
    metal_profile_stage_mode() != ProfileStageMode::Disabled
}

pub(crate) fn profile_now(enabled: bool) -> Option<ProfileInstant> {
    enabled.then(Instant::now)
}

pub(crate) fn elapsed_us(start: Option<ProfileInstant>) -> u128 {
    start.map_or(0, |start| start.elapsed().as_micros())
}

pub(crate) fn emit_metal_batch_profile_row(path: &str, row: &MetalBatchProfileRow<'_>) {
    let fields = format_metal_batch_profile_fields(row);
    j2k_profile::emit_profile_row(
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
) -> Vec<(String, String)> {
    vec![
        ("slice".to_string(), row.slice.to_string()),
        ("stage".to_string(), row.stage.to_string()),
        ("pipeline".to_string(), row.pipeline.to_string()),
        ("processor".to_string(), row.processor.to_string()),
        ("metric".to_string(), "wall_us".to_string()),
        ("metric_kind".to_string(), "wall_elapsed".to_string()),
        ("aggregation".to_string(), "exclusive".to_string()),
        ("route".to_string(), row.route.to_string()),
        ("backend".to_string(), row.backend.to_string()),
        ("fmt".to_string(), row.fmt.to_string()),
        ("request_count".to_string(), row.request_count.to_string()),
        ("output_count".to_string(), row.output_count.to_string()),
        ("elapsed_us".to_string(), row.elapsed_us.to_string()),
        ("outcome".to_string(), row.outcome.to_string()),
    ]
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
            backend: "Auto",
            fmt: "Rgb8",
            request_count: 16,
            output_count: 16,
            elapsed_us: 42,
            outcome: "metal_surface",
        });

        assert!(
            fields
                .iter()
                .any(|(name, value)| name == "pipeline" && value == "metal_cpu_hybrid"),
            "hybrid batch profile rows must identify the Metal/CPU hybrid pipeline"
        );
        assert!(
            fields
                .iter()
                .any(|(name, value)| name == "processor" && value == "hybrid"),
            "hybrid batch profile rows must identify whether time is CPU, Metal, transfer, wait, or scheduler work"
        );
        assert!(
            fields
                .iter()
                .any(|(name, value)| name == "metric_kind" && value == "wall_elapsed"),
            "hybrid batch profile rows must identify wall-time semantics"
        );
        assert!(
            fields
                .iter()
                .any(|(name, value)| name == "aggregation" && value == "exclusive"),
            "hybrid batch profile rows must identify whether elapsed time is exclusive or aggregated"
        );
        assert_eq!(
            fields,
            vec![
                ("slice".to_string(), "decode_batch".to_string()),
                ("stage".to_string(), "execute".to_string()),
                ("pipeline".to_string(), "metal_cpu_hybrid".to_string()),
                ("processor".to_string(), "hybrid".to_string()),
                ("metric".to_string(), "wall_us".to_string()),
                ("metric_kind".to_string(), "wall_elapsed".to_string()),
                ("aggregation".to_string(), "exclusive".to_string()),
                (
                    "route".to_string(),
                    "auto_repeated_region_scaled_direct_metal".to_string()
                ),
                ("backend".to_string(), "Auto".to_string()),
                ("fmt".to_string(), "Rgb8".to_string()),
                ("request_count".to_string(), "16".to_string()),
                ("output_count".to_string(), "16".to_string()),
                ("elapsed_us".to_string(), "42".to_string()),
                ("outcome".to_string(), "metal_surface".to_string()),
            ]
        );
    }
}
