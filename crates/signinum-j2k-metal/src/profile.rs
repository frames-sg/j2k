// SPDX-License-Identifier: Apache-2.0

use std::cell::RefCell;
use std::sync::OnceLock;
use std::time::Instant;

use signinum_profile::{profile_stage_mode_from_env, same_summary_labels, ProfileStageMode};

const METAL_PROFILE_STAGES_ENV: &str = "SIGNINUM_J2K_METAL_PROFILE_STAGES";

thread_local! {
    static METAL_BATCH_PROFILE_SUMMARY: RefCell<signinum_profile::ProfileSummary> =
        RefCell::new(signinum_profile::ProfileSummary::new(same_summary_labels(&[
            "slice", "stage", "route", "backend", "fmt", "outcome",
        ])));
}

pub(crate) type ProfileInstant = Instant;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MetalBatchProfileRow<'a> {
    pub(crate) slice: &'a str,
    pub(crate) stage: &'a str,
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

pub(crate) fn gpu_route_profile_enabled() -> bool {
    signinum_profile::gpu_route_profile_enabled()
}

pub(crate) fn emit_gpu_route_profile<K, V>(codec: &str, op: &str, path: &str, fields: &[(K, V)])
where
    K: AsRef<str>,
    V: AsRef<str>,
{
    debug_assert_eq!(op, "gpu_route");
    signinum_profile::emit_gpu_route_profile(codec, path, fields);
}

pub(crate) fn emit_metal_batch_profile_row(path: &str, row: &MetalBatchProfileRow<'_>) {
    let fields = format_metal_batch_profile_fields(row);
    signinum_profile::emit_profile_row(
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
    fn metal_batch_profile_fields_include_route_and_timing_context() {
        let fields = format_metal_batch_profile_fields(&MetalBatchProfileRow {
            slice: "decode_batch",
            stage: "execute",
            route: "auto_repeated_region_scaled_direct_metal",
            backend: "Auto",
            fmt: "Rgb8",
            request_count: 16,
            output_count: 16,
            elapsed_us: 42,
            outcome: "metal",
        });

        assert_eq!(
            fields,
            vec![
                ("slice".to_string(), "decode_batch".to_string()),
                ("stage".to_string(), "execute".to_string()),
                (
                    "route".to_string(),
                    "auto_repeated_region_scaled_direct_metal".to_string()
                ),
                ("backend".to_string(), "Auto".to_string()),
                ("fmt".to_string(), "Rgb8".to_string()),
                ("request_count".to_string(), "16".to_string()),
                ("output_count".to_string(), "16".to_string()),
                ("elapsed_us".to_string(), "42".to_string()),
                ("outcome".to_string(), "metal".to_string()),
            ]
        );
    }
}
