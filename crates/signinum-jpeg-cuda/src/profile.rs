// SPDX-License-Identifier: Apache-2.0

use core::fmt::Write;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::OnceLock;

const GPU_ROUTE_PROFILE_ENV: &str = "SIGNINUM_GPU_ROUTE_PROFILE";
const SUMMARY_LABEL_FIELDS: &[(&str, &str)] = &[
    ("op", "route_op"),
    ("request", "request"),
    ("fmt", "fmt"),
    ("decision", "decision"),
    ("reason", "reason"),
    ("has_fast_packet", "has_fast_packet"),
    ("supports_output_format", "supports_output_format"),
    ("hardware_decode", "hardware_decode"),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProfileStageMode {
    Off,
    Rows,
    Summary,
}

pub(crate) fn gpu_route_profile_enabled() -> bool {
    gpu_route_profile_stage_mode() != ProfileStageMode::Off
}

fn gpu_route_profile_stage_mode() -> ProfileStageMode {
    static MODE: OnceLock<ProfileStageMode> = OnceLock::new();
    *MODE.get_or_init(|| {
        profile_stage_mode_from_value(std::env::var(GPU_ROUTE_PROFILE_ENV).ok().as_deref())
    })
}

pub(crate) fn env_flag_from_value(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

pub(crate) fn profile_stage_mode_from_value(value: Option<&str>) -> ProfileStageMode {
    match value.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("summary" | "aggregate") => ProfileStageMode::Summary,
        Some(value) if env_flag_from_value(value) => ProfileStageMode::Rows,
        _ => ProfileStageMode::Off,
    }
}

pub(crate) fn format_profile_row(
    codec: &str,
    op: &str,
    path: &str,
    fields: &[(&str, &str)],
) -> String {
    let mut row = String::from("signinum_profile");
    let _ = write!(row, " codec={codec} op={op} path={path}");
    for (key, value) in fields {
        let _ = write!(row, " {key}={value}");
    }
    row
}

pub(crate) fn emit_gpu_route_profile(codec: &str, op: &str, path: &str, fields: &[(&str, &str)]) {
    match gpu_route_profile_stage_mode() {
        ProfileStageMode::Rows => eprintln!("{}", format_profile_row(codec, op, path, fields)),
        ProfileStageMode::Summary => {
            PROFILE_SUMMARY.with(|summary| {
                summary.borrow_mut().record_str(codec, op, path, fields);
            });
        }
        ProfileStageMode::Off => {}
    }
}

thread_local! {
    static PROFILE_SUMMARY: RefCell<ProfileSummary> = RefCell::new(ProfileSummary::default());
}

#[derive(Default)]
pub(crate) struct ProfileSummary {
    rows: BTreeMap<SummaryKey, SummaryRow>,
}

impl ProfileSummary {
    pub(crate) fn record_str(
        &mut self,
        codec: &str,
        op: &str,
        path: &str,
        fields: &[(&str, &str)],
    ) {
        let key = SummaryKey::new(codec, op, path, fields);
        self.rows.entry(key).or_default().count += 1;
    }

    pub(crate) fn format_rows(&self) -> Vec<String> {
        self.rows
            .iter()
            .map(|(key, row)| key.format_summary_row(row))
            .collect()
    }
}

impl Drop for ProfileSummary {
    fn drop(&mut self) {
        for row in self.format_rows() {
            eprintln!("{row}");
        }
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct SummaryKey {
    codec: String,
    op: String,
    path: String,
    labels: Vec<(String, String)>,
}

impl SummaryKey {
    fn new(codec: &str, op: &str, path: &str, fields: &[(&str, &str)]) -> Self {
        let labels = SUMMARY_LABEL_FIELDS
            .iter()
            .filter_map(|(field_name, output_name)| {
                fields
                    .iter()
                    .find(|(field, _)| field == field_name)
                    .map(|(_, value)| ((*output_name).to_owned(), (*value).to_owned()))
            })
            .collect();
        Self {
            codec: codec.to_owned(),
            op: op.to_owned(),
            path: path.to_owned(),
            labels,
        }
    }

    fn format_summary_row(&self, row: &SummaryRow) -> String {
        let mut output = String::from("signinum_profile_summary");
        let _ = write!(
            output,
            " codec={} op={} path={}",
            self.codec, self.op, self.path
        );
        for (label, value) in &self.labels {
            let _ = write!(output, " {label}={value}");
        }
        let _ = write!(output, " count={}", row.count);
        output
    }
}

#[derive(Default)]
struct SummaryRow {
    count: u128,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_flag_accepts_common_truthy_values() {
        for value in ["1", "true", "TRUE", "yes", "on", "On"] {
            assert!(
                env_flag_from_value(value),
                "{value} should enable profiling"
            );
        }
    }

    #[test]
    fn profile_row_uses_compact_key_value_format() {
        let row = format_profile_row(
            "jpeg",
            "gpu_route",
            "cuda",
            &[("request", "Cuda"), ("decision", "nvjpeg")],
        );
        assert_eq!(
            row,
            "signinum_profile codec=jpeg op=gpu_route path=cuda request=Cuda decision=nvjpeg"
        );
    }

    #[test]
    fn profile_stage_mode_parses_summary_mode() {
        assert_eq!(
            profile_stage_mode_from_value(Some("summary")),
            ProfileStageMode::Summary
        );
        assert_eq!(
            profile_stage_mode_from_value(Some("aggregate")),
            ProfileStageMode::Summary
        );
        assert_eq!(
            profile_stage_mode_from_value(Some("1")),
            ProfileStageMode::Rows
        );
        assert_eq!(
            profile_stage_mode_from_value(Some("0")),
            ProfileStageMode::Off
        );
    }

    #[test]
    fn profile_summary_counts_route_decisions() {
        let mut summary = ProfileSummary::default();
        summary.record_str(
            "jpeg",
            "gpu_route",
            "cuda",
            &[
                ("op", "batch_full"),
                ("request", "AutoOrCuda"),
                ("fmt", "Rgb8"),
                ("tiles", "64"),
                ("decision", "nvjpeg_batch"),
            ],
        );
        summary.record_str(
            "jpeg",
            "gpu_route",
            "cuda",
            &[
                ("op", "batch_full"),
                ("request", "AutoOrCuda"),
                ("fmt", "Rgb8"),
                ("tiles", "128"),
                ("decision", "nvjpeg_batch"),
            ],
        );

        assert_eq!(
            summary.format_rows(),
            vec![
                "signinum_profile_summary codec=jpeg op=gpu_route path=cuda route_op=batch_full request=AutoOrCuda fmt=Rgb8 decision=nvjpeg_batch count=2"
            ]
        );
    }
}
