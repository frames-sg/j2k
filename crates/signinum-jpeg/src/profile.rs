// SPDX-License-Identifier: Apache-2.0

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::OnceLock;
use std::time::Duration;

const JPEG_PROFILE_STAGES_ENV: &str = "SIGNINUM_JPEG_PROFILE_STAGES";
const SUMMARY_LABEL_FIELDS: &[&str] = &["mode", "fmt", "downscale", "scan_path"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProfileStageMode {
    Off,
    Rows,
    Summary,
}

pub(crate) fn jpeg_profile_stages_enabled() -> bool {
    jpeg_profile_stage_mode() != ProfileStageMode::Off
}

fn jpeg_profile_stage_mode() -> ProfileStageMode {
    static MODE: OnceLock<ProfileStageMode> = OnceLock::new();
    *MODE.get_or_init(|| {
        profile_stage_mode_from_value(std::env::var(JPEG_PROFILE_STAGES_ENV).ok().as_deref())
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

pub(crate) fn emit_jpeg_profile_row(op: &str, path: &str, fields: &[(&str, &str)]) {
    match jpeg_profile_stage_mode() {
        ProfileStageMode::Rows => eprintln!("{}", format_profile_row("jpeg", op, path, fields)),
        ProfileStageMode::Summary => {
            PROFILE_SUMMARY.with(|summary| {
                summary.borrow_mut().record_str("jpeg", op, path, fields);
            });
        }
        ProfileStageMode::Off => {}
    }
}

pub(crate) fn duration_us_string(duration: Duration) -> String {
    duration.as_micros().to_string()
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
        let row = self.rows.entry(key).or_default();
        row.count += 1;
        for (field, value) in fields {
            if is_timing_field(field) {
                if let Ok(value) = value.parse::<u128>() {
                    *row.metrics.entry((*field).to_owned()).or_default() += value;
                }
            }
        }
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
            .filter_map(|label| {
                fields
                    .iter()
                    .find(|(field, _)| field == label)
                    .map(|(_, value)| ((*label).to_owned(), (*value).to_owned()))
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
        for (metric, sum) in &row.metrics {
            let average = if row.count == 0 { 0 } else { sum / row.count };
            let _ = write!(output, " {metric}_sum={sum} {metric}_avg={average}");
        }
        output
    }
}

#[derive(Default)]
struct SummaryRow {
    count: u128,
    metrics: BTreeMap<String, u128>,
}

fn is_timing_field(field: &str) -> bool {
    field.ends_with("_us") || field.ends_with("_ns")
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
    fn env_flag_rejects_empty_false_and_zero_values() {
        for value in ["", "0", "false", "FALSE", "no", "off", "anything-else"] {
            assert!(
                !env_flag_from_value(value),
                "{value} should disable profiling"
            );
        }
    }

    #[test]
    fn profile_row_uses_compact_key_value_format() {
        let fields = [("width", "19"), ("height", "17"), ("total_us", "123")];
        let row = format_profile_row("jpeg", "encode", "cpu", &fields);
        assert_eq!(
            row,
            "signinum_profile codec=jpeg op=encode path=cpu width=19 height=17 total_us=123"
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
    fn profile_summary_groups_rows_and_averages_timing_fields() {
        let mut summary = ProfileSummary::default();
        summary.record_str(
            "jpeg",
            "decode",
            "cpu",
            &[
                ("mode", "full"),
                ("fmt", "Rgb8"),
                ("source_width", "16"),
                ("decode_us", "4"),
                ("total_us", "6"),
            ],
        );
        summary.record_str(
            "jpeg",
            "decode",
            "cpu",
            &[
                ("mode", "full"),
                ("fmt", "Rgb8"),
                ("source_width", "32"),
                ("decode_us", "8"),
                ("total_us", "10"),
            ],
        );

        assert_eq!(
            summary.format_rows(),
            vec![
                "signinum_profile_summary codec=jpeg op=decode path=cpu mode=full fmt=Rgb8 count=2 decode_us_sum=12 decode_us_avg=6 total_us_sum=16 total_us_avg=8"
            ]
        );
    }
}
