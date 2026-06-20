// SPDX-License-Identifier: Apache-2.0

use std::cell::RefCell;
use std::sync::OnceLock;

pub(crate) use j2k_profile::duration_us_string;
use j2k_profile::{
    profile_stage_mode_from_env, same_summary_labels, ProfileStageMode, SummaryLabel,
};

#[cfg(test)]
pub(crate) use j2k_profile::ProfileSummary;

const JPEG_PROFILE_STAGES_ENV: &str = "J2K_JPEG_PROFILE_STAGES";
const SUMMARY_LABEL_FIELD_KEYS: &[&str] = &["mode", "fmt", "downscale", "scan_path"];

pub(crate) fn jpeg_profile_stages_enabled() -> bool {
    jpeg_profile_stage_mode() != ProfileStageMode::Disabled
}

fn jpeg_profile_stage_mode() -> ProfileStageMode {
    static MODE: OnceLock<ProfileStageMode> = OnceLock::new();
    *MODE.get_or_init(|| profile_stage_mode_from_env(JPEG_PROFILE_STAGES_ENV))
}

pub(crate) fn emit_jpeg_profile_row(op: &str, path: &str, fields: &[(&str, &str)]) {
    j2k_profile::emit_profile_row_with_timing_summary(
        jpeg_profile_stage_mode(),
        &PROFILE_SUMMARY,
        "jpeg",
        op,
        path,
        fields,
        SUMMARY_LABEL_FIELD_KEYS,
    );
}

thread_local! {
    static PROFILE_SUMMARY: RefCell<j2k_profile::ProfileSummary> = RefCell::new(
       j2k_profile::ProfileSummary::new(summary_label_fields().iter().cloned()).emit_on_drop()
    );
}

fn summary_label_fields() -> &'static [SummaryLabel] {
    static SUMMARY_LABEL_FIELDS: OnceLock<Box<[SummaryLabel]>> = OnceLock::new();
    SUMMARY_LABEL_FIELDS
        .get_or_init(|| same_summary_labels(SUMMARY_LABEL_FIELD_KEYS).into_boxed_slice())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_summary_groups_rows_and_averages_timing_fields() {
        let mut summary = ProfileSummary::new(summary_label_fields().iter().cloned());
        j2k_profile::record_timing_summary_str(
            &mut summary,
            "jpeg",
            "decode",
            "cpu",
            &[
                ("mode", "full"),
                ("fmt", "Rgb8"),
                ("source_width", "16"),
                ("decode_us", "4"),
                ("output_bytes", "48"),
                ("total_us", "6"),
            ],
            SUMMARY_LABEL_FIELD_KEYS,
        );
        j2k_profile::record_timing_summary_str(
            &mut summary,
            "jpeg",
            "decode",
            "cpu",
            &[
                ("mode", "full"),
                ("fmt", "Rgb8"),
                ("source_width", "32"),
                ("decode_us", "8"),
                ("output_bytes", "96"),
                ("total_us", "10"),
            ],
            SUMMARY_LABEL_FIELD_KEYS,
        );

        assert_eq!(
            summary.format_rows(),
            vec![
                "j2k_profile_summary codec=jpeg op=decode path=cpu mode=full fmt=Rgb8 count=2 decode_us_sum=12 decode_us_avg=6 total_us_sum=16 total_us_avg=8"
            ]
        );
    }
}
