// SPDX-License-Identifier: MIT OR Apache-2.0

use std::cell::RefCell;

pub(crate) use j2k_profile::ProfileField;
use j2k_profile::{same_summary_labels, ProfileStageMode, StageModeCache};

#[cfg(test)]
pub(crate) use j2k_profile::ProfileSummary;

const JPEG_PROFILE_STAGES_ENV: &str = "J2K_JPEG_PROFILE_STAGES";
const SUMMARY_LABEL_FIELD_KEYS: &[&str] = &["mode", "fmt", "downscale", "scan_path"];

pub(crate) fn jpeg_profile_stages_enabled() -> bool {
    jpeg_profile_stage_mode() != ProfileStageMode::Disabled
}

fn jpeg_profile_stage_mode() -> ProfileStageMode {
    static MODE: StageModeCache = StageModeCache::new();
    MODE.mode_from_env(JPEG_PROFILE_STAGES_ENV)
}

pub(crate) fn emit_jpeg_profile_fields<const N: usize>(
    operation: &'static str,
    op: &str,
    path: &str,
    build: impl FnOnce() -> j2k_profile::ProfileResult<[ProfileField; N]>,
) {
    let mode = jpeg_profile_stage_mode();
    if mode == ProfileStageMode::Disabled {
        return;
    }
    match build() {
        Ok(fields) => {
            j2k_profile::emit_profile_fields(mode, &PROFILE_SUMMARY, "jpeg", op, path, &fields);
        }
        Err(error) => j2k_profile::emit_profile_error(operation, &error),
    }
}

thread_local! {
    static PROFILE_SUMMARY: RefCell<j2k_profile::ProfileSummary> =
        RefCell::new(new_profile_summary().emit_on_drop());
}

fn new_profile_summary() -> j2k_profile::ProfileSummary {
    match same_summary_labels(SUMMARY_LABEL_FIELD_KEYS).and_then(j2k_profile::ProfileSummary::new) {
        Ok(summary) => summary,
        Err(error) => {
            j2k_profile::emit_profile_error("jpeg_summary_init", &error);
            j2k_profile::ProfileSummary::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_summary_groups_rows_and_averages_timing_fields() {
        thread_local! {
            static TEST_SUMMARY: RefCell<ProfileSummary> =
                RefCell::new(new_profile_summary());
        }

        for (source_width, decode_us, output_bytes, total_us) in
            [(16_u32, 4_u128, 48_usize, 6_u128), (32, 8, 96, 10)]
        {
            let fields = [
                ProfileField::label("mode", "full").expect("mode field"),
                ProfileField::label("fmt", "Rgb8").expect("format field"),
                ProfileField::metric_with_summary("source_width", source_width, false)
                    .expect("source width field"),
                ProfileField::metric("decode_us", decode_us).expect("decode field"),
                ProfileField::metric_with_summary("output_bytes", output_bytes, false)
                    .expect("output field"),
                ProfileField::metric("total_us", total_us).expect("total field"),
            ];
            j2k_profile::emit_profile_fields(
                ProfileStageMode::Summary,
                &TEST_SUMMARY,
                "jpeg",
                "decode",
                "cpu",
                &fields,
            );
        }

        TEST_SUMMARY.with(|summary| {
            assert_eq!(
                summary.borrow().format_rows().expect("summary should format"),
                vec![
                    "j2k_profile_summary codec=jpeg op=decode path=cpu mode=full fmt=Rgb8 count=2 decode_us_sum=12 decode_us_avg=6 total_us_sum=16 total_us_avg=8"
                ]
            );
        });
    }
}
