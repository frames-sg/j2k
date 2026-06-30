//! Internal profiling helpers shared by the `j2k` workspace crates.

#![doc(hidden)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

#[cfg(feature = "std")]
mod emit;
mod env;
mod field;
mod format;
mod gpu_route;
mod parse;
mod summary;
mod timing;

#[cfg(feature = "std")]
pub use emit::{
    emit_profile_fields, emit_profile_line, emit_profile_row, emit_profile_row_now,
    emit_profile_row_u128, emit_profile_row_with_timing_summary, flush_profile_summary_to,
};
pub use env::profile_stage_mode_from_value;
#[cfg(feature = "std")]
pub use env::{env_flag_from_env, profile_stage_mode_from_env, StageModeCache};
pub use field::{MetricUnit, ProfileField};
pub use format::format_profile_key_value_fields;
#[cfg(feature = "std")]
pub use gpu_route::{
    emit_gpu_route_decision_profile, emit_gpu_route_fields, emit_gpu_route_profile,
    emit_gpu_route_surface_profile, gpu_route_profile_enabled,
};
pub use parse::{
    parse_profile_key_value_fields, parse_profile_line, ParsedProfileFields, ParsedProfileKind,
};
pub use summary::{same_summary_labels, ProfileSummary, SummaryLabel};
#[cfg(feature = "std")]
pub use timing::duration_us_string;
pub use timing::{elapsed_us, profile_now, ProfileInstant};

/// Controls profiling output for a profiling stage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProfileStageMode {
    /// Disable profiling output.
    Disabled,
    /// Emit one row per profiling event.
    Rows,
    /// Aggregate profiling events and emit summary rows.
    Summary,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_route::gpu_route_summary_labels;
    use alloc::string::String;
    use alloc::vec;

    #[cfg(feature = "std")]
    use std::cell::RefCell;

    #[test]
    fn parses_env_truthy_and_falsy_values() {
        for value in [
            Some("1"),
            Some("true"),
            Some("TRUE"),
            Some("yes"),
            Some("on"),
        ] {
            assert!(crate::env::env_flag_from_value(value));
        }

        for value in [
            None,
            Some(""),
            Some("0"),
            Some("false"),
            Some("FALSE"),
            Some("no"),
            Some("off"),
        ] {
            assert!(!crate::env::env_flag_from_value(value));
        }
    }

    #[test]
    fn parses_stage_mode_values() {
        assert_eq!(
            ProfileStageMode::Disabled,
            profile_stage_mode_from_value(None)
        );
        assert_eq!(
            ProfileStageMode::Disabled,
            profile_stage_mode_from_value(Some("off"))
        );
        assert_eq!(
            ProfileStageMode::Rows,
            profile_stage_mode_from_value(Some("1"))
        );
        assert_eq!(
            ProfileStageMode::Rows,
            profile_stage_mode_from_value(Some("true"))
        );

        for value in ["summary", "summaries", "aggregate", "aggregates"] {
            assert_eq!(
                ProfileStageMode::Summary,
                profile_stage_mode_from_value(Some(value))
            );
        }
    }

    #[test]
    fn formats_string_profile_rows() {
        let row = crate::format::format_profile_row(
            "jpeg",
            "decode",
            "tile/0",
            &[("rows", "4"), ("elapsed_us", "12")],
        );

        assert_eq!(
            "j2k_profile codec=jpeg op=decode path=tile/0 rows=4 elapsed_us=12",
            row
        );
    }

    #[test]
    fn formats_key_value_fields_without_forcing_prefix_order() {
        let fields = [
            ("codec", "transcode"),
            ("op", "transcode_batch"),
            ("request", "metal_auto"),
            ("path", "auto"),
        ];

        assert_eq!(
            " codec=transcode op=transcode_batch request=metal_auto path=auto",
            format_profile_key_value_fields(&fields)
        );
    }

    #[test]
    fn formats_u128_profile_rows() {
        let row = crate::format::format_profile_row_u128(
            "j2k",
            "decode",
            "tile/1",
            &[("elapsed_us", 34_u128), ("bytes", 99_u128)],
        );

        assert_eq!(
            "j2k_profile codec=j2k op=decode path=tile/1 elapsed_us=34 bytes=99",
            row
        );
    }

    #[test]
    fn summary_counts_and_remaps_labels() {
        let mut summary = ProfileSummary::new(vec![
            SummaryLabel::new("component", "stage"),
            SummaryLabel::same("backend"),
        ]);

        summary.record_str(
            "jpeg",
            "decode",
            "tile/0",
            &[
                ("component", "idct"),
                ("backend", "cpu"),
                ("quality", "fast"),
            ],
        );
        summary.record_str(
            "jpeg",
            "decode",
            "tile/0",
            &[("component", "idct"), ("backend", "cpu")],
        );

        assert_eq!(
            vec![
                "j2k_profile_summary codec=jpeg op=decode path=tile/0 stage=idct backend=cpu count=2"
            ],
            summary.format_rows()
        );
    }

    #[test]
    fn summary_emits_timing_sums_and_averages() {
        let mut summary = ProfileSummary::new([SummaryLabel::same("stage")]);

        summary.record_str(
            "jpeg",
            "decode",
            "tile/0",
            &[("stage", "entropy"), ("elapsed_us", "10"), ("bytes", "100")],
        );
        summary.record_str(
            "jpeg",
            "decode",
            "tile/0",
            &[("stage", "entropy"), ("elapsed_us", "20"), ("bytes", "50")],
        );

        assert_eq!(
            vec![
                "j2k_profile_summary codec=jpeg op=decode path=tile/0 stage=entropy count=2 bytes_sum=150 elapsed_us_sum=30 elapsed_us_avg=15"
            ],
            summary.format_rows()
        );
    }

    #[test]
    fn count_only_summary_omits_numeric_fields() {
        let mut summary = ProfileSummary::counts_only([SummaryLabel::same("route")]);

        summary.record_u128(
            "j2k-cuda",
            "decode",
            "tile/2",
            &[
                ("route", 1_u128),
                ("width", 512_u128),
                ("height", 512_u128),
                ("tiles", 4_u128),
            ],
        );
        summary.record_u128(
            "j2k-cuda",
            "decode",
            "tile/2",
            &[
                ("route", 1_u128),
                ("width", 256_u128),
                ("height", 256_u128),
                ("tiles", 2_u128),
            ],
        );

        assert_eq!(
            vec!["j2k_profile_summary codec=j2k-cuda op=decode path=tile/2 route=1 count=2"],
            summary.format_rows()
        );
    }

    #[test]
    fn summary_omits_absent_configured_labels() {
        let mut summary = ProfileSummary::new([
            SummaryLabel::same("backend"),
            SummaryLabel::same("missing_label"),
        ]);

        summary.record_str(
            "jpeg",
            "decode",
            "tile/0",
            &[("backend", "cpu"), ("elapsed_us", "8")],
        );

        assert_eq!(
            vec![
                "j2k_profile_summary codec=jpeg op=decode path=tile/0 backend=cpu count=1 elapsed_us_sum=8 elapsed_us_avg=8"
            ],
            summary.format_rows()
        );
    }

    #[test]
    fn summary_emits_u128_timing_summaries() {
        let mut summary = ProfileSummary::new([SummaryLabel::same("backend")]);

        summary.record_u128(
            "j2k",
            "decode",
            "tile/1",
            &[("backend", 7_u128), ("elapsed_ns", 3_u128)],
        );
        summary.record_u128(
            "j2k",
            "decode",
            "tile/1",
            &[("backend", 7_u128), ("elapsed_ns", 9_u128)],
        );

        assert_eq!(
            vec![
                "j2k_profile_summary codec=j2k op=decode path=tile/1 backend=7 count=2 elapsed_ns_sum=12 elapsed_ns_avg=6"
            ],
            summary.format_rows()
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn profile_summary_emit_on_drop_is_explicit_and_not_cloned() {
        let mut summary = ProfileSummary::new([SummaryLabel::same("stage")]).emit_on_drop();
        summary.record_str(
            "jpeg",
            "decode",
            "tile/0",
            &[("stage", "emit"), ("elapsed_ms", "4")],
        );

        let cloned = summary.clone();
        assert!(summary.emit_on_drop_enabled());
        assert!(!cloned.emit_on_drop_enabled());

        summary.flush_to_stderr();
        assert!(summary.format_rows().is_empty());
    }

    #[cfg(feature = "std")]
    #[test]
    fn emit_helpers_honor_stage_modes() {
        thread_local! {
            static TEST_SUMMARY: RefCell<ProfileSummary> =
                RefCell::new(ProfileSummary::new([SummaryLabel::same("stage")]));
        }

        emit_profile_row(
            ProfileStageMode::Disabled,
            &TEST_SUMMARY,
            "jpeg",
            "decode",
            "tile/0",
            &[("stage", "off"), ("elapsed_us", "10")],
        );
        TEST_SUMMARY.with(|summary| assert!(summary.borrow().format_rows().is_empty()));

        emit_profile_row(
            ProfileStageMode::Summary,
            &TEST_SUMMARY,
            "jpeg",
            "decode",
            "tile/0",
            &[("stage", "on"), ("elapsed_us", "10")],
        );
        emit_profile_row_u128(
            ProfileStageMode::Summary,
            &TEST_SUMMARY,
            "jpeg",
            "decode",
            "tile/0",
            &[("stage", 1_u128), ("elapsed_us", 5_u128)],
        );

        TEST_SUMMARY.with(|summary| {
            assert_eq!(
                vec![
                    "j2k_profile_summary codec=jpeg op=decode path=tile/0 stage=1 count=1 elapsed_us_sum=5 elapsed_us_avg=5",
                    "j2k_profile_summary codec=jpeg op=decode path=tile/0 stage=on count=1 elapsed_us_sum=10 elapsed_us_avg=10",
                ],
                summary.borrow().format_rows()
            );
        });
    }

    #[cfg(feature = "std")]
    #[test]
    fn parses_stage_mode_from_named_env_var() {
        const ENV_KEY: &str = "J2K_PROFILE_TEST_STAGE_MODE";

        std::env::set_var(ENV_KEY, "summary");
        assert_eq!(
            ProfileStageMode::Summary,
            profile_stage_mode_from_env(ENV_KEY)
        );
        std::env::set_var(ENV_KEY, "1");
        assert_eq!(ProfileStageMode::Rows, profile_stage_mode_from_env(ENV_KEY));
        std::env::remove_var(ENV_KEY);
        assert_eq!(
            ProfileStageMode::Disabled,
            profile_stage_mode_from_env(ENV_KEY)
        );
    }

    #[test]
    fn builds_same_summary_labels_from_field_keys() {
        let mut summary = ProfileSummary::new(same_summary_labels(&["mode", "fmt"]));
        summary.record_str(
            "jpeg",
            "decode",
            "cpu",
            &[("mode", "full"), ("fmt", "Rgb8"), ("total_us", "5")],
        );

        assert_eq!(
            vec![
                "j2k_profile_summary codec=jpeg op=decode path=cpu mode=full fmt=Rgb8 count=1 total_us_sum=5 total_us_avg=5"
            ],
            summary.format_rows()
        );
    }

    #[test]
    fn gpu_route_profile_stage_mode_uses_shared_env_parser() {
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
            ProfileStageMode::Disabled
        );
        assert_eq!(
            ProfileStageMode::Disabled,
            profile_stage_mode_from_value(None)
        );
        assert_ne!(
            ProfileStageMode::Disabled,
            profile_stage_mode_from_value(Some("on"))
        );
    }

    #[test]
    fn gpu_route_summary_counts_route_decisions() {
        let mut summary = ProfileSummary::counts_only(gpu_route_summary_labels());
        summary.record_str(
            "jpeg",
            "gpu_route",
            "metal",
            &[
                ("op", "full"),
                ("request", "Metal"),
                ("fmt", "Rgb8"),
                ("width", "16"),
                ("decision", "metal_kernel"),
                ("reason", "none"),
            ],
        );
        summary.record_str(
            "jpeg",
            "gpu_route",
            "metal",
            &[
                ("op", "full"),
                ("request", "Metal"),
                ("fmt", "Rgb8"),
                ("width", "32"),
                ("decision", "metal_kernel"),
                ("reason", "none"),
            ],
        );

        assert_eq!(
            summary.format_rows(),
            vec![
                "j2k_profile_summary codec=jpeg op=gpu_route path=metal route_op=full request=Metal fmt=Rgb8 decision=metal_kernel reason=none count=2"
            ]
        );
    }

    #[test]
    fn records_timing_summary_str_with_only_labels_and_timing_fields() {
        let mut summary = ProfileSummary::new(same_summary_labels(&["mode", "fmt", "downscale"]));
        crate::summary::record_timing_summary_str(
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
            &["mode", "fmt", "downscale"],
        );
        crate::summary::record_timing_summary_str(
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
            &["mode", "fmt", "downscale"],
        );

        assert_eq!(
            vec![
                "j2k_profile_summary codec=jpeg op=decode path=cpu mode=full fmt=Rgb8 count=2 decode_us_sum=12 decode_us_avg=6 total_us_sum=16 total_us_avg=8"
            ],
            summary.format_rows()
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn emits_string_profile_rows_with_timing_summary_filter() {
        thread_local! {
            static TEST_SUMMARY: RefCell<ProfileSummary> =
                RefCell::new(ProfileSummary::new(same_summary_labels(&["stage"])));
        }

        emit_profile_row_with_timing_summary(
            ProfileStageMode::Summary,
            &TEST_SUMMARY,
            "jpeg",
            "decode",
            "cpu",
            &[("stage", "entropy"), ("width", "512"), ("elapsed_us", "9")],
            &["stage"],
        );

        TEST_SUMMARY.with(|summary| {
            assert_eq!(
                vec![
                    "j2k_profile_summary codec=jpeg op=decode path=cpu stage=entropy count=1 elapsed_us_sum=9 elapsed_us_avg=9"
                ],
                summary.borrow().format_rows()
            );
        });
    }

    #[test]
    fn formats_typed_profile_fields_like_compat_rows() {
        let fields = [
            ProfileField::label("stage", "entropy"),
            ProfileField::metric("elapsed_us", 9_u128, MetricUnit::Microseconds),
        ];

        assert_eq!(
            "j2k_profile codec=jpeg op=decode path=cpu stage=entropy elapsed_us=9",
            crate::format::format_profile_fields("jpeg", "decode", "cpu", &fields)
        );
    }

    #[test]
    fn typed_summary_respects_explicit_metric_policy() {
        let mut summary = ProfileSummary::new(same_summary_labels(&["stage"]));
        let fields = [
            ProfileField::label("stage", "entropy"),
            ProfileField::metric("elapsed_us", 9_u128, MetricUnit::Microseconds),
            ProfileField::metric_with_summary("output_bytes", 512_u128, MetricUnit::Bytes, false),
        ];

        summary.record_fields("jpeg", "decode", "cpu", &fields);

        assert_eq!(
            vec![
                "j2k_profile_summary codec=jpeg op=decode path=cpu stage=entropy count=1 elapsed_us_sum=9 elapsed_us_avg=9"
            ],
            summary.format_rows()
        );
    }

    #[test]
    fn parses_profile_rows_into_key_value_fields() {
        let parsed = parse_profile_line(
            "j2k_profile codec=transcode op=transcode_batch path=metal total_us=42",
        )
        .expect("profile row should parse");

        assert_eq!(ParsedProfileKind::Row, parsed.kind());
        assert_eq!(Some("transcode"), parsed.get("codec"));
        assert_eq!(Some("transcode_batch"), parsed.get("op"));
        assert_eq!(Some("metal"), parsed.get("path"));
        assert_eq!(Some("42"), parsed.get("total_us"));
    }

    #[test]
    fn parses_profile_key_value_field_lists() {
        assert_eq!(
            vec![
                (String::from("mode"), String::from("auto")),
                (String::from("cpu_ms"), String::from("12.5")),
                (String::from("error"), String::from("launch")),
            ],
            parse_profile_key_value_fields("mode=auto cpu_ms=12.5 error=launch failed")
        );
    }

    #[test]
    fn take_formatted_rows_flushes_summary_state_without_stderr() {
        let mut summary = ProfileSummary::new(same_summary_labels(&["stage"]));
        summary.record_str(
            "jpeg",
            "decode",
            "cpu",
            &[("stage", "entropy"), ("elapsed_us", "5")],
        );

        assert_eq!(
            vec![
                "j2k_profile_summary codec=jpeg op=decode path=cpu stage=entropy count=1 elapsed_us_sum=5 elapsed_us_avg=5"
            ],
            summary.take_formatted_rows()
        );
        assert!(summary.format_rows().is_empty());
    }

    #[cfg(feature = "std")]
    #[test]
    fn formats_duration_as_microsecond_string() {
        assert_eq!(
            duration_us_string(std::time::Duration::from_nanos(1_234_567)),
            "1234"
        );
    }
}
