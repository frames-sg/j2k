//! Internal profiling helpers shared by the `j2k` workspace crates.

#![doc(hidden)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

#[cfg(feature = "std")]
mod emit;
mod env;
mod error;
mod field;
mod format;
mod gpu_route;
mod limits;
mod parse;
mod summary;
mod text;
mod timing;

mod allocation;

#[cfg(feature = "std")]
pub use emit::{
    emit_profile_error, emit_profile_fields, emit_profile_line, emit_profile_row,
    emit_profile_row_now, emit_profile_row_u128, emit_profile_row_with_timing_summary,
};
pub use env::profile_stage_mode_from_value;
#[cfg(feature = "std")]
pub use env::{env_flag_from_env, profile_stage_mode_from_env, StageModeCache};
pub use error::{ProfileError, ProfileResult};
pub use field::ProfileField;
#[cfg(any(feature = "std", test))]
pub use format::format_profile_row_u128;
pub use format::{format_profile_key_value_fields, format_profile_key_value_fields_with_limits};
#[cfg(feature = "std")]
pub use gpu_route::{
    emit_gpu_route_decision_profile, emit_gpu_route_fields, emit_gpu_route_surface_profile,
    gpu_route_profile_enabled,
};
pub use limits::ProfileLimits;
pub use parse::{
    parse_profile_key_value_fields, parse_profile_key_value_fields_with_limits, parse_profile_line,
    parse_profile_line_with_limits, ParsedProfileFields, ParsedProfileKind,
};
pub use summary::{
    same_summary_labels, same_summary_labels_with_limits, ProfileSummary, SummaryLabel,
};
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
        )
        .expect("bounded profile row should format");

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
            format_profile_key_value_fields(&fields).expect("bounded profile fields should format")
        );
    }

    #[test]
    fn formats_u128_profile_rows() {
        let row = crate::format::format_profile_row_u128(
            "j2k",
            "decode",
            "tile/1",
            &[("elapsed_us", 34_u128), ("bytes", 99_u128)],
        )
        .expect("bounded integer row should format");

        assert_eq!(
            "j2k_profile codec=j2k op=decode path=tile/1 elapsed_us=34 bytes=99",
            row
        );
    }

    #[test]
    fn summary_counts_and_remaps_labels() {
        let mut summary = ProfileSummary::new(vec![
            SummaryLabel::new("component", "stage").expect("valid remapped label"),
            SummaryLabel::same("backend").expect("valid same-key label"),
        ])
        .expect("valid summary");

        summary
            .record_str(
                "jpeg",
                "decode",
                "tile/0",
                &[
                    ("component", "idct"),
                    ("backend", "cpu"),
                    ("quality", "fast"),
                ],
            )
            .expect("first summary row should record");
        summary
            .record_str(
                "jpeg",
                "decode",
                "tile/0",
                &[("component", "idct"), ("backend", "cpu")],
            )
            .expect("second summary row should record");

        assert_eq!(
            vec![
                "j2k_profile_summary codec=jpeg op=decode path=tile/0 stage=idct backend=cpu count=2"
            ],
            summary.format_rows().expect("summary should format")
        );
    }

    #[test]
    fn summary_emits_timing_sums_and_averages() {
        let mut summary =
            ProfileSummary::new([SummaryLabel::same("stage").expect("valid summary label")])
                .expect("valid summary");

        summary
            .record_str(
                "jpeg",
                "decode",
                "tile/0",
                &[("stage", "entropy"), ("elapsed_us", "10"), ("bytes", "100")],
            )
            .expect("first timing row should record");
        summary
            .record_str(
                "jpeg",
                "decode",
                "tile/0",
                &[("stage", "entropy"), ("elapsed_us", "20"), ("bytes", "50")],
            )
            .expect("second timing row should record");

        assert_eq!(
            vec![
                "j2k_profile_summary codec=jpeg op=decode path=tile/0 stage=entropy count=2 bytes_sum=150 elapsed_us_sum=30 elapsed_us_avg=15"
            ],
            summary.format_rows().expect("summary should format")
        );
    }

    #[test]
    fn count_only_summary_omits_numeric_fields() {
        let mut summary =
            ProfileSummary::counts_only(
                [SummaryLabel::same("route").expect("valid summary label")],
            )
            .expect("valid count-only summary");

        summary
            .record_u128(
                "j2k-cuda",
                "decode",
                "tile/2",
                &[
                    ("route", 1_u128),
                    ("width", 512_u128),
                    ("height", 512_u128),
                    ("tiles", 4_u128),
                ],
            )
            .expect("first count row should record");
        summary
            .record_u128(
                "j2k-cuda",
                "decode",
                "tile/2",
                &[
                    ("route", 1_u128),
                    ("width", 256_u128),
                    ("height", 256_u128),
                    ("tiles", 2_u128),
                ],
            )
            .expect("second count row should record");

        assert_eq!(
            vec!["j2k_profile_summary codec=j2k-cuda op=decode path=tile/2 route=1 count=2"],
            summary.format_rows().expect("summary should format")
        );
    }

    #[test]
    fn summary_omits_absent_configured_labels() {
        let mut summary = ProfileSummary::new([
            SummaryLabel::same("backend").expect("valid summary label"),
            SummaryLabel::same("missing_label").expect("valid summary label"),
        ])
        .expect("valid summary");

        summary
            .record_str(
                "jpeg",
                "decode",
                "tile/0",
                &[("backend", "cpu"), ("elapsed_us", "8")],
            )
            .expect("summary row should record");

        assert_eq!(
            vec![
                "j2k_profile_summary codec=jpeg op=decode path=tile/0 backend=cpu count=1 elapsed_us_sum=8 elapsed_us_avg=8"
            ],
            summary.format_rows().expect("summary should format")
        );
    }

    #[test]
    fn summary_emits_u128_timing_summaries() {
        let mut summary =
            ProfileSummary::new([SummaryLabel::same("backend").expect("valid summary label")])
                .expect("valid summary");

        summary
            .record_u128(
                "j2k",
                "decode",
                "tile/1",
                &[("backend", 7_u128), ("elapsed_ns", 3_u128)],
            )
            .expect("first integer row should record");
        summary
            .record_u128(
                "j2k",
                "decode",
                "tile/1",
                &[("backend", 7_u128), ("elapsed_ns", 9_u128)],
            )
            .expect("second integer row should record");

        assert_eq!(
            vec![
                "j2k_profile_summary codec=j2k op=decode path=tile/1 backend=7 count=2 elapsed_ns_sum=12 elapsed_ns_avg=6"
            ],
            summary.format_rows().expect("summary should format")
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn profile_summary_emit_on_drop_is_explicit() {
        let mut summary =
            ProfileSummary::new([SummaryLabel::same("stage").expect("valid summary label")])
                .expect("valid summary")
                .emit_on_drop();
        summary
            .record_str(
                "jpeg",
                "decode",
                "tile/0",
                &[("stage", "emit"), ("elapsed_ms", "4")],
            )
            .expect("summary row should record");

        assert!(summary.emit_on_drop_enabled());

        summary.flush_to_stderr().expect("summary should flush");
        assert!(summary
            .format_rows()
            .expect("empty summary should format")
            .is_empty());
    }

    #[cfg(feature = "std")]
    #[test]
    fn emit_helpers_honor_stage_modes() {
        thread_local! {
            static TEST_SUMMARY: RefCell<ProfileSummary> =
                RefCell::new(ProfileSummary::new([
                    SummaryLabel::same("stage").expect("valid summary label")
                ]).expect("valid summary"));
        }

        emit_profile_row(
            ProfileStageMode::Disabled,
            &TEST_SUMMARY,
            "jpeg",
            "decode",
            "tile/0",
            &[("stage", "off"), ("elapsed_us", "10")],
        );
        TEST_SUMMARY.with(|summary| {
            assert!(summary
                .borrow()
                .format_rows()
                .expect("empty summary should format")
                .is_empty());
        });

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
                summary.borrow().format_rows().expect("summary should format")
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
        let labels = same_summary_labels(&["mode", "fmt"]).expect("valid summary labels");
        let mut summary = ProfileSummary::new(labels).expect("valid summary");
        summary
            .record_str(
                "jpeg",
                "decode",
                "cpu",
                &[("mode", "full"), ("fmt", "Rgb8"), ("total_us", "5")],
            )
            .expect("summary row should record");

        assert_eq!(
            vec![
                "j2k_profile_summary codec=jpeg op=decode path=cpu mode=full fmt=Rgb8 count=1 total_us_sum=5 total_us_avg=5"
            ],
            summary.format_rows().expect("summary should format")
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
        let labels = gpu_route_summary_labels().expect("valid GPU route labels");
        let mut summary = ProfileSummary::counts_only(labels).expect("valid count-only summary");
        summary
            .record_str(
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
            )
            .expect("first route row should record");
        summary
            .record_str(
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
            )
            .expect("second route row should record");

        assert_eq!(
            summary.format_rows().expect("summary should format"),
            vec![
                "j2k_profile_summary codec=jpeg op=gpu_route path=metal route_op=full request=Metal fmt=Rgb8 decision=metal_kernel reason=none count=2"
            ]
        );
    }

    #[test]
    fn records_timing_summary_str_with_only_labels_and_timing_fields() {
        let labels =
            same_summary_labels(&["mode", "fmt", "downscale"]).expect("valid summary labels");
        let mut summary = ProfileSummary::new(labels).expect("valid summary");
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
        )
        .expect("first timing summary row should record");
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
        )
        .expect("second timing summary row should record");

        assert_eq!(
            vec![
                "j2k_profile_summary codec=jpeg op=decode path=cpu mode=full fmt=Rgb8 count=2 decode_us_sum=12 decode_us_avg=6 total_us_sum=16 total_us_avg=8"
            ],
            summary.format_rows().expect("summary should format")
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn emits_string_profile_rows_with_timing_summary_filter() {
        thread_local! {
            static TEST_SUMMARY: RefCell<ProfileSummary> =
                RefCell::new(ProfileSummary::new(
                    same_summary_labels(&["stage"]).expect("valid summary labels")
                ).expect("valid summary"));
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
                summary.borrow().format_rows().expect("summary should format")
            );
        });
    }

    #[test]
    fn formats_typed_profile_fields_like_compat_rows() {
        let fields = [
            ProfileField::label("stage", "entropy").expect("valid label field"),
            ProfileField::metric("elapsed_us", 9_u128).expect("valid metric field"),
        ];

        assert_eq!(
            "j2k_profile codec=jpeg op=decode path=cpu stage=entropy elapsed_us=9",
            crate::format::format_profile_fields("jpeg", "decode", "cpu", &fields)
                .expect("typed profile row should format")
        );
    }

    #[test]
    fn typed_summary_respects_explicit_metric_policy() {
        let labels = same_summary_labels(&["stage"]).expect("valid summary labels");
        let mut summary = ProfileSummary::new(labels).expect("valid summary");
        let fields = [
            ProfileField::label("stage", "entropy").expect("valid label field"),
            ProfileField::metric("elapsed_us", 9_u128).expect("valid metric field"),
            ProfileField::metric_with_summary("output_bytes", 512_u128, false)
                .expect("valid non-summary metric field"),
        ];

        summary
            .record_fields("jpeg", "decode", "cpu", &fields)
            .expect("typed row should record");

        assert_eq!(
            vec![
                "j2k_profile_summary codec=jpeg op=decode path=cpu stage=entropy count=1 elapsed_us_sum=9 elapsed_us_avg=9"
            ],
            summary.format_rows().expect("summary should format")
        );
    }

    #[test]
    fn parses_profile_rows_into_key_value_fields() {
        let parsed = parse_profile_line(
            "j2k_profile codec=transcode op=transcode_batch path=metal total_us=42",
        )
        .expect("bounded parser should succeed")
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
            parse_profile_key_value_fields("mode=auto cpu_ms=12.5 error=launch")
                .expect("valid fields should parse")
        );
        assert_eq!(
            ProfileError::InvalidInput {
                what: "profile field is not key=value"
            },
            parse_profile_key_value_fields("mode=auto failed")
                .expect_err("malformed token must not be silently discarded")
        );
    }

    #[test]
    fn take_formatted_rows_flushes_summary_state_without_stderr() {
        let labels = same_summary_labels(&["stage"]).expect("valid summary labels");
        let mut summary = ProfileSummary::new(labels).expect("valid summary");
        summary
            .record_str(
                "jpeg",
                "decode",
                "cpu",
                &[("stage", "entropy"), ("elapsed_us", "5")],
            )
            .expect("summary row should record");

        assert_eq!(
            vec![
                "j2k_profile_summary codec=jpeg op=decode path=cpu stage=entropy count=1 elapsed_us_sum=5 elapsed_us_avg=5"
            ],
            summary
                .take_formatted_rows()
                .expect("summary should format and clear")
        );
        assert!(summary
            .format_rows()
            .expect("empty summary should format")
            .is_empty());
    }

    #[test]
    fn parser_and_formatter_enforce_explicit_limits() {
        let one_field = ProfileLimits::default().with_max_fields(1);
        assert_eq!(
            ProfileError::LimitExceeded {
                what: "field count",
                requested: 2,
                limit: 1,
            },
            parse_profile_key_value_fields_with_limits("a=1 b=2", one_field)
                .expect_err("field limit must be enforced")
        );

        assert_eq!(
            ProfileError::InvalidInput {
                what: "profile token contains whitespace",
            },
            format_profile_key_value_fields(&[("label", "two words")])
                .expect_err("ambiguous output must be rejected")
        );

        let invalid = ProfileLimits::default()
            .with_max_input_bytes(4)
            .with_max_token_bytes(5);
        assert_eq!(
            ProfileError::InvalidLimits {
                what: "token bytes exceed input bytes",
            },
            parse_profile_key_value_fields_with_limits("a=1", invalid)
                .expect_err("inconsistent limits must be rejected")
        );
    }

    #[test]
    fn existing_summary_updates_do_not_grow_retained_capacity() {
        let mut summary = ProfileSummary::default();
        summary
            .record_u128("j2k", "decode", "cpu", &[("elapsed_us", 4_u128)])
            .expect("first row should record");
        let retained = summary.retained_capacity_bytes();

        summary
            .record_u128("j2k", "decode", "cpu", &[("elapsed_us", 8_u128)])
            .expect("existing row should update");

        assert_eq!(retained, summary.retained_capacity_bytes());
        assert_eq!(
            vec![
                "j2k_profile_summary codec=j2k op=decode path=cpu count=2 elapsed_us_sum=12 elapsed_us_avg=6"
            ],
            summary.format_rows().expect("summary should format")
        );
    }

    #[test]
    fn row_limit_failure_preserves_the_complete_summary() {
        let limits = ProfileLimits::default().with_max_rows(1);
        let mut summary = ProfileSummary::new_with_limits([], limits).expect("valid limits");
        summary
            .record_str("jpeg", "decode", "first", &[("elapsed_us", "3")])
            .expect("first row should record");
        let before = summary.format_rows().expect("summary should format");
        let retained = summary.retained_capacity_bytes();

        assert_eq!(
            ProfileError::LimitExceeded {
                what: "summary row count",
                requested: 2,
                limit: 1,
            },
            summary
                .record_str("jpeg", "decode", "second", &[("elapsed_us", "5")])
                .expect_err("second distinct row must exceed the limit")
        );
        assert_eq!(1, summary.row_count());
        assert_eq!(retained, summary.retained_capacity_bytes());
        assert_eq!(
            before,
            summary.format_rows().expect("summary should remain valid")
        );
    }

    #[test]
    fn numeric_limit_failure_rolls_back_count_and_sums() {
        let limits = ProfileLimits::default().with_max_numeric_fields_per_row(1);
        let mut summary = ProfileSummary::new_with_limits([], limits).expect("valid limits");
        summary
            .record_u128("jpeg", "decode", "cpu", &[("elapsed_us", 3_u128)])
            .expect("first row should record");
        let before = summary.format_rows().expect("summary should format");

        assert_eq!(
            ProfileError::LimitExceeded {
                what: "summary numeric field count",
                requested: 2,
                limit: 1,
            },
            summary
                .record_u128(
                    "jpeg",
                    "decode",
                    "cpu",
                    &[("elapsed_us", 5_u128), ("bytes", 10_u128)],
                )
                .expect_err("new numeric key must exceed the limit")
        );
        assert_eq!(
            before,
            summary.format_rows().expect("summary should roll back")
        );
    }

    #[test]
    fn failed_take_keeps_summary_rows_and_retained_state() {
        let limits = ProfileLimits::default().with_max_output_bytes(1);
        let mut summary = ProfileSummary::new_with_limits([], limits).expect("valid limits");
        summary
            .record_str("jpeg", "decode", "cpu", &[("elapsed_us", "3")])
            .expect("summary row should record");
        let retained = summary.retained_capacity_bytes();

        assert!(matches!(
            summary.take_formatted_rows(),
            Err(ProfileError::LimitExceeded { .. })
        ));
        assert_eq!(1, summary.row_count());
        assert_eq!(retained, summary.retained_capacity_bytes());
    }

    #[test]
    fn owned_field_formatting_is_bounded_and_fallible() {
        let limits = ProfileLimits::default().with_max_token_bytes(4);
        assert_eq!(
            ProfileError::LimitExceeded {
                what: "field value",
                requested: 5,
                limit: 4,
            },
            ProfileField::label_with_limits("key", "12345", limits)
                .expect_err("oversized display output must fail")
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn duration_microseconds_use_the_typed_fallible_formatter() {
        let micros = std::time::Duration::from_nanos(1_234_567).as_micros();
        let field = ProfileField::metric("duration_us", micros).expect("bounded duration field");
        assert_eq!(field.value(), "1234");
    }
}
