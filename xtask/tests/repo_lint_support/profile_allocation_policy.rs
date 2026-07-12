// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bounded ownership and transactional behavior ratchets for profile telemetry.

use std::fs;

use super::{assert_pattern_checks, read_source_files, repo_root, PatternCheck};

fn read(relative: &str) -> String {
    fs::read_to_string(repo_root().join(relative))
        .unwrap_or_else(|error| panic!("read {relative}: {error}"))
}

fn source_section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    source
        .split_once(start)
        .unwrap_or_else(|| panic!("missing section start: {start}"))
        .1
        .split_once(end)
        .unwrap_or_else(|| panic!("missing section end: {end}"))
        .0
}

#[test]
fn profile_owners_remain_bounded_fallible_and_move_only() {
    let summary = read("crates/j2k-profile/src/summary.rs");
    let core = read_source_files(
        repo_root(),
        &[
            "crates/j2k-profile/src/allocation.rs",
            "crates/j2k-profile/src/error.rs",
            "crates/j2k-profile/src/emit.rs",
            "crates/j2k-profile/src/field.rs",
            "crates/j2k-profile/src/format.rs",
            "crates/j2k-profile/src/limits.rs",
            "crates/j2k-profile/src/parse.rs",
            "crates/j2k-profile/src/summary.rs",
            "crates/j2k-profile/src/summary/output.rs",
            "crates/j2k-profile/src/summary/record.rs",
            "crates/j2k-profile/src/timing.rs",
        ],
    );

    assert_pattern_checks(&[
        PatternCheck::new("profile bounded owner model", &core)
            .required(&[
                "pub struct ProfileLimits",
                "pub enum ProfileError",
                "ProfileError::AllocationFailed",
                "try_reserve_exact",
                ".capacity()",
                "max_retained_bytes",
                "max_output_bytes",
                "pub fn record_str",
                "pub fn format_rows(&self) -> ProfileResult<Vec<String>>",
                "pub fn take_formatted_rows(&mut self) -> ProfileResult<Vec<String>>",
                "pub fn emit_profile_error",
            ])
            .forbidden(&[
                "BTreeMap",
                "Vec::with_capacity",
                ".to_owned()",
                ".to_string()",
                "writing to String failed",
                "duration_us_string",
                "impl Clone for ProfileSummary",
                "#[derive(Clone, Debug, Eq, PartialEq)]\npub struct ParsedProfileFields",
            ]),
        PatternCheck::new("profile drop diagnostics", &summary)
            .required(&["crate::emit_profile_error(\"summary_drop\", &error)"])
            .forbidden(&["std::eprintln!(\"j2k_profile_error"]),
    ]);
}

#[test]
fn transcode_profile_row_uses_the_shared_fallible_contract() {
    let report = read("crates/j2k-transcode/src/jpeg_to_htj2k/report.rs");
    let profile_owner = report
        .split_once("/// Detailed timing and dispatch counters")
        .map_or(report.as_str(), |(profile_owner, _)| profile_owner);

    assert_pattern_checks(&[
        PatternCheck::new("transcode profile row owner", profile_owner)
            .required(&[
                "pub struct TranscodeBatchProfileRow",
                ") -> j2k_profile::ProfileResult<Self>",
                "try_reserve_exact(TRANSCODE_PROFILE_FIELD_COUNT)",
                "fields.capacity()",
                "checked_add(report.transform_us)",
                "ProfileField::metric_with_limits",
                "fn finish(self) -> j2k_profile::ProfileResult<TranscodeBatchProfileFields>",
            ])
            .forbidden(&[
                "#[derive(Debug, Clone, PartialEq, Eq)]\npub struct TranscodeBatchProfileRow",
                "Vec::with_capacity",
                ".to_string()",
                ".replace(' ', \"_\")",
                ".saturating_add(",
                "profile row includes required prefix field",
            ]),
    ]);
}

#[test]
fn cpu_jpeg_profile_callers_use_typed_fallible_fields() {
    let profile = read("crates/j2k-jpeg/src/profile.rs");
    let encoder = read("crates/j2k-jpeg/src/encoder.rs");
    let encode_profile = encoder
        .split_once("fn emit_cpu_encode_profile(")
        .and_then(|(_, tail)| tail.split_once("impl JpegSamples<'_>"))
        .map(|(body, _)| body)
        .expect("CPU encode profile owner");
    let lossless = read("crates/j2k-jpeg/src/decoder/lossless_helpers.rs");
    let lossless_profile = lossless
        .split_once("pub(super) fn emit_decode_scan_profile(")
        .and_then(|(_, tail)| tail.split_once("pub(super) fn consume_lossless_restart("))
        .map(|(body, _)| body)
        .expect("lossless scan profile owner");
    let routing = read("crates/j2k-jpeg/src/decoder/routing/profile.rs");
    let callers = [encode_profile, lossless_profile, routing.as_str()].concat();

    assert_pattern_checks(&[
        PatternCheck::new("JPEG typed profile emitter", &profile)
            .required(&[
                "fn emit_jpeg_profile_fields<const N: usize>",
                "impl FnOnce() -> j2k_profile::ProfileResult<[ProfileField; N]>",
                "j2k_profile::emit_profile_fields",
                "j2k_profile::emit_profile_error(operation, &error)",
            ])
            .forbidden(&["duration_us_string", "emit_jpeg_profile_row"]),
        PatternCheck::new("JPEG typed profile callers", &callers)
            .required(&[
                "ProfileField::label",
                "ProfileField::metric(",
                "ProfileField::metric_with_summary",
            ])
            .forbidden(&[".to_string()", "duration_us_string", "format!("]),
    ]);
}

#[test]
fn profile_regressions_cover_limits_capacity_and_rollback() {
    let tests = read_source_files(
        repo_root(),
        &[
            "crates/j2k-profile/src/allocation.rs",
            "crates/j2k-profile/src/lib.rs",
        ],
    );
    assert_pattern_checks(&[
        PatternCheck::new("profile allocation regressions", &tests).required(&[
            "allocator_reported_overcapacity_is_rejected",
            "allocator_reservation_failure_is_typed",
            "parser_and_formatter_enforce_explicit_limits",
            "existing_summary_updates_do_not_grow_retained_capacity",
            "row_limit_failure_preserves_the_complete_summary",
            "numeric_limit_failure_rolls_back_count_and_sums",
            "failed_take_keeps_summary_rows_and_retained_state",
            "owned_field_formatting_is_bounded_and_fallible",
            "duration_microseconds_use_the_typed_fallible_formatter",
        ]),
    ]);
}

#[test]
fn metal_profile_emitters_use_bounded_typed_fields() {
    let batch = read("crates/j2k-metal/src/profile.rs");
    let batch = batch
        .split_once("#[cfg(test)]")
        .map_or(batch.as_str(), |part| part.0);
    let direct = read("crates/j2k-metal/src/profile_env/direct.rs");
    let direct = direct
        .split_once("#[cfg(test)]")
        .map_or(direct.as_str(), |part| part.0);
    let jpeg = read("crates/j2k-jpeg-metal/src/compute/batch_support.rs");
    let jpeg = source_section(
        &jpeg,
        "impl FastBatchTiming {",
        "#[cfg(all(test, target_os = \"macos\"))]",
    );

    assert_pattern_checks(&[
        PatternCheck::new("J2K Metal batch profile emitter", batch)
            .required(&[
                "pub(crate) enum MetalBatchProfileValue<T>",
                "ProfileResult<[ProfileField; 14]>",
                "j2k_profile::emit_profile_fields",
                "j2k_profile::emit_profile_error(\"metal_batch_fields\"",
            ])
            .forbidden(&["vec![", ".to_string()", "format!(", "emit_profile_row("]),
        PatternCheck::new("J2K Metal direct profile emitter", direct)
            .required(&[
                "pub(crate) struct MetalDirectProfileRow",
                "ProfileResult<[ProfileField; 10]>",
                "decode_profile_label() -> ProfileResult<String>",
                "Err(std::env::VarError::NotUnicode(_))",
                "SanitizedProfileLabel",
                "j2k_profile::emit_profile_fields",
                "j2k_profile::emit_profile_error(\"metal_direct_fields\"",
            ])
            .forbidden(&[
                "vec![",
                ".to_string()",
                "format!(",
                ".collect()",
                "emit_profile_row(",
            ]),
        PatternCheck::new("JPEG Metal fast batch profile emitter", jpeg)
            .required(&[
                "j2k_profile::ProfileResult<[j2k_profile::ProfileField; 12]>",
                "format_args!(\"{}x{}\"",
                "j2k_profile::emit_profile_fields",
                "j2k_profile::emit_profile_error(\"jpeg_metal_fast420_fields\"",
            ])
            .forbidden(&["Vec<", "vec![", ".to_string()", "format!("]),
    ]);
}

#[test]
fn metal_profile_callers_do_not_preformat_owned_values() {
    let direct = read("crates/j2k-metal/src/compute/direct_profile.rs");
    let direct = source_section(
        &direct,
        "pub(super) fn emit_direct_hybrid_stage_timings(",
        "fn stage_processor",
    );
    let hybrid = read("crates/j2k-metal/src/hybrid.rs");
    let hybrid = source_section(
        &hybrid,
        "fn emit_region_scaled_color_plan_build_timings(",
        "fn plan_stage_processor",
    );
    let execute = read("crates/j2k-metal/src/batch/execute.rs");
    let execute = execute
        .split_once("#[cfg(test)]")
        .map_or(execute.as_str(), |part| part.0);

    assert_pattern_checks(&[
        PatternCheck::new("J2K Metal direct profile caller", direct)
            .required(&[
                "MetalDirectProfileRow",
                "MetalProfileFormat::Pixel(fmt)",
                "emit_profile_error(\"metal_direct_label\"",
            ])
            .forbidden(&[".to_string()", "format!(", "label.clone()"]),
        PatternCheck::new("J2K Metal hybrid-plan profile caller", hybrid)
            .required(&[
                "MetalDirectProfileRow",
                "MetalProfileFormat::Family(\"Rgb\")",
                "emit_profile_error(\"metal_hybrid_plan_label\"",
            ])
            .forbidden(&[".to_string()", "format!(", "label.clone()"]),
        PatternCheck::new("J2K Metal batch profile caller", execute)
            .required(&[
                "struct PendingBatchProfile",
                "BatchMetadataBudget::new",
                ".try_vec(",
                "uniform_profile_value",
                "MetalBatchProfileValue",
                "emit_profile_error(\"metal_batch_profile_context\"",
            ])
            .forbidden(&[
                "fn profile_backend_label",
                "fn profile_format_label",
                ".to_string()",
                "format!(",
                "collect::<Vec",
            ]),
    ]);
}

#[test]
fn accelerator_profile_diagnostics_use_the_shared_emitter() {
    let callers = read_source_files(
        repo_root(),
        &[
            "crates/j2k-metal/src/routing.rs",
            "crates/j2k-jpeg-metal/src/lib.rs",
            "crates/j2k-jpeg-cuda/src/profile.rs",
        ],
    );
    let shared = read("crates/j2k-profile/src/emit.rs");

    assert_pattern_checks(&[
        PatternCheck::new("accelerator profile error diagnostics", &callers)
            .required(&[
                "emit_profile_error(\"metal_gpu_route_fields\"",
                "emit_profile_error(\"jpeg_metal_gpu_route_fields\"",
                "emit_profile_error(operation, &error)",
            ])
            .forbidden(&["j2k_profile_error operation="]),
        PatternCheck::new("shared profile error diagnostic", &shared).required(&[
            "pub fn emit_profile_error<E: std::fmt::Display + ?Sized>",
            "j2k_profile_error operation={operation} error={error}",
        ]),
    ]);
}

#[test]
fn profile_modules_stay_below_initial_focus_ratchets() {
    for (relative, limit) in [
        ("crates/j2k-profile/src/field.rs", 205),
        ("crates/j2k-profile/src/format.rs", 260),
        ("crates/j2k-profile/src/parse.rs", 150),
        ("crates/j2k-profile/src/timing.rs", 50),
        ("crates/j2k-profile/src/summary.rs", 375),
        ("crates/j2k-profile/src/summary/output.rs", 150),
        ("crates/j2k-profile/src/summary/record.rs", 625),
        ("crates/j2k-metal/src/profile.rs", 300),
        ("crates/j2k-metal/src/profile_env.rs", 325),
        ("crates/j2k-metal/src/profile_env/direct.rs", 250),
        ("crates/j2k-metal/src/compute/direct_profile.rs", 400),
    ] {
        let lines = read(relative).lines().count();
        assert!(lines < limit, "{relative} must stay below {limit} lines");
    }
}
