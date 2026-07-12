// SPDX-License-Identifier: MIT OR Apache-2.0

//! Non-fatal, bounded CUDA profile and trace emission contracts.

use std::fs;

use crate::repo_lint_support::{assert_pattern_checks, repo_root, PatternCheck};

fn read(relative: &str) -> String {
    fs::read_to_string(repo_root().join(relative))
        .unwrap_or_else(|error| panic!("read {relative}: {error}"))
}

#[test]
fn cuda_profile_failures_are_bounded_observable_and_non_fatal() {
    let facade = read("crates/j2k-cuda/src/profile.rs");
    let emit = read("crates/j2k-cuda/src/profile/emit.rs");
    let trace = read("crates/j2k-cuda/src/profile/trace.rs");
    let tests = read("crates/j2k-cuda/src/profile/tests.rs");
    let decoder_profile = read("crates/j2k-cuda/src/decoder/profile.rs");
    let decoder_idwt = read("crates/j2k-cuda/src/decoder/resident/idwt.rs");
    let stage = read("crates/j2k-cuda/src/encode/stage.rs");
    let runtime = read("crates/j2k-cuda/src/runtime.rs");
    let shared_emit = read("crates/j2k-profile/src/emit.rs");
    let shared_format = read("crates/j2k-profile/src/format.rs");
    let production_profile = format!("{facade}\n{emit}\n{trace}\n{decoder_profile}");

    assert_pattern_checks(&[
        PatternCheck::new("focused CUDA profile facade", &facade).required(&[
            "mod emit;",
            "mod trace;",
            "mod tests;",
            "emit::emit_htj2k_profile_row(path, self);",
            "trace::export_trace_if_requested(path, self);",
            "emit::emit_htj2k_encode_profile_row(path, self);",
            "trace::export_encode_trace_if_requested(path, self);",
        ]),
        PatternCheck::new("fallible CUDA profile field construction", &emit)
            .required(&[
                "ProfileField::metric(",
                "ProfileField::label(",
                "fn build_profile_fields<const N: usize>(",
                "j2k_profile::emit_profile_error(operation, &error)",
                "j2k_profile::emit_profile_fields(",
            ])
            .forbidden(&[".to_string()", "format!(", "emit_profile_row("]),
        PatternCheck::new("bounded fallible CUDA trace formatting", &trace)
            .required(&[
                ") -> ProfileResult<String>",
                "ProfileLimits::default()",
                ".try_reserve_exact(",
                "self.output.capacity()",
                "ProfileError::AllocationFailed",
                "ProfileError::LimitExceeded",
                "j2k_profile::emit_profile_error(\"cuda_htj2k_trace_format\"",
                "emit_trace_write_error(\"cuda_htj2k_trace_write\"",
                "j2k_profile::emit_profile_error(\"cuda_htj2k_encode_trace_format\"",
                "emit_trace_write_error(\"cuda_htj2k_encode_trace_write\"",
                "fn emit_trace_write_error(",
                "j2k_profile::emit_profile_error(",
                "create_new(true)",
                "'\\n' => self.write_fragment(format_args!(\"\\\\n\"))?",
            ])
            .forbidden(&[
                "String::from(",
                ".to_string()",
                "format!(",
                ".expect(",
                "std::fs::write(",
            ]),
        PatternCheck::new("CUDA route profile callers", &format!("{stage}\n{runtime}")).required(
            &[
                "emit_optional_gpu_route_fields(",
                "ProfileField::label(",
                "ProfileField::metric(",
            ],
        ),
        PatternCheck::new("bounded CUDA IDWT host trace", &decoder_profile)
            .required(&[
                ") -> j2k_profile::ProfileResult<String>",
                "j2k_profile::format_profile_row_u128(",
                "fn emit_cuda_idwt_batch_host_trace_row(",
                "j2k_profile::emit_profile_line(row)",
                "j2k_profile::emit_profile_error(\"cuda_idwt_batch_host_trace\", &error)",
            ])
            .forbidden(&["format!(", ".to_string()"]),
        PatternCheck::new("non-fatal CUDA IDWT trace caller", &decoder_idwt)
            .required(&["emit_cuda_idwt_batch_host_trace_row(row);"])
            .forbidden(&["eprintln!(\"{}\", format_cuda_idwt_batch_host_trace_row"]),
        PatternCheck::new("shared bounded integer row formatter", &shared_format).required(&[
            "pub fn format_profile_row_u128<K>(",
            "ProfileLimits::default()",
            "try_output_string(",
        ]),
        PatternCheck::new("shared non-fatal profile sinks", &shared_emit).required(&[
            "emit_formatted(",
            "if let Err(error) = summary.borrow_mut().record_str",
            "if let Err(error) = summary.borrow_mut().record_fields",
            "emit_profile_error(\"summary_record\", &error)",
            "emit_profile_error(\"typed_summary_record\", &error)",
            "pub fn emit_profile_error<E: std::fmt::Display + ?Sized>(operation: &str, error: &E)",
        ]),
        PatternCheck::new("CUDA profile behavior regressions", &tests).required(&[
            "profile_field_build_failure_is_diagnostic_only",
            "trace_categories_are_json_escaped_and_bounded",
            "trace_file_write_refuses_to_overwrite_existing_file",
        ]),
        PatternCheck::new("CUDA profile lint integrity", &production_profile)
            .forbidden(&["#[allow(", "#[expect("]),
    ]);

    assert_focus_ratchets(&facade, &emit, &trace, &tests, &decoder_profile);
}

fn assert_focus_ratchets(
    facade: &str,
    emit: &str,
    trace: &str,
    tests: &str,
    decoder_profile: &str,
) {
    for (relative, source, limit) in [
        ("crates/j2k-cuda/src/profile.rs", facade, 225),
        ("crates/j2k-cuda/src/profile/emit.rs", emit, 225),
        ("crates/j2k-cuda/src/profile/trace.rs", trace, 275),
        ("crates/j2k-cuda/src/profile/tests.rs", tests, 225),
        (
            "crates/j2k-cuda/src/decoder/profile.rs",
            decoder_profile,
            250,
        ),
    ] {
        assert!(
            source.lines().count() < limit,
            "{relative} must remain below its {limit}-line focus ratchet"
        );
    }
}
