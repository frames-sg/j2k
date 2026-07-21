// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fail-closed release validation for the CUDA adapters.

use std::{collections::BTreeSet, env, ffi::OsString};

use crate::process::{self, cargo, CommandContext};

const CUDA_RELEASE_ENV: &[(&str, &str)] = &[
    ("J2K_REQUIRE_CUDA_RUNTIME", "1"),
    ("J2K_REQUIRE_CUDA_OXIDE_BUILD", "1"),
    ("J2K_REQUIRE_CUDA_JPEG_HARDWARE_DECODE", "1"),
    ("RUST_TEST_THREADS", "1"),
];

const CUDA_RUNTIME_TEST_TARGETS: &[&str] = &["--lib", "--bins", "--tests", "--examples"];

const GPU_TEST_SKIP_MARKER: &str = "J2K_GPU_TEST_SKIPPED";

const CUDA_SKIP_MARKERS: &[&str] = &[
    "skipping cuda",
    "cuda runtime is unavailable",
    "cuda device/runtime is unavailable",
    "reason=device-unavailable",
];

const HTJ2K_ENCODE_PARITY_TESTS: &[&str] = &[
    "cuda_deinterleave_matches_native_reference_when_required",
    "cuda_facade_byte_matches_native_across_matrix_when_required",
    "cuda_forward_rct_matches_native_reference_when_required",
    "cuda_htj2k_tile_encode_hook_rejects_subsampling_with_typed_err_when_cuda_runtime_required",
    "cuda_quantize_reversible_matches_native_reference_when_required",
    "cuda_runtime_required_implies_feature_compiled",
    "lossless_facade_in_scope_input_never_hits_ok_none_fallback",
];

const TRANSCODE_PARITY_TESTS: &[&str] = &[
    "cuda_dwt97_batch_matches_per_job_and_reports_stage_timings_when_required",
    "cuda_dwt97_batch_non_uniform_geometry_falls_back_to_per_job_when_required",
    "cuda_dwt97_matches_scalar_oracle_within_tolerance_when_required",
    "cuda_htj2k97_codeblock_batch_matches_oracle_when_required",
    "cuda_htj2k97_codeblock_batch_rejects_non_uniform_geometry_when_required",
    "cuda_resident_htj2k97_batch_matches_host_bounce_codestream_when_required",
    "cuda_reversible_dwt53_matches_scalar_oracle_when_required",
    "ycbcr_420_batch_transcodes_to_htj2k_with_explicit_cuda_97_codeblock_path",
];

const ML_CUDA_TESTS: &[&str] = &[
    "burn_direct_session_reuses_events_and_codec_memory_for_one_thousand_batches",
    "cuda_burn_batch_continues_after_one_group_submit_failure",
    "cuda_burn_regroups_prepared_images_and_keeps_settings_failures_indexed_without_cuda",
    "direct_cuda_batch_writes_exact_u8_pixels_and_reuses_the_session",
    "direct_cuda_burn_rgba_matches_cpu_across_codecs_types_geometry_and_layouts",
    "direct_cuda_preserves_native_u16_and_i16_samples",
    "direct_cuda_rgb_preserves_subnative_codes_and_burn_layout",
    "direct_cuda_signed_rgb_matches_cpu_for_geometry_and_burn_layout",
    "direct_cuda_supports_roi_and_reduction_without_host_staging",
    "dropping_submitted_burn_batch_retires_cuda_work_and_keeps_session_reusable",
    "empty_cuda_batch_uses_the_persistent_shared_codec_contract_without_initializing_work",
];

struct CudaRuntimeSuite {
    label: &'static str,
    package: &'static str,
    features: &'static str,
}

const CUDA_RUNTIME_SUITES: &[CudaRuntimeSuite] = &[
    CudaRuntimeSuite {
        label: "CUDA Oxide runtime",
        package: "j2k-cuda-runtime",
        features: "cuda-oxide",
    },
    CudaRuntimeSuite {
        label: "profiled CUDA Oxide runtime",
        package: "j2k-cuda-runtime",
        features: "cuda-profiling,cuda-oxide",
    },
    CudaRuntimeSuite {
        label: "JPEG CUDA adapter",
        package: "j2k-jpeg-cuda",
        features: "cuda-runtime",
    },
    CudaRuntimeSuite {
        label: "J2K CUDA adapter",
        package: "j2k-cuda",
        features: "cuda-runtime",
    },
    CudaRuntimeSuite {
        label: "transcode CUDA adapter",
        package: "j2k-transcode-cuda",
        features: "cuda-runtime",
    },
    CudaRuntimeSuite {
        label: "profiled J2K CUDA adapter",
        package: "j2k-cuda",
        features: "cuda-profiling",
    },
];

struct CudaClippySuite {
    package: &'static str,
    features: &'static str,
}

const CUDA_CLIPPY_SUITES: &[CudaClippySuite] = &[
    CudaClippySuite {
        package: "j2k-cuda-runtime",
        features: "cuda-oxide",
    },
    CudaClippySuite {
        package: "j2k-cuda-runtime",
        features: "cuda-profiling,cuda-oxide",
    },
    CudaClippySuite {
        package: "j2k-jpeg-cuda",
        features: "cuda-runtime",
    },
    CudaClippySuite {
        package: "j2k-cuda",
        features: "cuda-runtime",
    },
    CudaClippySuite {
        package: "j2k-transcode-cuda",
        features: "cuda-runtime",
    },
    CudaClippySuite {
        package: "j2k-cuda",
        features: "cuda-profiling",
    },
    CudaClippySuite {
        package: "j2k-ml",
        features: "cuda",
    },
];

struct ExactCudaSuite {
    label: &'static str,
    package: &'static str,
    features: &'static str,
    test_targets: &'static [&'static str],
    required_tests: &'static [&'static str],
}

const EXACT_CUDA_SUITES: &[ExactCudaSuite] = &[
    ExactCudaSuite {
        label: "Burn J2K CUDA direct tensor integration",
        package: "j2k-ml",
        features: "cuda",
        test_targets: &["cuda", "cuda_rgba", "cuda_batch_sessions"],
        required_tests: ML_CUDA_TESTS,
    },
    ExactCudaSuite {
        label: "HTJ2K encode CUDA parity inventory",
        package: "j2k-cuda",
        features: "cuda-runtime",
        test_targets: &["htj2k_encode_parity"],
        required_tests: HTJ2K_ENCODE_PARITY_TESTS,
    },
    ExactCudaSuite {
        label: "JPEG-to-HTJ2K CUDA transcode parity inventory",
        package: "j2k-transcode-cuda",
        features: "cuda-runtime",
        test_targets: &[
            "reversible_dwt53_parity",
            "dwt97_parity",
            "dwt97_batch_parity",
            "htj2k97_codeblock_parity",
            "jpeg_to_htj2k",
        ],
        required_tests: TRANSCODE_PARITY_TESTS,
    },
];

/// Runs the complete release-mode CUDA validation policy on a real Linux `x86_64` device.
pub(crate) fn release_cuda() -> Result<(), String> {
    run_release_cuda(env::consts::OS, env::consts::ARCH)
}

fn run_release_cuda(os: &str, arch: &str) -> Result<(), String> {
    require_cuda_host(os, arch)?;
    require_cuda_device()?;

    for suite in CUDA_CLIPPY_SUITES {
        let args = clippy_suite_args(suite);
        let label = format!("{} CUDA Clippy", suite.package);
        let output = run_cargo_captured(&args, CUDA_RELEASE_ENV, &label)?;
        reject_cuda_skip_markers(&output, &label)?;
    }

    for suite in CUDA_RUNTIME_SUITES {
        let args = runtime_suite_args(suite);
        let output = run_cargo_captured(&args, CUDA_RELEASE_ENV, suite.label)?;
        validate_complete_test_run(&output, suite.label)?;
    }

    for suite in EXACT_CUDA_SUITES {
        validate_exact_inventory(suite)?;
        let args = exact_suite_args(suite, false);
        let output = run_cargo_captured(&args, CUDA_RELEASE_ENV, suite.label)?;
        validate_exact_named_run(&output, suite.label, suite.required_tests)?;
    }

    Ok(())
}

fn require_cuda_host(os: &str, arch: &str) -> Result<(), String> {
    if os == "linux" && arch == "x86_64" {
        Ok(())
    } else {
        Err(format!(
            "release-cuda requires Linux x86_64, but this host is {os}/{arch}; refusing to report CUDA release success on an unaudited platform"
        ))
    }
}

fn require_cuda_device() -> Result<(), String> {
    const ARGS: &[&str] = &["--query-gpu=index,uuid", "--format=csv,noheader,nounits"];
    eprintln!("+ nvidia-smi {}", ARGS.join(" "));
    #[cfg(all(test, unix))]
    let program: OsString = test_support::nvidia_smi_program();
    #[cfg(not(all(test, unix)))]
    let program: OsString = OsString::from("nvidia-smi");
    let output = process::command_output(program, ARGS, CommandContext::new())?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    print!("{stdout}");
    eprint!("{stderr}");
    if !output.status.success() {
        return Err(format!(
            "`nvidia-smi {}` exited with {}; a working CUDA driver and real device are required",
            ARGS.join(" "),
            output.status
        ));
    }
    validate_cuda_device_probe(&format!("{stdout}\n{stderr}"))
}

fn validate_cuda_device_probe(output: &str) -> Result<(), String> {
    reject_cuda_skip_markers(output, "CUDA device probe")?;
    let device_rows = output
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .filter(|line| {
            let mut fields = line.split(',').map(str::trim);
            matches!(fields.next(), Some(index) if index.parse::<usize>().is_ok())
                && matches!(fields.next(), Some(uuid) if uuid.starts_with("GPU-") && uuid.len() > 4)
                && fields.next().is_none()
        })
        .count();
    if device_rows == 0 {
        Err(
            "nvidia-smi reported no parseable CUDA devices; refusing to run release validation"
                .to_string(),
        )
    } else {
        eprintln!("CUDA release validation found {device_rows} device(s)");
        Ok(())
    }
}

fn clippy_suite_args(suite: &CudaClippySuite) -> Vec<&'static str> {
    vec![
        "clippy",
        "--all-targets",
        "-p",
        suite.package,
        "--features",
        suite.features,
        "--",
        "-D",
        "warnings",
    ]
}

fn runtime_suite_args(suite: &CudaRuntimeSuite) -> Vec<&'static str> {
    let mut args = vec![
        "test",
        "--release",
        "-p",
        suite.package,
        "--features",
        suite.features,
    ];
    args.extend_from_slice(CUDA_RUNTIME_TEST_TARGETS);
    args.extend_from_slice(&["--", "--show-output"]);
    args
}

fn exact_suite_args(suite: &ExactCudaSuite, list_only: bool) -> Vec<&'static str> {
    let mut args = vec![
        "test",
        "--release",
        "-p",
        suite.package,
        "--features",
        suite.features,
    ];
    for target in suite.test_targets {
        args.extend_from_slice(&["--test", target]);
    }
    args.push("--");
    args.push(if list_only { "--list" } else { "--show-output" });
    args
}

fn validate_exact_inventory(suite: &ExactCudaSuite) -> Result<(), String> {
    let args = exact_suite_args(suite, true);
    let label = format!("list {}", suite.label);
    let output = run_cargo_captured(&args, CUDA_RELEASE_ENV, &label)?;
    reject_cuda_skip_markers(&output, &label)?;
    let actual = listed_rust_tests(&output);
    let expected = expected_test_set(suite.required_tests);
    compare_named_inventory(&actual, &expected, suite.label, "inventory")
}

fn validate_complete_test_run(output: &str, label: &str) -> Result<(), String> {
    reject_cuda_skip_markers(output, label)?;
    let summary_lines = output
        .lines()
        .filter(|line| line.trim().starts_with("test result:"))
        .collect::<Vec<_>>();
    let summaries = successful_test_summaries(output);
    if summary_lines.is_empty() {
        return Err(format!(
            "{label} emitted no successful Rust test summary; the suite may be missing or incomplete"
        ));
    }
    if summaries.len() != summary_lines.len() {
        return Err(format!(
            "{label} emitted an unsuccessful or unrecognized Rust test summary; the suite may have failed, been cancelled, or produced incomplete evidence"
        ));
    }

    let total = summaries
        .iter()
        .copied()
        .fold(TestSummary::default(), TestSummary::add);
    if total.failed != 0 || total.ignored != 0 || total.measured != 0 || total.filtered_out != 0 {
        return Err(format!(
            "{label} was partial: {} passed, {} failed, {} ignored, {} measured, {} filtered out",
            total.passed, total.failed, total.ignored, total.measured, total.filtered_out
        ));
    }
    if total.passed == 0 {
        return Err(format!(
            "{label} passed zero tests; refusing false-green CUDA release evidence"
        ));
    }
    Ok(())
}

fn validate_exact_named_run(
    output: &str,
    label: &str,
    required_tests: &[&str],
) -> Result<(), String> {
    validate_complete_test_run(output, label)?;
    let expected = expected_test_set(required_tests);
    let actual = passed_rust_tests(output);
    compare_named_inventory(&actual, &expected, label, "passed-test")?;

    let passed = successful_test_summaries(output)
        .iter()
        .map(|summary| summary.passed)
        .sum::<usize>();
    if passed != expected.len() {
        return Err(format!(
            "{label} passed {passed} tests, expected exactly {} named tests",
            expected.len()
        ));
    }
    Ok(())
}

fn expected_test_set(tests: &[&str]) -> BTreeSet<String> {
    tests.iter().map(|test| (*test).to_string()).collect()
}

fn compare_named_inventory(
    actual: &BTreeSet<String>,
    expected: &BTreeSet<String>,
    label: &str,
    kind: &str,
) -> Result<(), String> {
    if actual == expected {
        return Ok(());
    }
    let missing = expected.difference(actual).cloned().collect::<Vec<_>>();
    let unexpected = actual.difference(expected).cloned().collect::<Vec<_>>();
    Err(format!(
        "{label} {kind} mismatch; missing: {missing:?}; unexpected: {unexpected:?}"
    ))
}

fn run_cargo_captured(args: &[&str], envs: &[(&str, &str)], label: &str) -> Result<String, String> {
    let env_display = envs
        .iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join(" ");
    eprintln!(
        "+ {env_display} {} {}",
        cargo().to_string_lossy(),
        args.join(" ")
    );
    let output = process::command_output(cargo(), args, CommandContext::new().envs(envs))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    print!("{stdout}");
    eprint!("{stderr}");
    if !output.status.success() {
        return Err(format!(
            "`{}` exited with {} while running {label}",
            cargo().to_string_lossy(),
            output.status
        ));
    }
    Ok(format!("{stdout}\n{stderr}"))
}

fn reject_cuda_skip_markers(output: &str, label: &str) -> Result<(), String> {
    let lowercase = output.to_ascii_lowercase();
    let marker = if output.contains(GPU_TEST_SKIP_MARKER) {
        Some(GPU_TEST_SKIP_MARKER)
    } else {
        CUDA_SKIP_MARKERS
            .iter()
            .find(|marker| lowercase.contains(**marker))
            .copied()
    };
    if let Some(marker) = marker {
        Err(format!(
            "{label} emitted CUDA skip evidence `{marker}`; release validation must fail rather than skip"
        ))
    } else {
        Ok(())
    }
}

fn listed_rust_tests(output: &str) -> BTreeSet<String> {
    output
        .lines()
        .filter_map(|line| line.trim().strip_suffix(": test"))
        .map(str::to_string)
        .collect()
}

fn passed_rust_tests(output: &str) -> BTreeSet<String> {
    output
        .lines()
        .filter_map(|line| {
            let rest = line.trim().strip_prefix("test ")?;
            rest.strip_suffix(" ... ok").map(str::to_string)
        })
        .collect()
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct TestSummary {
    passed: usize,
    failed: usize,
    ignored: usize,
    measured: usize,
    filtered_out: usize,
}

impl TestSummary {
    fn parse(line: &str) -> Option<Self> {
        let counts = line.trim().strip_prefix("test result: ok.")?;
        let mut fields = counts.split(';').map(str::trim);
        let passed = parse_count(fields.next()?, "passed")?;
        let failed = parse_count(fields.next()?, "failed")?;
        let ignored = parse_count(fields.next()?, "ignored")?;
        let measured = parse_count(fields.next()?, "measured")?;
        let filtered_out = parse_count(fields.next()?, "filtered out")?;
        if fields
            .next()
            .is_some_and(|timing| !timing.starts_with("finished in "))
            || fields.next().is_some()
        {
            return None;
        }
        Some(Self {
            passed,
            failed,
            ignored,
            measured,
            filtered_out,
        })
    }

    const fn add(self, other: Self) -> Self {
        Self {
            passed: self.passed + other.passed,
            failed: self.failed + other.failed,
            ignored: self.ignored + other.ignored,
            measured: self.measured + other.measured,
            filtered_out: self.filtered_out + other.filtered_out,
        }
    }
}

fn parse_count(field: &str, suffix: &str) -> Option<usize> {
    field
        .strip_suffix(suffix)?
        .trim()
        .trim_end_matches('.')
        .parse()
        .ok()
}

fn successful_test_summaries(output: &str) -> Vec<TestSummary> {
    output.lines().filter_map(TestSummary::parse).collect()
}

#[cfg(all(test, unix))]
mod test_support;

#[cfg(test)]
mod tests;
