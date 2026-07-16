// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    bench_build, bench_report, best_version_line, compile_benchmark_args, j2k_bench_signoff,
    one_line, parse_bench_lane, render_benchmark_report, split_semicolon_list, BenchmarkLane,
    BenchmarkReport, COMPILE_BENCHMARKS,
};

#[cfg(unix)]
use crate::{command_support::use_test_cargo_program, test_command::RecordingProgram};

#[test]
fn shared_registry_declares_transcode_metal_features_and_runtime_gate() {
    let benchmark = COMPILE_BENCHMARKS
        .iter()
        .find(|benchmark| benchmark.package == "j2k-transcode-metal")
        .copied()
        .expect("transcode Metal benchmark registry entry");
    assert_eq!(
        compile_benchmark_args(benchmark),
        [
            "bench",
            "-p",
            "j2k-transcode-metal",
            "--bench",
            "dct97",
            "--features",
            "bench-internals",
            "--no-run",
        ]
    );
    assert!(benchmark
        .runtime_env
        .contains(&("J2K_REQUIRE_METAL_RUNTIME", "1")));
}

#[cfg(unix)]
#[test]
fn benchmark_build_and_signoff_execute_the_complete_fake_cargo_plan() {
    let recording = RecordingProgram::new(
        "benchmark-command-test",
        "if [ \"$1\" = test ]; then printf 'test result: ok. 100 passed; 0 failed;\\n'; fi",
    );
    let _cargo = use_test_cargo_program(recording.program().as_os_str().to_owned());

    bench_build(std::iter::empty()).expect("benchmark build plan");
    j2k_bench_signoff().expect("benchmark signoff plan");

    let log = recording.log();
    assert!(log.contains("bench -p j2k --bench public_api --no-run|"));
    assert!(log.contains(
        "bench -p j2k-transcode-metal --bench dct97 --features bench-internals --no-run|"
    ));
    assert!(log.contains("bench -p j2k-ml --bench tensor_decode --features cpu --no-run|"));
    assert!(log.contains("test -p j2k-compare --test in_process_parity -- --nocapture|"));
    assert!(log.contains("test -p j2k-jpeg --features bench-libjpeg-turbo --test libjpeg_turbo_compare -- --nocapture|"));
    assert_eq!(log.lines().count(), 19);
}

#[cfg(unix)]
#[test]
fn benchmark_build_lanes_never_compile_the_other_accelerator() {
    for (lane, expected_lines, required, forbidden) in [
        (
            "host",
            9,
            "j2k-ml --bench tensor_decode --features cpu",
            "j2k-cuda",
        ),
        ("cuda", 4, "j2k-cuda --bench htj2k_decode", "j2k-metal"),
        ("metal", 2, "j2k-jpeg-metal", "j2k-cuda"),
    ] {
        let recording = RecordingProgram::new("benchmark-lane-test", "");
        let _cargo = use_test_cargo_program(recording.program().as_os_str().to_owned());

        bench_build(["--lane".to_string(), lane.to_string()].into_iter())
            .expect("lane benchmark plan");

        let log = recording.log();
        assert_eq!(log.lines().count(), expected_lines, "lane {lane}: {log}");
        assert!(log.contains(required), "lane {lane}: {log}");
        assert!(!log.contains(forbidden), "lane {lane}: {log}");
    }
}

#[test]
fn benchmark_lane_parser_defaults_all_and_rejects_invalid_input() {
    assert_eq!(
        parse_bench_lane(std::iter::empty()).unwrap(),
        BenchmarkLane::All
    );
    assert_eq!(
        parse_bench_lane(["--lane".to_string(), "metal".to_string()].into_iter()).unwrap(),
        BenchmarkLane::Metal
    );
    assert!(parse_bench_lane(["--lane".to_string()].into_iter()).is_err());
    assert!(parse_bench_lane(["--lane".to_string(), "other".to_string()].into_iter()).is_err());
}

#[test]
fn benchmark_report_renderer_preserves_provenance_comparators_and_skips() {
    let report = BenchmarkReport {
        command: "cargo bench --all".to_string(),
        host: "test-host".to_string(),
        rustc: "rustc 1.96\nverbose details".to_string(),
        cargo: "cargo 1.96\nverbose details".to_string(),
        git_revision: "0123456789abcdef".to_string(),
        workspace_version: "0.7.0".to_string(),
        input_source: "external fixtures".to_string(),
        compare_threads: "4".to_string(),
        comparator_versions: vec![
            ("OpenJPEG".to_string(), "openjp2 2.5.3".to_string()),
            ("Grok".to_string(), "grok 15.0".to_string()),
        ],
        skipped_rows: vec!["unsupported alpha".to_string(), "missing codec".to_string()],
    };

    let rendered = render_benchmark_report(&report);

    assert!(rendered.starts_with("# Benchmark publication report\n\n"));
    assert!(rendered.contains("- command: cargo bench --all"));
    assert!(rendered.contains("- rustc: rustc 1.96\n"));
    assert!(!rendered.contains("verbose details"));
    assert!(rendered.contains("- OpenJPEG: openjp2 2.5.3"));
    assert!(rendered.contains("- Grok: grok 15.0"));
    assert!(rendered.contains("## skipped rows\n- unsupported alpha\n- missing codec\n"));
}

#[test]
fn benchmark_report_renderer_marks_an_empty_skip_inventory() {
    let report = BenchmarkReport {
        command: "not recorded".to_string(),
        host: "host".to_string(),
        rustc: String::new(),
        cargo: String::new(),
        git_revision: "revision".to_string(),
        workspace_version: "0.7.0".to_string(),
        input_source: "not recorded".to_string(),
        compare_threads: "not set".to_string(),
        comparator_versions: Vec::new(),
        skipped_rows: Vec::new(),
    };

    let rendered = render_benchmark_report(&report);

    assert!(rendered.contains("## comparator versions\n\n## skipped rows\n- none recorded\n"));
}

#[test]
fn benchmark_text_parsers_are_ordered_trimmed_and_fail_closed() {
    assert_eq!(
        split_semicolon_list(" first ; ;second; third "),
        ["first", "second", "third"]
    );
    assert!(split_semicolon_list(" ; ").is_empty());
    assert_eq!(one_line("first\nsecond"), "first");
    assert_eq!(one_line(""), "");

    assert_eq!(
        best_version_line("banner\ncompiled against OpenJPEG 2.5\nversion 9"),
        "compiled against OpenJPEG 2.5"
    );
    assert_eq!(
        best_version_line("\nfirst nonempty\nsecond"),
        "first nonempty"
    );
    assert_eq!(best_version_line("\n\n"), "version unavailable");
}

#[test]
fn benchmark_report_arguments_reject_unknown_and_missing_values_before_probing_tools() {
    for (args, expected) in [
        (vec!["--unknown"], "unknown bench-report argument"),
        (vec!["--command"], "--command requires a value"),
        (vec!["--input-source"], "--input-source requires a value"),
        (vec!["--skipped-row"], "--skipped-row requires a value"),
        (vec!["--out"], "--out requires a value"),
    ] {
        let error = bench_report(args.into_iter().map(str::to_string))
            .expect_err("invalid bench-report arguments");
        assert!(error.contains(expected), "unexpected error: {error}");
    }
    assert_eq!(bench_report(["--help".to_string()].into_iter()), Ok(()));
}
