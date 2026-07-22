// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    bench_build, bench_report, best_version_line, j2k_bench_signoff, j2k_ml_batch_bench_cuda,
    j2k_ml_batch_bench_metal, one_line, render_benchmark_report, split_semicolon_list,
    transcode_metal_bench_args, BenchmarkReport,
};

#[cfg(unix)]
use crate::{command_support::use_test_cargo_program, test_command::RecordingProgram};

#[test]
fn transcode_metal_bench_enables_its_declared_internal_surface() {
    assert_eq!(
        transcode_metal_bench_args(),
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
}

#[cfg(unix)]
#[test]
fn benchmark_build_and_signoff_execute_the_complete_fake_cargo_plan() {
    let recording = RecordingProgram::new(
        "benchmark-command-test",
        "if [ \"$1\" = test ]; then printf 'test result: ok. 100 passed; 0 failed;\\n'; fi",
    );
    let _cargo = use_test_cargo_program(recording.program().as_os_str().to_owned());

    bench_build().expect("benchmark build plan");
    j2k_bench_signoff().expect("benchmark signoff plan");

    let log = recording.log();
    assert!(log.contains("bench -p j2k --bench public_api --no-run|"));
    assert!(log.contains(
        "bench -p j2k-transcode-metal --bench dct97 --features bench-internals --no-run|"
    ));
    assert!(log.contains("bench -p j2k-ml --bench batch_decode --features cpu --no-run|"));
    assert!(
        log.contains("bench -p j2k-ml --bench batch_decode_metal --features cpu,metal --no-run|")
    );
    assert!(log.contains("bench -p j2k-ml --bench batch_decode_cuda --features cpu,cuda --no-run|"));
    assert!(log.contains("test -p j2k-compare --test in_process_parity -- --nocapture|"));
    assert!(log.contains("test -p j2k-jpeg --features bench-libjpeg-turbo --test libjpeg_turbo_compare -- --nocapture|"));
    assert_eq!(log.lines().count(), 21);
}

#[cfg(unix)]
#[test]
fn accelerator_batch_benchmark_commands_select_one_explicit_backend() {
    let recording = RecordingProgram::new("j2k-ml-benchmark-command-test", "");
    let _cargo = use_test_cargo_program(recording.program().as_os_str().to_owned());

    j2k_ml_batch_bench_metal().expect("Metal batch benchmark command");
    j2k_ml_batch_bench_cuda().expect("CUDA batch benchmark command");

    let log = recording.log();
    let lines = log.lines().collect::<Vec<_>>();
    assert_eq!(lines.len(), 2);
    assert!(
        lines[0].starts_with("bench -p j2k-ml --bench batch_decode_metal --features cpu,metal|")
    );
    assert!(lines[1].starts_with("bench -p j2k-ml --bench batch_decode_cuda --features cpu,cuda|"));
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
