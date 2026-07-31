// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    bench_build, compile_benchmark_args, j2k_bench_signoff, j2k_ml_batch_bench_cuda,
    j2k_ml_batch_bench_metal, parse_bench_lane, BenchmarkLane, COMPILE_BENCHMARKS,
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

#[cfg(unix)]
#[test]
fn benchmark_build_lanes_never_compile_the_other_accelerator() {
    for (lane, expected_lines, required, forbidden) in [
        (
            "host",
            9,
            "j2k-ml --bench batch_decode --features cpu",
            "j2k-cuda",
        ),
        (
            "cuda",
            5,
            "j2k-ml --bench batch_decode_cuda --features cpu,cuda",
            "j2k-metal",
        ),
        (
            "metal",
            3,
            "j2k-ml --bench batch_decode_metal --features cpu,metal",
            "j2k-cuda",
        ),
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
