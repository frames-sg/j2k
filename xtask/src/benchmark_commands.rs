use crate::benchmark_registry::{BenchmarkLane, CompileBenchmark, COMPILE_BENCHMARKS};
use crate::command_support::{run_cargo, run_cargo_test_with_pass_floor};

pub(super) fn bench_build(args: impl Iterator<Item = String>) -> Result<(), String> {
    let lane = parse_bench_lane(args)?;
    for benchmark in COMPILE_BENCHMARKS
        .iter()
        .filter(|benchmark| lane.selects(benchmark.lane))
    {
        run_cargo(&compile_benchmark_args(*benchmark))?;
    }
    Ok(())
}

fn parse_bench_lane(mut args: impl Iterator<Item = String>) -> Result<BenchmarkLane, String> {
    let Some(argument) = args.next() else {
        return Ok(BenchmarkLane::All);
    };
    if argument != "--lane" {
        return Err(format!(
            "unknown bench-build argument `{argument}`; expected --lane host|cuda|metal|all"
        ));
    }
    let value = args
        .next()
        .ok_or_else(|| "--lane requires host, cuda, metal, or all".to_string())?;
    if let Some(extra) = args.next() {
        return Err(format!("unexpected bench-build argument `{extra}`"));
    }
    BenchmarkLane::parse(&value)
}

pub(super) fn j2k_ml_batch_bench_metal() -> Result<(), String> {
    run_cargo(&[
        "bench",
        "-p",
        "j2k-ml",
        "--bench",
        "batch_decode_metal",
        "--features",
        "cpu,metal",
    ])
}

pub(super) fn j2k_ml_batch_bench_cuda() -> Result<(), String> {
    run_cargo(&[
        "bench",
        "-p",
        "j2k-ml",
        "--bench",
        "batch_decode_cuda",
        "--features",
        "cpu,cuda",
    ])
}

fn compile_benchmark_args(benchmark: CompileBenchmark) -> Vec<&'static str> {
    let mut args = vec!["bench", "-p", benchmark.package];
    if let Some(bench) = benchmark.bench {
        args.extend_from_slice(&["--bench", bench]);
    }
    if let Some(features) = benchmark.features {
        args.extend_from_slice(&["--features", features]);
    }
    args.push("--no-run");
    args
}

pub(super) fn j2k_bench_signoff() -> Result<(), String> {
    run_cargo_test_with_pass_floor(
        &["test", "-p", "j2k-compare", "--test", "in_process_parity"],
        &[("J2K_REQUIRE_OPENJPEG", "1"), ("J2K_REQUIRE_GROK", "1")],
        8,
        "in-process OpenJPEG/Grok parity",
    )?;
    run_cargo_test_with_pass_floor(
        &["test", "-p", "j2k", "--test", "openjpeg_parity"],
        &[("J2K_REQUIRE_OPENJPEG", "1")],
        7,
        "OpenJPEG CLI parity",
    )?;
    run_cargo_test_with_pass_floor(
        &["test", "-p", "j2k", "--test", "grok_parity"],
        &[("J2K_REQUIRE_GROK", "1")],
        12,
        "Grok CLI parity",
    )?;
    run_cargo_test_with_pass_floor(
        &[
            "test",
            "-p",
            "j2k-jpeg",
            "--features",
            "bench-libjpeg-turbo",
            "--test",
            "libjpeg_turbo_compare",
        ],
        &[("J2K_REQUIRE_LIBJPEG_TURBO", "1")],
        1,
        "libjpeg-turbo JPEG parity",
    )
}

#[cfg(test)]
mod tests;
