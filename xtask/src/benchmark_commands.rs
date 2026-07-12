use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

use crate::command_support::{
    command_output, command_output_allow_failure, command_output_os, run_cargo,
    run_cargo_test_with_pass_floor, workspace_version,
};
use crate::process::cargo;

#[expect(
    clippy::too_many_lines,
    reason = "the benchmark build command intentionally lists the complete fail-fast benchmark matrix"
)]
pub(super) fn bench_build() -> Result<(), String> {
    run_cargo(&["bench", "-p", "j2k", "--bench", "public_api", "--no-run"])?;
    run_cargo(&[
        "bench",
        "-p",
        "j2k-native",
        "--bench",
        "tier1_bitplane",
        "--no-run",
    ])?;
    run_cargo(&[
        "bench",
        "-p",
        "j2k-native",
        "--bench",
        "htj2k_sigprop_phase",
        "--no-run",
    ])?;
    run_cargo(&[
        "bench",
        "-p",
        "j2k-native",
        "--bench",
        "direct_cpu",
        "--no-run",
    ])?;
    run_cargo(&[
        "bench",
        "-p",
        "j2k-jpeg",
        "--bench",
        "encode_cpu",
        "--no-run",
    ])?;
    run_cargo(&[
        "bench",
        "-p",
        "j2k-jpeg",
        "--features",
        "bench-libjpeg-turbo",
        "--no-run",
    ])?;
    run_cargo(&["bench", "-p", "j2k-jpeg-metal", "--no-run"])?;
    run_cargo(&[
        "bench",
        "-p",
        "j2k-jpeg-cuda",
        "--bench",
        "device_decode",
        "--features",
        "cuda-runtime",
        "--no-run",
    ])?;
    run_cargo(&[
        "bench",
        "-p",
        "j2k-cuda",
        "--bench",
        "encode_stages",
        "--features",
        "cuda-runtime",
        "--no-run",
    ])?;
    run_cargo(&[
        "bench",
        "-p",
        "j2k-cuda",
        "--bench",
        "htj2k_decode",
        "--features",
        "cuda-runtime",
        "--no-run",
    ])?;
    run_cargo(&[
        "bench",
        "-p",
        "j2k-cuda",
        "--bench",
        "htj2k_encode",
        "--features",
        "cuda-runtime",
        "--no-run",
    ])?;
    run_cargo(&[
        "bench",
        "-p",
        "j2k-tilecodec",
        "--bench",
        "compare",
        "--no-run",
    ])?;
    run_cargo(&[
        "bench",
        "-p",
        "j2k-transcode",
        "--bench",
        "dct53",
        "--no-run",
    ])?;
    run_cargo(transcode_metal_bench_args())
}

fn transcode_metal_bench_args() -> &'static [&'static str] {
    &[
        "bench",
        "-p",
        "j2k-transcode-metal",
        "--bench",
        "dct97",
        "--features",
        "bench-internals",
        "--no-run",
    ]
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

#[derive(Debug)]
struct BenchmarkReport {
    command: String,
    host: String,
    rustc: String,
    cargo: String,
    git_revision: String,
    workspace_version: String,
    input_source: String,
    compare_threads: String,
    comparator_versions: Vec<(String, String)>,
    skipped_rows: Vec<String>,
}

pub(super) fn bench_report(args: impl Iterator<Item = String>) -> Result<(), String> {
    let mut command = env::var("J2K_BENCH_COMMAND").unwrap_or_else(|_| "not recorded".into());
    let mut input_source = env::var("J2K_BENCH_INPUT_SOURCE")
        .or_else(|_| env::var("J2K_BENCH_INPUTS"))
        .unwrap_or_else(|_| "not recorded".into());
    let mut out_path = None::<PathBuf>;
    let mut skipped_rows = env::var("J2K_BENCH_SKIPPED_ROWS")
        .ok()
        .map(|rows| split_semicolon_list(&rows))
        .unwrap_or_default();

    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--command" => {
                command = args
                    .next()
                    .ok_or_else(|| "--command requires a value".to_string())?;
            }
            "--input-source" => {
                input_source = args
                    .next()
                    .ok_or_else(|| "--input-source requires a value".to_string())?;
            }
            "--skipped-row" => {
                skipped_rows.push(
                    args.next()
                        .ok_or_else(|| "--skipped-row requires a value".to_string())?,
                );
            }
            "--out" => {
                out_path = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--out requires a value".to_string())?,
                ));
            }
            "--help" | "-h" => {
                print_bench_report_help();
                return Ok(());
            }
            other => return Err(format!("unknown bench-report argument `{other}`")),
        }
    }

    let report = BenchmarkReport {
        command,
        host: host_description(),
        rustc: command_output("rustc", &["-Vv"])
            .unwrap_or_else(|err| format!("unavailable: {err}")),
        cargo: command_output_os(cargo(), &["-V"])
            .unwrap_or_else(|err| format!("unavailable: {err}")),
        git_revision: command_output("git", &["rev-parse", "HEAD"])
            .unwrap_or_else(|err| format!("unavailable: {err}")),
        workspace_version: workspace_version()?,
        input_source,
        compare_threads: env::var("J2K_COMPARE_THREADS").unwrap_or_else(|_| "not set".to_string()),
        comparator_versions: comparator_versions(),
        skipped_rows,
    };
    let rendered = render_benchmark_report(&report);

    if let Some(path) = out_path {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)
                .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
        }
        fs::write(&path, rendered)
            .map_err(|err| format!("failed to write {}: {err}", path.display()))
    } else {
        print!("{rendered}");
        Ok(())
    }
}

fn host_description() -> String {
    command_output("uname", &["-a"])
        .unwrap_or_else(|_| format!("{} {}", env::consts::OS, env::consts::ARCH))
}

fn comparator_versions() -> Vec<(String, String)> {
    vec![
        (
            "OpenJPEG".to_string(),
            comparator_command_version("J2K_OPENJPEG_DECOMPRESS_BIN", "opj_decompress", &["-h"]),
        ),
        ("Grok".to_string(), grok_comparator_version()),
        (
            "libjpeg-turbo".to_string(),
            command_output("pkg-config", &["--modversion", "libturbojpeg"]).map_or_else(
                |err| format!("unavailable: {err}"),
                |version| format!("pkg-config libturbojpeg {version}"),
            ),
        ),
    ]
}

fn grok_comparator_version() -> String {
    if let Ok(version) = command_output("pkg-config", &["--modversion", "libgrokj2k"]) {
        let lib_dir = command_output("pkg-config", &["--variable", "libdir", "libgrokj2k"])
            .unwrap_or_else(|err| format!("libdir unavailable: {err}"));
        return format!("pkg-config libgrokj2k {version}; libdir: {lib_dir}");
    }
    env::var("J2K_GROK_ROOT").map_or_else(
        |_| "unavailable: pkg-config libgrokj2k and J2K_GROK_ROOT not set".to_string(),
        |root| format!("configured root: {root}"),
    )
}

fn comparator_command_version(env_var: &str, fallback: &str, args: &[&str]) -> String {
    let program = env::var(env_var).unwrap_or_else(|_| fallback.to_string());
    let path = program.clone();
    command_output_allow_failure(&program, args).map_or_else(
        |err| format!("unavailable: {err}; path: {path}"),
        |version| format!("{}; path: {path}", best_version_line(&version)),
    )
}

fn best_version_line(output: &str) -> &str {
    output
        .lines()
        .find(|line| line.contains("compiled against") || line.contains("version"))
        .or_else(|| output.lines().find(|line| !line.trim().is_empty()))
        .unwrap_or("version unavailable")
}

fn render_benchmark_report(report: &BenchmarkReport) -> String {
    let mut out = String::new();
    writeln!(&mut out, "# Benchmark publication report").unwrap();
    writeln!(&mut out).unwrap();
    writeln!(&mut out, "- command: {}", report.command).unwrap();
    writeln!(&mut out, "- host: {}", report.host).unwrap();
    writeln!(&mut out, "- rustc: {}", one_line(&report.rustc)).unwrap();
    writeln!(&mut out, "- cargo: {}", one_line(&report.cargo)).unwrap();
    writeln!(&mut out, "- crate revision: {}", report.git_revision).unwrap();
    writeln!(
        &mut out,
        "- workspace version: {}",
        report.workspace_version
    )
    .unwrap();
    writeln!(&mut out, "- input source: {}", report.input_source).unwrap();
    writeln!(
        &mut out,
        "- J2K_COMPARE_THREADS: {}",
        report.compare_threads
    )
    .unwrap();
    writeln!(&mut out).unwrap();
    writeln!(&mut out, "## comparator versions").unwrap();
    for (name, version) in &report.comparator_versions {
        writeln!(&mut out, "- {name}: {version}").unwrap();
    }
    writeln!(&mut out).unwrap();
    writeln!(&mut out, "## skipped rows").unwrap();
    if report.skipped_rows.is_empty() {
        writeln!(&mut out, "- none recorded").unwrap();
    } else {
        for row in &report.skipped_rows {
            writeln!(&mut out, "- {row}").unwrap();
        }
    }
    out
}

fn one_line(value: &str) -> String {
    value.lines().next().unwrap_or(value).to_string()
}

fn split_semicolon_list(value: &str) -> Vec<String> {
    value
        .split(';')
        .map(str::trim)
        .filter(|row| !row.is_empty())
        .map(str::to_string)
        .collect()
}

fn print_bench_report_help() {
    println!(
        "usage: cargo xtask bench-report [--command <command>] [--input-source <source>] \
         [--skipped-row <row>]... [--out <path>]"
    );
}

#[cfg(test)]
mod tests {
    use super::transcode_metal_bench_args;

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
}
