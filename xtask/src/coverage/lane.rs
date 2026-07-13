// SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::Path;

use crate::process::{self, cargo, CommandContext};

use super::{
    accelerator_ownership::shared_accelerator_packages,
    build_outputs::{BuildOutputEvidence, CurrentBuildTarget},
    model::CoverageLane,
};

const REQUIRED_CARGO_LLVM_COV_VERSION: &str = "0.8.7";

const METAL_COVERAGE_ENV: &[(&str, &str)] = &[
    ("J2K_REQUIRE_METAL_RUNTIME", "1"),
    ("RUST_TEST_THREADS", "1"),
];

const CUDA_COVERAGE_ENV: &[(&str, &str)] = &[
    ("J2K_REQUIRE_CUDA_RUNTIME", "1"),
    ("J2K_REQUIRE_CUDA_OXIDE_BUILD", "1"),
    ("J2K_REQUIRE_CUDA_JPEG_HARDWARE_DECODE", "1"),
    ("RUST_TEST_THREADS", "1"),
];

pub(super) struct CoverageLaneRun {
    pub(super) cargo_llvm_cov_version: String,
    pub(super) build_output_evidence: BuildOutputEvidence,
}

pub(super) fn run_lane(
    root: &Path,
    lane: CoverageLane,
    lcov_path: &Path,
    compiler_regions_path: &Path,
) -> Result<CoverageLaneRun, String> {
    let cargo_llvm_cov_version = coverage_tool_version()?;
    // A unique empty target gives every scanned build-script output current-run
    // provenance. It avoids relying on cargo-llvm-cov's best-effort clean or on
    // byte/mtime comparisons that cannot distinguish deterministic reruns.
    let current_build_target = CurrentBuildTarget::create(root)?;
    let target_dir = current_build_target.path()?;
    match lane {
        CoverageLane::Host => run_host_coverage(lcov_path, compiler_regions_path, target_dir),
        CoverageLane::Metal => run_metal_coverage(lcov_path, compiler_regions_path, target_dir),
        CoverageLane::Cuda => run_cuda_coverage(lcov_path, compiler_regions_path, target_dir),
    }?;
    let build_output_evidence = BuildOutputEvidence::capture(current_build_target)?;
    Ok(CoverageLaneRun {
        cargo_llvm_cov_version,
        build_output_evidence,
    })
}

fn run_host_coverage(
    lcov_path: &Path,
    compiler_regions_path: &Path,
    target_dir: &Path,
) -> Result<(), String> {
    let output = path_arg(lcov_path)?;
    run_llvm_cov(&host_coverage_args(&output), &[], target_dir)?;
    report_compiler_regions(compiler_regions_path, &[], target_dir)
}

fn run_metal_coverage(
    lcov_path: &Path,
    compiler_regions_path: &Path,
    target_dir: &Path,
) -> Result<(), String> {
    let args = accelerator_coverage_args(CoverageLane::Metal)?;
    run_llvm_cov(&args, METAL_COVERAGE_ENV, target_dir)?;
    run_llvm_cov(
        metal_hardware_coverage_args(),
        METAL_COVERAGE_ENV,
        target_dir,
    )?;
    report_lcov(lcov_path, METAL_COVERAGE_ENV, target_dir)?;
    report_compiler_regions(compiler_regions_path, METAL_COVERAGE_ENV, target_dir)
}

fn run_cuda_coverage(
    lcov_path: &Path,
    compiler_regions_path: &Path,
    target_dir: &Path,
) -> Result<(), String> {
    let args = accelerator_coverage_args(CoverageLane::Cuda)?;
    run_llvm_cov(&args, CUDA_COVERAGE_ENV, target_dir)?;
    report_lcov(lcov_path, CUDA_COVERAGE_ENV, target_dir)?;
    report_compiler_regions(compiler_regions_path, CUDA_COVERAGE_ENV, target_dir)
}

fn host_coverage_args(output: &str) -> Vec<&str> {
    vec![
        "llvm-cov",
        "--include-build-script",
        "--workspace",
        "--all-features",
        "--lib",
        "--bins",
        "--tests",
        "--no-fail-fast",
        "--coverage-host-only",
        "--lcov",
        "--output-path",
        output,
    ]
}

fn accelerator_coverage_args(lane: CoverageLane) -> Result<Vec<&'static str>, String> {
    let mut base = vec![
        "llvm-cov",
        "--include-build-script",
        "--no-report",
        "--no-clean",
        "--all-features",
        "--lib",
        "--tests",
        "--no-fail-fast",
    ];
    match lane {
        CoverageLane::Metal => base.insert(6, "--bins"),
        CoverageLane::Cuda => base.push("--coverage-host-only"),
        CoverageLane::Host => {
            return Err("host coverage cannot use accelerator package selection".to_string());
        }
    }
    Ok(package_coverage_args(&base, lane))
}

const fn metal_hardware_coverage_args() -> &'static [&'static str] {
    &[
        "llvm-cov",
        "--include-build-script",
        "--no-report",
        "--no-clean",
        "--all-features",
        "--lib",
        "-p",
        "j2k-metal",
        "--",
        "--ignored",
        "--test-threads=1",
    ]
}

fn package_coverage_args(base: &[&'static str], lane: CoverageLane) -> Vec<&'static str> {
    let mut args = base.to_vec();
    for package in lane
        .coverage_packages()
        .chain(shared_accelerator_packages())
    {
        args.push("-p");
        args.push(package);
    }
    args
}

fn report_lcov(lcov_path: &Path, envs: &[(&str, &str)], target_dir: &Path) -> Result<(), String> {
    let output = path_arg(lcov_path)?;
    run_llvm_cov(&report_lcov_args(&output), envs, target_dir)
}

fn report_lcov_args(output: &str) -> Vec<&str> {
    vec![
        "llvm-cov",
        "report",
        "--include-build-script",
        "--lcov",
        "--output-path",
        output,
    ]
}

fn report_compiler_regions(
    compiler_regions_path: &Path,
    envs: &[(&str, &str)],
    target_dir: &Path,
) -> Result<(), String> {
    let output = path_arg(compiler_regions_path)?;
    run_llvm_cov(&report_compiler_regions_args(&output), envs, target_dir)
}

fn report_compiler_regions_args(output: &str) -> Vec<&str> {
    vec![
        "llvm-cov",
        "report",
        "--include-build-script",
        "--json",
        "--output-path",
        output,
    ]
}

fn coverage_tool_version() -> Result<String, String> {
    let output =
        process::command_output(cargo(), &["llvm-cov", "--version"], CommandContext::new())?;
    let stdout = String::from_utf8(output.stdout)
        .map_err(|error| format!("cargo llvm-cov --version stdout is not UTF-8: {error}"))?;
    let stderr = String::from_utf8(output.stderr)
        .map_err(|error| format!("cargo llvm-cov --version stderr is not UTF-8: {error}"))?;
    let rendered = format!("{stdout}\n{stderr}");
    if !output.status.success() {
        return Err(format!(
            "cargo llvm-cov --version failed with {}: {}",
            output.status,
            rendered.trim()
        ));
    }
    let observed = parse_coverage_tool_version(&rendered)?;
    if observed != REQUIRED_CARGO_LLVM_COV_VERSION {
        return Err(format!(
            "cargo-llvm-cov {REQUIRED_CARGO_LLVM_COV_VERSION} is required, found {observed}"
        ));
    }
    Ok(observed.to_string())
}

fn parse_coverage_tool_version(rendered: &str) -> Result<&str, String> {
    rendered
        .lines()
        .find_map(|line| {
            line.trim()
                .strip_prefix("cargo-llvm-cov ")
                .and_then(|version| version.split_whitespace().next())
        })
        .ok_or_else(|| {
            format!(
                "cargo llvm-cov --version did not report a named version: {}",
                rendered.trim()
            )
        })
}

fn run_llvm_cov(args: &[&str], envs: &[(&str, &str)], target_dir: &Path) -> Result<(), String> {
    let target_dir = path_arg(target_dir)?;
    let current_build_env = current_build_env(envs, target_dir.as_str());
    process::run_command(
        cargo(),
        args,
        CommandContext::new().envs(&current_build_env),
    )
}

fn current_build_env<'a>(
    envs: &[(&'a str, &'a str)],
    target_dir: &'a str,
) -> Vec<(&'a str, &'a str)> {
    let mut current_build_env = envs.to_vec();
    current_build_env.push(("CARGO_LLVM_COV_TARGET_DIR", target_dir));
    current_build_env.push(("CARGO_LLVM_COV_BUILD_DIR", target_dir));
    current_build_env
}

fn path_arg(path: &Path) -> Result<String, String> {
    path.to_str()
        .map(ToOwned::to_owned)
        .ok_or_else(|| format!("coverage path is not valid UTF-8: {}", path.display()))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::path::Path;

    use crate::coverage::accelerator_ownership::{
        shared_accelerator_packages, shared_accelerator_sources,
    };
    use crate::process::use_test_cargo_program;
    use crate::test_command::RecordingProgram;

    use super::{
        accelerator_coverage_args, current_build_env, host_coverage_args,
        metal_hardware_coverage_args, package_coverage_args, parse_coverage_tool_version,
        report_compiler_regions_args, report_lcov_args, run_lane, CoverageLane, CUDA_COVERAGE_ENV,
        METAL_COVERAGE_ENV,
    };

    #[test]
    fn lane_orchestrators_execute_complete_hermetic_cargo_plans() {
        let recording = RecordingProgram::new(
            "coverage-lane-command-test",
            "if [ \"$1\" = llvm-cov ] && [ \"$2\" = --version ]; then printf 'cargo-llvm-cov 0.8.7\\n'; fi",
        );
        let _cargo = use_test_cargo_program(recording.program().as_os_str().to_owned());
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask manifest has workspace parent");

        for lane in [CoverageLane::Host, CoverageLane::Metal, CoverageLane::Cuda] {
            let result = run_lane(
                root,
                lane,
                &root.join(format!("target/test-{}.info", lane.name())),
                &root.join(format!("target/test-{}-regions.json", lane.name())),
            )
            .expect("hermetic coverage lane");
            assert_eq!(result.cargo_llvm_cov_version, "0.8.7");
        }

        let log = recording.log();
        assert!(log.contains("llvm-cov --version|"));
        assert!(log.contains("--workspace --all-features --lib --bins --tests"));
        assert!(log.contains("-p j2k-metal -- --ignored --test-threads=1"));
        assert!(log.contains("-p j2k-cuda-runtime"));
        assert!(log.contains("llvm-cov report --include-build-script --lcov"));
        assert!(log.contains("llvm-cov report --include-build-script --json"));
    }

    #[test]
    fn llvm_cov_commands_share_unique_target_and_build_directory() {
        let target = "/tmp/j2k-current-coverage-test";
        let env = current_build_env(METAL_COVERAGE_ENV, target);

        assert!(env.contains(&("CARGO_LLVM_COV_TARGET_DIR", target)));
        assert!(env.contains(&("CARGO_LLVM_COV_BUILD_DIR", target)));
    }

    #[test]
    fn accelerator_lane_package_args_include_every_shared_source_owner() {
        for lane in [CoverageLane::Metal, CoverageLane::Cuda] {
            let args = package_coverage_args(&[], lane);
            for package in shared_accelerator_packages() {
                assert!(
                    args.windows(2).any(|pair| pair == ["-p", package]),
                    "accelerator coverage omitted shared source owner {package}"
                );
            }
        }
    }

    #[test]
    fn lane_spec_drives_package_args_and_source_ownership() {
        for lane in [CoverageLane::Metal, CoverageLane::Cuda] {
            let args = package_coverage_args(&[], lane);
            for package in lane.coverage_packages() {
                assert!(args.windows(2).any(|pair| pair == ["-p", package]));
                assert!(lane.owns_path(&format!("crates/{package}/src/lib.rs")));
            }
        }
    }

    #[test]
    fn shared_accelerator_source_owners_drive_lane_package_selection() {
        let path_owners = shared_accelerator_sources()
            .iter()
            .map(|source| source.package)
            .collect::<BTreeSet<_>>();
        let selected_owners = shared_accelerator_packages();

        assert_eq!(path_owners, selected_owners);
    }

    #[test]
    fn coverage_tool_version_parser_requires_named_record() {
        assert_eq!(
            parse_coverage_tool_version("warning: rust 1.90\ncargo-llvm-cov 0.8.7\n"),
            Ok("0.8.7")
        );
        assert!(parse_coverage_tool_version("warning: rust 1.90\n").is_err());
    }

    #[test]
    fn host_command_captures_the_complete_workspace_and_writes_lcov() {
        assert_eq!(
            host_coverage_args("host.info"),
            [
                "llvm-cov",
                "--include-build-script",
                "--workspace",
                "--all-features",
                "--lib",
                "--bins",
                "--tests",
                "--no-fail-fast",
                "--coverage-host-only",
                "--lcov",
                "--output-path",
                "host.info",
            ]
        );
    }

    #[test]
    fn accelerator_commands_preserve_lane_specific_execution_contracts() {
        assert!(accelerator_coverage_args(CoverageLane::Host).is_err());
        let metal = accelerator_coverage_args(CoverageLane::Metal).unwrap();
        assert!(metal.contains(&"--bins"));
        assert!(!metal.contains(&"--coverage-host-only"));
        assert!(metal.contains(&"--no-report") && metal.contains(&"--no-clean"));

        let cuda = accelerator_coverage_args(CoverageLane::Cuda).unwrap();
        assert!(!cuda.contains(&"--bins"));
        assert!(cuda.contains(&"--coverage-host-only"));
        assert!(cuda.contains(&"--no-report") && cuda.contains(&"--no-clean"));

        for (lane, args) in [(CoverageLane::Metal, metal), (CoverageLane::Cuda, cuda)] {
            let package_values = args
                .windows(2)
                .filter_map(|pair| (pair[0] == "-p").then_some(pair[1]))
                .collect::<Vec<_>>();
            assert_eq!(
                package_values.len(),
                package_values.iter().collect::<BTreeSet<_>>().len(),
                "{lane:?} command contains duplicate package selections"
            );
        }
    }

    #[test]
    fn metal_hardware_and_report_commands_cannot_drop_required_flags() {
        assert_eq!(
            metal_hardware_coverage_args(),
            [
                "llvm-cov",
                "--include-build-script",
                "--no-report",
                "--no-clean",
                "--all-features",
                "--lib",
                "-p",
                "j2k-metal",
                "--",
                "--ignored",
                "--test-threads=1",
            ]
        );
        assert_eq!(
            report_lcov_args("lane.info"),
            [
                "llvm-cov",
                "report",
                "--include-build-script",
                "--lcov",
                "--output-path",
                "lane.info",
            ]
        );
        assert_eq!(
            report_compiler_regions_args("lane-regions.json"),
            [
                "llvm-cov",
                "report",
                "--include-build-script",
                "--json",
                "--output-path",
                "lane-regions.json",
            ]
        );
    }

    #[test]
    fn accelerator_environments_require_real_serial_hardware_execution() {
        assert_eq!(
            METAL_COVERAGE_ENV,
            &[
                ("J2K_REQUIRE_METAL_RUNTIME", "1"),
                ("RUST_TEST_THREADS", "1")
            ]
        );
        assert_eq!(
            CUDA_COVERAGE_ENV,
            &[
                ("J2K_REQUIRE_CUDA_RUNTIME", "1"),
                ("J2K_REQUIRE_CUDA_OXIDE_BUILD", "1"),
                ("J2K_REQUIRE_CUDA_JPEG_HARDWARE_DECODE", "1"),
                ("RUST_TEST_THREADS", "1"),
            ]
        );
    }
}
