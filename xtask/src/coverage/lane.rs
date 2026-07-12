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
) -> Result<CoverageLaneRun, String> {
    let cargo_llvm_cov_version = coverage_tool_version()?;
    // A unique empty target gives every scanned build-script output current-run
    // provenance. It avoids relying on cargo-llvm-cov's best-effort clean or on
    // byte/mtime comparisons that cannot distinguish deterministic reruns.
    let current_build_target = CurrentBuildTarget::create(root)?;
    let target_dir = current_build_target.path()?;
    match lane {
        CoverageLane::Host => run_host_coverage(lcov_path, target_dir),
        CoverageLane::Metal => run_metal_coverage(lcov_path, target_dir),
        CoverageLane::Cuda => run_cuda_coverage(lcov_path, target_dir),
    }?;
    let build_output_evidence = BuildOutputEvidence::capture(current_build_target)?;
    Ok(CoverageLaneRun {
        cargo_llvm_cov_version,
        build_output_evidence,
    })
}

fn run_host_coverage(lcov_path: &Path, target_dir: &Path) -> Result<(), String> {
    let output = path_arg(lcov_path)?;
    run_llvm_cov(
        &[
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
            &output,
        ],
        &[],
        target_dir,
    )
}

fn run_metal_coverage(lcov_path: &Path, target_dir: &Path) -> Result<(), String> {
    let args = package_coverage_args(
        &[
            "llvm-cov",
            "--include-build-script",
            "--no-report",
            "--no-clean",
            "--all-features",
            "--lib",
            "--bins",
            "--tests",
            "--no-fail-fast",
        ],
        CoverageLane::Metal,
    );
    run_llvm_cov(&args, METAL_COVERAGE_ENV, target_dir)?;
    run_llvm_cov(
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
        ],
        METAL_COVERAGE_ENV,
        target_dir,
    )?;
    report_lcov(lcov_path, METAL_COVERAGE_ENV, target_dir)
}

fn run_cuda_coverage(lcov_path: &Path, target_dir: &Path) -> Result<(), String> {
    let args = package_coverage_args(
        &[
            "llvm-cov",
            "--include-build-script",
            "--no-report",
            "--no-clean",
            "--all-features",
            "--lib",
            "--tests",
            "--no-fail-fast",
            "--coverage-host-only",
        ],
        CoverageLane::Cuda,
    );
    run_llvm_cov(&args, CUDA_COVERAGE_ENV, target_dir)?;
    report_lcov(lcov_path, CUDA_COVERAGE_ENV, target_dir)
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
    run_llvm_cov(
        &[
            "llvm-cov",
            "report",
            "--include-build-script",
            "--lcov",
            "--output-path",
            &output,
        ],
        envs,
        target_dir,
    )
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

    use crate::coverage::accelerator_ownership::{
        shared_accelerator_packages, shared_accelerator_sources,
    };

    use super::{
        current_build_env, package_coverage_args, parse_coverage_tool_version, CoverageLane,
        METAL_COVERAGE_ENV,
    };

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
}
