// SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::Path;

use crate::process::{self, cargo, CommandContext};

use super::model::CoverageLane;

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

pub(super) fn run_lane(lane: CoverageLane, lcov_path: &Path) -> Result<(), String> {
    run_llvm_cov(&["llvm-cov", "clean", "--workspace"], &[])?;
    match lane {
        CoverageLane::Host => run_host_coverage(lcov_path),
        CoverageLane::Metal => run_metal_coverage(lcov_path),
        CoverageLane::Cuda => run_cuda_coverage(lcov_path),
    }
}

fn run_host_coverage(lcov_path: &Path) -> Result<(), String> {
    let output = path_arg(lcov_path)?;
    run_llvm_cov(
        &[
            "llvm-cov",
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
    )
}

fn run_metal_coverage(lcov_path: &Path) -> Result<(), String> {
    run_llvm_cov(
        &[
            "llvm-cov",
            "--no-report",
            "--no-clean",
            "--all-features",
            "--lib",
            "--bins",
            "--tests",
            "--no-fail-fast",
            "-p",
            "j2k-metal-support",
            "-p",
            "j2k-jpeg-metal",
            "-p",
            "j2k-metal",
            "-p",
            "j2k-transcode-metal",
        ],
        METAL_COVERAGE_ENV,
    )?;
    run_llvm_cov(
        &[
            "llvm-cov",
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
    )?;
    report_lcov(lcov_path, METAL_COVERAGE_ENV)
}

fn run_cuda_coverage(lcov_path: &Path) -> Result<(), String> {
    run_llvm_cov(
        &[
            "llvm-cov",
            "--no-report",
            "--no-clean",
            "--all-features",
            "--lib",
            "--tests",
            "--no-fail-fast",
            "--coverage-host-only",
            "-p",
            "j2k-cuda-runtime",
            "-p",
            "j2k-jpeg-cuda",
            "-p",
            "j2k-cuda",
            "-p",
            "j2k-transcode-cuda",
        ],
        CUDA_COVERAGE_ENV,
    )?;
    report_lcov(lcov_path, CUDA_COVERAGE_ENV)
}

fn report_lcov(lcov_path: &Path, envs: &[(&str, &str)]) -> Result<(), String> {
    let output = path_arg(lcov_path)?;
    run_llvm_cov(
        &["llvm-cov", "report", "--lcov", "--output-path", &output],
        envs,
    )
}

fn run_llvm_cov(args: &[&str], envs: &[(&str, &str)]) -> Result<(), String> {
    process::run_command(cargo(), args, CommandContext::new().envs(envs))
}

fn path_arg(path: &Path) -> Result<String, String> {
    path.to_str()
        .map(ToOwned::to_owned)
        .ok_or_else(|| format!("coverage path is not valid UTF-8: {}", path.display()))
}
