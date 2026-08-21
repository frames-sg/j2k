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
    let mut current_build_target = CurrentBuildTarget::create(root)?;
    let target_dir = current_build_target.path()?;
    let build_output_evidence = match lane {
        CoverageLane::Host => {
            run_host_coverage(lcov_path, compiler_regions_path, target_dir)?;
            BuildOutputEvidence::snapshot(&current_build_target)?
        }
        CoverageLane::Metal => run_metal_coverage(
            lcov_path,
            compiler_regions_path,
            target_dir,
            &current_build_target,
        )?,
        CoverageLane::Cuda => run_cuda_coverage(
            lcov_path,
            compiler_regions_path,
            target_dir,
            &current_build_target,
        )?,
    };
    current_build_target.cleanup()?;
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
    current_build_target: &CurrentBuildTarget,
) -> Result<BuildOutputEvidence, String> {
    let args = accelerator_coverage_args(CoverageLane::Metal)?;
    run_llvm_cov(&args, METAL_COVERAGE_ENV, target_dir)?;
    // The primary all-feature package graph defines cfg provenance for source
    // reachability. Follow-up feature-scoped and hardware passes accumulate
    // execution profiles, but may legitimately rebuild shared dependencies
    // under narrower feature scopes in the same Cargo target directory.
    let build_output_evidence = BuildOutputEvidence::snapshot(current_build_target)?;
    run_feature_coverage(CoverageLane::Metal, METAL_COVERAGE_ENV, target_dir)?;
    run_llvm_cov(
        metal_hardware_coverage_args(),
        METAL_COVERAGE_ENV,
        target_dir,
    )?;
    report_lcov(lcov_path, METAL_COVERAGE_ENV, target_dir)?;
    report_compiler_regions(compiler_regions_path, METAL_COVERAGE_ENV, target_dir)?;
    Ok(build_output_evidence)
}

fn run_cuda_coverage(
    lcov_path: &Path,
    compiler_regions_path: &Path,
    target_dir: &Path,
    current_build_target: &CurrentBuildTarget,
) -> Result<BuildOutputEvidence, String> {
    let args = accelerator_coverage_args(CoverageLane::Cuda)?;
    run_llvm_cov(&args, CUDA_COVERAGE_ENV, target_dir)?;
    // See the Metal lane: the primary all-feature graph is the lane's cfg
    // provenance authority; the CUDA-only ML pass contributes profiles only.
    let build_output_evidence = BuildOutputEvidence::snapshot(current_build_target)?;
    run_feature_coverage(CoverageLane::Cuda, CUDA_COVERAGE_ENV, target_dir)?;
    report_lcov(lcov_path, CUDA_COVERAGE_ENV, target_dir)?;
    report_compiler_regions(compiler_regions_path, CUDA_COVERAGE_ENV, target_dir)?;
    Ok(build_output_evidence)
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
        "--no-report",
        "--all-features",
        "--lib",
        "--tests",
        "--no-fail-fast",
    ];
    match lane {
        CoverageLane::Metal => base.insert(5, "--bins"),
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
        .all_feature_coverage_packages()
        .chain(shared_accelerator_packages())
    {
        args.push("-p");
        args.push(package);
    }
    args
}

fn feature_coverage_args(package: &'static str, feature: &'static str) -> Vec<&'static str> {
    vec![
        "llvm-cov",
        "--no-clean",
        "--no-default-features",
        "--features",
        feature,
        "--lib",
        "--tests",
        "--no-fail-fast",
        "-p",
        package,
    ]
}

fn run_feature_coverage(
    lane: CoverageLane,
    envs: &[(&str, &str)],
    target_dir: &Path,
) -> Result<(), String> {
    for (package, feature) in lane.feature_coverage_packages() {
        let mut args = feature_coverage_args(package, feature);
        if lane == CoverageLane::Cuda {
            args.push("--coverage-host-only");
        }
        run_llvm_cov(&args, envs, target_dir)?;
    }
    Ok(())
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
        accelerator_coverage_args, current_build_env, feature_coverage_args, host_coverage_args,
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
        assert!(log.contains("--no-default-features --features metal --lib --tests"));
        assert!(log.contains("--no-default-features --features cuda --lib --tests"));
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
            let feature_packages = lane
                .feature_coverage_packages()
                .map(|(package, _)| package)
                .collect::<BTreeSet<_>>();
            for package in lane.coverage_packages() {
                assert!(
                    args.windows(2).any(|pair| pair == ["-p", package])
                        || feature_packages.contains(package)
                );
            }
            for prefix in lane.accelerator_source_prefixes() {
                assert!(lane.owns_path(prefix));
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
        assert!(metal.contains(&"--no-report") && !metal.contains(&"--no-clean"));
        assert!(!metal.contains(&"--include-build-script"));

        let cuda = accelerator_coverage_args(CoverageLane::Cuda).unwrap();
        assert!(!cuda.contains(&"--bins"));
        assert!(cuda.contains(&"--coverage-host-only"));
        assert!(cuda.contains(&"--no-report") && !cuda.contains(&"--no-clean"));
        assert!(!cuda.contains(&"--include-build-script"));

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
    fn cuda_coverage_does_not_enable_j2k_ml_metal_dependencies() {
        let cuda = accelerator_coverage_args(CoverageLane::Cuda).unwrap();
        let packages = cuda
            .windows(2)
            .filter_map(|pair| (pair[0] == "-p").then_some(pair[1]))
            .collect::<BTreeSet<_>>();

        for package in [
            "j2k-cuda-build-support",
            "j2k-cuda-runtime",
            "j2k-cuda-j2k-engine",
            "j2k-cuda-jpeg-engine",
            "j2k-cuda-transcode-engine",
            "j2k-jpeg-cuda",
            "j2k-cuda",
            "j2k-transcode-cuda",
        ] {
            assert!(
                packages.contains(package),
                "CUDA coverage omitted {package}"
            );
        }
        for package in ["j2k-metal", "j2k-jpeg-metal", "j2k-transcode-metal"] {
            assert!(
                !packages.contains(package),
                "CUDA coverage selected Metal-only package {package}"
            );
        }
        assert!(
            !packages.contains("j2k-ml"),
            "CUDA coverage must not select j2k-ml under --all-features because that enables its Metal dependencies"
        );
        assert_eq!(
            feature_coverage_args("j2k-ml", "cuda"),
            [
                "llvm-cov",
                "--no-clean",
                "--no-default-features",
                "--features",
                "cuda",
                "--lib",
                "--tests",
                "--no-fail-fast",
                "-p",
                "j2k-ml",
            ]
        );
    }

    #[test]
    fn feature_coverage_uses_a_cargo_llvm_cov_0_8_7_compatible_accumulation_plan() {
        for feature in ["metal", "cuda"] {
            let args = feature_coverage_args("j2k-ml", feature);
            assert!(args.contains(&"--no-clean"));
            assert!(
                !(args.contains(&"--no-clean") && args.contains(&"--no-report")),
                "cargo-llvm-cov 0.8.7 rejects --no-clean together with --no-report"
            );
        }
    }

    #[test]
    fn cuda_lane_orchestrator_selects_only_cuda_feature_family() {
        let recording = RecordingProgram::new(
            "cuda-coverage-package-selection-test",
            "if [ \"$1\" = llvm-cov ] && [ \"$2\" = --version ]; then printf 'cargo-llvm-cov 0.8.7\\n'; fi",
        );
        let _cargo = use_test_cargo_program(recording.program().as_os_str().to_owned());
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask manifest has workspace parent");

        run_lane(
            root,
            CoverageLane::Cuda,
            &root.join("target/test-cuda-selection.info"),
            &root.join("target/test-cuda-selection-regions.json"),
        )
        .expect("hermetic CUDA coverage lane");

        let commands = recording.log();
        let primary = commands
            .lines()
            .find(|line| line.contains("llvm-cov --no-report --all-features"))
            .expect("CUDA lane runs an all-feature primary package command");
        assert!(!primary.contains("-p j2k-ml"));
        let ml = commands
            .lines()
            .find(|line| line.contains("--features cuda") && line.contains("-p j2k-ml"))
            .expect("CUDA lane runs j2k-ml with only its CUDA feature");
        assert!(ml.contains("--no-default-features"));
        assert!(ml.contains("--coverage-host-only"));
        assert!(!ml.contains("--all-features"));
        for package in ["j2k-metal", "j2k-jpeg-metal", "j2k-transcode-metal"] {
            assert!(
                !commands.contains(&format!("-p {package}")),
                "CUDA coverage selected Metal-only package {package}"
            );
        }
    }

    #[test]
    fn cuda_lane_uses_primary_build_scope_for_cfg_provenance() {
        let recording = RecordingProgram::new(
            "cuda-coverage-primary-provenance-test",
            r#"
if [ "$1" = llvm-cov ] && [ "$2" = --version ]; then
    printf 'cargo-llvm-cov 0.8.7\n'
fi
case "$*" in
  *'llvm-cov --no-report --all-features'*)
    scope=j2k-cuda-j2k-engine-cd0e0123456789ab
    cfg='cargo::rustc-check-cfg=cfg(j2k_cuda_oxide_j2k_ml_built)\ncargo::rustc-cfg=j2k_cuda_oxide_j2k_ml_built\n'
    ;;
  *'llvm-cov --no-clean --no-default-features --features cuda'*)
    scope=j2k-cuda-j2k-engine-d1aa0123456789ab
    cfg='cargo::rustc-check-cfg=cfg(j2k_cuda_oxide_j2k_ml_built)\n'
    ;;
  *) exit 0 ;;
esac
directory="$CARGO_LLVM_COV_TARGET_DIR/debug/build/$scope"
mkdir -p "$directory"
printf '%b' "$cfg" > "$directory/output"
"#,
        );
        let _cargo = use_test_cargo_program(recording.program().as_os_str().to_owned());
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask manifest has workspace parent");

        let result = run_lane(
            root,
            CoverageLane::Cuda,
            &root.join("target/test-cuda-provenance.info"),
            &root.join("target/test-cuda-provenance-regions.json"),
        )
        .expect("hermetic CUDA coverage lane");
        let package = "j2k-cuda-j2k-engine".to_string();
        let packages = BTreeSet::from([package.clone()]);
        let flags = result
            .build_output_evidence
            .current_cfg_flags(&packages, &packages)
            .expect("primary all-feature cfg provenance");

        assert!(flags[&package]["j2k_cuda_oxide_j2k_ml_built"]);
    }

    #[test]
    fn metal_hardware_and_report_commands_cannot_drop_required_flags() {
        assert_eq!(
            metal_hardware_coverage_args(),
            [
                "llvm-cov",
                "--include-build-script",
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
