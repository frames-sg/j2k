// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use super::accelerator_ownership::is_shared_accelerator_path;
use super::compiler_regions::CompilerRegionReport;
use super::source_analysis::SourceRole;

pub(super) const CHANGED_LINE_THRESHOLD_PERCENT: u64 = 80;
const HOST_LCOV_PATH: &str = "lcov-host.info";
const METAL_LCOV_PATH: &str = "lcov-metal.info";
const CUDA_LCOV_PATH: &str = "lcov-cuda.info";
const HOST_COMPILER_REGIONS_PATH: &str = "coverage-host-regions.json";
const METAL_COMPILER_REGIONS_PATH: &str = "coverage-metal-regions.json";
const CUDA_COMPILER_REGIONS_PATH: &str = "coverage-cuda-regions.json";
const HOST_SUMMARY_PATH: &str = "coverage-host-summary.json";
const METAL_SUMMARY_PATH: &str = "coverage-metal-summary.json";
const CUDA_SUMMARY_PATH: &str = "coverage-cuda-summary.json";

struct AcceleratorLaneSpec {
    packages: &'static [AcceleratorPackageSpec],
}

struct AcceleratorPackageSpec {
    name: &'static str,
    source_prefix: &'static str,
}

const METAL_ACCELERATOR_LANE: AcceleratorLaneSpec = AcceleratorLaneSpec {
    packages: &[
        accelerator_package("j2k-metal-support", "crates/j2k-metal-support/"),
        accelerator_package("j2k-jpeg-metal", "crates/j2k-jpeg-metal/"),
        accelerator_package("j2k-metal", "crates/j2k-metal/"),
        accelerator_package("j2k-transcode-metal", "crates/j2k-transcode-metal/"),
    ],
};

const CUDA_ACCELERATOR_LANE: AcceleratorLaneSpec = AcceleratorLaneSpec {
    packages: &[
        accelerator_package("j2k-cuda-runtime", "crates/j2k-cuda-runtime/"),
        accelerator_package("j2k-jpeg-cuda", "crates/j2k-jpeg-cuda/"),
        accelerator_package("j2k-cuda", "crates/j2k-cuda/"),
        accelerator_package("j2k-transcode-cuda", "crates/j2k-transcode-cuda/"),
    ],
};

const fn accelerator_package(
    name: &'static str,
    source_prefix: &'static str,
) -> AcceleratorPackageSpec {
    AcceleratorPackageSpec {
        name,
        source_prefix,
    }
}

const METAL_VENDOR_PATHS: &[&str] = &["third_party/block-0.1.6-patched/src/lib.rs"];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum CoverageLane {
    Host,
    Metal,
    Cuda,
}

impl CoverageLane {
    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::Host => "host",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }

    pub(super) const fn lcov_path(self) -> &'static str {
        match self {
            Self::Host => HOST_LCOV_PATH,
            Self::Metal => METAL_LCOV_PATH,
            Self::Cuda => CUDA_LCOV_PATH,
        }
    }

    pub(super) const fn summary_path(self) -> &'static str {
        match self {
            Self::Host => HOST_SUMMARY_PATH,
            Self::Metal => METAL_SUMMARY_PATH,
            Self::Cuda => CUDA_SUMMARY_PATH,
        }
    }

    pub(super) const fn compiler_regions_path(self) -> &'static str {
        match self {
            Self::Host => HOST_COMPILER_REGIONS_PATH,
            Self::Metal => METAL_COMPILER_REGIONS_PATH,
            Self::Cuda => CUDA_COMPILER_REGIONS_PATH,
        }
    }

    pub(super) const fn scope_name(self) -> &'static str {
        match self {
            Self::Host => "non-accelerator-production",
            Self::Metal => "metal-and-shared-accelerator-production",
            Self::Cuda => "cuda-and-shared-accelerator-production",
        }
    }

    pub(super) const fn enforces_line_threshold(self) -> bool {
        !matches!(self, Self::Metal)
    }

    pub(super) const fn line_threshold_mode(self) -> &'static str {
        if self.enforces_line_threshold() {
            "release-gate"
        } else {
            "audited-evidence"
        }
    }

    pub(super) fn owns_path(self, path: &str) -> bool {
        match self {
            Self::Host => !is_accelerator_path(path),
            Self::Metal => {
                METAL_ACCELERATOR_LANE.owns_path(path)
                    || is_shared_accelerator_path(path)
                    || METAL_VENDOR_PATHS.contains(&path)
            }
            Self::Cuda => CUDA_ACCELERATOR_LANE.owns_path(path) || is_shared_accelerator_path(path),
        }
    }

    pub(super) fn coverage_packages(self) -> impl Iterator<Item = &'static str> {
        self.accelerator_packages()
            .iter()
            .map(|package| package.name)
    }

    pub(super) fn includes_source(self, path: &str, role: SourceRole) -> bool {
        role.is_measurable() && self.owns_path(path)
    }

    const fn accelerator_packages(self) -> &'static [AcceleratorPackageSpec] {
        match self {
            Self::Host => &[],
            Self::Metal => METAL_ACCELERATOR_LANE.packages,
            Self::Cuda => CUDA_ACCELERATOR_LANE.packages,
        }
    }
}

#[derive(Debug, Default)]
pub(super) struct LcovReport {
    pub(super) lines: BTreeMap<String, BTreeMap<usize, u64>>,
    pub(super) compiler_regions: CompilerRegionReport,
}

#[derive(Debug)]
pub(super) struct CoverageOptions {
    pub(super) lane: CoverageLane,
    pub(super) base: Option<String>,
    pub(super) output: Option<PathBuf>,
}

#[derive(Debug, Default)]
pub(super) struct CoverageCounts {
    pub(super) measurable: usize,
    pub(super) covered: usize,
}

#[derive(Debug, Default)]
pub(super) struct SourceDispositionCounts {
    pub(super) changed_lines: usize,
    pub(super) files: BTreeSet<String>,
}

#[derive(Debug)]
pub(super) struct ChangedCoverageResult {
    pub(super) overall: CoverageCounts,
    pub(super) critical: CoverageCounts,
    pub(super) accelerator: CoverageCounts,
    pub(super) changed_files: BTreeSet<String>,
    pub(super) uncovered: Vec<(String, usize)>,
    pub(super) unmeasured: Vec<(String, usize)>,
    pub(super) exclusions: BTreeMap<&'static str, usize>,
    pub(super) source_dispositions: BTreeMap<&'static str, SourceDispositionCounts>,
    pub(super) absent_instrumentable_files: Vec<String>,
    pub(super) changed_functions_without_covered_body: Vec<String>,
    pub(super) changed_executable_bodies_without_covered_body: Vec<String>,
    pub(super) changed_deferred_bodies_without_covered_compiler_region: Vec<String>,
    pub(super) compiler_noninstrumentable_deferred_bodies: Vec<String>,
    pub(super) compiler_noninstrumentable_lines: Vec<String>,
    pub(super) mixed_test_production_lines: Vec<String>,
    pub(super) changed_opaque_macros: Vec<String>,
}

pub(super) fn parse_options(args: impl Iterator<Item = String>) -> Result<CoverageOptions, String> {
    let mut lane = CoverageLane::Host;
    let mut base = None;
    let mut output = None;
    let mut args = args.peekable();

    if args.peek().is_some_and(|arg| !arg.starts_with('-')) {
        lane = match args.next().as_deref() {
            Some("host") => CoverageLane::Host,
            Some("metal") => CoverageLane::Metal,
            Some("cuda") => CoverageLane::Cuda,
            Some(other) => {
                return Err(format!(
                    "unknown coverage lane `{other}`; expected host, metal, or cuda"
                ));
            }
            None => {
                return Err("coverage lane argument disappeared after lookahead".to_string());
            }
        };
    }

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--base" => {
                base = Some(
                    args.next()
                        .ok_or_else(|| "--base requires a git revision".to_string())?,
                );
            }
            "--output" => {
                output = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--output requires a path".to_string())?,
                ));
            }
            other => return Err(format!("unknown coverage option `{other}`")),
        }
    }

    Ok(CoverageOptions { lane, base, output })
}

pub(super) fn is_accelerator_path(path: &str) -> bool {
    METAL_ACCELERATOR_LANE.owns_path(path)
        || CUDA_ACCELERATOR_LANE.owns_path(path)
        || is_shared_accelerator_path(path)
}

impl AcceleratorLaneSpec {
    fn owns_path(&self, path: &str) -> bool {
        self.packages
            .iter()
            .any(|package| path.starts_with(package.source_prefix))
    }
}

#[cfg(test)]
mod tests {
    use super::{accelerator_package, CoverageLane};

    #[test]
    fn accelerator_package_preserves_registry_metadata() {
        let package = accelerator_package(
            std::hint::black_box("j2k-test-accelerator"),
            std::hint::black_box("crates/j2k-test-accelerator/"),
        );

        assert_eq!(package.name, "j2k-test-accelerator");
        assert_eq!(package.source_prefix, "crates/j2k-test-accelerator/");
    }

    #[test]
    fn coverage_lane_artifacts_are_lane_specific() {
        let cases = [
            (
                CoverageLane::Host,
                "lcov-host.info",
                "coverage-host-summary.json",
                "coverage-host-regions.json",
            ),
            (
                CoverageLane::Metal,
                "lcov-metal.info",
                "coverage-metal-summary.json",
                "coverage-metal-regions.json",
            ),
            (
                CoverageLane::Cuda,
                "lcov-cuda.info",
                "coverage-cuda-summary.json",
                "coverage-cuda-regions.json",
            ),
        ];

        for (lane, lcov, summary, compiler_regions) in cases {
            assert_eq!(lane.lcov_path(), lcov);
            assert_eq!(lane.summary_path(), summary);
            assert_eq!(lane.compiler_regions_path(), compiler_regions);
        }
    }
}
