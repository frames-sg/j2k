// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

pub(super) const CHANGED_LINE_THRESHOLD_PERCENT: u64 = 80;
const HOST_LCOV_PATH: &str = "lcov-host.info";
const METAL_LCOV_PATH: &str = "lcov-metal.info";
const CUDA_LCOV_PATH: &str = "lcov-cuda.info";
const HOST_SUMMARY_PATH: &str = "coverage-host-summary.json";
const METAL_SUMMARY_PATH: &str = "coverage-metal-summary.json";
const CUDA_SUMMARY_PATH: &str = "coverage-cuda-summary.json";

const METAL_CRATE_PREFIXES: &[&str] = &[
    "crates/j2k-metal/",
    "crates/j2k-metal-support/",
    "crates/j2k-jpeg-metal/",
    "crates/j2k-transcode-metal/",
];

const CUDA_CRATE_PREFIXES: &[&str] = &[
    "crates/j2k-cuda-runtime/",
    "crates/j2k-cuda/",
    "crates/j2k-jpeg-cuda/",
    "crates/j2k-transcode-cuda/",
];

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

    pub(super) fn includes_path(self, path: &str) -> bool {
        match self {
            Self::Host => is_production_rust(path),
            Self::Metal => is_production_rust(path) && has_any_prefix(path, METAL_CRATE_PREFIXES),
            Self::Cuda => is_production_rust(path) && has_any_prefix(path, CUDA_CRATE_PREFIXES),
        }
    }
}

#[derive(Debug, Default)]
pub(super) struct LcovReport {
    pub(super) lines: BTreeMap<String, BTreeMap<usize, u64>>,
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

#[derive(Debug)]
pub(super) struct ChangedCoverageResult {
    pub(super) overall: CoverageCounts,
    pub(super) accelerator: CoverageCounts,
    pub(super) changed_files: BTreeSet<String>,
    pub(super) uncovered: Vec<(String, usize)>,
    pub(super) unmeasured: Vec<(String, usize)>,
    pub(super) exclusions: BTreeMap<&'static str, usize>,
    pub(super) absent_instrumentable_files: Vec<String>,
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
            None => unreachable!("peeked coverage lane"),
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

fn is_production_rust(path: &str) -> bool {
    Path::new(path)
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("rs"))
        && (path.starts_with("xtask/src/")
            || (path.starts_with("crates/") && path.contains("/src/")))
        && !path.contains("/tests/")
        && !path.ends_with("/tests.rs")
        && !path.ends_with("_tests.rs")
        && !path.ends_with("/test_helpers.rs")
}

pub(super) fn is_accelerator_path(path: &str) -> bool {
    has_any_prefix(path, METAL_CRATE_PREFIXES) || has_any_prefix(path, CUDA_CRATE_PREFIXES)
}

fn has_any_prefix(path: &str, prefixes: &[&str]) -> bool {
    prefixes.iter().any(|prefix| path.starts_with(prefix))
}
