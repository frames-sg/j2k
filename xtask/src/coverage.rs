// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::process::{self, cargo, CommandContext};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};

const CHANGED_LINE_THRESHOLD_PERCENT: u64 = 80;
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

const CUDA_SIMT_EVIDENCE: &[EvidenceTest] = &[
    EvidenceTest {
        path: "crates/j2k-cuda-runtime/src/tests.rs",
        name: "cuda_oxide_copy_u8_matches_builtin_copy_and_cpu_when_required",
    },
    EvidenceTest {
        path: "crates/j2k-cuda/tests/htj2k_encode_parity.rs",
        name: "cuda_facade_byte_matches_native_across_matrix_when_required",
    },
    EvidenceTest {
        path: "crates/j2k-transcode-cuda/tests/jpeg_to_htj2k.rs",
        name: "ycbcr_420_batch_transcodes_to_htj2k_with_explicit_cuda_97_codeblock_path",
    },
];

const CUDA_SCAFFOLD_EVIDENCE: &[EvidenceTest] = &[EvidenceTest {
    path: "crates/j2k-cuda-runtime/src/tests.rs",
    name: "kernel_module_names_cover_htj2k_decode_and_encode_stages",
}];

const CUDA_FFI_EVIDENCE: &[EvidenceTest] = &[EvidenceTest {
    path: "crates/j2k-cuda-runtime/src/tests.rs",
    name: "runtime_raii_primitives_smoke_when_required",
}];

const METAL_SHADER_EVIDENCE: &[EvidenceTest] = &[
    EvidenceTest {
        path: "crates/j2k-metal/tests/shader_integrity.rs",
        name: "metal_kernels_are_wired_to_host_pipelines",
    },
    EvidenceTest {
        path: "crates/j2k-metal/tests/device.rs",
        name: "full_classic_grayscale_decode_to_metal_matches_host_decode",
    },
];

const COVERAGE_EXCLUSIONS: &[CoverageExclusion] = &[
    CoverageExclusion {
        id: "cuda-simt-device-rust",
        reason: "CUDA SIMT device Rust is cross-compiled to PTX and cannot be instrumented by host LLVM coverage",
        matcher: ExclusionMatcher::PathPattern {
            prefix: "crates/j2k-cuda-runtime/src/cuda_oxide_",
            contains: Some("/simt/src/"),
            excludes: None,
            suffix: "main.rs",
        },
        evidence: CUDA_SIMT_EVIDENCE,
    },
    CoverageExclusion {
        id: "cuda-generated-host-scaffold",
        reason: "generated cuda-oxide host project scaffolds contain only the build entry point",
        matcher: ExclusionMatcher::PathPattern {
            prefix: "crates/j2k-cuda-runtime/src/cuda_oxide_",
            contains: Some("/src/"),
            excludes: Some("/simt/"),
            suffix: "main.rs",
        },
        evidence: CUDA_SCAFFOLD_EVIDENCE,
    },
    CoverageExclusion {
        id: "cuda-shared-simt-prelude",
        reason: "shared CUDA SIMT device helpers are included into PTX crates and are not host-instrumentable",
        matcher: ExclusionMatcher::WholeFile {
            path: "crates/j2k-cuda-runtime/src/cuda_oxide_simt_prelude.rs",
        },
        evidence: CUDA_SIMT_EVIDENCE,
    },
    CoverageExclusion {
        id: "cuda-driver-ffi-declarations",
        reason: "CUDA Driver and NVTX FFI type declarations have no executable host coverage region",
        matcher: ExclusionMatcher::MarkerSpan {
            path: "crates/j2k-cuda-runtime/src/driver.rs",
            start: "pub(crate) type CuResult = c_int;",
            end: "pub(crate) type NvtxRangePop = unsafe extern \"C\" fn() -> c_int;",
        },
        evidence: CUDA_FFI_EVIDENCE,
    },
    CoverageExclusion {
        id: "metal-embedded-shader-body",
        reason: "the embedded Metal shader body is MSL text, not executable host Rust",
        matcher: ExclusionMatcher::MarkerSpan {
            path: "crates/j2k-metal/src/compute/shader_source.rs",
            start: "        r\"",
            end: "\",",
        },
        evidence: METAL_SHADER_EVIDENCE,
    },
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CoverageLane {
    Host,
    Metal,
    Cuda,
}

impl CoverageLane {
    const fn name(self) -> &'static str {
        match self {
            Self::Host => "host",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }

    const fn lcov_path(self) -> &'static str {
        match self {
            Self::Host => HOST_LCOV_PATH,
            Self::Metal => METAL_LCOV_PATH,
            Self::Cuda => CUDA_LCOV_PATH,
        }
    }

    const fn summary_path(self) -> &'static str {
        match self {
            Self::Host => HOST_SUMMARY_PATH,
            Self::Metal => METAL_SUMMARY_PATH,
            Self::Cuda => CUDA_SUMMARY_PATH,
        }
    }

    fn includes_path(self, path: &str) -> bool {
        match self {
            Self::Host => is_production_rust(path),
            Self::Metal => is_production_rust(path) && has_any_prefix(path, METAL_CRATE_PREFIXES),
            Self::Cuda => is_production_rust(path) && has_any_prefix(path, CUDA_CRATE_PREFIXES),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct EvidenceTest {
    path: &'static str,
    name: &'static str,
}

#[derive(Clone, Copy, Debug)]
struct CoverageExclusion {
    id: &'static str,
    reason: &'static str,
    matcher: ExclusionMatcher,
    evidence: &'static [EvidenceTest],
}

#[derive(Clone, Copy, Debug)]
enum ExclusionMatcher {
    WholeFile {
        path: &'static str,
    },
    PathPattern {
        prefix: &'static str,
        contains: Option<&'static str>,
        excludes: Option<&'static str>,
        suffix: &'static str,
    },
    MarkerSpan {
        path: &'static str,
        start: &'static str,
        end: &'static str,
    },
}

#[derive(Debug, Default)]
struct LcovReport {
    lines: BTreeMap<String, BTreeMap<usize, u64>>,
}

#[derive(Debug)]
struct CoverageOptions {
    lane: CoverageLane,
    base: Option<String>,
    output: Option<PathBuf>,
}

#[derive(Debug, Default)]
struct CoverageCounts {
    measurable: usize,
    covered: usize,
}

#[derive(Debug)]
struct ChangedCoverageResult {
    overall: CoverageCounts,
    accelerator: CoverageCounts,
    changed_files: BTreeSet<String>,
    uncovered: Vec<(String, usize)>,
    unmeasured: Vec<(String, usize)>,
    exclusions: BTreeMap<&'static str, usize>,
    absent_instrumentable_files: Vec<String>,
}

pub(crate) fn coverage(args: impl Iterator<Item = String>) -> Result<(), String> {
    let options = parse_options(args)?;
    let root =
        env::current_dir().map_err(|err| format!("failed to locate repository root: {err}"))?;
    validate_exclusion_policy(&root)?;

    let lcov_path = root.join(options.lane.lcov_path());
    run_lane(options.lane, &lcov_path)?;

    let base = resolve_diff_base(options.base.as_deref())?;
    let merge_base = git_output(&["merge-base", "HEAD", &base])?;
    let diff = git_output(&[
        "diff",
        "--unified=0",
        "--no-ext-diff",
        "--diff-filter=ACMR",
        &merge_base,
        "--",
        "*.rs",
    ])?;
    let changed = parse_changed_lines(&diff)?;
    let lcov = fs::read_to_string(&lcov_path)
        .map_err(|err| format!("failed to read {}: {err}", lcov_path.display()))?;
    let report = parse_lcov(&lcov, &root)?;
    if report.lines.is_empty() {
        return Err(format!(
            "{} did not contain any Rust coverage records",
            lcov_path.display()
        ));
    }

    let result = evaluate_changed_coverage(options.lane, &root, &changed, &report)?;
    let violations = coverage_violations(options.lane, &result);
    let summary_path = options
        .output
        .unwrap_or_else(|| root.join(options.lane.summary_path()));
    write_summary(
        &summary_path,
        options.lane,
        &base,
        &merge_base,
        &lcov_path,
        &result,
        &violations,
    )?;
    print_summary(options.lane, &summary_path, &result);

    if violations.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "changed-line coverage failed:\n{}",
            violations
                .iter()
                .map(|violation| format!("- {violation}"))
                .collect::<Vec<_>>()
                .join("\n")
        ))
    }
}

fn parse_options(args: impl Iterator<Item = String>) -> Result<CoverageOptions, String> {
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

fn run_lane(lane: CoverageLane, lcov_path: &Path) -> Result<(), String> {
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

fn resolve_diff_base(explicit: Option<&str>) -> Result<String, String> {
    if let Some(base) = explicit {
        verify_git_revision(base)?;
        return Ok(base.to_string());
    }
    if let Ok(base) = env::var("J2K_COVERAGE_BASE") {
        if base.trim().is_empty() {
            return Err("J2K_COVERAGE_BASE must not be empty".to_string());
        }
        verify_git_revision(&base)?;
        return Ok(base);
    }
    if let Ok(base_ref) = env::var("GITHUB_BASE_REF") {
        if !base_ref.trim().is_empty() {
            for candidate in [format!("origin/{base_ref}"), base_ref] {
                if verify_git_revision(&candidate).is_ok() {
                    return Ok(candidate);
                }
            }
            return Err(
                "GITHUB_BASE_REF is not available locally; coverage checkout must use fetch-depth: 0"
                    .to_string(),
            );
        }
    }

    let fallback = "HEAD^";
    verify_git_revision(fallback).map_err(|_| {
        "cannot resolve a changed-line coverage base; pass --base or set J2K_COVERAGE_BASE"
            .to_string()
    })?;
    Ok(fallback.to_string())
}

fn verify_git_revision(revision: &str) -> Result<(), String> {
    git_output(&["rev-parse", "--verify", &format!("{revision}^{{commit}}")]).map(|_| ())
}

fn git_output(args: &[&str]) -> Result<String, String> {
    let output = process::command_output(OsString::from("git"), args, CommandContext::new())?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(format!(
            "`git {}` exited with {}{}",
            args.join(" "),
            output.status,
            if stderr.is_empty() {
                String::new()
            } else {
                format!(": {stderr}")
            }
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn parse_changed_lines(diff: &str) -> Result<BTreeMap<String, BTreeSet<usize>>, String> {
    let mut changed = BTreeMap::<String, BTreeSet<usize>>::new();
    let mut current_path = None::<String>;

    for line in diff.lines() {
        if let Some(path) = line.strip_prefix("+++ b/") {
            current_path = Some(path.to_string());
            continue;
        }
        if !line.starts_with("@@ ") {
            continue;
        }
        let path = current_path
            .as_ref()
            .ok_or_else(|| format!("diff hunk has no destination path: {line}"))?;
        let added = line
            .split_whitespace()
            .find(|part| part.starts_with('+'))
            .ok_or_else(|| format!("diff hunk has no added range: {line}"))?;
        let range = added
            .trim_start_matches('+')
            .split_once(',')
            .map_or((added.trim_start_matches('+'), "1"), |(start, count)| {
                (start, count)
            });
        let start = range
            .0
            .parse::<usize>()
            .map_err(|err| format!("invalid diff hunk start in `{line}`: {err}"))?;
        let count = range
            .1
            .parse::<usize>()
            .map_err(|err| format!("invalid diff hunk count in `{line}`: {err}"))?;
        if count == 0 {
            continue;
        }
        let end = start
            .checked_add(count)
            .ok_or_else(|| format!("diff hunk range overflows in `{line}`"))?;
        changed.entry(path.clone()).or_default().extend(start..end);
    }
    Ok(changed)
}

fn parse_lcov(input: &str, root: &Path) -> Result<LcovReport, String> {
    let mut report = LcovReport::default();
    let mut current_path = None::<String>;

    for line in input.lines() {
        if let Some(path) = line.strip_prefix("SF:") {
            current_path = Some(normalize_lcov_path(path, root)?);
            continue;
        }
        let Some(data) = line.strip_prefix("DA:") else {
            continue;
        };
        let path = current_path
            .as_ref()
            .ok_or_else(|| format!("LCOV DA record has no source file: {line}"))?;
        let mut fields = data.split(',');
        let line_number = fields
            .next()
            .ok_or_else(|| format!("LCOV DA record has no line number: {line}"))?
            .parse::<usize>()
            .map_err(|err| format!("invalid LCOV line number in `{line}`: {err}"))?;
        let count = fields
            .next()
            .ok_or_else(|| format!("LCOV DA record has no execution count: {line}"))?
            .parse::<u64>()
            .map_err(|err| format!("invalid LCOV execution count in `{line}`: {err}"))?;
        report
            .lines
            .entry(path.clone())
            .or_default()
            .entry(line_number)
            .and_modify(|existing| *existing = (*existing).max(count))
            .or_insert(count);
    }
    Ok(report)
}

fn normalize_lcov_path(path: &str, root: &Path) -> Result<String, String> {
    let path = Path::new(path);
    let relative = if path.is_absolute() {
        path.strip_prefix(root).map_err(|_| {
            format!(
                "LCOV source {} is outside repository root {}",
                path.display(),
                root.display()
            )
        })?
    } else {
        path.strip_prefix("./").unwrap_or(path)
    };
    Ok(relative
        .components()
        .map(|part| part.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/"))
}

fn evaluate_changed_coverage(
    lane: CoverageLane,
    root: &Path,
    changed: &BTreeMap<String, BTreeSet<usize>>,
    report: &LcovReport,
) -> Result<ChangedCoverageResult, String> {
    let mut result = ChangedCoverageResult {
        overall: CoverageCounts::default(),
        accelerator: CoverageCounts::default(),
        changed_files: BTreeSet::new(),
        uncovered: Vec::new(),
        unmeasured: Vec::new(),
        exclusions: BTreeMap::new(),
        absent_instrumentable_files: Vec::new(),
    };

    for (path, lines) in changed {
        if !lane.includes_path(path) {
            continue;
        }
        let source_path = root.join(path);
        let source = fs::read_to_string(&source_path).map_err(|err| {
            format!(
                "failed to read changed source {}: {err}",
                source_path.display()
            )
        })?;
        let source_lines = source.lines().collect::<Vec<_>>();
        let test_module_start = terminal_test_module_start(&source_lines);
        let file_coverage = report.lines.get(path);
        let mut changed_unexcluded = false;

        result.changed_files.insert(path.clone());
        for &line_number in lines {
            if line_number == 0 || line_number > source_lines.len() {
                continue;
            }
            if test_module_start.is_some_and(|start| line_number >= start) {
                continue;
            }
            if let Some(exclusion) = matching_exclusion(path, line_number, &source_lines)? {
                *result.exclusions.entry(exclusion.id).or_default() += 1;
                continue;
            }
            changed_unexcluded = true;
            let Some(count) = file_coverage.and_then(|coverage| coverage.get(&line_number)) else {
                result.unmeasured.push((path.clone(), line_number));
                continue;
            };

            result.overall.measurable += 1;
            if *count > 0 {
                result.overall.covered += 1;
            } else {
                result.uncovered.push((path.clone(), line_number));
            }
            if is_accelerator_path(path) {
                result.accelerator.measurable += 1;
                if *count > 0 {
                    result.accelerator.covered += 1;
                }
            }
        }

        if lane != CoverageLane::Host
            && changed_unexcluded
            && file_coverage.is_none()
            && source_has_instrumentable_function(path, &source_lines)?
        {
            result.absent_instrumentable_files.push(path.clone());
        }
    }

    Ok(result)
}

fn terminal_test_module_start(source: &[&str]) -> Option<usize> {
    source.windows(3).enumerate().find_map(|(index, lines)| {
        let first = lines[0].trim();
        let second = lines[1].trim();
        let third = lines[2].trim();
        if first == "#[cfg(test)]"
            && (second.starts_with("mod tests") || third.starts_with("mod tests"))
        {
            Some(index + 1)
        } else {
            None
        }
    })
}

fn source_has_instrumentable_function(path: &str, source: &[&str]) -> Result<bool, String> {
    for (index, line) in source.iter().enumerate() {
        let line_number = index + 1;
        if terminal_test_module_start(source).is_some_and(|start| line_number >= start)
            || matching_exclusion(path, line_number, source)?.is_some()
        {
            continue;
        }
        let trimmed = line.trim_start();
        if !trimmed.starts_with("//")
            && !trimmed.starts_with('*')
            && (trimmed.starts_with("fn ")
                || trimmed.starts_with("pub fn ")
                || trimmed.starts_with("pub(crate) fn ")
                || trimmed.starts_with("pub(super) fn ")
                || trimmed.starts_with("const fn ")
                || trimmed.starts_with("pub const fn "))
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn matching_exclusion(
    path: &str,
    line_number: usize,
    source: &[&str],
) -> Result<Option<&'static CoverageExclusion>, String> {
    for exclusion in COVERAGE_EXCLUSIONS {
        if exclusion_matches(exclusion, path, line_number, source)? {
            return Ok(Some(exclusion));
        }
    }
    Ok(None)
}

fn exclusion_matches(
    exclusion: &CoverageExclusion,
    path: &str,
    line_number: usize,
    source: &[&str],
) -> Result<bool, String> {
    match exclusion.matcher {
        ExclusionMatcher::WholeFile { path: exact } => Ok(path == exact),
        ExclusionMatcher::PathPattern {
            prefix,
            contains,
            excludes,
            suffix,
        } => Ok(path.starts_with(prefix)
            && contains.is_none_or(|needle| path.contains(needle))
            && excludes.is_none_or(|needle| !path.contains(needle))
            && path.ends_with(suffix)),
        ExclusionMatcher::MarkerSpan {
            path: exact,
            start,
            end,
        } => {
            if path != exact {
                return Ok(false);
            }
            let start_line = unique_marker_line(source, start, exclusion.id)?;
            let end_line = unique_marker_line(source, end, exclusion.id)?;
            if start_line > end_line {
                return Err(format!(
                    "coverage exclusion `{}` marker order is invalid",
                    exclusion.id
                ));
            }
            Ok((start_line..=end_line).contains(&line_number))
        }
    }
}

fn unique_marker_line(source: &[&str], marker: &str, id: &str) -> Result<usize, String> {
    let matches = source
        .iter()
        .enumerate()
        .filter_map(|(index, line)| (*line == marker).then_some(index + 1))
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [line] => Ok(*line),
        [] => Err(format!(
            "coverage exclusion `{id}` marker `{marker}` is missing"
        )),
        _ => Err(format!(
            "coverage exclusion `{id}` marker `{marker}` is ambiguous"
        )),
    }
}

fn coverage_violations(lane: CoverageLane, result: &ChangedCoverageResult) -> Vec<String> {
    let mut violations = Vec::new();
    if !meets_threshold(&result.overall) {
        violations.push(format!(
            "{} changed executable Rust lines are {:.2}% covered ({} / {}), below {}%",
            lane.name(),
            coverage_percent(&result.overall).unwrap_or(0.0),
            result.overall.covered,
            result.overall.measurable,
            CHANGED_LINE_THRESHOLD_PERCENT
        ));
    }
    if result.accelerator.measurable > 0 && !meets_threshold(&result.accelerator) {
        violations.push(format!(
            "{} changed accelerator host lines are {:.2}% covered ({} / {}), below {}%",
            lane.name(),
            coverage_percent(&result.accelerator).unwrap_or(0.0),
            result.accelerator.covered,
            result.accelerator.measurable,
            CHANGED_LINE_THRESHOLD_PERCENT
        ));
    }
    if !result.absent_instrumentable_files.is_empty() {
        violations.push(format!(
            "instrumentable accelerator source files are absent from the {} LCOV artifact: {}",
            lane.name(),
            result.absent_instrumentable_files.join(", ")
        ));
    }
    violations
}

fn meets_threshold(counts: &CoverageCounts) -> bool {
    counts.measurable == 0
        || counts.covered.saturating_mul(100)
            >= counts
                .measurable
                .saturating_mul(CHANGED_LINE_THRESHOLD_PERCENT as usize)
}

fn coverage_percent(counts: &CoverageCounts) -> Option<f64> {
    (counts.measurable > 0).then(|| counts.covered as f64 * 100.0 / counts.measurable as f64)
}

fn write_summary(
    path: &Path,
    lane: CoverageLane,
    base: &str,
    merge_base: &str,
    lcov_path: &Path,
    result: &ChangedCoverageResult,
    violations: &[String],
) -> Result<(), String> {
    let exclusions = COVERAGE_EXCLUSIONS
        .iter()
        .map(|exclusion| {
            json!({
                "id": exclusion.id,
                "reason": exclusion.reason,
                "changed_lines_excluded": result.exclusions.get(exclusion.id).copied().unwrap_or(0),
                "evidence_tests": exclusion.evidence.iter().map(|evidence| {
                    format!("{}::{}", evidence.path, evidence.name)
                }).collect::<Vec<_>>(),
            })
        })
        .collect::<Vec<_>>();
    let document = json!({
        "schema": "j2k-changed-line-coverage-v1",
        "lane": lane.name(),
        "status": if violations.is_empty() { "passed" } else { "failed" },
        "base": base,
        "merge_base": merge_base,
        "threshold_percent": CHANGED_LINE_THRESHOLD_PERCENT,
        "lcov_artifact": lcov_path.file_name().map_or_else(String::new, |name| name.to_string_lossy().into_owned()),
        "changed_files": result.changed_files,
        "overall": {
            "measurable_lines": result.overall.measurable,
            "covered_lines": result.overall.covered,
            "coverage_percent": coverage_percent(&result.overall),
        },
        "accelerator_host_rust": {
            "measurable_lines": result.accelerator.measurable,
            "covered_lines": result.accelerator.covered,
            "coverage_percent": coverage_percent(&result.accelerator),
        },
        "uncovered_lines": result.uncovered.iter().map(|(path, line)| format!("{path}:{line}")).collect::<Vec<_>>(),
        "non_executable_or_not_instrumented_lines": result.unmeasured.iter().map(|(path, line)| format!("{path}:{line}")).collect::<Vec<_>>(),
        "absent_instrumentable_files": result.absent_instrumentable_files,
        "narrow_exclusions": exclusions,
        "violations": violations,
    });
    let rendered = serde_json::to_string_pretty(&document)
        .map_err(|err| format!("failed to render coverage summary: {err}"))?;
    fs::write(path, format!("{rendered}\n"))
        .map_err(|err| format!("failed to write {}: {err}", path.display()))
}

fn print_summary(lane: CoverageLane, summary_path: &Path, result: &ChangedCoverageResult) {
    let percent = coverage_percent(&result.overall)
        .map_or_else(|| "n/a".to_string(), |value| format!("{value:.2}%"));
    let accelerator_percent = coverage_percent(&result.accelerator)
        .map_or_else(|| "n/a".to_string(), |value| format!("{value:.2}%"));
    eprintln!(
        "{} changed-line coverage: {} ({} / {} measurable lines)",
        lane.name(),
        percent,
        result.overall.covered,
        result.overall.measurable
    );
    eprintln!(
        "{} accelerator host coverage: {} ({} / {} measurable lines)",
        lane.name(),
        accelerator_percent,
        result.accelerator.covered,
        result.accelerator.measurable
    );
    eprintln!("coverage evidence: {}", summary_path.display());
}

fn validate_exclusion_policy(root: &Path) -> Result<(), String> {
    let mut ids = BTreeSet::new();
    for exclusion in COVERAGE_EXCLUSIONS {
        if !ids.insert(exclusion.id) {
            return Err(format!(
                "duplicate coverage exclusion id `{}`",
                exclusion.id
            ));
        }
        if exclusion.reason.trim().is_empty() || exclusion.evidence.is_empty() {
            return Err(format!(
                "coverage exclusion `{}` needs a reason and evidence tests",
                exclusion.id
            ));
        }
        for evidence in exclusion.evidence {
            let source = fs::read_to_string(root.join(evidence.path)).map_err(|err| {
                format!(
                    "coverage exclusion `{}` evidence {} is unavailable: {err}",
                    exclusion.id, evidence.path
                )
            })?;
            if !source.contains(&format!("fn {}(", evidence.name)) {
                return Err(format!(
                    "coverage exclusion `{}` evidence test `{}::{}` is missing",
                    exclusion.id, evidence.path, evidence.name
                ));
            }
        }
        validate_exclusion_matcher(root, exclusion)?;
    }
    Ok(())
}

fn validate_exclusion_matcher(root: &Path, exclusion: &CoverageExclusion) -> Result<(), String> {
    match exclusion.matcher {
        ExclusionMatcher::WholeFile { path } => {
            if !root.join(path).is_file() {
                return Err(format!(
                    "coverage exclusion `{}` file `{path}` is missing",
                    exclusion.id
                ));
            }
        }
        ExclusionMatcher::PathPattern { .. } => {
            let cuda_runtime_src = root.join("crates/j2k-cuda-runtime/src");
            let matched = collect_rust_files(&cuda_runtime_src, root)?
                .into_iter()
                .any(|path| exclusion_matches(exclusion, &path, 1, &[]).unwrap_or(false));
            if !matched {
                return Err(format!(
                    "coverage exclusion `{}` path pattern matches no Rust file",
                    exclusion.id
                ));
            }
        }
        ExclusionMatcher::MarkerSpan { path, .. } => {
            let source_path = root.join(path);
            let source = fs::read_to_string(&source_path).map_err(|err| {
                format!(
                    "coverage exclusion `{}` cannot read {}: {err}",
                    exclusion.id,
                    source_path.display()
                )
            })?;
            let lines = source.lines().collect::<Vec<_>>();
            let matched = (1..=lines.len())
                .any(|line| exclusion_matches(exclusion, path, line, &lines).unwrap_or(false));
            if !matched {
                return Err(format!(
                    "coverage exclusion `{}` marker span matches no line",
                    exclusion.id
                ));
            }
        }
    }
    Ok(())
}

fn collect_rust_files(directory: &Path, root: &Path) -> Result<Vec<String>, String> {
    let mut files = Vec::new();
    let mut pending = vec![directory.to_path_buf()];
    while let Some(current) = pending.pop() {
        for entry in fs::read_dir(&current)
            .map_err(|err| format!("failed to inspect {}: {err}", current.display()))?
        {
            let entry = entry.map_err(|err| format!("failed to inspect directory entry: {err}"))?;
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                files.push(
                    path.strip_prefix(root)
                        .map_err(|err| format!("failed to normalize {}: {err}", path.display()))?
                        .components()
                        .map(|part| part.as_os_str().to_string_lossy())
                        .collect::<Vec<_>>()
                        .join("/"),
                );
            }
        }
    }
    Ok(files)
}

fn is_production_rust(path: &str) -> bool {
    path.ends_with(".rs")
        && (path.starts_with("xtask/src/")
            || (path.starts_with("crates/") && path.contains("/src/")))
        && !path.contains("/tests/")
        && !path.ends_with("/tests.rs")
        && !path.ends_with("_tests.rs")
        && !path.ends_with("/test_helpers.rs")
}

fn is_accelerator_path(path: &str) -> bool {
    has_any_prefix(path, METAL_CRATE_PREFIXES) || has_any_prefix(path, CUDA_CRATE_PREFIXES)
}

fn has_any_prefix(path: &str, prefixes: &[&str]) -> bool {
    prefixes.iter().any(|prefix| path.starts_with(prefix))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_result(measurable: usize, covered: usize) -> ChangedCoverageResult {
        ChangedCoverageResult {
            overall: CoverageCounts {
                measurable,
                covered,
            },
            accelerator: CoverageCounts {
                measurable,
                covered,
            },
            changed_files: BTreeSet::new(),
            uncovered: Vec::new(),
            unmeasured: Vec::new(),
            exclusions: BTreeMap::new(),
            absent_instrumentable_files: Vec::new(),
        }
    }

    #[test]
    fn parses_added_diff_hunks_without_counting_deletions() {
        let diff = "\
diff --git a/crates/a/src/lib.rs b/crates/a/src/lib.rs
--- a/crates/a/src/lib.rs
+++ b/crates/a/src/lib.rs
@@ -2,0 +3,2 @@
+first
+second
@@ -8 +10 @@
-old
+new
";

        let changed = parse_changed_lines(diff).unwrap();

        assert_eq!(changed["crates/a/src/lib.rs"], BTreeSet::from([3, 4, 10]));
    }

    #[test]
    fn lcov_parser_merges_duplicate_line_records_by_max_count() {
        let root = Path::new("/repo");
        let lcov = "\
SF:/repo/crates/a/src/lib.rs
DA:3,0
DA:4,2
end_of_record
SF:/repo/crates/a/src/lib.rs
DA:3,1
end_of_record
";

        let report = parse_lcov(lcov, root).unwrap();

        assert_eq!(report.lines["crates/a/src/lib.rs"][&3], 1);
        assert_eq!(report.lines["crates/a/src/lib.rs"][&4], 2);
    }

    #[test]
    fn eighty_percent_changed_line_coverage_passes_exactly() {
        let result = synthetic_result(5, 4);
        assert!(coverage_violations(CoverageLane::Cuda, &result).is_empty());
    }

    #[test]
    fn accelerator_threshold_cannot_be_masked_by_cpu_coverage() {
        let mut result = synthetic_result(100, 99);
        result.accelerator = CoverageCounts {
            measurable: 5,
            covered: 3,
        };

        let violations = coverage_violations(CoverageLane::Host, &result);

        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("accelerator host lines"));
    }

    #[test]
    fn hosted_changed_line_gate_includes_all_production_rust() {
        assert!(CoverageLane::Host.includes_path("crates/j2k-cuda/src/error.rs"));
        assert!(CoverageLane::Host.includes_path("crates/j2k-metal/src/error.rs"));
        assert!(CoverageLane::Host.includes_path("crates/j2k/src/error.rs"));
        assert!(CoverageLane::Host.includes_path("xtask/src/coverage.rs"));
        assert!(!CoverageLane::Host.includes_path("crates/j2k/tests/decode.rs"));
    }

    #[test]
    fn metal_raw_shader_span_is_narrower_than_the_host_source_file() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let path = "crates/j2k-metal/src/compute/shader_source.rs";
        let source = fs::read_to_string(root.join(path)).unwrap();
        let lines = source.lines().collect::<Vec<_>>();

        assert_eq!(
            matching_exclusion(path, 7, &lines)
                .unwrap()
                .map(|rule| rule.id),
            Some("metal-embedded-shader-body")
        );
        assert!(matching_exclusion(path, 563, &lines).unwrap().is_none());
    }

    #[test]
    fn exclusion_policy_maps_every_narrow_rule_to_existing_tests() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        validate_exclusion_policy(&root).unwrap();
        assert!(COVERAGE_EXCLUSIONS
            .iter()
            .all(|rule| !rule.evidence.is_empty()));
        assert!(!COVERAGE_EXCLUSIONS.iter().any(|rule| {
            matches!(
                rule.matcher,
                ExclusionMatcher::WholeFile {
                    path: "crates/j2k-cuda/" | "crates/j2k-metal/"
                }
            )
        }));
    }

    #[test]
    fn coverage_cli_defaults_to_host_and_accepts_explicit_lanes() {
        let default = parse_options(std::iter::empty()).unwrap();
        let metal = parse_options(
            [
                "metal".to_string(),
                "--base".to_string(),
                "HEAD^".to_string(),
            ]
            .into_iter(),
        )
        .unwrap();
        let cuda = parse_options(["cuda".to_string()].into_iter()).unwrap();

        assert_eq!(default.lane, CoverageLane::Host);
        assert_eq!(metal.lane, CoverageLane::Metal);
        assert_eq!(metal.base.as_deref(), Some("HEAD^"));
        assert_eq!(cuda.lane, CoverageLane::Cuda);
    }
}
