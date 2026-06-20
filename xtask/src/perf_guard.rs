use std::{
    collections::BTreeMap,
    env,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
};

use crate::process::{self, CommandContext};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BenchEstimate {
    pub(crate) id: String,
    pub(crate) median_ns: f64,
    pub(crate) median_lower_ns: f64,
    pub(crate) median_upper_ns: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RegressionOutcome {
    pub(crate) id: String,
    pub(crate) baseline_ns: f64,
    pub(crate) current_ns: f64,
    pub(crate) delta_percent: f64,
    pub(crate) enforced: bool,
    pub(crate) threshold_exceeded: bool,
    pub(crate) regressed: bool,
}

#[derive(Debug, Clone)]
struct PerfGuardOptions {
    mode: PerfGuardMode,
    threshold_percent: f64,
    quick: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum PerfGuardMode {
    GitRef { baseline_ref: String },
    RecordCurrent { name: String },
    CompareCurrent { name: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BenchCommand {
    package: &'static str,
    bench: &'static str,
    filter: Option<&'static str>,
    features: Option<&'static str>,
    env: &'static [(&'static str, &'static str)],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BenchManifestStanza {
    path: &'static str,
    bench_name: &'static str,
    stanza: &'static str,
}

const DEFAULT_BASELINE_REF: &str = "j2k-bench-original";
const DEFAULT_THRESHOLD_PERCENT: f64 = 10.0;
const MIN_ABSOLUTE_REGRESSION_NS: f64 = 100.0;
const BENCH_COMMANDS: &[BenchCommand] = &[
    BenchCommand {
        package: "j2k",
        bench: "public_api",
        filter: None,
        features: None,
        env: &[],
    },
    BenchCommand {
        package: "j2k-jpeg",
        bench: "encode_cpu",
        filter: Some("jpeg_cpu_encode_runtime/"),
        features: None,
        env: &[],
    },
    BenchCommand {
        package: "j2k-native",
        bench: "tier1_bitplane",
        filter: Some("htj2k_cleanup_decode/"),
        features: None,
        env: &[],
    },
    BenchCommand {
        package: "j2k-native",
        bench: "tier1_bitplane",
        filter: Some("htj2k_refinement_fixture_decode"),
        features: None,
        env: &[],
    },
    BenchCommand {
        package: "j2k-native",
        bench: "tier1_bitplane",
        filter: Some("htj2k_refinement_block_decode"),
        features: None,
        env: &[],
    },
    BenchCommand {
        package: "j2k-native",
        bench: "tier1_bitplane",
        filter: Some("htj2k_cleanup_encode/"),
        features: None,
        env: &[],
    },
    BenchCommand {
        package: "j2k-native",
        bench: "tier1_bitplane",
        filter: Some("htj2k_cleanup_encode_distribution"),
        features: None,
        env: &[],
    },
    BenchCommand {
        package: "j2k-native",
        bench: "htj2k_sigprop_phase",
        filter: None,
        features: None,
        env: &[],
    },
    BenchCommand {
        package: "j2k-cuda",
        bench: "htj2k_decode",
        filter: Some("j2k_cuda_htj2k_"),
        features: Some("cuda-runtime"),
        env: &[],
    },
    BenchCommand {
        package: "j2k-cuda",
        bench: "htj2k_encode",
        filter: Some("j2k_cuda_htj2k_"),
        features: Some("cuda-runtime"),
        env: &[],
    },
];
const BENCH_SOURCE_FILES: &[&str] = &[
    "crates/j2k/benches/public_api.rs",
    "crates/j2k-jpeg/benches/encode_cpu.rs",
    "crates/j2k-native/benches/tier1_bitplane.rs",
    "crates/j2k-native/benches/htj2k_sigprop_phase.rs",
    "crates/j2k-native/fixtures/htj2k/openhtj2k_ds0_ht_09_b11.j2k",
    "crates/j2k-cuda/benches/htj2k_decode.rs",
    "crates/j2k-cuda/benches/htj2k_encode.rs",
];
const BENCH_MANIFEST_STANZAS: &[BenchManifestStanza] = &[
    BenchManifestStanza {
        path: "crates/j2k-jpeg/Cargo.toml",
        bench_name: "encode_cpu",
        stanza: "[[bench]]\nname = \"encode_cpu\"\nharness = false\n",
    },
    BenchManifestStanza {
        path: "crates/j2k-cuda/Cargo.toml",
        bench_name: "htj2k_decode",
        stanza: "[[bench]]\nname = \"htj2k_decode\"\nharness = false\n",
    },
    BenchManifestStanza {
        path: "crates/j2k-cuda/Cargo.toml",
        bench_name: "htj2k_encode",
        stanza: "[[bench]]\nname = \"htj2k_encode\"\nharness = false\nrequired-features = [\"cuda-runtime\"]\n",
    },
];

pub(crate) fn j2k_perf_guard(args: impl Iterator<Item = String>) -> Result<(), String> {
    let options = PerfGuardOptions::parse(args)?;
    let root = repo_root()?;
    let perf_root = root.join("target").join("j2k-perf");
    fs::create_dir_all(&perf_root)
        .map_err(|err| format!("failed to create {}: {err}", perf_root.display()))?;

    let outcomes = match &options.mode {
        PerfGuardMode::GitRef { baseline_ref } => {
            let baseline_worktree = perf_root.join("baseline-worktree");
            recreate_baseline_worktree(&root, &baseline_worktree, baseline_ref)?;
            sync_benchmark_sources(&root, &baseline_worktree)?;

            let baseline_target = perf_root.join("baseline-target");
            let current_target = perf_root.join("current-target");
            reset_dir(&baseline_target)?;
            reset_dir(&current_target)?;

            run_benches(&baseline_worktree, &baseline_target, options.quick)?;
            run_benches(&root, &current_target, options.quick)?;

            let baseline = discover_estimates(&baseline_target.join("criterion"))?;
            let current = discover_estimates(&current_target.join("criterion"))?;
            compare_estimates(&baseline, &current, options.threshold_percent)?
        }
        PerfGuardMode::RecordCurrent { name } => {
            let target = perf_root.join("current-record-target");
            reset_dir(&target)?;
            run_benches(&root, &target, options.quick)?;
            let estimates = discover_estimates(&target.join("criterion"))?;
            let snapshot = current_snapshot_path(&perf_root, name)?;
            write_estimate_snapshot(&snapshot, &estimates)?;
            eprintln!(
                "Recorded current-tree performance baseline `{name}` at {}",
                snapshot.display()
            );
            return Ok(());
        }
        PerfGuardMode::CompareCurrent { name } => {
            let snapshot = current_snapshot_path(&perf_root, name)?;
            let baseline = read_estimate_snapshot(&snapshot)?;
            let target = perf_root.join("current-compare-target");
            reset_dir(&target)?;
            run_benches(&root, &target, options.quick)?;
            let current = discover_estimates(&target.join("criterion"))?;
            compare_estimates(&baseline, &current, options.threshold_percent)?
        }
    };
    emit_report(&outcomes, options.threshold_percent);

    if outcomes.iter().any(|outcome| outcome.regressed) {
        Err("Codec performance guard found regressions".to_string())
    } else {
        Ok(())
    }
}

fn current_snapshot_path(perf_root: &Path, name: &str) -> Result<PathBuf, String> {
    validate_snapshot_name(name)?;
    Ok(perf_root
        .join("current-tree-baselines")
        .join(format!("{name}.json")))
}

fn validate_snapshot_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("current-tree baseline name must not be empty".to_string());
    }
    if name
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        Ok(())
    } else {
        Err(format!(
            "current-tree baseline name `{name}` may only contain ASCII letters, digits, '.', '-', and '_'"
        ))
    }
}

fn write_estimate_snapshot(path: &Path, estimates: &[BenchEstimate]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
    }
    let mut sorted = estimates.to_vec();
    sorted.sort_by(|a, b| a.id.cmp(&b.id));
    let values = sorted
        .iter()
        .map(|estimate| {
            serde_json::json!({
                "id": estimate.id,
                "median_ns": estimate.median_ns,
                "median_lower_ns": estimate.median_lower_ns,
                "median_upper_ns": estimate.median_upper_ns,
            })
        })
        .collect::<Vec<_>>();
    let value = serde_json::json!({
        "version": 1,
        "estimates": values,
    });
    let data = serde_json::to_string_pretty(&value)
        .map_err(|err| format!("failed to serialize estimate snapshot: {err}"))?;
    fs::write(path, format!("{data}\n"))
        .map_err(|err| format!("failed to write {}: {err}", path.display()))
}

fn read_estimate_snapshot(path: &Path) -> Result<Vec<BenchEstimate>, String> {
    let data = fs::read_to_string(path)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&data)
        .map_err(|err| format!("failed to parse {}: {err}", path.display()))?;
    let version = value
        .get("version")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| format!("{} is missing version", path.display()))?;
    if version != 1 {
        return Err(format!(
            "{} has unsupported estimate snapshot version {version}",
            path.display()
        ));
    }
    let raw_estimates = value
        .get("estimates")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("{} is missing estimates", path.display()))?;
    let mut estimates = Vec::with_capacity(raw_estimates.len());
    for raw in raw_estimates {
        let id = raw
            .get("id")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| format!("{} contains an estimate without id", path.display()))?
            .to_string();
        let median_ns = raw
            .get("median_ns")
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| format!("{} estimate {id} is missing median_ns", path.display()))?;
        let median_lower_ns = raw
            .get("median_lower_ns")
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| {
                format!(
                    "{} estimate {id} is missing median_lower_ns",
                    path.display()
                )
            })?;
        let median_upper_ns = raw
            .get("median_upper_ns")
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| {
                format!(
                    "{} estimate {id} is missing median_upper_ns",
                    path.display()
                )
            })?;
        estimates.push(BenchEstimate {
            id,
            median_ns,
            median_lower_ns,
            median_upper_ns,
        });
    }
    estimates.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(estimates)
}

pub(crate) fn sync_benchmark_sources(source_root: &Path, target_root: &Path) -> Result<(), String> {
    for relative in BENCH_SOURCE_FILES {
        let source = source_root.join(relative);
        let target = target_root.join(relative);
        let parent = target.parent().ok_or_else(|| {
            format!(
                "benchmark source target has no parent directory: {}",
                target.display()
            )
        })?;
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
        fs::copy(&source, &target).map_err(|err| {
            format!(
                "failed to copy benchmark source {} to {}: {err}",
                source.display(),
                target.display()
            )
        })?;
    }

    for manifest in BENCH_MANIFEST_STANZAS {
        ensure_benchmark_manifest_stanza(target_root, *manifest)?;
    }

    Ok(())
}

fn ensure_benchmark_manifest_stanza(
    target_root: &Path,
    manifest: BenchManifestStanza,
) -> Result<(), String> {
    let path = target_root.join(manifest.path);
    let mut contents = fs::read_to_string(&path)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    if contents.contains(manifest.stanza) {
        return Ok(());
    }
    if contents.contains(&format!("name = \"{}\"", manifest.bench_name)) {
        return Err(format!(
            "{} declares benchmark `{}` without the required Criterion harness stanza",
            path.display(),
            manifest.bench_name
        ));
    }

    if !contents.ends_with('\n') {
        contents.push('\n');
    }
    contents.push('\n');
    contents.push_str(manifest.stanza);
    fs::write(&path, contents).map_err(|err| format!("failed to write {}: {err}", path.display()))
}

pub(crate) fn compare_estimates(
    baseline: &[BenchEstimate],
    current: &[BenchEstimate],
    threshold_percent: f64,
) -> Result<Vec<RegressionOutcome>, String> {
    let baseline_by_id = baseline
        .iter()
        .map(|estimate| (estimate.id.as_str(), estimate))
        .collect::<BTreeMap<_, _>>();
    let current_by_id = current
        .iter()
        .map(|estimate| (estimate.id.as_str(), estimate))
        .collect::<BTreeMap<_, _>>();
    let mut outcomes = Vec::with_capacity(baseline.len());
    for base in baseline {
        let Some(now) = current_by_id.get(base.id.as_str()) else {
            if is_enforced_perf_id(&base.id) {
                return Err(format!("missing current benchmark result for {}", base.id));
            }
            continue;
        };
        if base.median_ns <= 0.0 {
            return Err(format!(
                "baseline benchmark {} has non-positive median {}",
                base.id, base.median_ns
            ));
        }
        let delta_percent = ((now.median_ns - base.median_ns) / base.median_ns) * 100.0;
        let confident_absolute_delta_ns = now.median_lower_ns - base.median_upper_ns;
        let confident_delta_percent = (confident_absolute_delta_ns / base.median_upper_ns) * 100.0;
        let enforced = is_enforced_perf_id(&base.id);
        let threshold_exceeded = confident_delta_percent > threshold_percent
            && confident_absolute_delta_ns > MIN_ABSOLUTE_REGRESSION_NS;
        outcomes.push(RegressionOutcome {
            id: base.id.clone(),
            baseline_ns: base.median_ns,
            current_ns: now.median_ns,
            delta_percent,
            enforced,
            threshold_exceeded,
            regressed: enforced && threshold_exceeded,
        });
    }
    for now in current {
        if is_enforced_perf_id(&now.id) && !baseline_by_id.contains_key(now.id.as_str()) {
            return Err(format!("missing baseline benchmark result for {}", now.id));
        }
    }
    Ok(outcomes)
}

fn is_enforced_perf_id(id: &str) -> bool {
    let id = CriterionPerfId::new(id);
    if REPORT_ONLY_PERF_GROUPS.contains(&id.group) {
        return false;
    }
    ENFORCED_PERF_GROUPS.contains(&id.group)
        || (id.group == "j2k_public_decode"
            && id.first_case_segment().is_some_and(segment_has_htj2k_token))
        || id.has_htj2k_token()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CriterionPerfId<'a> {
    group: &'a str,
    rest: &'a str,
}

impl<'a> CriterionPerfId<'a> {
    fn new(id: &'a str) -> Self {
        let (group, rest) = id.split_once('/').unwrap_or((id, ""));
        Self { group, rest }
    }

    fn first_case_segment(self) -> Option<&'a str> {
        self.rest.split('/').find(|segment| !segment.is_empty())
    }

    fn has_htj2k_token(self) -> bool {
        self.rest.split('/').any(segment_has_htj2k_token)
    }
}

const ENFORCED_PERF_GROUPS: &[&str] = &[
    "jpeg_cpu_encode_runtime",
    "htj2k_cleanup_decode",
    "htj2k_cleanup_encode",
    "htj2k_cleanup_encode_distribution",
    "htj2k_refinement_fixture_decode",
    "htj2k_refinement_block_decode",
    "htj2k_refinement_sigprop_phase",
    "htj2k_cpuupload_decode_batch",
];

const REPORT_ONLY_PERF_GROUPS: &[&str] = &[
    "htj2k_cleanup_encode_parallel_batch_size",
    "htj2k_region_scaled_plan_build",
    "htj2k_feeder_coalesce",
    "htj2k_metal_route",
    "wsi_tile_batch_region_scaled_rgb_q4",
];

fn segment_has_htj2k_token(segment: &str) -> bool {
    segment.split('_').any(|token| token == "htj2k")
}

pub(crate) fn discover_estimates(criterion_root: &Path) -> Result<Vec<BenchEstimate>, String> {
    let mut out = Vec::new();
    discover_estimates_inner(criterion_root, criterion_root, &mut out)?;
    out.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(out)
}

fn discover_estimates_inner(
    criterion_root: &Path,
    dir: &Path,
    out: &mut Vec<BenchEstimate>,
) -> Result<(), String> {
    if !dir.exists() {
        return Err(format!(
            "Criterion output directory does not exist: {}",
            criterion_root.display()
        ));
    }
    for entry in
        fs::read_dir(dir).map_err(|err| format!("failed to read {}: {err}", dir.display()))?
    {
        let entry =
            entry.map_err(|err| format!("failed to read {} entry: {err}", dir.display()))?;
        let path = entry.path();
        if path.is_dir() {
            discover_estimates_inner(criterion_root, &path, out)?;
            continue;
        }
        if path.file_name().and_then(|name| name.to_str()) != Some("estimates.json") {
            continue;
        }
        if path
            .parent()
            .and_then(Path::file_name)
            .and_then(|name| name.to_str())
            != Some("new")
        {
            continue;
        }
        let id = estimate_id(criterion_root, &path)?;
        let (median_ns, median_lower_ns, median_upper_ns) = read_median_estimate(&path)?;
        out.push(BenchEstimate {
            id,
            median_ns,
            median_lower_ns,
            median_upper_ns,
        });
    }
    Ok(())
}

fn estimate_id(criterion_root: &Path, estimate_path: &Path) -> Result<String, String> {
    let bench_path = estimate_path
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| {
            format!(
                "invalid Criterion estimate path {}",
                estimate_path.display()
            )
        })?;
    let rel = bench_path.strip_prefix(criterion_root).map_err(|err| {
        format!(
            "failed to strip Criterion root {} from {}: {err}",
            criterion_root.display(),
            bench_path.display()
        )
    })?;
    let mut parts = Vec::new();
    for component in rel.components() {
        parts.push(component.as_os_str().to_string_lossy().into_owned());
    }
    Ok(parts.join("/"))
}

fn read_median_estimate(path: &Path) -> Result<(f64, f64, f64), String> {
    let data = fs::read_to_string(path)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&data)
        .map_err(|err| format!("failed to parse {}: {err}", path.display()))?;
    let median = value
        .get("median")
        .ok_or_else(|| format!("{} is missing median", path.display()))?;
    let point_estimate = median
        .get("point_estimate")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| format!("{} is missing median.point_estimate", path.display()))?;
    let confidence_interval = median
        .get("confidence_interval")
        .ok_or_else(|| format!("{} is missing median.confidence_interval", path.display()))?;
    let lower_bound = confidence_interval
        .get("lower_bound")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| {
            format!(
                "{} is missing median.confidence_interval.lower_bound",
                path.display()
            )
        })?;
    let upper_bound = confidence_interval
        .get("upper_bound")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| {
            format!(
                "{} is missing median.confidence_interval.upper_bound",
                path.display()
            )
        })?;
    Ok((point_estimate, lower_bound, upper_bound))
}

fn emit_report(outcomes: &[RegressionOutcome], threshold_percent: f64) {
    eprintln!("Codec performance guard threshold: +{threshold_percent:.2}% median");
    for outcome in outcomes {
        let status = if outcome.regressed {
            "REGRESSED"
        } else if outcome.threshold_exceeded && !outcome.enforced {
            "report"
        } else {
            "ok"
        };
        eprintln!(
            "{status:9} {:>9.2}% baseline={:.2}ns current={:.2}ns {}",
            outcome.delta_percent, outcome.baseline_ns, outcome.current_ns, outcome.id
        );
    }
}

fn run_benches(workdir: &Path, target_dir: &Path, quick: bool) -> Result<(), String> {
    for bench in BENCH_COMMANDS {
        let args = bench_args(*bench, quick);
        process::run_command(
            cargo(),
            &args,
            CommandContext::new()
                .current_dir(workdir)
                .target_dir(target_dir)
                .envs(bench.env),
        )?;
    }
    Ok(())
}

fn bench_args(bench: BenchCommand, quick: bool) -> Vec<&'static str> {
    let mut args = vec!["bench", "-p", bench.package, "--bench", bench.bench];
    if let Some(features) = bench.features {
        args.push("--features");
        args.push(features);
    }
    if bench.filter.is_some() || quick {
        args.push("--");
        if let Some(filter) = bench.filter {
            args.push(filter);
        }
        if quick {
            args.push("--quick");
        }
    }
    args
}

fn recreate_baseline_worktree(
    root: &Path,
    worktree: &Path,
    baseline_ref: &str,
) -> Result<(), String> {
    if worktree.exists() {
        process::run_command(
            OsString::from("git"),
            &["worktree", "remove", "--force", path_str(worktree)?],
            CommandContext::new().current_dir(root),
        )?;
    }
    process::run_command(
        OsString::from("git"),
        &[
            "worktree",
            "add",
            "--detach",
            path_str(worktree)?,
            baseline_ref,
        ],
        CommandContext::new().current_dir(root),
    )
}

fn reset_dir(path: &Path) -> Result<(), String> {
    if path.exists() {
        fs::remove_dir_all(path)
            .map_err(|err| format!("failed to remove {}: {err}", path.display()))?;
    }
    fs::create_dir_all(path).map_err(|err| format!("failed to create {}: {err}", path.display()))
}

fn repo_root() -> Result<PathBuf, String> {
    let path =
        process::command_output_os(OsString::from("git"), &["rev-parse", "--show-toplevel"])?;
    Ok(PathBuf::from(path))
}

impl PerfGuardOptions {
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut options = Self {
            mode: PerfGuardMode::GitRef {
                baseline_ref: DEFAULT_BASELINE_REF.to_string(),
            },
            threshold_percent: DEFAULT_THRESHOLD_PERCENT,
            quick: false,
        };
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--baseline-ref" => {
                    let baseline_ref = args
                        .next()
                        .ok_or_else(|| "--baseline-ref requires a value".to_string())?;
                    options.set_mode(PerfGuardMode::GitRef { baseline_ref })?;
                }
                "--record-current" => {
                    let name = args
                        .next()
                        .ok_or_else(|| "--record-current requires a value".to_string())?;
                    validate_snapshot_name(&name)?;
                    options.set_mode(PerfGuardMode::RecordCurrent { name })?;
                }
                "--compare-current" => {
                    let name = args
                        .next()
                        .ok_or_else(|| "--compare-current requires a value".to_string())?;
                    validate_snapshot_name(&name)?;
                    options.set_mode(PerfGuardMode::CompareCurrent { name })?;
                }
                "--threshold-percent" => {
                    let value = args
                        .next()
                        .ok_or_else(|| "--threshold-percent requires a value".to_string())?;
                    options.threshold_percent = value
                        .parse::<f64>()
                        .map_err(|err| format!("invalid --threshold-percent `{value}`: {err}"))?;
                    if options.threshold_percent < 0.0 {
                        return Err("--threshold-percent must be non-negative".to_string());
                    }
                }
                "--quick" => options.quick = true,
                "--help" | "-h" => return Err(help_text()),
                other => {
                    return Err(format!(
                        "unknown j2k-perf-guard argument `{other}`\n{}",
                        help_text()
                    ))
                }
            }
        }
        Ok(options)
    }

    fn set_mode(&mut self, mode: PerfGuardMode) -> Result<(), String> {
        if self.mode
            != (PerfGuardMode::GitRef {
                baseline_ref: DEFAULT_BASELINE_REF.to_string(),
            })
        {
            return Err("choose only one baseline mode".to_string());
        }
        self.mode = mode;
        Ok(())
    }
}

fn help_text() -> String {
    "usage: cargo xtask j2k-perf-guard [--baseline-ref REF | --record-current NAME | --compare-current NAME] [--threshold-percent N] [--quick]".to_string()
}

fn path_str(path: &Path) -> Result<&str, String> {
    path.to_str()
        .ok_or_else(|| format!("path is not valid UTF-8: {}", path.display()))
}

fn cargo() -> OsString {
    env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"))
}

#[cfg(test)]
mod tests {
    use super::{
        bench_args, compare_estimates, is_enforced_perf_id, read_estimate_snapshot,
        write_estimate_snapshot, BenchCommand, BenchEstimate, PerfGuardMode, PerfGuardOptions,
        BENCH_COMMANDS,
    };
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn perf_guard_parses_current_tree_record_mode() {
        let options = PerfGuardOptions::parse(
            ["--record-current", "htj2k-roi-baseline", "--quick"]
                .into_iter()
                .map(str::to_string),
        )
        .unwrap();

        assert_eq!(
            options.mode,
            PerfGuardMode::RecordCurrent {
                name: "htj2k-roi-baseline".to_string()
            }
        );
        assert!(options.quick);
    }

    #[test]
    fn perf_guard_parses_current_tree_compare_mode() {
        let options = PerfGuardOptions::parse(
            [
                "--compare-current",
                "htj2k-roi-baseline",
                "--threshold-percent",
                "7.5",
            ]
            .into_iter()
            .map(str::to_string),
        )
        .unwrap();

        assert_eq!(
            options.mode,
            PerfGuardMode::CompareCurrent {
                name: "htj2k-roi-baseline".to_string()
            }
        );
        assert_eq!(options.threshold_percent, 7.5);
    }

    #[test]
    fn perf_guard_rejects_multiple_baseline_modes() {
        let error = PerfGuardOptions::parse(
            ["--record-current", "one", "--compare-current", "two"]
                .into_iter()
                .map(str::to_string),
        )
        .unwrap_err();

        assert!(
            error.contains("choose only one baseline mode"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn estimate_snapshot_round_trips_sorted_estimates() {
        let root = temp_dir("j2k-perf-snapshot-test");
        let path = root.join("baselines").join("htj2k.json");
        let estimates = vec![
            BenchEstimate {
                id: "z_group/z_case".to_string(),
                median_ns: 200.0,
                median_lower_ns: 190.0,
                median_upper_ns: 210.0,
            },
            BenchEstimate {
                id: "a_group/a_case".to_string(),
                median_ns: 100.0,
                median_lower_ns: 95.0,
                median_upper_ns: 105.0,
            },
        ];

        write_estimate_snapshot(&path, &estimates).unwrap();
        let round_trip = read_estimate_snapshot(&path).unwrap();

        assert_eq!(
            round_trip,
            vec![
                BenchEstimate {
                    id: "a_group/a_case".to_string(),
                    median_ns: 100.0,
                    median_lower_ns: 95.0,
                    median_upper_ns: 105.0,
                },
                BenchEstimate {
                    id: "z_group/z_case".to_string(),
                    median_ns: 200.0,
                    median_lower_ns: 190.0,
                    median_upper_ns: 210.0,
                },
            ]
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn perf_guard_tracks_htj2k_maturation_benchmarks() {
        let expected = [
            BenchCommand {
                package: "j2k-jpeg",
                bench: "encode_cpu",
                filter: Some("jpeg_cpu_encode_runtime/"),
                features: None,
                env: &[],
            },
            BenchCommand {
                package: "j2k-native",
                bench: "tier1_bitplane",
                filter: Some("htj2k_cleanup_encode/"),
                features: None,
                env: &[],
            },
            BenchCommand {
                package: "j2k-native",
                bench: "tier1_bitplane",
                filter: Some("htj2k_cleanup_decode/"),
                features: None,
                env: &[],
            },
            BenchCommand {
                package: "j2k-native",
                bench: "htj2k_sigprop_phase",
                filter: None,
                features: None,
                env: &[],
            },
        ];

        for command in expected {
            assert!(
                BENCH_COMMANDS.contains(&command),
                "J2K perf guard must track {command:?}"
            );
        }
    }

    #[test]
    fn perf_guard_tracks_cuda_htj2k_benchmarks() {
        for (bench, filter) in [
            ("htj2k_decode", "j2k_cuda_htj2k_"),
            ("htj2k_encode", "j2k_cuda_htj2k_"),
        ] {
            assert!(
                BENCH_COMMANDS.iter().any(|command| {
                    command.package == "j2k-cuda"
                        && command.bench == bench
                        && command.filter == Some(filter)
                        && command.features == Some("cuda-runtime")
                }),
                "J2K perf guard must track CUDA HTJ2K benchmark `{bench}`"
            );
        }
    }

    #[test]
    fn perf_guard_errors_when_enforced_result_is_missing() {
        let error = compare_estimates(
            &[estimate(
                "htj2k_cleanup_encode/encode_64x64/2459041792",
                1_000.0,
            )],
            &[],
            10.0,
        )
        .unwrap_err();

        assert!(
            error.contains("missing current benchmark result"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn perf_guard_errors_when_enforced_baseline_result_is_missing() {
        let error = compare_estimates(
            &[],
            &[estimate(
                "jpeg_cpu_encode_runtime/rgb8_512_420_default",
                1_000.0,
            )],
            10.0,
        )
        .unwrap_err();

        assert!(
            error.contains("missing baseline benchmark result"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn filtered_bench_command_passes_filter_before_quick_flag() {
        let command = BenchCommand {
            package: "j2k-native",
            bench: "tier1_bitplane",
            filter: Some("htj2k_cleanup_encode/"),
            features: None,
            env: &[],
        };

        assert_eq!(
            bench_args(command, true),
            vec![
                "bench",
                "-p",
                "j2k-native",
                "--bench",
                "tier1_bitplane",
                "--",
                "htj2k_cleanup_encode/",
                "--quick",
            ]
        );
    }

    #[test]
    fn feature_bench_command_passes_features_before_filter_separator() {
        let command = BenchCommand {
            package: "j2k-cuda",
            bench: "htj2k_encode",
            filter: Some("j2k_cuda_htj2k_"),
            features: Some("cuda-runtime"),
            env: &[],
        };

        assert_eq!(
            bench_args(command, true),
            vec![
                "bench",
                "-p",
                "j2k-cuda",
                "--bench",
                "htj2k_encode",
                "--features",
                "cuda-runtime",
                "--",
                "j2k_cuda_htj2k_",
                "--quick",
            ]
        );
    }

    #[test]
    fn perf_guard_enforces_stable_htj2k_rows_only() {
        assert!(is_enforced_perf_id(
            "jpeg_cpu_encode_runtime/rgb8_512_420_default"
        ));
        assert!(is_enforced_perf_id(
            "htj2k_cleanup_encode/encode_64x64/2459041792"
        ));
        assert!(!is_enforced_perf_id(
            "wsi_tile_batch_region_scaled_rgb_q4/j2k_htj2k_rgb_512_batch_16"
        ));
        assert!(!is_enforced_perf_id(
            "htj2k_region_scaled_plan_build/j2k-metal-resident"
        ));
        assert!(!is_enforced_perf_id(
            "htj2k_feeder_coalesce/j2k-metal-resident"
        ));
        assert!(!is_enforced_perf_id(
            "htj2k_cleanup_encode_parallel_batch_size/rayon_par_iter_global_blocks/128"
        ));
        assert!(!is_enforced_perf_id(
            "tier1_bitplane_encode/encode_64x64/default"
        ));
        assert!(!is_enforced_perf_id(
            "wsi_tile_batch_region_scaled_rgb_q4/j2k-cpu-staged-metal_htj2k_rgb_512_batch_16"
        ));
        assert!(!is_enforced_perf_id(
            "htj2k_metal_route/j2k-metal-resident_htj2k_rgb_512_batch_16"
        ));
        assert!(!is_enforced_perf_id(
            "j2k_public_decode/nothtj2k_accidental_substring"
        ));
        assert!(is_enforced_perf_id("j2k_public_decode/htj2k_rgb8_lossless"));
        assert!(is_enforced_perf_id(
            "j2k_public_cpu_encode_matrix/rgb8_512_htj2k_external"
        ));
    }

    #[test]
    fn perf_guard_reports_out_of_scope_regressions_without_failing() {
        let baseline = vec![
            estimate("htj2k_cleanup_encode/encode_64x64/2459041792", 1_000.0),
            estimate(
                "htj2k_cleanup_encode_parallel_batch_size/rayon_par_iter_global_blocks/128",
                1_000.0,
            ),
            estimate("tier1_bitplane_encode/encode_64x64/default", 1_000.0),
            estimate(
                "wsi_tile_batch_region_scaled_rgb_q4/j2k-cpu-staged-metal_htj2k_rgb_512_batch_16",
                1_000.0,
            ),
        ];
        let current = baseline
            .iter()
            .map(|estimate| BenchEstimate {
                id: estimate.id.clone(),
                median_ns: 1_300.0,
                median_lower_ns: 1_300.0,
                median_upper_ns: 1_310.0,
            })
            .collect::<Vec<_>>();

        let outcomes = compare_estimates(&baseline, &current, 10.0).unwrap();

        assert!(outcomes[0].enforced);
        assert!(outcomes[0].threshold_exceeded);
        assert!(outcomes[0].regressed);
        for outcome in &outcomes[1..] {
            assert!(!outcome.enforced, "{outcome:?}");
            assert!(outcome.threshold_exceeded, "{outcome:?}");
            assert!(!outcome.regressed, "{outcome:?}");
        }
    }

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let mut dir = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        dir.push(format!("{name}-{}-{nanos}", std::process::id()));
        dir
    }

    fn estimate(id: &str, median_ns: f64) -> BenchEstimate {
        BenchEstimate {
            id: id.to_string(),
            median_ns,
            median_lower_ns: median_ns,
            median_upper_ns: median_ns,
        }
    }
}
