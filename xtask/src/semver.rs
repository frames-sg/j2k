// SPDX-License-Identifier: MIT OR Apache-2.0

//! Release-version and reviewed public-API compatibility gates.

use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    ffi::OsString,
    fmt::Write as _,
    fs,
    path::Path,
};

use crate::process::{self, cargo, CommandContext};

const CARGO_SEMVER_CHECKS_VERSION: &str = "0.48.0";
const SEMVER_BASELINE_VERSION: &str = "0.6.2";
const SEMVER_BASELINE_TAG: &str = "v0.6.2";
const SEMVER_BASELINE_COMMIT: &str = "55ee746e1b49f7309e4d030cc01a69d580173920";
const API_DIFF_REPORT: &str = "engineering/reviewed-public-api-diff-0.7.0.md";
const API_REVIEW_CONFIG: &str = "engineering/public-api-review-0.7.0.yml";

const SEMVER_BASELINE_PACKAGES: &[&str] = &[
    "j2k",
    "j2k-core",
    "j2k-jpeg",
    "j2k-tilecodec",
    "j2k-jpeg-metal",
    "j2k-metal",
    "j2k-jpeg-cuda",
    "j2k-cuda",
    "j2k-transcode",
    "j2k-transcode-cuda",
    "j2k-metal-support",
    "j2k-transcode-metal",
    "j2k-native",
    "j2k-types",
    "j2k-cuda-runtime",
    "j2k-profile",
];

const SEMVER_NEW_PACKAGES: &[&str] = &["j2k-codec-math"];

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct Version {
    major: u64,
    minor: u64,
    patch: u64,
}

impl Version {
    fn parse(value: &str) -> Result<Self, String> {
        if value.contains(['-', '+']) {
            return Err(format!(
                "release version `{value}` must not contain prerelease or build metadata"
            ));
        }
        let mut parts = value.split('.');
        let major = parse_version_component(parts.next(), value, "major")?;
        let minor = parse_version_component(parts.next(), value, "minor")?;
        let patch = parse_version_component(parts.next(), value, "patch")?;
        if parts.next().is_some() {
            return Err(format!(
                "release version `{value}` must have exactly three components"
            ));
        }
        Ok(Self {
            major,
            minor,
            patch,
        })
    }
}

fn parse_version_component(
    component: Option<&str>,
    version: &str,
    label: &str,
) -> Result<u64, String> {
    component
        .ok_or_else(|| format!("release version `{version}` is missing its {label} component"))?
        .parse::<u64>()
        .map_err(|err| {
            format!("release version `{version}` has an invalid {label} component: {err}")
        })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReleaseType {
    Major,
    Minor,
    Patch,
}

impl ReleaseType {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Major => "major",
            Self::Minor => "minor",
            Self::Patch => "patch",
        }
    }
}

fn computed_release_type(baseline: Version, candidate: Version) -> Result<ReleaseType, String> {
    if candidate <= baseline {
        return Err(format!(
            "candidate version {candidate:?} must be newer than baseline {baseline:?}"
        ));
    }

    if candidate.major != baseline.major {
        return Ok(ReleaseType::Major);
    }
    if candidate.minor != baseline.minor {
        return Ok(if candidate.major == 0 {
            ReleaseType::Major
        } else {
            ReleaseType::Minor
        });
    }
    Ok(match (candidate.major, candidate.minor) {
        (0, 0) => ReleaseType::Major,
        (0, _) => ReleaseType::Minor,
        (_, _) => ReleaseType::Patch,
    })
}

#[derive(Debug)]
struct PackageApiDiff {
    package: String,
    candidate_version: String,
    release_type: Option<ReleaseType>,
    baseline_count: usize,
    candidate_count: usize,
    added: BTreeSet<String>,
    removed: BTreeSet<String>,
}

impl PackageApiDiff {
    fn added_fingerprint(&self) -> String {
        fingerprint(&self.added)
    }

    fn removed_fingerprint(&self) -> String {
        fingerprint(&self.removed)
    }
}

#[derive(Debug)]
struct ReviewEntry {
    removed_fingerprint: Option<String>,
    added_fingerprint: Option<String>,
    rationale: String,
}

#[derive(Debug)]
struct ReviewConfig {
    candidate_version: String,
    reviews: BTreeMap<String, ReviewEntry>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Options {
    write_report: bool,
}

pub(crate) fn semver(
    args: impl Iterator<Item = String>,
    stable_packages: &[&str],
    cargo_public_api_version: &str,
) -> Result<(), String> {
    let options = parse_options(args)?;
    require_macos()?;
    verify_baseline_tag()?;
    verify_tool_versions(cargo_public_api_version, options.write_report)?;
    validate_package_partition(stable_packages)?;

    let versions = workspace_package_versions()?;
    let candidate_version = common_candidate_version(stable_packages, &versions)?;
    let baseline = Version::parse(SEMVER_BASELINE_VERSION)?;
    let baseline_snapshot = baseline_api_snapshot(cargo_public_api_version)?;
    let baseline_apis = parse_api_snapshot(&baseline_snapshot)?;
    let current_snapshot = current_api_snapshot(cargo_public_api_version)?;
    let snapshot_apis = parse_api_snapshot(&current_snapshot)?;
    let current_apis = if options.write_report {
        collect_current_apis(stable_packages)?
    } else {
        snapshot_apis.clone()
    };
    let stale_snapshot_packages = snapshot_drift(stable_packages, &snapshot_apis, &current_apis);

    let baseline_set = SEMVER_BASELINE_PACKAGES
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut diffs = Vec::with_capacity(stable_packages.len());
    for package in stable_packages {
        let current = current_apis.get(*package).cloned().ok_or_else(|| {
            format!("current public API snapshot is missing stable package `{package}`")
        })?;
        let candidate = versions
            .get(*package)
            .ok_or_else(|| format!("workspace metadata is missing stable package `{package}`"))?;
        if baseline_set.contains(package) {
            let baseline_api = baseline_apis.get(*package).ok_or_else(|| {
                format!(
                    "{SEMVER_BASELINE_TAG} public API snapshot is missing baseline package `{package}`"
                )
            })?;
            let release_type = computed_release_type(baseline, Version::parse(candidate)?)?;
            diffs.push(PackageApiDiff {
                package: (*package).to_string(),
                candidate_version: candidate.clone(),
                release_type: Some(release_type),
                baseline_count: baseline_api.len(),
                candidate_count: current.len(),
                added: current.difference(baseline_api).cloned().collect(),
                removed: baseline_api.difference(&current).cloned().collect(),
            });
        } else {
            diffs.push(PackageApiDiff {
                package: (*package).to_string(),
                candidate_version: candidate.clone(),
                release_type: None,
                baseline_count: 0,
                candidate_count: current.len(),
                added: current,
                removed: BTreeSet::new(),
            });
        }
    }

    let report = render_report(&candidate_version, &diffs, cargo_public_api_version);
    verify_or_write_report(options, &report)?;
    if !stale_snapshot_packages.is_empty() {
        return Err(format!(
            "wrote {API_DIFF_REPORT} from the current workspace, but docs/stable-api-1.0.public-api.txt is stale for packages {stale_snapshot_packages:?}; run `cargo xtask stable-api --write` on macOS with cargo-public-api 0.52.0 and review that diff before treating the API report as verified"
        ));
    }

    let review_config = load_review_config()?;
    validate_reviews(&review_config, &candidate_version, &diffs)?;
    run_semver_checks(&diffs)?;
    Ok(())
}

fn parse_options(args: impl Iterator<Item = String>) -> Result<Options, String> {
    let mut write_report = false;
    for arg in args {
        match arg.as_str() {
            "--write-report" if !write_report => write_report = true,
            "--write-report" => {
                return Err("duplicate semver argument `--write-report`".to_string())
            }
            other => return Err(format!("unknown semver argument `{other}`")),
        }
    }
    Ok(Options { write_report })
}

fn require_macos() -> Result<(), String> {
    if env::consts::OS == "macos" {
        Ok(())
    } else {
        Err(format!(
            "semver/API review must run on macOS so Metal public APIs are included; current host is {}",
            env::consts::OS
        ))
    }
}

fn verify_baseline_tag() -> Result<(), String> {
    let revision = capture_command(
        OsString::from("git"),
        &["rev-parse", &format!("{SEMVER_BASELINE_TAG}^{{commit}}")],
        &[],
        "resolve semver baseline tag",
    )?;
    if revision.trim() == SEMVER_BASELINE_COMMIT {
        Ok(())
    } else {
        Err(format!(
            "{SEMVER_BASELINE_TAG} must peel to {SEMVER_BASELINE_COMMIT}, found `{}`",
            revision.trim()
        ))
    }
}

fn verify_tool_versions(
    cargo_public_api_version: &str,
    require_public_api_binary: bool,
) -> Result<(), String> {
    let toolchain = env::var("J2K_SEMVER_TOOLCHAIN").unwrap_or_else(|_| "1.96".to_string());
    let toolchain_arg = format!("+{toolchain}");
    let semver_version = capture_command(
        OsString::from("cargo"),
        &[toolchain_arg.as_str(), "semver-checks", "--version"],
        &[],
        "detect cargo-semver-checks",
    )?;
    require_version_token(
        &semver_version,
        "cargo-semver-checks",
        CARGO_SEMVER_CHECKS_VERSION,
    )?;

    if !require_public_api_binary {
        return Ok(());
    }

    let public_api_version = capture_command(
        cargo(),
        &["public-api", "--version"],
        &[],
        "detect cargo-public-api",
    )?;
    require_version_token(
        &public_api_version,
        "cargo-public-api",
        cargo_public_api_version,
    )
}

fn require_version_token(output: &str, tool: &str, expected: &str) -> Result<(), String> {
    let mut words = output.split_whitespace();
    let actual_tool = words.next().unwrap_or_default();
    let actual_version = words.next().unwrap_or_default();
    if actual_tool == tool && actual_version == expected {
        Ok(())
    } else {
        Err(format!(
            "{tool} version must be {expected}; found `{}`",
            output.trim()
        ))
    }
}

fn validate_package_partition(stable_packages: &[&str]) -> Result<(), String> {
    let stable = stable_packages.iter().copied().collect::<BTreeSet<_>>();
    if stable.len() != stable_packages.len() {
        return Err("STABLE_SEMVER_PACKAGES contains duplicates".to_string());
    }
    let baseline = SEMVER_BASELINE_PACKAGES
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let new = SEMVER_NEW_PACKAGES.iter().copied().collect::<BTreeSet<_>>();
    if baseline.len() != SEMVER_BASELINE_PACKAGES.len() {
        return Err("SEMVER_BASELINE_PACKAGES contains duplicates".to_string());
    }
    if new.len() != SEMVER_NEW_PACKAGES.len() {
        return Err("SEMVER_NEW_PACKAGES contains duplicates".to_string());
    }
    let overlap = baseline.intersection(&new).copied().collect::<Vec<_>>();
    if !overlap.is_empty() {
        return Err(format!(
            "semver baseline/new package lists overlap: {overlap:?}"
        ));
    }
    let configured = baseline.union(&new).copied().collect::<BTreeSet<_>>();
    if configured == stable {
        Ok(())
    } else {
        let missing = stable.difference(&configured).copied().collect::<Vec<_>>();
        let unexpected = configured.difference(&stable).copied().collect::<Vec<_>>();
        Err(format!(
            "semver baseline/new package partition drifted; missing: {missing:?}; unexpected: {unexpected:?}"
        ))
    }
}

fn workspace_package_versions() -> Result<BTreeMap<String, String>, String> {
    let output = capture_command(
        cargo(),
        &["metadata", "--locked", "--no-deps", "--format-version=1"],
        &[],
        "read workspace package versions",
    )?;
    let metadata: serde_json::Value =
        serde_json::from_str(&output).map_err(|err| format!("parse cargo metadata JSON: {err}"))?;
    let packages = metadata
        .get("packages")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "cargo metadata did not contain a packages array".to_string())?;
    Ok(packages
        .iter()
        .filter_map(|package| {
            Some((
                package.get("name")?.as_str()?.to_string(),
                package.get("version")?.as_str()?.to_string(),
            ))
        })
        .collect())
}

fn common_candidate_version(
    stable_packages: &[&str],
    versions: &BTreeMap<String, String>,
) -> Result<String, String> {
    let mut candidates = stable_packages
        .iter()
        .map(|package| {
            versions
                .get(*package)
                .cloned()
                .ok_or_else(|| format!("workspace metadata is missing stable package `{package}`"))
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    if candidates.len() != 1 {
        return Err(format!(
            "stable semver packages must share one candidate version, found {candidates:?}"
        ));
    }
    candidates
        .pop_first()
        .ok_or_else(|| "stable semver package list is empty".to_string())
}

fn baseline_api_snapshot(cargo_public_api_version: &str) -> Result<String, String> {
    let object = format!("{SEMVER_BASELINE_TAG}:docs/stable-api-1.0.public-api.txt");
    let snapshot = capture_command(
        OsString::from("git"),
        &["show", object.as_str()],
        &[],
        "read tagged public API snapshot",
    )?;
    let expected_generator = format!("Generator: `cargo-public-api {cargo_public_api_version}`.");
    if snapshot.contains(&expected_generator) {
        Ok(snapshot)
    } else {
        Err(format!(
            "{SEMVER_BASELINE_TAG} public API snapshot was not generated by cargo-public-api {cargo_public_api_version}"
        ))
    }
}

fn current_api_snapshot(cargo_public_api_version: &str) -> Result<String, String> {
    let path = "docs/stable-api-1.0.public-api.txt";
    let snapshot = fs::read_to_string(path).map_err(|err| format!("read {path}: {err}"))?;
    let expected_generator = format!("Generator: `cargo-public-api {cargo_public_api_version}`.");
    if snapshot.contains(&expected_generator) {
        Ok(snapshot)
    } else {
        Err(format!(
            "{path} was not generated by cargo-public-api {cargo_public_api_version}"
        ))
    }
}

fn collect_current_apis(
    stable_packages: &[&str],
) -> Result<BTreeMap<String, BTreeSet<String>>, String> {
    let mut current = BTreeMap::new();
    for package in stable_packages {
        eprintln!("collecting current public API for `{package}`");
        current.insert((*package).to_string(), current_public_api(package)?);
    }
    Ok(current)
}

fn snapshot_drift<'a>(
    stable_packages: &[&'a str],
    snapshot_apis: &BTreeMap<String, BTreeSet<String>>,
    current_apis: &BTreeMap<String, BTreeSet<String>>,
) -> Vec<&'a str> {
    stable_packages
        .iter()
        .copied()
        .filter(|package| snapshot_apis.get(*package) != current_apis.get(*package))
        .collect()
}

fn current_public_api(package: &str) -> Result<BTreeSet<String>, String> {
    let output = capture_command(
        cargo(),
        &[
            "public-api",
            "-p",
            package,
            "--all-features",
            "-sss",
            "--color",
            "never",
        ],
        &[],
        &format!("generate current public API for {package}"),
    )?;
    let api = output
        .lines()
        .map(str::trim_end)
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect::<BTreeSet<_>>();
    if api.is_empty() {
        Err(format!(
            "cargo-public-api returned no public items for stable package `{package}`"
        ))
    } else {
        Ok(api)
    }
}

fn parse_api_snapshot(snapshot: &str) -> Result<BTreeMap<String, BTreeSet<String>>, String> {
    let mut sections = BTreeMap::<String, BTreeSet<String>>::new();
    let mut package = None::<String>;
    let mut in_api = false;
    for line in snapshot.lines() {
        if let Some(name) = line
            .strip_prefix("## `")
            .and_then(|rest| rest.strip_suffix('`'))
        {
            package = Some(name.to_string());
            in_api = false;
            continue;
        }
        if line == "```text" {
            in_api = true;
            continue;
        }
        if line == "```" {
            in_api = false;
            continue;
        }
        if in_api && !line.is_empty() {
            let name = package.as_ref().ok_or_else(|| {
                "public API snapshot contains a text fence before a package heading".to_string()
            })?;
            sections
                .entry(name.clone())
                .or_default()
                .insert(line.to_string());
        }
    }
    if sections.is_empty() {
        Err("public API snapshot did not contain package API sections".to_string())
    } else {
        Ok(sections)
    }
}

fn fingerprint(items: &BTreeSet<String>) -> String {
    if items.is_empty() {
        return "none".to_string();
    }
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for item in items {
        for byte in item.as_bytes().iter().copied().chain([b'\n']) {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    format!("fnv1a64:{hash:016x}")
}

fn render_report(
    candidate_version: &str,
    diffs: &[PackageApiDiff],
    cargo_public_api_version: &str,
) -> String {
    let mut out = String::new();
    writeln!(
        &mut out,
        "# Reviewed public API diff for j2k {candidate_version}"
    )
    .unwrap();
    writeln!(&mut out).unwrap();
    writeln!(
        &mut out,
        "This report is generated by `cargo xtask semver --write-report`. Normal `cargo xtask semver` regenerates it in memory and fails if this committed file is stale. Breaking fingerprints require a matching rationale in `{API_REVIEW_CONFIG}`; report regeneration never updates that review config."
    )
    .unwrap();
    writeln!(&mut out).unwrap();
    writeln!(
        &mut out,
        "- Baseline registry version: `{SEMVER_BASELINE_VERSION}`"
    )
    .unwrap();
    writeln!(
        &mut out,
        "- Baseline source snapshot: `{SEMVER_BASELINE_TAG}` peeled to `{SEMVER_BASELINE_COMMIT}`"
    )
    .unwrap();
    writeln!(&mut out, "- Candidate version: `{candidate_version}`").unwrap();
    writeln!(
        &mut out,
        "- Tool pins: `cargo-semver-checks {CARGO_SEMVER_CHECKS_VERSION}`, `cargo-public-api {cargo_public_api_version}`"
    )
    .unwrap();
    writeln!(&mut out).unwrap();
    writeln!(&mut out, "## Summary").unwrap();
    writeln!(&mut out).unwrap();
    writeln!(
        &mut out,
        "| Package | Baseline | Candidate | Computed release type | Added | Removed/changed | Removed fingerprint | Added fingerprint |"
    )
    .unwrap();
    writeln!(
        &mut out,
        "| --- | --- | --- | --- | ---: | ---: | --- | --- |"
    )
    .unwrap();
    for diff in diffs {
        let baseline = if diff.release_type.is_some() {
            SEMVER_BASELINE_VERSION
        } else {
            "new/unpublished"
        };
        let release_type = diff.release_type.map_or("new", ReleaseType::as_str);
        writeln!(
            &mut out,
            "| `{}` | `{baseline}` | `{}` | `{release_type}` | {} | {} | `{}` | `{}` |",
            diff.package,
            diff.candidate_version,
            diff.added.len(),
            diff.removed.len(),
            diff.removed_fingerprint(),
            diff.added_fingerprint(),
        )
        .unwrap();
    }

    writeln!(&mut out).unwrap();
    writeln!(
        &mut out,
        "## New packages without a 0.6.2 registry baseline"
    )
    .unwrap();
    writeln!(&mut out).unwrap();
    for diff in diffs.iter().filter(|diff| diff.release_type.is_none()) {
        writeln!(
            &mut out,
            "- `{}` `{}`: {} public API items; fingerprint `{}`.",
            diff.package,
            diff.candidate_version,
            diff.candidate_count,
            diff.added_fingerprint()
        )
        .unwrap();
    }

    writeln!(&mut out).unwrap();
    writeln!(&mut out, "## Published-package details").unwrap();
    for diff in diffs.iter().filter(|diff| diff.release_type.is_some()) {
        writeln!(&mut out).unwrap();
        writeln!(&mut out, "### `{}`", diff.package).unwrap();
        writeln!(&mut out).unwrap();
        writeln!(
            &mut out,
            "Baseline items: {}. Candidate items: {}. Computed release type: `{}`.",
            diff.baseline_count,
            diff.candidate_count,
            diff.release_type
                .expect("published diff release type")
                .as_str()
        )
        .unwrap();
        render_diff_items(
            &mut out,
            "Removed or changed baseline API items",
            &diff.removed,
        );
        render_diff_items(&mut out, "Added candidate API items", &diff.added);
    }
    out
}

fn render_diff_items(out: &mut String, heading: &str, items: &BTreeSet<String>) {
    writeln!(out).unwrap();
    writeln!(out, "#### {heading}").unwrap();
    writeln!(out).unwrap();
    if items.is_empty() {
        writeln!(out, "None.").unwrap();
        return;
    }
    writeln!(out, "```text").unwrap();
    for item in items {
        writeln!(out, "{item}").unwrap();
    }
    writeln!(out, "```").unwrap();
}

fn verify_or_write_report(options: Options, rendered: &str) -> Result<(), String> {
    let path = Path::new(API_DIFF_REPORT);
    if options.write_report {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|err| format!("create {}: {err}", parent.display()))?;
        }
        fs::write(path, rendered).map_err(|err| format!("write {}: {err}", path.display()))?;
        eprintln!(
            "wrote {}; review its diff before editing {API_REVIEW_CONFIG}",
            path.display()
        );
        return Ok(());
    }

    let committed = fs::read_to_string(path).map_err(|err| {
        format!(
            "read {}: {err}; run `cargo xtask semver --write-report` and review the generated diff",
            path.display()
        )
    })?;
    if committed == rendered {
        Ok(())
    } else {
        Err(format!(
            "{} is stale; run `cargo xtask semver --write-report`, review the diff, and update {API_REVIEW_CONFIG} only for intentional API changes",
            path.display()
        ))
    }
}

fn load_review_config() -> Result<ReviewConfig, String> {
    let source = fs::read_to_string(API_REVIEW_CONFIG)
        .map_err(|err| format!("read {API_REVIEW_CONFIG}: {err}"))?;
    let value: serde_yaml_ng::Value = serde_yaml_ng::from_str(&source)
        .map_err(|err| format!("parse {API_REVIEW_CONFIG}: {err}"))?;
    parse_review_config(&value)
}

fn parse_review_config(value: &serde_yaml_ng::Value) -> Result<ReviewConfig, String> {
    let root = value
        .as_mapping()
        .ok_or_else(|| "API review config root must be a mapping".to_string())?;
    reject_unknown_keys(
        root,
        &[
            "version",
            "baseline_tag",
            "baseline_version",
            "candidate_version",
            "reviews",
        ],
        "API review config",
    )?;
    if required_u64(root, "version")? != 1 {
        return Err("API review config version must be 1".to_string());
    }
    require_exact(root, "baseline_tag", SEMVER_BASELINE_TAG)?;
    require_exact(root, "baseline_version", SEMVER_BASELINE_VERSION)?;
    let candidate_version = required_string(root, "candidate_version")?.to_string();
    let review_values = required_value(root, "reviews")?
        .as_mapping()
        .ok_or_else(|| "API review config `reviews` must be a mapping".to_string())?;
    let mut reviews = BTreeMap::new();
    for (package, value) in review_values {
        let package = package
            .as_str()
            .ok_or_else(|| "API review package keys must be strings".to_string())?;
        let entry = value
            .as_mapping()
            .ok_or_else(|| format!("API review for `{package}` must be a mapping"))?;
        reject_unknown_keys(
            entry,
            &["removed_fingerprint", "added_fingerprint", "rationale"],
            &format!("API review for `{package}`"),
        )?;
        let removed_fingerprint = optional_string(entry, "removed_fingerprint")?;
        let added_fingerprint = optional_string(entry, "added_fingerprint")?;
        if removed_fingerprint.is_none() && added_fingerprint.is_none() {
            return Err(format!(
                "API review for `{package}` must declare a removed or added fingerprint"
            ));
        }
        let rationale = required_string(entry, "rationale")?.trim().to_string();
        if rationale.len() < 20 {
            return Err(format!(
                "API review rationale for `{package}` must be at least 20 characters"
            ));
        }
        if reviews
            .insert(
                package.to_string(),
                ReviewEntry {
                    removed_fingerprint,
                    added_fingerprint,
                    rationale,
                },
            )
            .is_some()
        {
            return Err(format!("duplicate API review for `{package}`"));
        }
    }
    Ok(ReviewConfig {
        candidate_version,
        reviews,
    })
}

fn validate_reviews(
    config: &ReviewConfig,
    candidate_version: &str,
    diffs: &[PackageApiDiff],
) -> Result<(), String> {
    let mut errors = Vec::new();
    if config.candidate_version != candidate_version {
        errors.push(format!(
            "review config candidate {} does not match workspace candidate {candidate_version}",
            config.candidate_version
        ));
    }
    let by_package = diffs
        .iter()
        .map(|diff| (diff.package.as_str(), diff))
        .collect::<BTreeMap<_, _>>();
    for diff in diffs {
        if diff.removed.is_empty() {
            continue;
        }
        let Some(review) = config.reviews.get(&diff.package) else {
            errors.push(format!(
                "`{}` has {} removed/changed API items with fingerprint {} but no review entry",
                diff.package,
                diff.removed.len(),
                diff.removed_fingerprint()
            ));
            continue;
        };
        if review.removed_fingerprint.as_deref() != Some(diff.removed_fingerprint().as_str()) {
            errors.push(format!(
                "`{}` removed/changed fingerprint is {}, but its review records {:?}",
                diff.package,
                diff.removed_fingerprint(),
                review.removed_fingerprint
            ));
        }
    }
    for (package, review) in &config.reviews {
        let Some(diff) = by_package.get(package.as_str()) else {
            errors.push(format!(
                "review config contains unknown package `{package}`"
            ));
            continue;
        };
        if let Some(expected) = &review.removed_fingerprint {
            let actual = diff.removed_fingerprint();
            if expected != &actual {
                errors.push(format!(
                    "`{package}` reviewed removed fingerprint {expected} is stale; actual is {actual}"
                ));
            }
        }
        if let Some(expected) = &review.added_fingerprint {
            let actual = diff.added_fingerprint();
            if expected != &actual {
                errors.push(format!(
                    "`{package}` reviewed added fingerprint {expected} is stale; actual is {actual}"
                ));
            }
        }
        if review.rationale.trim().len() < 20 {
            errors.push(format!("`{package}` review rationale is too short"));
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "unreviewed or stale public API changes:\n- {}\nReview {API_DIFF_REPORT}, then update {API_REVIEW_CONFIG} with the exact generated fingerprint and a package-specific rationale.",
            errors.join("\n- ")
        ))
    }
}

fn required_value<'a>(
    mapping: &'a serde_yaml_ng::Mapping,
    key: &str,
) -> Result<&'a serde_yaml_ng::Value, String> {
    mapping
        .get(serde_yaml_ng::Value::String(key.to_string()))
        .ok_or_else(|| format!("API review config is missing `{key}`"))
}

fn required_string<'a>(mapping: &'a serde_yaml_ng::Mapping, key: &str) -> Result<&'a str, String> {
    required_value(mapping, key)?
        .as_str()
        .ok_or_else(|| format!("API review config `{key}` must be a string"))
}

fn optional_string(mapping: &serde_yaml_ng::Mapping, key: &str) -> Result<Option<String>, String> {
    mapping
        .get(serde_yaml_ng::Value::String(key.to_string()))
        .map(|value| {
            value
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| format!("API review config `{key}` must be a string"))
        })
        .transpose()
}

fn required_u64(mapping: &serde_yaml_ng::Mapping, key: &str) -> Result<u64, String> {
    required_value(mapping, key)?
        .as_u64()
        .ok_or_else(|| format!("API review config `{key}` must be an unsigned integer"))
}

fn require_exact(
    mapping: &serde_yaml_ng::Mapping,
    key: &str,
    expected: &str,
) -> Result<(), String> {
    let actual = required_string(mapping, key)?;
    if actual == expected {
        Ok(())
    } else {
        Err(format!(
            "API review config `{key}` must be `{expected}`, found `{actual}`"
        ))
    }
}

fn reject_unknown_keys(
    mapping: &serde_yaml_ng::Mapping,
    allowed: &[&str],
    context: &str,
) -> Result<(), String> {
    let allowed = allowed.iter().copied().collect::<BTreeSet<_>>();
    let unknown = mapping
        .keys()
        .map(|key| {
            key.as_str()
                .ok_or_else(|| format!("{context} keys must be strings"))
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .filter(|key| !allowed.contains(key))
        .collect::<Vec<_>>();
    if unknown.is_empty() {
        Ok(())
    } else {
        Err(format!("{context} contains unknown keys: {unknown:?}"))
    }
}

fn run_semver_checks(diffs: &[PackageApiDiff]) -> Result<(), String> {
    let toolchain = env::var("J2K_SEMVER_TOOLCHAIN").unwrap_or_else(|_| "1.96".to_string());
    let toolchain_arg = format!("+{toolchain}");
    for diff in diffs.iter().filter(|diff| diff.release_type.is_some()) {
        let release_type = diff.release_type.expect("published diff release type");
        process::run_command(
            OsString::from("cargo"),
            &[
                toolchain_arg.as_str(),
                "semver-checks",
                "check-release",
                "--package",
                diff.package.as_str(),
                "--baseline-version",
                SEMVER_BASELINE_VERSION,
                "--release-type",
                release_type.as_str(),
                "--color",
                "never",
            ],
            CommandContext::new(),
        )?;
    }
    Ok(())
}

fn capture_command(
    program: OsString,
    args: &[&str],
    envs: &[(&str, &str)],
    label: &str,
) -> Result<String, String> {
    eprintln!("+ {} {}", program.to_string_lossy(), args.join(" "));
    let output = process::command_output(program.clone(), args, CommandContext::new().envs(envs))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if output.status.success() {
        Ok(stdout.trim().to_string())
    } else {
        Err(format!(
            "{} exited with {} while attempting to {label}:\n{}",
            program.to_string_lossy(),
            output.status,
            format!("{stdout}\n{stderr}").trim()
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::{
        computed_release_type, fingerprint, parse_api_snapshot, parse_options, parse_review_config,
        validate_package_partition, validate_reviews, PackageApiDiff, ReleaseType, ReviewConfig,
        ReviewEntry, Version, SEMVER_BASELINE_PACKAGES, SEMVER_NEW_PACKAGES,
    };

    #[test]
    fn release_type_uses_cargo_pre_one_compatibility_rules() {
        assert_eq!(
            computed_release_type(
                Version::parse("0.6.2").unwrap(),
                Version::parse("0.7.0").unwrap()
            )
            .unwrap(),
            ReleaseType::Major
        );
        assert_eq!(
            computed_release_type(
                Version::parse("1.6.2").unwrap(),
                Version::parse("1.7.0").unwrap()
            )
            .unwrap(),
            ReleaseType::Minor
        );
        assert_eq!(
            computed_release_type(
                Version::parse("1.6.2").unwrap(),
                Version::parse("1.6.3").unwrap()
            )
            .unwrap(),
            ReleaseType::Patch
        );
        assert_eq!(
            computed_release_type(
                Version::parse("0.6.2").unwrap(),
                Version::parse("0.6.3").unwrap()
            )
            .unwrap(),
            ReleaseType::Minor
        );
        assert_eq!(
            computed_release_type(
                Version::parse("0.0.2").unwrap(),
                Version::parse("0.0.3").unwrap()
            )
            .unwrap(),
            ReleaseType::Major
        );
        assert!(computed_release_type(
            Version::parse("0.7.0").unwrap(),
            Version::parse("0.6.2").unwrap()
        )
        .is_err());
    }

    #[test]
    fn package_partition_lists_new_packages_explicitly() {
        let mut stable = SEMVER_BASELINE_PACKAGES.to_vec();
        stable.extend_from_slice(SEMVER_NEW_PACKAGES);
        assert!(validate_package_partition(&stable).is_ok());
        stable.pop();
        assert!(validate_package_partition(&stable).is_err());
    }

    #[test]
    fn parses_snapshot_sections_as_sets() {
        let snapshot = "## `alpha`\n\n```text\npub fn alpha::b()\npub fn alpha::a()\n```\n\n## `beta`\n\n```text\npub struct beta::B\n```\n";
        let parsed = parse_api_snapshot(snapshot).unwrap();
        assert_eq!(
            parsed["alpha"].iter().cloned().collect::<Vec<_>>(),
            ["pub fn alpha::a()", "pub fn alpha::b()"]
        );
        assert_eq!(parsed["beta"].len(), 1);
    }

    #[test]
    fn fingerprints_are_order_independent_and_empty_is_none() {
        let first = ["b".to_string(), "a".to_string()]
            .into_iter()
            .collect::<BTreeSet<_>>();
        let second = ["a".to_string(), "b".to_string()]
            .into_iter()
            .collect::<BTreeSet<_>>();
        assert_eq!(fingerprint(&first), fingerprint(&second));
        assert_eq!(fingerprint(&BTreeSet::new()), "none");
    }

    #[test]
    fn report_regeneration_flag_is_explicit() {
        assert!(!parse_options(std::iter::empty()).unwrap().write_report);
        assert!(
            parse_options(["--write-report".to_string()].into_iter())
                .unwrap()
                .write_report
        );
        assert!(parse_options(["--write".to_string()].into_iter()).is_err());
    }

    #[test]
    fn parses_review_config_and_rejects_unknown_fields() {
        let source = "\
version: 1
baseline_tag: v0.6.2
baseline_version: 0.6.2
candidate_version: 0.7.0
reviews:
  alpha:
    removed_fingerprint: 'fnv1a64:1234'
    rationale: 'Intentional reviewed compatibility change.'
";
        let value: serde_yaml_ng::Value = serde_yaml_ng::from_str(source).unwrap();
        let parsed = parse_review_config(&value).unwrap();
        assert_eq!(parsed.candidate_version, "0.7.0");
        assert_eq!(parsed.reviews.len(), 1);

        let invalid = source.replace("rationale:", "unknown: nope\n    rationale:");
        let value: serde_yaml_ng::Value = serde_yaml_ng::from_str(&invalid).unwrap();
        assert!(parse_review_config(&value).is_err());
    }

    #[test]
    fn unreviewed_and_stale_breaking_fingerprints_fail() {
        let removed = ["pub fn alpha::old()".to_string()]
            .into_iter()
            .collect::<BTreeSet<_>>();
        let added = ["pub fn alpha::new()".to_string()]
            .into_iter()
            .collect::<BTreeSet<_>>();
        let diff = PackageApiDiff {
            package: "alpha".to_string(),
            candidate_version: "0.7.0".to_string(),
            release_type: Some(ReleaseType::Major),
            baseline_count: 1,
            candidate_count: 1,
            added,
            removed,
        };
        let empty = ReviewConfig {
            candidate_version: "0.7.0".to_string(),
            reviews: BTreeMap::new(),
        };
        assert!(validate_reviews(&empty, "0.7.0", &[diff]).is_err());

        let removed = ["pub fn alpha::old()".to_string()]
            .into_iter()
            .collect::<BTreeSet<_>>();
        let diff = PackageApiDiff {
            package: "alpha".to_string(),
            candidate_version: "0.7.0".to_string(),
            release_type: Some(ReleaseType::Major),
            baseline_count: 1,
            candidate_count: 0,
            added: BTreeSet::new(),
            removed,
        };
        let reviews = [(
            "alpha".to_string(),
            ReviewEntry {
                removed_fingerprint: Some("fnv1a64:stale".to_string()),
                added_fingerprint: None,
                rationale: "Intentional reviewed compatibility change.".to_string(),
            },
        )]
        .into_iter()
        .collect();
        let stale = ReviewConfig {
            candidate_version: "0.7.0".to_string(),
            reviews,
        };
        assert!(validate_reviews(&stale, "0.7.0", &[diff]).is_err());
    }
}
