use std::{
    env,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use crate::publication_gate::{collect_publication_gate_issues, PUBLICATION_GATE_KEYS};

use super::options::AdoptionBenchmarkOptions;
use super::parsing::read_tsv_metadata;

pub(super) fn enforce_publication_gate(options: &AdoptionBenchmarkOptions) -> Result<(), String> {
    if options.quick || options.include_generated {
        return Ok(());
    }
    let fixture_metadata = read_tsv_metadata(
        &options.out_dir.join("cpu-fixture-compare.out"),
        &PUBLICATION_GATE_KEYS,
    )?;
    let encode_metadata = read_tsv_metadata(
        &options.out_dir.join("cpu-encode-compare.out"),
        &PUBLICATION_GATE_KEYS,
    )?;
    let mut issues = Vec::new();
    collect_publication_gate_issues("cpu-fixture-compare", Some(&fixture_metadata), &mut issues);
    collect_publication_gate_issues("cpu-encode-compare", Some(&encode_metadata), &mut issues);
    if issues.is_empty() {
        return Ok(());
    }
    Err(format!(
        "adoption benchmark is not publishable: {}; artifacts were written under {}. Use --quick or --include-generated only for smoke/diagnostic runs.",
        issues.join("; "),
        options.out_dir.display()
    ))
}

pub(super) fn criterion_target_dir(options: &AdoptionBenchmarkOptions, step_name: &str) -> PathBuf {
    absolute_path(&options.out_dir)
        .join("cargo-target")
        .join(step_name)
}

pub(super) fn absolute_path(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    }
}

pub(super) fn benchmark_env_path(path: &Path, label: &str) -> Result<String, String> {
    let path = canonical_benchmark_path(path, label)?;
    let display = path.display().to_string();
    path.into_os_string()
        .into_string()
        .map_err(|_| format!("{label} path contains non-Unicode data: {display}"))
}

pub(super) fn benchmark_env_path_list(path_list: &str, label: &str) -> Result<String, String> {
    let paths = env::split_paths(path_list)
        .map(|path| canonical_benchmark_path(&path, label))
        .collect::<Result<Vec<_>, _>>()?;
    if paths.is_empty() {
        return Err(format!("{label} must include at least one path"));
    }
    let joined = env::join_paths(paths)
        .map_err(|error| format!("{label} path-list cannot be represented: {error}"))?;
    let display = joined.to_string_lossy().into_owned();
    joined
        .into_string()
        .map_err(|_| format!("{label} path-list contains non-Unicode data: {display}"))
}

pub(super) fn canonical_benchmark_path(path: &Path, label: &str) -> Result<PathBuf, String> {
    absolute_path(path).canonicalize().map_err(|error| {
        format!(
            "{label} path {} cannot be canonicalized: {error}",
            path.display()
        )
    })
}

pub(super) fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}
