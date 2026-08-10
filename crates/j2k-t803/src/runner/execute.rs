// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{fs, path::Path, path::PathBuf, process::Command, sync::Arc};

use crate::{EncoderEvidence, IutIdentity, PlatformIdentity, ReportStatus, T803Report, T803Suite};

use super::{cache, cases, oracle};

pub(super) struct IutConfig {
    pub(super) name: &'static str,
    pub(super) claim: &'static str,
    pub(super) report_stem: &'static str,
    pub(super) features: Vec<String>,
    pub(super) platform: PlatformIdentity,
}

pub(super) fn run(
    cache_dir: &Path,
    output_dir: Option<PathBuf>,
    development: bool,
    suite: T803Suite,
    mut config: IutConfig,
    encoder: impl FnOnce() -> Result<EncoderEvidence, String>,
    decode: impl FnMut(Arc<[u8]>, u8) -> Result<cases::DecodedImage, cases::DecodeFailure>,
) -> Result<(), String> {
    let (manifest, corpus) = cache::verify_cached(cache_dir)?;
    let candidate_sha = git_output(&["rev-parse", "HEAD"])?;
    let dirty = !git_output(&["status", "--porcelain", "--untracked-files=normal"])?.is_empty();
    if dirty && !development {
        return Err(
            "the source tree is dirty; commit the exact candidate or pass --development for non-release evidence"
            .to_string(),
        );
    }
    let encoder = encoder()?;

    let decoder_cases = manifest
        .decoder_cases_for_suite(suite)
        .map_err(|error| error.to_string())?;
    let mut cases = cases::run_decoder_cases(&decoder_cases, &corpus, decode);
    if matches!(suite, T803Suite::Part1 | T803Suite::All) {
        cases.extend(cases::run_jp2_cases(&manifest, &corpus));
    }
    if matches!(suite, T803Suite::Part15 | T803Suite::All) {
        cases.extend(cases::run_jph_cases(&manifest, &corpus));
    }
    let native_component_oracles = oracle::run(&manifest, &corpus)?;
    if dirty {
        config
            .features
            .push("development-dirty-worktree".to_string());
    }
    config.features.sort_unstable();
    config.features.dedup();
    let report = T803Report::new(
        suite,
        IutIdentity {
            name: config.name.to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            candidate_sha,
            claim: config.claim.to_string(),
        },
        config.platform,
        manifest.source.archive_sha256.clone(),
        config.features,
        manifest.files.clone(),
        native_component_oracles,
        cases,
        encoder,
    )
    .map_err(|error| error.to_string())?;

    let output_dir = output_dir.unwrap_or_else(|| cache_dir.join("reports"));
    fs::create_dir_all(&output_dir)
        .map_err(|error| format!("create {}: {error}", output_dir.display()))?;
    let stem = if dirty {
        format!("{}-development", config.report_stem)
    } else {
        config.report_stem.to_string()
    };
    let json_path = output_dir.join(format!("{stem}.json"));
    let markdown_path = output_dir.join(format!("{stem}.md"));
    fs::write(
        &json_path,
        report.to_json().map_err(|error| error.to_string())?,
    )
    .map_err(|error| format!("write {}: {error}", json_path.display()))?;
    fs::write(
        &markdown_path,
        report.to_markdown().map_err(|error| error.to_string())?,
    )
    .map_err(|error| format!("write {}: {error}", markdown_path.display()))?;

    println!("wrote {}", json_path.display());
    println!("wrote {}", markdown_path.display());
    if report.status == ReportStatus::Pass {
        Ok(())
    } else {
        Err(format!(
            "T.803 {} IUT failed; complete evidence was written to {}",
            config.name,
            json_path.display()
        ))
    }
}

fn git_output(args: &[&str]) -> Result<String, String> {
    let output = Command::new("git")
        .args(args)
        .output()
        .map_err(|error| format!("start git {}: {error}", args.join(" ")))?;
    if !output.status.success() {
        return Err(format!(
            "git {} exited with {}",
            args.join(" "),
            output.status
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}
