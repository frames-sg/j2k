// SPDX-License-Identifier: MIT OR Apache-2.0

use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};

use crate::process::{self, CommandContext};

mod config;
mod report;
mod stage;

use config::{validate_clone_config, DUPLICATED_LINE_THRESHOLD};
use report::validate_jscpd_report;
use stage::{reset_generated_directory, stage_production_sources};

const JSCPD_VERSION: &str = "4.0.5";
const JSCPD_PACKAGE: &str = "jscpd@4.0.5";
const CONFIG_RELATIVE: &str = ".jscpd.json";
const AUDIT_ROOT_RELATIVE: &str = "target/clone-audit";
const STAGE_RELATIVE: &str = "target/clone-audit/production";
const REPORT_RELATIVE: &str = "target/clone-audit/report";

pub(crate) fn clone_audit(args: impl Iterator<Item = String>) -> Result<(), String> {
    let arguments = args.collect::<Vec<_>>();
    if !arguments.is_empty() {
        return Err(format!(
            "clone-audit accepts no arguments; received {}",
            arguments.join(" ")
        ));
    }
    let root = workspace_root()?;
    let config_path = root.join(CONFIG_RELATIVE);
    validate_clone_config(&config_path)?;

    let audit_root = root.join(AUDIT_ROOT_RELATIVE);
    let stage_root = root.join(STAGE_RELATIVE);
    let report_root = root.join(REPORT_RELATIVE);
    reset_generated_directory(&audit_root, &stage_root)?;
    reset_generated_directory(&audit_root, &report_root)?;
    fs::create_dir_all(&stage_root)
        .map_err(|error| format!("create clone-audit stage {}: {error}", stage_root.display()))?;
    fs::create_dir_all(&report_root).map_err(|error| {
        format!(
            "create clone-audit report {}: {error}",
            report_root.display()
        )
    })?;

    let summary = stage_production_sources(&root, &stage_root)?;
    let scanner_args = jscpd_args(&stage_root, &config_path, &report_root)?;
    process::run_command_owned(
        OsString::from("npx"),
        &scanner_args,
        CommandContext::new().current_dir(&root),
    )
    .map_err(|error| {
        format!("pinned jscpd {JSCPD_VERSION} production clone audit failed: {error}")
    })?;
    validate_jscpd_report(
        &report_root.join("jscpd-report.json"),
        DUPLICATED_LINE_THRESHOLD,
    )?;
    println!(
        "production clone audit passed across {} staged Rust sources; masked {} test-only syntax nodes on {} mixed lines; report: {}",
        summary.files,
        summary.masked_nodes,
        summary.mixed_lines,
        report_root.join("jscpd-report.json").display()
    );
    Ok(())
}

fn workspace_root() -> Result<PathBuf, String> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| "xtask manifest directory has no repository parent".to_string())
}

fn jscpd_args(
    stage_root: &Path,
    config_path: &Path,
    report_root: &Path,
) -> Result<Vec<String>, String> {
    Ok(vec![
        "--yes".to_string(),
        JSCPD_PACKAGE.to_string(),
        path_text(stage_root)?,
        "--config".to_string(),
        path_text(config_path)?,
        "--output".to_string(),
        path_text(report_root)?,
        "--silent".to_string(),
    ])
}

fn path_text(path: &Path) -> Result<String, String> {
    path.to_str()
        .map(str::to_string)
        .ok_or_else(|| format!("clone-audit path is not UTF-8: {}", path.display()))
}

#[cfg(test)]
mod tests;
