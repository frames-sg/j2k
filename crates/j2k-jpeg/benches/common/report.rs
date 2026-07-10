// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub(crate) fn report_iterations(default_iters: usize) -> usize {
    std::env::var("J2K_REPORT_ITERS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|&iters| iters > 0)
        .unwrap_or(default_iters)
}

pub(crate) fn report_ratio(numerator: u128, denominator: u128) -> f64 {
    nanos_as_secs(numerator) / nanos_as_secs(denominator)
}

pub(crate) fn escape_csv(raw: &str) -> String {
    raw.replace('"', "\"\"")
}

pub(crate) fn escape_markdown_table_cell(raw: &str) -> String {
    raw.replace('\\', "\\\\")
        .replace('|', "\\|")
        .replace('\n', "<br>")
}

pub(crate) fn nanos_as_secs(ns: u128) -> f64 {
    let capped = u64::try_from(ns).unwrap_or(u64::MAX);
    Duration::from_nanos(capped).as_secs_f64()
}

pub(crate) struct ReportPaths {
    pub(crate) csv: PathBuf,
    pub(crate) markdown: PathBuf,
}

pub(crate) fn write_reports(
    report_dir: impl AsRef<Path>,
    stem: &str,
    csv: &str,
    markdown: &str,
) -> ReportPaths {
    let report_dir = report_dir.as_ref();
    fs::create_dir_all(report_dir).expect("create benchmark report directory");
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock after unix epoch")
        .as_secs();
    let csv_path = report_dir.join(format!("{stem}-{timestamp}.csv"));
    let md_path = report_dir.join(format!("{stem}-{timestamp}.md"));

    fs::write(&csv_path, csv).expect("write CSV report");
    fs::write(&md_path, markdown).expect("write Markdown report");
    fs::write(report_dir.join(format!("{stem}-latest.csv")), csv).expect("write latest CSV report");
    fs::write(report_dir.join(format!("{stem}-latest.md")), markdown)
        .expect("write latest Markdown report");

    ReportPaths {
        csv: csv_path,
        markdown: md_path,
    }
}
