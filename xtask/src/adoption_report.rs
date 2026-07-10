use std::{
    collections::BTreeMap,
    fs,
    io::ErrorKind,
    path::{Path, PathBuf},
};

use serde_json::Value;

use crate::publication_gate::collect_publication_gate_issues;

mod render;

const DEFAULT_REPORT_NAME: &str = "adoption-report.md";

#[derive(Debug)]
struct AdoptionReportOptions {
    run_dir: PathBuf,
    out: Option<PathBuf>,
    allow_nonpublishable: bool,
}

#[derive(Debug)]
struct TsvTable {
    headers: Vec<String>,
    rows: Vec<BTreeMap<String, String>>,
}

#[derive(Debug)]
struct AdoptionReportEvidence {
    summary: Value,
    publication_issues: Vec<String>,
}

#[derive(Debug)]
struct AdoptionReportModel {
    run_dir: PathBuf,
    summary: Value,
    cpu_fixture_compare: TsvTable,
    cpu_encode_compare: TsvTable,
    publication_issues: Vec<String>,
}

pub(crate) fn adoption_report(args: impl Iterator<Item = String>) -> Result<(), String> {
    let args = args.collect::<Vec<_>>();
    if args
        .iter()
        .any(|arg| matches!(arg.as_str(), "--help" | "-h"))
    {
        println!("{}", help_text());
        return Ok(());
    }
    let options = AdoptionReportOptions::parse(args.into_iter())?;
    let evidence = AdoptionReportEvidence::collect(&options.run_dir)?;
    if !evidence.publication_issues.is_empty() && !options.allow_nonpublishable {
        return Err(format!(
            "adoption benchmark bundle is not publishable: {}. Re-run with --allow-nonpublishable only for diagnostic reports.",
            evidence.publication_issues.join("; ")
        ));
    }

    let model = AdoptionReportModel::collect(options.run_dir.clone(), evidence)?;
    let report = render::render_report(&model);
    let out = options
        .out
        .clone()
        .unwrap_or_else(|| options.run_dir.join(DEFAULT_REPORT_NAME));
    if let Some(parent) = out.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .map_err(|error| format!("create {}: {error}", parent.display()))?;
    }
    fs::write(&out, report).map_err(|error| format!("write {}: {error}", out.display()))?;
    eprintln!("wrote adoption report {}", out.display());
    Ok(())
}

impl AdoptionReportEvidence {
    fn collect(run_dir: &Path) -> Result<Self, String> {
        let summary_path = run_dir.join("summary.json");
        let summary_text = fs::read_to_string(&summary_path).map_err(|error| {
            if error.kind() == ErrorKind::NotFound {
                format!(
                    "adoption benchmark summary is missing at {}. Run `cargo xtask adoption-benchmark` with external fixtures/comparators or pass `--run-dir` to a completed bundle. Public benchmark/adoption claims remain blocked until this external evidence exists.",
                    summary_path.display()
                )
            } else {
                format!("read {}: {error}", summary_path.display())
            }
        })?;
        let summary: Value = serde_json::from_str(&summary_text)
            .map_err(|error| format!("parse {}: {error}", summary_path.display()))?;
        let publication_issues = publication_issues(&summary);
        Ok(Self {
            summary,
            publication_issues,
        })
    }
}

impl AdoptionReportModel {
    fn collect(run_dir: PathBuf, evidence: AdoptionReportEvidence) -> Result<Self, String> {
        let cpu_fixture_compare = read_tsv_table(&run_dir.join("cpu-fixture-compare.out"))?;
        let cpu_encode_compare = read_tsv_table(&run_dir.join("cpu-encode-compare.out"))?;
        Ok(Self {
            run_dir,
            summary: evidence.summary,
            cpu_fixture_compare,
            cpu_encode_compare,
            publication_issues: evidence.publication_issues,
        })
    }

    #[cfg(test)]
    fn serialized_schema(&self) -> Value {
        serde_json::json!({
            "run_dir": self.run_dir.display().to_string(),
            "summary": &self.summary,
            "cpu_fixture_compare": {
                "headers": &self.cpu_fixture_compare.headers,
                "rows": &self.cpu_fixture_compare.rows,
            },
            "cpu_encode_compare": {
                "headers": &self.cpu_encode_compare.headers,
                "rows": &self.cpu_encode_compare.rows,
            },
            "publication_issues": &self.publication_issues,
        })
    }
}

impl AdoptionReportOptions {
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut options = Self {
            run_dir: PathBuf::from("target/j2k-adoption-benchmark/full"),
            out: None,
            allow_nonpublishable: false,
        };
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--run-dir" => {
                    options.run_dir = PathBuf::from(
                        args.next()
                            .ok_or_else(|| "--run-dir requires a value".to_string())?,
                    );
                }
                "--out" => {
                    options.out = Some(PathBuf::from(
                        args.next()
                            .ok_or_else(|| "--out requires a value".to_string())?,
                    ));
                }
                "--allow-nonpublishable" => options.allow_nonpublishable = true,
                "--help" | "-h" => unreachable!("help handled before option parsing"),
                other => {
                    return Err(format!(
                        "unknown adoption-report argument `{other}`\n{}",
                        help_text()
                    ));
                }
            }
        }
        Ok(options)
    }
}

fn help_text() -> String {
    "usage: cargo xtask adoption-report [--run-dir DIR] [--out FILE] [--allow-nonpublishable]"
        .to_string()
}

fn publication_issues(summary: &Value) -> Vec<String> {
    let mut issues = Vec::new();
    collect_publication_gate_issues(
        "cpu-fixture-compare",
        summary.get("cpu_fixture_compare"),
        &mut issues,
    );
    collect_publication_gate_issues(
        "cpu-encode-compare",
        summary.get("cpu_encode_compare"),
        &mut issues,
    );
    if summary
        .get("mode")
        .and_then(Value::as_str)
        .is_some_and(|mode| mode != "full")
    {
        issues.push("run mode is not full".to_string());
    }
    if summary
        .get("include_generated")
        .and_then(Value::as_bool)
        .unwrap_or(true)
    {
        issues.push("generated fixtures included".to_string());
    }
    if summary
        .get("require_cuda")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        collect_required_cuda_issues(summary, &mut issues);
    }
    if summary
        .get("require_metal")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        collect_required_metal_issues(summary, &mut issues);
    }
    issues
}

fn collect_required_cuda_issues(summary: &Value, issues: &mut Vec<String>) {
    if !summary
        .get("cuda_requested")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        issues.push("CUDA required but cuda_requested is not true".to_string());
    }
    collect_step_ran_issue(summary, "cuda-htj2k-decode", "CUDA decode", issues);
    collect_step_ran_issue(summary, "cuda-htj2k-encode", "CUDA encode", issues);
    let Some(decode) = collect_metadata_present_issue(
        summary,
        "cuda_htj2k_decode",
        "CUDA decode metadata",
        issues,
    ) else {
        return;
    };
    collect_metadata_error_issue(decode, "CUDA decode metadata", issues);
    collect_required_string_issue(
        decode,
        "j2k_cuda_decode_io_policy",
        "cuda-rows-return-device-resident-surfaces",
        "CUDA decode io_policy",
        issues,
    );
    collect_not_set_issue(
        decode,
        "j2k_cuda_decode_input_dirs",
        "CUDA decode input dirs",
        issues,
    );
    collect_not_set_issue(
        decode,
        "j2k_cuda_decode_manifest",
        "CUDA decode manifest",
        issues,
    );
    collect_equals_issue(
        decode,
        "j2k_cuda_decode_generated_included",
        "false",
        "CUDA decode generated inclusion",
        issues,
    );
    collect_positive_count_issue(
        decode,
        "j2k_cuda_decode_external_case_count",
        "CUDA decode external case count",
        issues,
    );

    let Some(encode) = collect_metadata_present_issue(
        summary,
        "cuda_htj2k_encode",
        "CUDA encode metadata",
        issues,
    ) else {
        return;
    };
    collect_metadata_error_issue(encode, "CUDA encode metadata", issues);
    collect_required_string_issue(
        encode,
        "j2k_cuda_encode_io_policy",
        "cuda-host-input-rows-include-public-api-host-submission-and-device-encode-work",
        "CUDA encode io_policy",
        issues,
    );
    collect_not_set_issue(
        encode,
        "j2k_cuda_encode_input_dirs",
        "CUDA encode input dirs",
        issues,
    );
    collect_not_set_issue(
        encode,
        "j2k_cuda_encode_manifest",
        "CUDA encode manifest",
        issues,
    );
    collect_equals_issue(
        encode,
        "j2k_cuda_encode_generated_host_input_included",
        "false",
        "CUDA encode generated host-input inclusion",
        issues,
    );
    collect_equals_issue(
        encode,
        "j2k_cuda_encode_external_input_format",
        "staged-pnm-p5-p6",
        "CUDA encode external input format",
        issues,
    );
    collect_positive_count_issue(
        encode,
        "j2k_cuda_encode_external_case_count",
        "CUDA encode external case count",
        issues,
    );
    collect_criterion_step_issue(summary, "cuda-htj2k-decode", "CUDA decode", issues);
    collect_criterion_step_issue(summary, "cuda-htj2k-encode", "CUDA encode", issues);
}

fn collect_required_metal_issues(summary: &Value, issues: &mut Vec<String>) {
    if !summary
        .get("metal_requested")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        issues.push("Metal required but metal_requested is not true".to_string());
    }
    collect_required_metal_decode_issues(summary, issues);
    collect_required_metal_encode_issues(summary, issues);
    collect_required_metal_transcode_issues(summary, issues);
}

fn collect_required_metal_decode_issues(summary: &Value, issues: &mut Vec<String>) {
    collect_step_ran_issue(
        summary,
        "metal-decode-benchmark",
        "Metal decode benchmark",
        issues,
    );
    let Some(metal) = collect_metadata_present_issue(
        summary,
        "metal_decode_benchmark",
        "Metal decode metadata",
        issues,
    ) else {
        return;
    };
    if metal.get("status").and_then(Value::as_str) != Some("ran") {
        issues.push(format!(
            "Metal decode status is {}",
            metal
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("not-recorded")
        ));
    }
    if let Some(error) = metal.get("error").and_then(Value::as_str) {
        issues.push(format!("Metal decode output error: {error}"));
    }
    let bench_count = metal
        .get("bench_count")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let verified_count = metal
        .get("verified_bench_count")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    if bench_count == 0 {
        issues.push("Metal decode has no benchmark rows".to_string());
    }
    if verified_count == 0 {
        issues.push("Metal decode has no verified CPU/Metal benchmark rows".to_string());
    }
    if bench_count > 0
        && metal
            .get("skipped_bench_count")
            .and_then(Value::as_u64)
            .unwrap_or(bench_count)
            == bench_count
    {
        issues.push("Metal decode has no measured benchmark rows".to_string());
    }
    if !has_verified_external_metal_decode_row(metal) {
        issues.push("Metal decode has no verified external benchmark rows".to_string());
    }
    let Some(metadata) = metal.get("metadata") else {
        issues.push("Metal decode metadata missing".to_string());
        return;
    };
    collect_required_string_issue(
        metadata,
        "j2k_metal_decode_io_policy",
        "metal_resident_ms-does-not-readback",
        "Metal decode io_policy",
        issues,
    );
    collect_not_set_issue(
        metadata,
        "j2k_metal_decode_input_dirs",
        "Metal decode input dirs",
        issues,
    );
    collect_not_set_issue(
        metadata,
        "j2k_metal_decode_manifest",
        "Metal decode manifest",
        issues,
    );
    collect_equals_issue(
        metadata,
        "j2k_metal_decode_generated_included",
        "false",
        "Metal decode generated inclusion",
        issues,
    );
    collect_positive_count_issue(
        metadata,
        "j2k_metal_decode_external_case_count",
        "Metal decode external case count",
        issues,
    );
}

#[expect(
    clippy::too_many_lines,
    reason = "all required Metal encode publication gates are evaluated together fail-closed"
)]
fn collect_required_metal_encode_issues(summary: &Value, issues: &mut Vec<String>) {
    collect_step_ran_issue(
        summary,
        "metal-encode-auto-routing",
        "Metal encode auto-routing",
        issues,
    );
    let Some(metal) = collect_metadata_present_issue(
        summary,
        "metal_encode_auto_routing",
        "Metal encode metadata",
        issues,
    ) else {
        return;
    };
    if metal.get("status").and_then(Value::as_str) != Some("ran") {
        issues.push(format!(
            "Metal encode status is {}",
            metal
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("not-recorded")
        ));
    }
    if metal
        .get("probe_error_count")
        .and_then(Value::as_u64)
        .unwrap_or(1)
        != 0
    {
        issues.push("Metal encode has probe errors".to_string());
    }
    if metal
        .get("auto_bench_count")
        .and_then(Value::as_u64)
        .unwrap_or(0)
        == 0
    {
        issues.push("Metal encode has no auto benchmark rows".to_string());
    }
    let resident_count = metal
        .get("resident_bench_count")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let resident_verified_count = metal
        .get("resident_verified_bench_count")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    if resident_count == 0 {
        issues.push("Metal encode has no resident benchmark rows".to_string());
    }
    if resident_verified_count == 0 {
        issues.push(
            "Metal encode has no verified resident packetization/codestream rows".to_string(),
        );
    }
    if resident_count > 0
        && metal
            .get("skipped_resident_bench_count")
            .and_then(Value::as_u64)
            .unwrap_or(resident_count)
            == resident_count
    {
        issues.push("Metal encode has no measured resident benchmark rows".to_string());
    }
    let Some(metadata) = metal.get("metadata") else {
        issues.push("Metal encode metadata missing".to_string());
        return;
    };
    collect_required_string_issue(
        metadata,
        "j2k_metal_encode_io_policy",
        "auto-rows-include-public-api-host-submission-and-metal-auto-route-work",
        "Metal encode io_policy",
        issues,
    );
    collect_not_set_issue(
        metadata,
        "j2k_metal_encode_input_dirs",
        "Metal encode input dirs",
        issues,
    );
    collect_not_set_issue(
        metadata,
        "j2k_metal_encode_manifest",
        "Metal encode manifest",
        issues,
    );
    collect_equals_issue(
        metadata,
        "j2k_metal_encode_generated_host_input_included",
        "false",
        "Metal encode generated host-input inclusion",
        issues,
    );
    collect_equals_issue(
        metadata,
        "j2k_metal_encode_external_input_format",
        "staged-pnm-p5-p6",
        "Metal encode external input format",
        issues,
    );
    collect_positive_count_issue(
        metadata,
        "j2k_metal_encode_external_case_count",
        "Metal encode external case count",
        issues,
    );
}

fn collect_required_metal_transcode_issues(summary: &Value, issues: &mut Vec<String>) {
    collect_step_ran_issue(
        summary,
        "metal-transcode-benchmark",
        "Metal transcode benchmark",
        issues,
    );
    let Some(metal) = collect_metadata_present_issue(
        summary,
        "metal_transcode_benchmark",
        "Metal transcode metadata",
        issues,
    ) else {
        return;
    };
    if metal.get("status").and_then(Value::as_str) != Some("ran") {
        issues.push(format!(
            "Metal transcode status is {}",
            metal
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("not-recorded")
        ));
    }
    if let Some(error) = metal.get("error").and_then(Value::as_str) {
        issues.push(format!("Metal transcode output error: {error}"));
    }
    if metal
        .get("profile_count")
        .and_then(Value::as_u64)
        .unwrap_or(0)
        == 0
    {
        issues.push("Metal transcode has no profile rows".to_string());
    }
    if metal
        .get("verified_profile_count")
        .and_then(Value::as_u64)
        .unwrap_or(0)
        == 0
    {
        issues.push("Metal transcode has no verified Metal-dispatch profile rows".to_string());
    }
    if metal
        .get("comparison_context_count")
        .and_then(Value::as_u64)
        .unwrap_or(0)
        == 0
    {
        issues.push("Metal transcode has no comparable CPU/Metal profile context".to_string());
    }
    let auto_count = metal
        .get("auto_metal_profile_count")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let explicit_count = metal
        .get("explicit_metal_profile_count")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    if auto_count + explicit_count == 0 {
        issues.push("Metal transcode has no Metal-requested profile rows".to_string());
    }
}

fn has_verified_external_metal_decode_row(metal: &Value) -> bool {
    metal
        .get("benches")
        .and_then(Value::as_array)
        .is_some_and(|rows| {
            rows.iter().any(|row| {
                value_field(row, "source").starts_with("external:")
                    && numeric_field(row, "cpu_ms").is_some()
                    && numeric_field(row, "metal_resident_ms").is_some()
                    && numeric_field(row, "metal_readback_ms").is_some()
            })
        })
}

fn collect_metadata_present_issue<'a>(
    summary: &'a Value,
    key: &str,
    label: &str,
    issues: &mut Vec<String>,
) -> Option<&'a Value> {
    let metadata = summary.get(key);
    if metadata.is_none() {
        issues.push(format!("{label} missing"));
    }
    metadata
}

fn collect_metadata_error_issue(metadata: &Value, label: &str, issues: &mut Vec<String>) {
    if let Some(error) = metadata.get("metadata_error").and_then(Value::as_str) {
        issues.push(format!("{label} error: {error}"));
    }
}

fn collect_step_ran_issue(summary: &Value, name: &str, label: &str, issues: &mut Vec<String>) {
    let status = summary
        .get("steps")
        .and_then(Value::as_array)
        .and_then(|steps| {
            steps.iter().find_map(|step| {
                (step.get("name").and_then(Value::as_str) == Some(name)).then(|| {
                    step.get("status")
                        .and_then(Value::as_str)
                        .unwrap_or("missing")
                })
            })
        });
    if status != Some("ran") {
        issues.push(format!(
            "{label} step status is {}",
            status.unwrap_or("missing")
        ));
    }
}

fn collect_criterion_step_issue(
    summary: &Value,
    name: &str,
    label: &str,
    issues: &mut Vec<String>,
) {
    let count = summary
        .get("criterion")
        .and_then(|criterion| criterion.get("steps"))
        .and_then(Value::as_array)
        .and_then(|steps| {
            steps.iter().find_map(|step| {
                (step.get("step").and_then(Value::as_str) == Some(name))
                    .then(|| step.get("count").and_then(Value::as_u64).unwrap_or(0))
            })
        })
        .unwrap_or(0);
    if count == 0 {
        issues.push(format!("{label} Criterion estimates missing"));
    }
}

fn collect_required_string_issue(
    metadata: &Value,
    key: &str,
    required_substring: &str,
    label: &str,
    issues: &mut Vec<String>,
) {
    let value = metadata.get(key).and_then(Value::as_str).unwrap_or("");
    if !value.contains(required_substring) {
        issues.push(format!("{label} missing `{required_substring}`"));
    }
}

fn collect_not_set_issue(metadata: &Value, key: &str, label: &str, issues: &mut Vec<String>) {
    if metadata
        .get(key)
        .and_then(Value::as_str)
        .is_none_or(|value| matches!(value, "" | "not set" | "not-recorded"))
    {
        issues.push(format!("{label} not recorded"));
    }
}

fn collect_equals_issue(
    metadata: &Value,
    key: &str,
    expected: &str,
    label: &str,
    issues: &mut Vec<String>,
) {
    if metadata.get(key).and_then(Value::as_str) != Some(expected) {
        issues.push(format!("{label} is not `{expected}`"));
    }
}

fn collect_positive_count_issue(
    metadata: &Value,
    key: &str,
    label: &str,
    issues: &mut Vec<String>,
) {
    let count = metadata
        .get(key)
        .and_then(Value::as_str)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    if count == 0 {
        issues.push(format!("{label} is zero or missing"));
    }
}

fn numeric_field(row: &Value, key: &str) -> Option<f64> {
    match row.get(key)? {
        Value::Number(number) => number.as_f64(),
        Value::String(value) => value.parse::<f64>().ok(),
        _ => None,
    }
}

fn value_field(row: &Value, key: &str) -> String {
    match row.get(key) {
        Some(Value::String(value)) => value.clone(),
        Some(Value::Number(value)) => value.to_string(),
        Some(Value::Bool(value)) => value.to_string(),
        Some(Value::Null) | None => "not-recorded".to_string(),
        Some(other) => other.to_string(),
    }
}

fn read_tsv_table(path: &Path) -> Result<TsvTable, String> {
    let text =
        fs::read_to_string(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let mut lines = text.lines();
    let header = lines
        .by_ref()
        .find(|line| line.starts_with("decoder\tcase\t") || line.starts_with("encoder\tcase\t"))
        .ok_or_else(|| {
            format!(
                "{} did not contain a benchmark table header",
                path.display()
            )
        })?;
    let headers = header.split('\t').map(str::to_string).collect::<Vec<_>>();
    let mut rows = Vec::new();
    for line in lines.take_while(|line| !line.starts_with("benchmark_complete\t")) {
        if line.trim().is_empty() {
            continue;
        }
        let values = line.split('\t').collect::<Vec<_>>();
        if values.len() != headers.len() {
            return Err(format!(
                "{} row has {} columns but header has {}: {line}",
                path.display(),
                values.len(),
                headers.len()
            ));
        }
        rows.push(
            headers
                .iter()
                .cloned()
                .zip(values.into_iter().map(str::to_string))
                .collect(),
        );
    }
    Ok(TsvTable { headers, rows })
}

#[cfg(test)]
mod tests;
