use std::{
    collections::BTreeMap,
    fs,
    io::ErrorKind,
    path::{Path, PathBuf},
};

use serde_json::Value;

use crate::markdown::{escape_inline_code as escape_inline, markdown_header, markdown_row};
use crate::publication_gate::collect_publication_gate_issues;

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
    let summary_path = options.run_dir.join("summary.json");
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

    let issues = publication_issues(&summary);
    if !issues.is_empty() && !options.allow_nonpublishable {
        return Err(format!(
            "adoption benchmark bundle is not publishable: {}. Re-run with --allow-nonpublishable only for diagnostic reports.",
            issues.join("; ")
        ));
    }

    let fixture = read_tsv_table(&options.run_dir.join("cpu-fixture-compare.out"))?;
    let encode = read_tsv_table(&options.run_dir.join("cpu-encode-compare.out"))?;
    let report = render_report(&options, &summary, &fixture, &encode, &issues);
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

fn render_report(
    options: &AdoptionReportOptions,
    summary: &Value,
    fixture: &TsvTable,
    encode: &TsvTable,
    issues: &[String],
) -> String {
    let mut out = String::new();
    out.push_str("# J2K Adoption Benchmark Report\n\n");
    if issues.is_empty() {
        out.push_str("Status: publishable for the requested benchmark scopes according to recorded comparator and hardware gates.\n\n");
    } else {
        out.push_str("Status: diagnostic only. Do not use for marketing claims.\n\n");
        out.push_str("Blocking issues:\n");
        for issue in issues {
            out.push_str(&format!("- {issue}\n"));
        }
        out.push('\n');
    }

    out.push_str("## Bundle\n\n");
    metadata_list(
        &mut out,
        &[
            ("run_dir", options.run_dir.display().to_string()),
            ("mode", scalar_label(summary, "mode")),
            (
                "include_generated",
                scalar_label(summary, "include_generated"),
            ),
            ("input_dirs", scalar_label(summary, "input_dirs")),
            ("manifest", scalar_label(summary, "manifest")),
            (
                "encode_input_dirs",
                scalar_label(summary, "encode_input_dirs"),
            ),
            ("encode_manifest", scalar_label(summary, "encode_manifest")),
            (
                "cuda_decode_batch_sizes",
                scalar_label(summary, "cuda_decode_batch_sizes"),
            ),
            ("cuda_requested", scalar_label(summary, "cuda_requested")),
            ("metal_requested", scalar_label(summary, "metal_requested")),
            ("require_cuda", scalar_label(summary, "require_cuda")),
            ("require_metal", scalar_label(summary, "require_metal")),
        ],
    );

    out.push_str("\n## Publication Gates\n\n");
    metadata_table(
        &mut out,
        &[
            ("cpu-fixture-compare", summary.get("cpu_fixture_compare")),
            ("cpu-encode-compare", summary.get("cpu_encode_compare")),
        ],
        &[
            "publication_eligible",
            "publication_blockers",
            "benchmark_complete",
            "case_batch_sizes",
            "mixed_batch_sizes",
            "external_unique_input_count",
            "external_native_case_count",
            "external_materialized_case_count",
            "external_native_unique_input_count",
            "mixed_external_batch_group_count",
            "mixed_external_min_distinct_inputs",
            "mixed_external_max_distinct_inputs",
            "mixed_external_group_distinct_inputs",
            "publication_gate_skipped_comparators",
        ],
    );

    out.push_str("\n## Methodology\n\n");
    if let Some(scope) = summary
        .get("fixture_comparability_scope")
        .and_then(Value::as_str)
    {
        out.push_str(scope);
        out.push_str("\n\n");
    }
    if let Some(note) = summary.get("publication_note").and_then(Value::as_str) {
        out.push_str(note);
        out.push_str("\n\n");
    }
    methodology_metadata(&mut out, summary);

    out.push_str("\n## CPU Decode Rows\n\n");
    measured_table(
        &mut out,
        fixture,
        &[
            "decoder",
            "case",
            "benchmark_mode",
            "decode_method",
            "input_source",
            "corpus_category",
            "corpus_name",
            "license_status",
            "encode_command",
            "manifest_status",
            "codec",
            "container",
            "operation",
            "format",
            "dimensions",
            "source_fnv1a64",
            "batch_size",
            "median_us",
            "tiles_per_second_median",
            "decoded_mib_per_second_median",
        ],
        40,
    );

    out.push_str("\n## CPU Decode Mixed Winner Summary\n\n");
    mixed_winner_summary(
        &mut out,
        fixture,
        "decoder",
        "decoded_mib_per_second_median",
        |row| {
            row.get("input_source")
                .is_some_and(|value| value == "external:mixed")
        },
    );

    out.push_str("\n## CPU Decode Mixed Batch Rows\n\n");
    measured_table_filtered(
        &mut out,
        fixture,
        &[
            "decoder",
            "case",
            "benchmark_mode",
            "decode_method",
            "corpus_category",
            "corpus_name",
            "license_status",
            "codec",
            "container",
            "operation",
            "format",
            "dimensions",
            "batch_size",
            "input_bytes",
            "median_us",
            "tiles_per_second_median",
            "decoded_mib_per_second_median",
            "decoded_bytes_per_repeat",
        ],
        80,
        |row| {
            row.get("input_source")
                .is_some_and(|value| value == "external:mixed")
        },
    );

    out.push_str("\n## CPU Encode Rows\n\n");
    measured_table(
        &mut out,
        encode,
        &[
            "encoder",
            "case",
            "benchmark_mode",
            "encode_method",
            "input_source",
            "corpus_category",
            "corpus_name",
            "license_status",
            "source_command",
            "manifest_status",
            "format",
            "dimensions",
            "batch_size",
            "median_us",
            "images_per_second_median",
            "input_mib_per_second_median",
            "encoded_bytes_per_repeat",
        ],
        40,
    );

    out.push_str("\n## CPU Encode Mixed Winner Summary\n\n");
    mixed_winner_summary(
        &mut out,
        encode,
        "encoder",
        "input_mib_per_second_median",
        |row| {
            row.get("input_source")
                .is_some_and(|value| value == "external:mixed")
        },
    );

    out.push_str("\n## CPU Encode Mixed Batch Rows\n\n");
    measured_table_filtered(
        &mut out,
        encode,
        &[
            "encoder",
            "case",
            "benchmark_mode",
            "encode_method",
            "corpus_category",
            "corpus_name",
            "license_status",
            "format",
            "dimensions",
            "batch_size",
            "input_bytes",
            "median_us",
            "images_per_second_median",
            "input_mib_per_second_median",
            "encoded_bytes_per_repeat",
        ],
        80,
        |row| {
            row.get("input_source")
                .is_some_and(|value| value == "external:mixed")
        },
    );

    out.push_str("\n## Skipped And Context Rows\n\n");
    skipped_summary(&mut out, "decode", fixture);
    skipped_summary(&mut out, "encode", encode);

    out.push_str("\n## Hybrid Summary\n\n");
    metadata_table(
        &mut out,
        &[
            ("cuda-htj2k-decode", summary.get("cuda_htj2k_decode")),
            ("cuda-htj2k-encode", summary.get("cuda_htj2k_encode")),
        ],
        &[
            "j2k_cuda_decode_batch_sizes",
            "j2k_cuda_decode_io_policy",
            "j2k_cuda_decode_external_case_count",
            "j2k_cuda_decode_external_skipped_non_htj2k_count",
            "j2k_cuda_encode_io_policy",
            "j2k_cuda_encode_external_case_count",
            "j2k_cuda_encode_external_input_format",
        ],
    );
    criterion_estimate_table(
        &mut out,
        summary,
        &["cuda-htj2k-decode", "cuda-htj2k-encode"],
    );
    if let Some(metal) = summary.get("metal_decode_benchmark") {
        out.push_str("\nMetal decode benchmark summary:\n\n");
        out.push_str(&format!(
            "- status: {}\n",
            metal
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("not-recorded")
        ));
        if let Some(metadata) = metal.get("metadata") {
            metadata_list(
                &mut out,
                &[
                    (
                        "j2k_metal_decode_io_policy",
                        scalar_label(metadata, "j2k_metal_decode_io_policy"),
                    ),
                    (
                        "j2k_metal_decode_external_case_count",
                        scalar_label(metadata, "j2k_metal_decode_external_case_count"),
                    ),
                    (
                        "j2k_metal_decode_generated_included",
                        scalar_label(metadata, "j2k_metal_decode_generated_included"),
                    ),
                    ("bench_count", scalar_label(metal, "bench_count")),
                    (
                        "skipped_bench_count",
                        scalar_label(metal, "skipped_bench_count"),
                    ),
                    (
                        "verified_bench_count",
                        scalar_label(metal, "verified_bench_count"),
                    ),
                    (
                        "skipped_case_count",
                        scalar_label(metal, "skipped_case_count"),
                    ),
                ],
            );
        }
        metal_decode_summary(&mut out, metal);
    }
    if let Some(metal) = summary.get("metal_encode_auto_routing") {
        out.push_str("\nMetal auto-routing summary:\n\n");
        out.push_str(&format!(
            "- status: {}\n",
            metal
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("not-recorded")
        ));
        if let Some(metadata) = metal.get("metadata") {
            metadata_list(
                &mut out,
                &[
                    (
                        "j2k_metal_encode_io_policy",
                        scalar_label(metadata, "j2k_metal_encode_io_policy"),
                    ),
                    (
                        "j2k_metal_encode_external_case_count",
                        scalar_label(metadata, "j2k_metal_encode_external_case_count"),
                    ),
                    (
                        "j2k_metal_encode_external_input_format",
                        scalar_label(metadata, "j2k_metal_encode_external_input_format"),
                    ),
                    (
                        "j2k_metal_encode_resident_batch_sizes",
                        scalar_label(metadata, "j2k_metal_encode_resident_batch_sizes"),
                    ),
                    ("auto_bench_count", scalar_label(metal, "auto_bench_count")),
                    (
                        "skipped_auto_bench_count",
                        scalar_label(metal, "skipped_auto_bench_count"),
                    ),
                    (
                        "probe_error_count",
                        scalar_label(metal, "probe_error_count"),
                    ),
                    (
                        "resident_bench_count",
                        scalar_label(metal, "resident_bench_count"),
                    ),
                    (
                        "skipped_resident_bench_count",
                        scalar_label(metal, "skipped_resident_bench_count"),
                    ),
                    (
                        "resident_verified_bench_count",
                        scalar_label(metal, "resident_verified_bench_count"),
                    ),
                ],
            );
        }
        metal_auto_summary(&mut out, metal);
        metal_resident_summary(&mut out, metal);
    }
    if let Some(metal) = summary.get("metal_transcode_benchmark") {
        out.push_str("\nMetal transcode benchmark summary:\n\n");
        out.push_str(&format!(
            "- status: {}\n",
            metal
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("not-recorded")
        ));
        metadata_list(
            &mut out,
            &[
                ("bench_filter", scalar_label(metal, "bench_filter")),
                ("profile_count", scalar_label(metal, "profile_count")),
                (
                    "verified_profile_count",
                    scalar_label(metal, "verified_profile_count"),
                ),
                (
                    "comparison_context_count",
                    scalar_label(metal, "comparison_context_count"),
                ),
                (
                    "auto_metal_profile_count",
                    scalar_label(metal, "auto_metal_profile_count"),
                ),
                (
                    "explicit_metal_profile_count",
                    scalar_label(metal, "explicit_metal_profile_count"),
                ),
            ],
        );
        metal_transcode_summary(&mut out, metal);
    }

    out
}

fn methodology_metadata(out: &mut String, summary: &Value) {
    metadata_table(
        out,
        &[
            ("cpu-fixture-compare", summary.get("cpu_fixture_compare")),
            ("cpu-encode-compare", summary.get("cpu_encode_compare")),
        ],
        &[
            "benchmark_mode",
            "build_profile",
            "debug_assertions",
            "git_revision",
            "git_dirty",
            "host_hardware",
            "openjpeg_version",
            "grok_version",
            "openjpeg_compress_available",
            "grok_compress_available",
            "kakadu_included",
        ],
    );
}

fn measured_table(out: &mut String, table: &TsvTable, columns: &[&str], limit: usize) {
    measured_table_filtered(out, table, columns, limit, |_| true);
}

fn measured_table_filtered(
    out: &mut String,
    table: &TsvTable,
    columns: &[&str],
    limit: usize,
    include: impl Fn(&BTreeMap<String, String>) -> bool,
) {
    let missing_columns = columns
        .iter()
        .filter(|column| !table.headers.iter().any(|header| header == **column))
        .copied()
        .collect::<Vec<_>>();
    if !missing_columns.is_empty() {
        out.push_str(&format!(
            "Missing expected raw columns: `{}`.\n\n",
            missing_columns.join("`, `")
        ));
    }
    let measured_rows = table
        .rows
        .iter()
        .filter(|row| row.get("skip_reason").is_none_or(|value| value.is_empty()))
        .filter(|row| include(row))
        .collect::<Vec<_>>();
    let rows = measured_rows
        .iter()
        .take(limit)
        .copied()
        .collect::<Vec<_>>();
    if rows.is_empty() {
        out.push_str("No measured rows recorded.\n");
        return;
    }
    markdown_header(out, columns);
    for row in rows {
        markdown_row(out, columns.iter().map(|column| row_value(row, column)));
    }
    if measured_rows.len() > limit {
        out.push_str(&format!(
            "\nShowing first {limit} measured rows. See raw TSV outputs for the full table.\n"
        ));
    }
}

fn skipped_summary(out: &mut String, label: &str, table: &TsvTable) {
    let mut counts = BTreeMap::<String, usize>::new();
    for row in &table.rows {
        let reason = row.get("skip_reason").map(String::as_str).unwrap_or("");
        if !reason.is_empty() {
            *counts.entry(reason.to_string()).or_default() += 1;
        }
    }
    if counts.is_empty() {
        out.push_str(&format!("- {label}: none\n"));
    } else {
        for (reason, count) in counts {
            out.push_str(&format!("- {label}: {reason} ({count} rows)\n"));
        }
    }
}

fn mixed_winner_summary(
    out: &mut String,
    table: &TsvTable,
    participant_column: &str,
    metric_column: &str,
    include: impl Fn(&BTreeMap<String, String>) -> bool,
) {
    const FIRST_CLASS_PARTICIPANTS: &[&str] = &["j2k", "openjpeg", "grok"];

    let mut groups = BTreeMap::<(String, String), BTreeMap<String, f64>>::new();
    for row in &table.rows {
        if row
            .get("skip_reason")
            .is_some_and(|value| !value.is_empty())
            || !include(row)
        {
            continue;
        }
        let participant = row_value(row, participant_column);
        if !FIRST_CLASS_PARTICIPANTS.contains(&participant.as_str()) {
            continue;
        }
        let Some(metric) = numeric_row_field(row, metric_column) else {
            continue;
        };
        let group = (row_value(row, "case"), row_value(row, "batch_size"));
        groups.entry(group).or_default().insert(participant, metric);
    }
    if groups.is_empty() {
        out.push_str("No mixed rows recorded.\n");
        return;
    }

    out.push_str(
        "Winner eligibility is limited to first-class comparable rows: `j2k`, `openjpeg`, and `grok`. Optional CLI context rows such as OpenJPH or Kakadu remain in raw tables but do not decide this summary.\n\n",
    );

    let columns = [
        "case",
        "batch_size",
        "j2k_mib_per_s",
        "openjpeg_mib_per_s",
        "grok_mib_per_s",
        "winner",
        "winner_mib_per_s",
        "j2k_vs_winner",
    ];
    markdown_header(out, &columns);
    for ((case, batch_size), participants) in groups {
        let (winner_name, winner_value) = participants
            .iter()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(name, value)| (name.as_str(), *value))
            .unwrap_or(("NA", f64::NAN));
        let j2k_vs_winner = participants
            .get("j2k")
            .filter(|_| winner_value.is_finite() && winner_value > 0.0)
            .map(|value| format!("{:.3}x", value / winner_value))
            .unwrap_or_else(|| "NA".to_string());
        markdown_row(
            out,
            [
                case,
                batch_size,
                metric_label(participants.get("j2k").copied()),
                metric_label(participants.get("openjpeg").copied()),
                metric_label(participants.get("grok").copied()),
                winner_name.to_string(),
                metric_label(Some(winner_value)),
                j2k_vs_winner,
            ],
        );
    }
}

fn metadata_table(out: &mut String, groups: &[(&str, Option<&Value>)], keys: &[&str]) {
    let mut columns = Vec::with_capacity(keys.len() + 1);
    columns.push("section");
    columns.extend(keys.iter().copied());
    markdown_header(out, &columns);
    for (label, value) in groups {
        let mut row = Vec::with_capacity(keys.len() + 1);
        row.push((*label).to_string());
        for key in keys {
            row.push(value.map_or_else(|| "missing".to_string(), |v| scalar_label(v, key)));
        }
        markdown_row(out, row);
    }
}

fn metadata_list(out: &mut String, values: &[(&str, String)]) {
    for (key, value) in values {
        out.push_str(&format!("- `{key}`: `{}`\n", escape_inline(value)));
    }
}

fn criterion_estimate_table(out: &mut String, summary: &Value, step_names: &[&str]) {
    let Some(steps) = summary
        .get("criterion")
        .and_then(|criterion| criterion.get("steps"))
        .and_then(Value::as_array)
    else {
        out.push_str("\nNo Criterion estimates recorded.\n");
        return;
    };
    let mut rows = Vec::new();
    for step_name in step_names {
        let Some(step) = steps
            .iter()
            .find(|step| step.get("step").and_then(Value::as_str) == Some(*step_name))
        else {
            continue;
        };
        let Some(estimates) = step.get("estimates").and_then(Value::as_array) else {
            continue;
        };
        for estimate in estimates {
            rows.push((
                (*step_name).to_string(),
                value_field(estimate, "id"),
                numeric_field(estimate, "median_ns"),
                numeric_field(estimate, "median_lower_ns"),
                numeric_field(estimate, "median_upper_ns"),
            ));
        }
    }
    if rows.is_empty() {
        out.push_str("\nNo CUDA Criterion estimate rows recorded.\n");
        return;
    }

    out.push_str("\nCUDA Criterion estimate rows:\n\n");
    let columns = [
        "step",
        "id",
        "median_ms",
        "median_lower_ms",
        "median_upper_ms",
    ];
    markdown_header(out, &columns);
    for (step, id, median, lower, upper) in rows {
        markdown_row(
            out,
            [
                step,
                id,
                ns_to_ms_label(median),
                ns_to_ms_label(lower),
                ns_to_ms_label(upper),
            ],
        );
    }
}

#[derive(Default)]
struct MetalAutoGroup {
    rows: usize,
    cpu_ms_total: f64,
    auto_ms_total: f64,
}

#[derive(Default)]
struct MetalResidentGroup {
    rows: usize,
    cpu_ms_total: f64,
    hybrid_ms_total: f64,
    hybrid_rows: usize,
    resident_host_ms_total: f64,
    resident_buffer_ms_total: f64,
    host_readback_ms_total: f64,
}

#[derive(Default)]
struct MetalDecodeGroup {
    rows: usize,
    cpu_ms_total: f64,
    resident_ms_total: f64,
    readback_ms_total: f64,
}

#[derive(Default)]
struct MetalTranscodeGroup {
    rows: usize,
    total_ms_total: f64,
    transfer_bytes_total: u64,
    dct_handoffs_total: u64,
    dwt_handoffs_total: u64,
    dispatches_total: u64,
    tiles_total: u64,
}

fn metal_decode_summary(out: &mut String, metal: &Value) {
    let Some(rows) = metal.get("benches").and_then(Value::as_array) else {
        out.push_str("\nNo Metal decode benchmark rows recorded.\n");
        return;
    };
    let mut groups =
        BTreeMap::<(String, String, String, String, String, String), MetalDecodeGroup>::new();
    for row in rows {
        let Some(cpu_ms) = numeric_field(row, "cpu_ms") else {
            continue;
        };
        let Some(resident_ms) = numeric_field(row, "metal_resident_ms") else {
            continue;
        };
        let Some(readback_ms) = numeric_field(row, "metal_readback_ms") else {
            continue;
        };
        let key = (
            metal_decode_source_category(row),
            value_field(row, "codec"),
            value_field(row, "container"),
            value_field(row, "operation"),
            value_field(row, "fmt"),
            value_field(row, "size"),
        );
        let group = groups.entry(key).or_default();
        group.rows += 1;
        group.cpu_ms_total += cpu_ms;
        group.resident_ms_total += resident_ms;
        group.readback_ms_total += readback_ms;
    }
    if groups.is_empty() {
        out.push_str("\nNo measured Metal decode benchmark rows recorded.\n");
        return;
    }

    out.push_str("\nMetal decode row summary:\n\n");
    let columns = [
        "source",
        "codec",
        "container",
        "operation",
        "fmt",
        "size",
        "rows",
        "cpu_ms_avg",
        "metal_resident_ms_avg",
        "metal_readback_ms_avg",
        "readback_vs_cpu",
        "winner",
    ];
    markdown_header(out, &columns);
    for ((source, codec, container, operation, fmt, size), group) in groups {
        let rows = group.rows as f64;
        let cpu_avg = group.cpu_ms_total / rows;
        let resident_avg = group.resident_ms_total / rows;
        let readback_avg = group.readback_ms_total / rows;
        let ratio = readback_avg / cpu_avg;
        let winner = if readback_avg < cpu_avg {
            "metal-readback"
        } else if cpu_avg < readback_avg {
            "cpu"
        } else {
            "tie"
        };
        markdown_row(
            out,
            [
                source,
                codec,
                container,
                operation,
                fmt,
                size,
                group.rows.to_string(),
                format!("{cpu_avg:.3}"),
                format!("{resident_avg:.3}"),
                format!("{readback_avg:.3}"),
                format!("{ratio:.3}x"),
                winner.to_string(),
            ],
        );
    }
}

fn metal_decode_source_category(row: &Value) -> String {
    let source = value_field(row, "source");
    if source.starts_with("external:") {
        "external".to_string()
    } else {
        source
    }
}

fn metal_auto_summary(out: &mut String, metal: &Value) {
    let Some(rows) = metal.get("auto_benches").and_then(Value::as_array) else {
        out.push_str("\nNo Metal auto benchmark rows recorded.\n");
        return;
    };
    let mut groups = BTreeMap::<(String, String, String, String), MetalAutoGroup>::new();
    for row in rows {
        let Some(cpu_ms) = numeric_field(row, "cpu_ms") else {
            continue;
        };
        let Some(auto_ms) = numeric_field(row, "auto_ms") else {
            continue;
        };
        let key = (
            value_field(row, "mode"),
            value_field(row, "codec"),
            value_field(row, "components"),
            value_field(row, "size"),
        );
        let group = groups.entry(key).or_default();
        group.rows += 1;
        group.cpu_ms_total += cpu_ms;
        group.auto_ms_total += auto_ms;
    }
    if groups.is_empty() {
        out.push_str("\nNo measured Metal auto benchmark rows recorded.\n");
        return;
    }

    out.push_str("\nMetal auto external row summary:\n\n");
    let columns = [
        "mode",
        "codec",
        "components",
        "size",
        "rows",
        "cpu_ms_avg",
        "metal_auto_ms_avg",
        "metal_vs_cpu",
        "winner",
    ];
    markdown_header(out, &columns);
    for ((mode, codec, components, size), group) in groups {
        let cpu_avg = group.cpu_ms_total / group.rows as f64;
        let auto_avg = group.auto_ms_total / group.rows as f64;
        let ratio = auto_avg / cpu_avg;
        let winner = if auto_avg < cpu_avg {
            "metal-auto"
        } else if cpu_avg < auto_avg {
            "cpu"
        } else {
            "tie"
        };
        markdown_row(
            out,
            [
                mode,
                codec,
                components,
                size,
                group.rows.to_string(),
                format!("{cpu_avg:.3}"),
                format!("{auto_avg:.3}"),
                format!("{ratio:.3}x"),
                winner.to_string(),
            ],
        );
    }
}

fn metal_resident_summary(out: &mut String, metal: &Value) {
    let Some(rows) = metal.get("resident_benches").and_then(Value::as_array) else {
        out.push_str("\nNo Metal resident benchmark rows recorded.\n");
        return;
    };
    let mut groups =
        BTreeMap::<(String, String, String, String, String), MetalResidentGroup>::new();
    for row in rows {
        if row.get("packetization_used").and_then(Value::as_bool) != Some(true)
            || row.get("codestream_assembly_used").and_then(Value::as_bool) != Some(true)
        {
            continue;
        }
        let Some(cpu_ms) = numeric_field(row, "cpu_ms") else {
            continue;
        };
        let Some(resident_host_ms) = numeric_field(row, "resident_host_ms") else {
            continue;
        };
        let Some(resident_buffer_ms) = numeric_field(row, "resident_buffer_ms") else {
            continue;
        };
        let key = (
            value_field(row, "mode"),
            value_field(row, "codec"),
            value_field(row, "components"),
            value_field(row, "size"),
            value_field(row, "batch_size"),
        );
        let group = groups.entry(key).or_default();
        group.rows += 1;
        group.cpu_ms_total += cpu_ms;
        group.resident_host_ms_total += resident_host_ms;
        group.resident_buffer_ms_total += resident_buffer_ms;
        if let Some(hybrid_ms) = numeric_field(row, "hybrid_cpu_packet_ms") {
            group.hybrid_rows += 1;
            group.hybrid_ms_total += hybrid_ms;
        }
        if let Some(host_readback_ms) = numeric_field(row, "host_readback_ms") {
            group.host_readback_ms_total += host_readback_ms;
        }
    }
    if groups.is_empty() {
        out.push_str("\nNo verified Metal resident packetization rows recorded.\n");
        return;
    }

    out.push_str("\nMetal resident packetization summary:\n\n");
    let columns = [
        "mode",
        "codec",
        "components",
        "size",
        "batch_size",
        "rows",
        "cpu_ms_avg",
        "hybrid_cpu_packet_ms_avg",
        "resident_host_ms_avg",
        "resident_buffer_ms_avg",
        "host_readback_ms_avg",
        "resident_host_vs_cpu",
        "winner",
    ];
    markdown_header(out, &columns);
    for ((mode, codec, components, size, batch_size), group) in groups {
        let rows = group.rows as f64;
        let cpu_avg = group.cpu_ms_total / rows;
        let resident_host_avg = group.resident_host_ms_total / rows;
        let resident_buffer_avg = group.resident_buffer_ms_total / rows;
        let host_readback_avg = group.host_readback_ms_total / rows;
        let hybrid_avg =
            (group.hybrid_rows > 0).then(|| group.hybrid_ms_total / group.hybrid_rows as f64);
        let ratio = resident_host_avg / cpu_avg;
        let winner = if resident_host_avg < cpu_avg {
            "resident-host"
        } else if cpu_avg < resident_host_avg {
            "cpu"
        } else {
            "tie"
        };
        markdown_row(
            out,
            [
                mode,
                codec,
                components,
                size,
                batch_size,
                group.rows.to_string(),
                format!("{cpu_avg:.3}"),
                metric_label(hybrid_avg),
                format!("{resident_host_avg:.3}"),
                format!("{resident_buffer_avg:.3}"),
                format!("{host_readback_avg:.3}"),
                format!("{ratio:.3}x"),
                winner.to_string(),
            ],
        );
    }
}

fn metal_transcode_summary(out: &mut String, metal: &Value) {
    let Some(rows) = metal.get("profiles").and_then(Value::as_array) else {
        out.push_str("\nNo Metal transcode profile rows recorded.\n");
        return;
    };
    let mut groups = BTreeMap::<(String, String, String, String), MetalTranscodeGroup>::new();
    for row in rows {
        let Some(total_us) = numeric_field(row, "total_us") else {
            continue;
        };
        let key = (
            value_field(row, "context"),
            value_field(row, "request"),
            value_field(row, "transform_processor"),
            value_field(row, "pipeline"),
        );
        let group = groups.entry(key).or_default();
        group.rows += 1;
        group.total_ms_total += total_us / 1000.0;
        group.transfer_bytes_total += integer_field(row, "host_to_device_transfer_bytes")
            .unwrap_or(0)
            + integer_field(row, "device_to_host_transfer_bytes").unwrap_or(0);
        group.dct_handoffs_total +=
            integer_field(row, "dwt97_batch_resident_dct_handoff_count").unwrap_or(0);
        group.dwt_handoffs_total +=
            integer_field(row, "dwt97_batch_resident_dwt_handoff_count").unwrap_or(0);
        group.dispatches_total += integer_field(row, "accelerator_dispatches").unwrap_or(0);
        group.tiles_total += integer_field(row, "successful_tiles").unwrap_or(0);
    }
    if groups.is_empty() {
        out.push_str("\nNo measured Metal transcode profile rows recorded.\n");
        return;
    }

    out.push_str("\nMetal transcode profile summary:\n\n");
    let columns = [
        "context",
        "request",
        "transform_processor",
        "pipeline",
        "rows",
        "total_ms_avg",
        "successful_tiles",
        "dct_handoffs",
        "dwt_handoffs",
        "accelerator_dispatches",
        "transfer_bytes",
    ];
    markdown_header(out, &columns);
    for ((context, request, transform_processor, pipeline), group) in groups {
        markdown_row(
            out,
            [
                context,
                request,
                transform_processor,
                pipeline,
                group.rows.to_string(),
                format!("{:.3}", group.total_ms_total / group.rows as f64),
                group.tiles_total.to_string(),
                group.dct_handoffs_total.to_string(),
                group.dwt_handoffs_total.to_string(),
                group.dispatches_total.to_string(),
                group.transfer_bytes_total.to_string(),
            ],
        );
    }
}

fn numeric_field(row: &Value, key: &str) -> Option<f64> {
    match row.get(key)? {
        Value::Number(number) => number.as_f64(),
        Value::String(value) => value.parse::<f64>().ok(),
        _ => None,
    }
}

fn integer_field(row: &Value, key: &str) -> Option<u64> {
    match row.get(key)? {
        Value::Number(number) => number.as_u64(),
        Value::String(value) => value.parse::<u64>().ok(),
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

fn ns_to_ms_label(value: Option<f64>) -> String {
    value
        .map(|ns| format!("{:.3}", ns / 1_000_000.0))
        .unwrap_or_else(|| "NA".to_string())
}

fn row_value(row: &BTreeMap<String, String>, column: &str) -> String {
    row.get(column)
        .filter(|value| !value.is_empty())
        .cloned()
        .unwrap_or_else(|| "NA".to_string())
}

fn numeric_row_field(row: &BTreeMap<String, String>, column: &str) -> Option<f64> {
    row.get(column)?.parse::<f64>().ok()
}

fn metric_label(value: Option<f64>) -> String {
    value
        .filter(|value| value.is_finite())
        .map(|value| format!("{value:.3}"))
        .unwrap_or_else(|| "NA".to_string())
}

fn scalar_label(value: &Value, key: &str) -> String {
    match value.get(key) {
        Some(Value::String(value)) => value.clone(),
        Some(Value::Bool(value)) => value.to_string(),
        Some(Value::Number(value)) => value.to_string(),
        Some(Value::Null) => "null".to_string(),
        Some(other) => other.to_string(),
        None => "not-recorded".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::{adoption_report, publication_issues, read_tsv_table};
    use crate::publication_gate::collect_publication_gate_issues;
    use serde_json::json;
    use std::path::Path;

    #[test]
    fn publication_issues_require_clean_cpu_gates_and_full_external_run() {
        let summary = json!({
            "mode": "quick",
            "include_generated": true,
            "cpu_fixture_compare": {
                "publication_eligible": "false",
                "publication_blockers": "generated-fixtures-included",
                "benchmark_complete": "true"
            },
            "cpu_encode_compare": {
                "publication_eligible": "true",
                "publication_blockers": "none",
                "benchmark_complete": "true"
            }
        });

        let issues = publication_issues(&summary);

        assert!(issues
            .iter()
            .any(|issue| issue.contains("cpu-fixture-compare")));
        assert!(issues
            .iter()
            .any(|issue| issue.contains("run mode is not full")));
        assert!(issues
            .iter()
            .any(|issue| issue.contains("generated fixtures included")));
    }

    #[test]
    fn publication_issues_use_shared_gate_for_writer_metadata() {
        let failed_metadata = json!({
            "publication_eligible": "false",
            "publication_blockers": "generated-fixtures-included",
            "benchmark_complete": "false"
        });
        let clean_metadata = json!({
            "publication_eligible": "true",
            "publication_blockers": "none",
            "benchmark_complete": "true"
        });
        let summary = json!({
            "mode": "full",
            "include_generated": false,
            "cpu_fixture_compare": failed_metadata,
            "cpu_encode_compare": clean_metadata
        });
        let mut expected = Vec::new();
        collect_publication_gate_issues(
            "cpu-fixture-compare",
            summary.get("cpu_fixture_compare"),
            &mut expected,
        );

        let issues = publication_issues(&summary);
        let cpu_fixture_issues = issues
            .into_iter()
            .filter(|issue| issue.contains("cpu-fixture-compare"))
            .collect::<Vec<_>>();

        assert_eq!(cpu_fixture_issues, expected);
    }

    #[test]
    fn publication_issues_require_requested_cuda_evidence() {
        let summary = json!({
            "mode": "full",
            "include_generated": false,
            "cuda_requested": true,
            "require_cuda": true,
            "cpu_fixture_compare": {
                "publication_eligible": "true",
                "publication_blockers": "none",
                "benchmark_complete": "true"
            },
            "cpu_encode_compare": {
                "publication_eligible": "true",
                "publication_blockers": "none",
                "benchmark_complete": "true"
            },
            "cuda_htj2k_decode": {"metadata_error": "missing"},
            "cuda_htj2k_encode": {},
            "steps": [
                {"name": "cuda-htj2k-decode", "status": "skipped"},
                {"name": "cuda-htj2k-encode", "status": "ran"}
            ],
            "criterion": {"steps": []}
        });

        let issues = publication_issues(&summary);

        assert!(issues
            .iter()
            .any(|issue| issue.contains("CUDA decode step status")));
        assert!(issues
            .iter()
            .any(|issue| issue.contains("CUDA decode metadata error")));
        assert!(issues
            .iter()
            .any(|issue| issue.contains("CUDA encode manifest not recorded")));
    }

    #[test]
    fn publication_issues_accept_complete_required_cuda_evidence() {
        let summary = json!({
            "mode": "full",
            "include_generated": false,
            "cuda_requested": true,
            "require_cuda": true,
            "cpu_fixture_compare": {
                "publication_eligible": "true",
                "publication_blockers": "none",
                "benchmark_complete": "true"
            },
            "cpu_encode_compare": {
                "publication_eligible": "true",
                "publication_blockers": "none",
                "benchmark_complete": "true"
            },
            "cuda_htj2k_decode": {
                "j2k_cuda_decode_io_policy": "host-memory-fixture-bytes-preloaded-no-filesystem-io-in-timed-loop;cuda-rows-return-device-resident-surfaces",
                "j2k_cuda_decode_input_dirs": "/fixtures",
                "j2k_cuda_decode_manifest": "/fixtures.tsv",
                "j2k_cuda_decode_generated_included": "false",
                "j2k_cuda_decode_external_case_count": "12"
            },
            "cuda_htj2k_encode": {
                "j2k_cuda_encode_io_policy": "staged-pnm-pixels-preloaded-no-filesystem-io-in-timed-loop;cuda-host-input-rows-include-public-api-host-submission-and-device-encode-work",
                "j2k_cuda_encode_input_dirs": "/pnm",
                "j2k_cuda_encode_manifest": "/encode.tsv",
                "j2k_cuda_encode_generated_host_input_included": "false",
                "j2k_cuda_encode_external_input_format": "staged-pnm-p5-p6",
                "j2k_cuda_encode_external_case_count": "24"
            },
            "steps": [
                {"name": "cuda-htj2k-decode", "status": "ran"},
                {"name": "cuda-htj2k-encode", "status": "ran"}
            ],
            "criterion": {
                "steps": [
                    {"step": "cuda-htj2k-decode", "count": 3},
                    {"step": "cuda-htj2k-encode", "count": 2}
                ]
            }
        });

        let issues = publication_issues(&summary);

        assert!(issues.is_empty(), "unexpected issues: {issues:?}");
    }

    #[test]
    fn publication_issues_require_requested_metal_evidence() {
        let summary = json!({
            "mode": "full",
            "include_generated": false,
            "metal_requested": true,
            "require_metal": true,
            "cpu_fixture_compare": {
                "publication_eligible": "true",
                "publication_blockers": "none",
                "benchmark_complete": "true"
            },
            "cpu_encode_compare": {
                "publication_eligible": "true",
                "publication_blockers": "none",
                "benchmark_complete": "true"
            },
            "metal_decode_benchmark": {
                "status": "ran",
                "bench_count": 1,
                "skipped_bench_count": 1,
                "verified_bench_count": 0,
                "metadata": {}
            },
            "metal_encode_auto_routing": {
                "status": "ran",
                "auto_bench_count": 1,
                "skipped_auto_bench_count": 1,
                "probe_error_count": 0,
                "resident_bench_count": 1,
                "skipped_resident_bench_count": 1,
                "resident_verified_bench_count": 0,
                "metadata": {}
            },
            "metal_transcode_benchmark": {
                "status": "ran",
                "profile_count": 1,
                "verified_profile_count": 0,
                "cpu_profile_count": 1,
                "auto_metal_profile_count": 0,
                "explicit_metal_profile_count": 0,
                "comparison_context_count": 0,
                "profiles": []
            },
            "steps": [
                {"name": "metal-encode-auto-routing", "status": "ran"}
            ]
        });

        let issues = publication_issues(&summary);

        assert!(issues.iter().any(|issue| issue
            .contains("Metal encode has no verified resident packetization/codestream rows")));
        assert!(issues
            .iter()
            .any(|issue| issue.contains("Metal encode has no measured resident benchmark rows")));
        assert!(issues
            .iter()
            .any(|issue| issue.contains("Metal encode manifest not recorded")));
        assert!(issues
            .iter()
            .any(|issue| issue.contains("Metal decode has no verified CPU/Metal benchmark rows")));
        assert!(issues
            .iter()
            .any(|issue| issue.contains("Metal decode manifest not recorded")));
        assert!(issues
            .iter()
            .any(|issue| issue.contains("Metal transcode benchmark step status")));
        assert!(issues.iter().any(|issue| {
            issue.contains("Metal transcode has no verified Metal-dispatch profile rows")
        }));
        assert!(issues.iter().any(|issue| {
            issue.contains("Metal transcode has no comparable CPU/Metal profile context")
        }));
    }

    #[test]
    fn publication_issues_accept_complete_required_metal_evidence() {
        let summary = json!({
            "mode": "full",
            "include_generated": false,
            "metal_requested": true,
            "require_metal": true,
            "cpu_fixture_compare": {
                "publication_eligible": "true",
                "publication_blockers": "none",
                "benchmark_complete": "true"
            },
            "cpu_encode_compare": {
                "publication_eligible": "true",
                "publication_blockers": "none",
                "benchmark_complete": "true"
            },
            "metal_decode_benchmark": {
                "status": "ran",
                "bench_count": 2,
                "skipped_bench_count": 0,
                "verified_bench_count": 2,
                "skipped_case_count": 1,
                "metadata": {
                    "j2k_metal_decode_io_policy": "generated-fixtures-and-preloaded-external-codestreams;timed-full-rows-include-decode-work;metal_resident_ms-does-not-readback;metal_readback_ms-includes-host-visible-byte-access",
                    "j2k_metal_decode_input_dirs": "/fixtures",
                    "j2k_metal_decode_manifest": "/fixtures.tsv",
                    "j2k_metal_decode_generated_included": "false",
                    "j2k_metal_decode_external_case_count": "1"
                },
                "benches": [
                    {
                        "case": "external_gray8",
                        "source": "external:/fixtures/external_gray8.j2k",
                        "codec": "j2k",
                        "container": "raw-codestream",
                        "operation": "full",
                        "fmt": "gray8",
                        "size": "512x512",
                        "cpu_ms": 3.0,
                        "metal_resident_ms": 2.0,
                        "metal_readback_ms": 2.5,
                        "output_bytes": 262144
                    },
                    {
                        "case": "external_gray8",
                        "source": "external:/fixtures/external_gray8.j2k",
                        "codec": "j2k",
                        "container": "raw-codestream",
                        "operation": "region_scaled",
                        "fmt": "gray8",
                        "size": "256x256",
                        "cpu_ms": 1.5,
                        "metal_resident_ms": 1.0,
                        "metal_readback_ms": 1.2,
                        "output_bytes": 65536
                    }
                ]
            },
            "metal_encode_auto_routing": {
                "status": "ran",
                "auto_bench_count": 2,
                "skipped_auto_bench_count": 1,
                "probe_error_count": 0,
                "resident_bench_count": 4,
                "skipped_resident_bench_count": 0,
                "resident_verified_bench_count": 4,
                "metadata": {
                    "j2k_metal_encode_io_policy": "staged-pnm-pixels-preloaded-no-filesystem-io-in-timed-loop;auto-rows-include-public-api-host-submission-and-metal-auto-route-work",
                    "j2k_metal_encode_input_dirs": "/pnm",
                    "j2k_metal_encode_manifest": "/encode.tsv",
                    "j2k_metal_encode_generated_host_input_included": "false",
                    "j2k_metal_encode_external_input_format": "staged-pnm-p5-p6",
                    "j2k_metal_encode_external_case_count": "24"
                }
            },
            "metal_transcode_benchmark": {
                "status": "ran",
                "bench_filter": "jpeg_to_htj2k_wsi_integer_53_tile_batch/srgb_ybr420_224_batch_128",
                "profile_count": 3,
                "verified_profile_count": 2,
                "cpu_profile_count": 1,
                "auto_metal_profile_count": 1,
                "explicit_metal_profile_count": 1,
                "comparison_context_count": 1,
                "profiles": []
            },
            "steps": [
                {"name": "metal-decode-benchmark", "status": "ran"},
                {"name": "metal-encode-auto-routing", "status": "ran"},
                {"name": "metal-transcode-benchmark", "status": "ran"}
            ]
        });

        let issues = publication_issues(&summary);

        assert!(issues.is_empty(), "unexpected issues: {issues:?}");
    }

    #[test]
    fn parses_benchmark_tsv_rows_after_metadata_preamble() {
        let dir = std::env::current_dir()
            .expect("current dir")
            .join("target")
            .join("adoption-report-test")
            .join(std::process::id().to_string());
        std::fs::create_dir_all(&dir).expect("create dir");
        let path = dir.join("cpu-fixture-compare.out");
        std::fs::write(
            &path,
            "publication_eligible\ttrue\n\
decoder\tcase\tbenchmark_mode\tdecode_method\tskip_reason\n\
j2k\tcase_a\tportable-native\tnative\t\n\
openjpeg\tcase_a\tportable-native\tskipped\topenjpeg-unavailable\n\
benchmark_complete\ttrue\n",
        )
        .expect("write table");

        let table = read_tsv_table(&path).expect("parse table");

        assert_eq!(table.headers[0], "decoder");
        assert_eq!(table.rows.len(), 2);
        assert_eq!(table.rows[1]["skip_reason"], "openjpeg-unavailable");
    }

    #[test]
    fn report_requires_completed_external_benchmark_bundle() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        let dir = std::env::current_dir()
            .expect("current dir")
            .join("target")
            .join("adoption-report-missing-test")
            .join(format!("{}-{unique}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create dir");

        let error =
            adoption_report(["--run-dir".to_string(), dir.display().to_string()].into_iter())
                .expect_err("missing adoption benchmark bundle should fail");

        assert!(error.contains("adoption benchmark summary is missing"));
        assert!(error.contains("Public benchmark/adoption claims remain blocked"));
    }

    #[test]
    fn report_refuses_nonpublishable_bundle_by_default() {
        let dir = std::env::current_dir()
            .expect("current dir")
            .join("target")
            .join("adoption-report-refuse-test")
            .join(std::process::id().to_string());
        std::fs::create_dir_all(&dir).expect("create dir");
        write_minimal_bundle(&dir, false);

        let error = adoption_report(
            [
                "--run-dir".to_string(),
                dir.display().to_string(),
                "--out".to_string(),
                dir.join("report.md").display().to_string(),
            ]
            .into_iter(),
        )
        .expect_err("nonpublishable report should fail");

        assert!(error.contains("not publishable"));
    }

    #[test]
    fn report_allows_nonpublishable_bundle_when_explicit() {
        let dir = std::env::current_dir()
            .expect("current dir")
            .join("target")
            .join("adoption-report-allow-test")
            .join(std::process::id().to_string());
        std::fs::create_dir_all(&dir).expect("create dir");
        write_minimal_bundle(&dir, false);
        let out = dir.join("report.md");

        adoption_report(
            [
                "--run-dir".to_string(),
                dir.display().to_string(),
                "--out".to_string(),
                out.display().to_string(),
                "--allow-nonpublishable".to_string(),
            ]
            .into_iter(),
        )
        .expect("diagnostic report should write");

        let report = std::fs::read_to_string(out).expect("read report");
        assert!(report.contains("Status: diagnostic only"));
        assert!(report.contains("CPU Decode Rows"));
        assert!(report.contains("CPU Decode Mixed Winner Summary"));
        assert!(report.contains(
            "Winner eligibility is limited to first-class comparable rows: `j2k`, `openjpeg`, and `grok`."
        ));
        assert!(report.contains("CPU Decode Mixed Batch Rows"));
        assert!(report.contains("external_mixed_decode"));
        assert!(report.contains(
            "| external_mixed_decode | 16 | 200.000 | 180.000 | 150.000 | j2k | 200.000 | 1.000x |"
        ));
        assert!(report.contains("CPU Encode Mixed Winner Summary"));
        assert!(report.contains("CPU Encode Mixed Batch Rows"));
        assert!(report.contains("external_mixed_encode"));
        assert!(report.contains("| external_mixed_encode | 16 | 200.000 | 140.000 | 220.000 | grok | 220.000 | 0.909x |"));
        assert!(report.contains("CUDA Criterion estimate rows"));
        assert!(report.contains("cuda_decode_external_gray8"));
        assert!(report.contains("1.500"));
        assert!(report.contains("Metal decode row summary"));
        assert!(report.contains("metal-readback"));
        assert!(report.contains("Metal auto external row summary"));
        assert!(report.contains("metal-auto"));
        assert!(report.contains("0.400x"));
        assert!(report.contains("Metal transcode profile summary"));
        assert!(report.contains("metal_auto"));
        assert!(report.contains("57.000"));
    }

    fn write_minimal_bundle(dir: &Path, publishable: bool) {
        let eligible = if publishable { "true" } else { "false" };
        let blockers = if publishable {
            "none"
        } else {
            "generated-fixtures-included"
        };
        let summary = json!({
            "mode": if publishable { "full" } else { "quick" },
            "include_generated": !publishable,
            "cpu_fixture_compare": {
                "publication_eligible": eligible,
                "publication_blockers": blockers,
                "benchmark_complete": "true"
            },
            "cpu_encode_compare": {
                "publication_eligible": eligible,
                "publication_blockers": blockers,
                "benchmark_complete": "true"
            },
            "cuda_htj2k_decode": {},
            "cuda_htj2k_encode": {},
            "criterion": {
                "steps": [
                    {
                        "step": "cuda-htj2k-decode",
                        "count": 1,
                        "estimates": [
                            {
                                "id": "cuda_decode_external_gray8",
                                "median_ns": 1500000.0,
                                "median_lower_ns": 1400000.0,
                                "median_upper_ns": 1600000.0
                            }
                        ]
                    },
                    {
                        "step": "cuda-htj2k-encode",
                        "count": 1,
                        "estimates": [
                            {
                                "id": "cuda_encode_external_rgb8",
                                "median_ns": 2500000.0,
                                "median_lower_ns": 2400000.0,
                                "median_upper_ns": 2600000.0
                            }
                        ]
                    }
                ]
            },
            "metal_decode_benchmark": {
                "status": "ran",
                "bench_count": 1,
                "skipped_bench_count": 0,
                "verified_bench_count": 1,
                "skipped_case_count": 0,
                "metadata": {
                    "j2k_metal_decode_io_policy": "generated-fixtures-and-preloaded-external-codestreams;timed-full-rows-include-decode-work;metal_resident_ms-does-not-readback;metal_readback_ms-includes-host-visible-byte-access",
                    "j2k_metal_decode_external_case_count": "0",
                    "j2k_metal_decode_generated_included": "true"
                },
                "benches": [
                    {
                        "case": "generated_gray8",
                        "source": "generated",
                        "codec": "j2k",
                        "container": "raw-codestream",
                        "operation": "full",
                        "fmt": "gray8",
                        "size": "512x512",
                        "cpu_ms": 1.0,
                        "metal_resident_ms": 0.5,
                        "metal_readback_ms": 0.75,
                        "output_bytes": 262144
                    }
                ]
            },
            "metal_encode_auto_routing": {
                "status": "ran",
                "auto_bench_count": 2,
                "skipped_auto_bench_count": 0,
                "probe_error_count": 0,
                "metadata": {
                    "j2k_metal_encode_io_policy": "staged-pnm-pixels-preloaded-no-filesystem-io-in-timed-loop;auto-rows-include-public-api-host-submission-and-metal-auto-route-work",
                    "j2k_metal_encode_external_case_count": "2",
                    "j2k_metal_encode_external_input_format": "staged-pnm-p5-p6"
                },
                "auto_benches": [
                    {
                        "mode": "lossless_external",
                        "codec": "htj2k",
                        "components": "gray8",
                        "size": "512x512",
                        "cpu_ms": 10.0,
                        "auto_ms": 4.0
                    },
                    {
                        "mode": "lossless_external",
                        "codec": "htj2k",
                        "components": "gray8",
                        "size": "512x512",
                        "cpu_ms": 15.0,
                        "auto_ms": 6.0
                    }
                ]
            },
            "metal_transcode_benchmark": {
                "status": "ran",
                "bench_filter": "jpeg_to_htj2k_wsi_integer_53_tile_batch/srgb_ybr420_224_batch_128",
                "profile_count": 2,
                "verified_profile_count": 1,
                "cpu_profile_count": 1,
                "auto_metal_profile_count": 1,
                "explicit_metal_profile_count": 0,
                "comparison_context_count": 1,
                "profiles": [
                    {
                        "request": "cpu",
                        "path": "cpu",
                        "pipeline": "jpeg_to_htj2k",
                        "context": "srgb_ybr420_224_batch_128",
                        "transform_processor": "cpu",
                        "total_us": 86000,
                        "successful_tiles": 128,
                        "dwt97_batch_resident_dct_handoff_count": 0,
                        "dwt97_batch_resident_dwt_handoff_count": 0,
                        "accelerator_dispatches": 0,
                        "host_to_device_transfer_bytes": 0,
                        "device_to_host_transfer_bytes": 0
                    },
                    {
                        "request": "metal_auto",
                        "path": "auto",
                        "pipeline": "jpeg_to_htj2k",
                        "context": "srgb_ybr420_224_batch_128",
                        "transform_processor": "metal",
                        "total_us": 57000,
                        "successful_tiles": 128,
                        "dwt97_batch_resident_dct_handoff_count": 384,
                        "dwt97_batch_resident_dwt_handoff_count": 1536,
                        "accelerator_dispatches": 1,
                        "host_to_device_transfer_bytes": 65536,
                        "device_to_host_transfer_bytes": 65536
                    }
                ]
            }
        });
        std::fs::write(
            dir.join("summary.json"),
            serde_json::to_string_pretty(&summary).expect("summary json"),
        )
        .expect("write summary");
        std::fs::write(
            dir.join("cpu-fixture-compare.out"),
            "decoder\tcase\tbenchmark_mode\tdecode_method\tinput_source\tcorpus_category\tcodec\tcontainer\toperation\tformat\tdimensions\tbatch_size\tinput_bytes\tmedian_us\ttiles_per_second_median\tdecoded_mib_per_second_median\tdecoded_bytes_per_repeat\tskip_reason\n\
j2k\tcase_a\tportable-native\tnative\texternal:case-a\tnatural-image\tj2k\tjp2\tfull\trgb8\t128x128\t1\t1024\t10.0\t100.0\t20.0\t2048\t\n\
j2k\texternal_mixed_decode\tportable-native\tnative-mixed-external-batch\texternal:mixed\tnatural-image\tmixed\tmixed\tfull\trgb8\tmixed\t16\t16384\t20.0\t800.0\t200.0\t32768\t\n\
openjpeg\texternal_mixed_decode\tportable-native\tnative-mixed-external-batch\texternal:mixed\tnatural-image\tmixed\tmixed\tfull\trgb8\tmixed\t16\t16384\t22.0\t700.0\t180.0\t32768\t\n\
grok\texternal_mixed_decode\tportable-native\tnative-mixed-external-batch\texternal:mixed\tnatural-image\tmixed\tmixed\tfull\trgb8\tmixed\t16\t16384\t26.0\t600.0\t150.0\t32768\t\n\
openjph\texternal_mixed_decode\tportable-native\topenjph-cli-process-output-pnm\texternal:mixed\tnatural-image\tmixed\tmixed\tfull\trgb8\tmixed\t16\t16384\t5.0\t1000.0\t400.0\t32768\t\n\
benchmark_complete\ttrue\n",
        )
        .expect("write fixture output");
        std::fs::write(
            dir.join("cpu-encode-compare.out"),
            "encoder\tcase\tbenchmark_mode\tencode_method\tinput_source\tcorpus_category\tformat\tdimensions\tbatch_size\tinput_bytes\tmedian_us\timages_per_second_median\tinput_mib_per_second_median\tencoded_bytes_per_repeat\tskip_reason\n\
j2k\tcase_a\tclassic-lossless-cli\tpnm-input-cli-process-output-jp2\texternal:case-a\tnatural-image\tpng\t128x128\t1\t1024\t10.0\t100.0\t20.0\t1234\t\n\
j2k\texternal_mixed_encode\tclassic-lossless-cli\tpnm-input-cli-process-output-jp2\texternal:mixed\tnatural-image\tmixed\tmixed\t16\t16384\t20.0\t800.0\t200.0\t12345\t\n\
openjpeg\texternal_mixed_encode\tclassic-lossless-cli\tpnm-input-cli-process-output-jp2\texternal:mixed\tnatural-image\tmixed\tmixed\t16\t16384\t28.0\t500.0\t140.0\t12345\t\n\
grok\texternal_mixed_encode\tclassic-lossless-cli\tpnm-input-cli-process-output-jp2\texternal:mixed\tnatural-image\tmixed\tmixed\t16\t16384\t18.0\t900.0\t220.0\t12345\t\n\
kakadu\texternal_mixed_encode\tclassic-lossless-cli\tkakadu-cli-process-output-jp2\texternal:mixed\tnatural-image\tmixed\tmixed\t16\t16384\t10.0\t1600.0\t390.0\t12345\t\n\
benchmark_complete\ttrue\n",
        )
        .expect("write encode output");
    }
}
