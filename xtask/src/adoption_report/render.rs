use std::{collections::BTreeMap, fmt::Write as _};

use serde_json::Value;

use crate::markdown::{escape_inline_code as escape_inline, markdown_header, markdown_row};

use super::{numeric_field, value_field, AdoptionReportModel, TsvTable};

mod metal;

use metal::{
    metal_auto_summary, metal_decode_summary, metal_resident_summary, metal_transcode_summary,
};

pub(super) fn render_report(model: &AdoptionReportModel) -> String {
    let mut out = String::new();
    render_overview(&mut out, model);
    render_cpu_decode(&mut out, &model.cpu_fixture_compare);
    render_cpu_encode(&mut out, &model.cpu_encode_compare);
    out.push_str("\n## Skipped And Context Rows\n\n");
    skipped_summary(&mut out, "decode", &model.cpu_fixture_compare);
    skipped_summary(&mut out, "encode", &model.cpu_encode_compare);
    render_hybrid_summary(&mut out, &model.summary);
    out
}

fn render_overview(out: &mut String, model: &AdoptionReportModel) {
    let AdoptionReportModel {
        run_dir,
        summary,
        publication_issues: issues,
        ..
    } = model;
    out.push_str("# J2K Adoption Benchmark Report\n\n");
    if issues.is_empty() {
        out.push_str("Status: publishable for the requested benchmark scopes according to recorded comparator and hardware gates.\n\n");
    } else {
        out.push_str("Status: diagnostic only. Do not use for marketing claims.\n\n");
        out.push_str("Blocking issues:\n");
        for issue in issues {
            append_format(out, format_args!("- {issue}\n"));
        }
        out.push('\n');
    }

    out.push_str("## Bundle\n\n");
    metadata_list(
        out,
        &[
            ("run_dir", run_dir.display().to_string()),
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
        out,
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
    methodology_metadata(out, summary);
}

fn render_cpu_encode(out: &mut String, encode: &TsvTable) {
    out.push_str("\n## CPU Encode Rows\n\n");
    measured_table(
        out,
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
        out,
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
        out,
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
}

fn render_cpu_decode(out: &mut String, fixture: &TsvTable) {
    out.push_str("\n## CPU Decode Rows\n\n");
    measured_table(
        out,
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
        out,
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
        out,
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
}

#[expect(
    clippy::too_many_lines,
    reason = "the hybrid report section preserves one stable ordered Markdown schema"
)]
fn render_hybrid_summary(out: &mut String, summary: &Value) {
    out.push_str("\n## Hybrid Summary\n\n");
    metadata_table(
        out,
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
    criterion_estimate_table(out, summary, &["cuda-htj2k-decode", "cuda-htj2k-encode"]);
    if let Some(metal) = summary.get("metal_decode_benchmark") {
        out.push_str("\nMetal decode benchmark summary:\n\n");
        append_format(
            out,
            format_args!(
                "- status: {}\n",
                metal
                    .get("status")
                    .and_then(Value::as_str)
                    .unwrap_or("not-recorded")
            ),
        );
        if let Some(metadata) = metal.get("metadata") {
            metadata_list(
                out,
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
        metal_decode_summary(out, metal);
    }
    if let Some(metal) = summary.get("metal_encode_auto_routing") {
        out.push_str("\nMetal auto-routing summary:\n\n");
        append_format(
            out,
            format_args!(
                "- status: {}\n",
                metal
                    .get("status")
                    .and_then(Value::as_str)
                    .unwrap_or("not-recorded")
            ),
        );
        if let Some(metadata) = metal.get("metadata") {
            metadata_list(
                out,
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
        metal_auto_summary(out, metal);
        metal_resident_summary(out, metal);
    }
    if let Some(metal) = summary.get("metal_transcode_benchmark") {
        out.push_str("\nMetal transcode benchmark summary:\n\n");
        append_format(
            out,
            format_args!(
                "- status: {}\n",
                metal
                    .get("status")
                    .and_then(Value::as_str)
                    .unwrap_or("not-recorded")
            ),
        );
        metadata_list(
            out,
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
        metal_transcode_summary(out, metal);
    }
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
        append_format(
            out,
            format_args!(
                "Missing expected raw columns: `{}`.\n\n",
                missing_columns.join("`, `")
            ),
        );
    }
    let measured_rows = table
        .rows
        .iter()
        .filter(|row| {
            row.get("skip_reason")
                .is_none_or(std::string::String::is_empty)
        })
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
        append_format(
            out,
            format_args!(
                "\nShowing first {limit} measured rows. See raw TSV outputs for the full table.\n"
            ),
        );
    }
}

fn skipped_summary(out: &mut String, label: &str, table: &TsvTable) {
    let mut counts = BTreeMap::<String, usize>::new();
    for row in &table.rows {
        let reason = row.get("skip_reason").map_or("", String::as_str);
        if !reason.is_empty() {
            *counts.entry(reason.to_string()).or_default() += 1;
        }
    }
    if counts.is_empty() {
        append_format(out, format_args!("- {label}: none\n"));
    } else {
        for (reason, count) in counts {
            append_format(out, format_args!("- {label}: {reason} ({count} rows)\n"));
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
            .map_or(("NA", f64::NAN), |(name, value)| (name.as_str(), *value));
        let j2k_vs_winner = participants
            .get("j2k")
            .filter(|_| winner_value.is_finite() && winner_value > 0.0)
            .map_or_else(
                || "NA".to_string(),
                |value| format!("{:.3}x", value / winner_value),
            );
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
        append_format(out, format_args!("- `{key}`: `{}`\n", escape_inline(value)));
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

fn ns_to_ms_label(value: Option<f64>) -> String {
    value.map_or_else(|| "NA".to_string(), |ns| format!("{:.3}", ns / 1_000_000.0))
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
        .map_or_else(|| "NA".to_string(), |value| format!("{value:.3}"))
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

fn append_format(out: &mut String, arguments: std::fmt::Arguments<'_>) {
    out.write_fmt(arguments)
        .expect("writing formatted text to a String cannot fail");
}
