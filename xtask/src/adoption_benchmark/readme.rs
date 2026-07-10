use std::{fmt::Write as _, fs};

use crate::markdown::{escape_inline_code, markdown_header, markdown_row};
use crate::perf_guard::discover_estimates;

use super::options::AdoptionBenchmarkOptions;
use super::parsing::read_tsv_metadata;
use super::summary::{AdoptionStep, StepStatus};

#[expect(
    clippy::too_many_lines,
    reason = "the generated artifact README is a single ordered publication-evidence document"
)]
pub(super) fn write_readme(
    options: &AdoptionBenchmarkOptions,
    steps: &[AdoptionStep],
) -> Result<(), String> {
    let cpu_metadata = read_tsv_metadata(
        &options.out_dir.join("cpu-fixture-compare.out"),
        &[
            "publication_eligible",
            "publication_blockers",
            "benchmark_complete",
            "case_batch_sizes",
            "mixed_batch_sizes",
            "external_unique_input_count",
            "external_component_group_count",
            "external_dimension_count",
            "external_source_format_count",
            "mixed_external_batch_group_count",
            "mixed_external_min_distinct_inputs",
            "mixed_external_max_distinct_inputs",
            "mixed_external_group_distinct_inputs",
            "publication_gate_skipped_comparators",
            "openjph_included",
            "openjph_available",
            "openjph_expand_command",
            "openjph_version",
            "kakadu_included",
            "kakadu_available",
            "kakadu_expand_command",
            "kakadu_version",
        ],
    )
    .ok();
    let encode_metadata = read_tsv_metadata(
        &options.out_dir.join("cpu-encode-compare.out"),
        &[
            "publication_eligible",
            "publication_blockers",
            "benchmark_complete",
            "case_batch_sizes",
            "mixed_batch_sizes",
            "external_unique_input_count",
            "mixed_external_batch_group_count",
            "mixed_external_min_distinct_inputs",
            "mixed_external_max_distinct_inputs",
            "mixed_external_group_distinct_inputs",
            "kakadu_included",
            "kakadu_compress_available",
            "kakadu_compress_command",
            "kakadu_version",
        ],
    )
    .ok();
    let mut out = String::new();
    out.push_str("# J2K Adoption Benchmark Run\n\n");
    out.push_str("This directory is a benchmark artifact bundle. Treat `summary.json` as the machine-readable index.\n\n");
    out.push_str("## Inputs\n\n");
    append_format(
        &mut out,
        format_args!(
            "- Fixture dirs: `{}`\n",
            options.input_dirs.as_deref().unwrap_or("not set")
        ),
    );
    append_format(
        &mut out,
        format_args!(
            "- Fixture manifest: `{}`\n",
            options
                .manifest
                .as_ref()
                .map_or_else(|| "not set".to_string(), |path| path.display().to_string())
        ),
    );
    append_format(
        &mut out,
        format_args!(
            "- Encode source dirs: `{}`\n",
            options.encode_input_dirs.as_deref().unwrap_or("not set")
        ),
    );
    append_format(
        &mut out,
        format_args!(
            "- Encode manifest: `{}`\n",
            options
                .encode_manifest
                .as_ref()
                .map_or_else(|| "not set".to_string(), |path| path.display().to_string())
        ),
    );
    append_format(
        &mut out,
        format_args!(
            "- Generated fixtures included: `{}`\n",
            options.include_generated
        ),
    );
    append_format(
        &mut out,
        format_args!("- OpenJPH comparator requested: `{}`\n", options.openjph),
    );
    append_format(
        &mut out,
        format_args!("- Kakadu comparator requested: `{}`\n", options.kakadu),
    );
    append_format(
        &mut out,
        format_args!("- Quick mode: `{}`\n\n", options.quick),
    );
    out.push_str("## Steps\n\n");
    markdown_header(&mut out, &["Step", "Status", "Output", "Error log"]);
    for step in steps {
        let status = match &step.status {
            StepStatus::Ran => "ran".to_string(),
            StepStatus::Skipped { reason } => format!("skipped: {reason}"),
        };
        let name = format!("`{}`", escape_inline_code(step.name));
        let stdout = format!(
            "`{}`",
            escape_inline_code(&step.stdout.display().to_string())
        );
        let stderr = format!(
            "`{}`",
            escape_inline_code(&step.stderr.display().to_string())
        );
        markdown_row(&mut out, [name, status, stdout, stderr]);
    }
    out.push_str("\n## Publication Gate\n\n");
    out.push_str("Do not publish this bundle unless `cpu-fixture-compare.out` and `cpu-encode-compare.out` both contain `publication_eligible\ttrue`, `publication_blockers\tnone`, `benchmark_complete\ttrue`, and mixed external batch rows. CPU decode publication requires independent native compressed classic J2K and HTJ2K coverage; repo-materialized natural-image codestreams are diagnostic workload rows, not enough by themselves. CPU encode rows compare the same staged PNM bytes for classic lossless JP2 only. Optional OpenJPH rows are CLI/file-output HTJ2K/JPH-compatible context rows and must be labeled separately from the default in-process decoder matrix. Optional Kakadu rows are proprietary CLI/file-output context rows and must be labeled separately from the default matrix. CUDA decode hardware rows must be run with `--require-cuda` and the same pinned fixture manifest for supported-HTJ2K-subset claims. Metal decode hardware rows must be run with `--require-metal` and the same pinned fixture manifest before they are used for Metal decode speed claims. CUDA and Metal encode hardware rows must be run with `--require-cuda` or `--require-metal` and manifest-backed staged PGM/PPM sources before they are described as using the same encode source matrix. Metal transcode rows must be run with `--require-metal` for same-geometry batch Metal speed claims and must remain labeled as generated batch-route evidence until external corpus transcode rows exist. For Metal decode claims, compare `metal_readback_ms` with `cpu_ms` for host-observable speed and keep `metal_resident_ms` labeled as no-readback context. For Metal resident encode claims, compare `resident_host_ms` with `cpu_ms` only on rows where `packetization_used=true` and `codestream_assembly_used=true`; `resident_buffer_ms` is GPU-pipeline context, not a host-codec apples-to-apples number.\n");
    if let Some(metadata) = cpu_metadata {
        out.push_str("\nCurrent CPU fixture status:\n\n");
        for key in [
            "publication_eligible",
            "publication_blockers",
            "benchmark_complete",
            "case_batch_sizes",
            "mixed_batch_sizes",
            "external_unique_input_count",
            "external_component_group_count",
            "external_dimension_count",
            "external_source_format_count",
            "mixed_external_batch_group_count",
            "mixed_external_min_distinct_inputs",
            "mixed_external_max_distinct_inputs",
            "mixed_external_group_distinct_inputs",
            "publication_gate_skipped_comparators",
            "openjph_included",
            "openjph_available",
            "openjph_expand_command",
            "openjph_version",
            "kakadu_included",
            "kakadu_available",
            "kakadu_expand_command",
            "kakadu_version",
        ] {
            if let Some(value) = metadata.get(key).and_then(serde_json::Value::as_str) {
                append_format(&mut out, format_args!("- `{key}`: `{value}`\n"));
            }
        }
    }
    if let Some(metadata) = encode_metadata {
        out.push_str("\nCurrent CPU encode status:\n\n");
        for key in [
            "publication_eligible",
            "publication_blockers",
            "benchmark_complete",
            "case_batch_sizes",
            "mixed_batch_sizes",
            "external_unique_input_count",
            "mixed_external_batch_group_count",
            "mixed_external_min_distinct_inputs",
            "mixed_external_max_distinct_inputs",
            "mixed_external_group_distinct_inputs",
            "kakadu_included",
            "kakadu_compress_available",
            "kakadu_compress_command",
            "kakadu_version",
        ] {
            if let Some(value) = metadata.get(key).and_then(serde_json::Value::as_str) {
                append_format(&mut out, format_args!("- `{key}`: `{value}`\n"));
            }
        }
    }
    let mut criterion_rows = 0_usize;
    for step in steps {
        let Some(root) = &step.criterion_root else {
            continue;
        };
        if !matches!(&step.status, StepStatus::Ran) || !root.exists() {
            continue;
        }
        match discover_estimates(root) {
            Ok(estimates) => {
                criterion_rows += estimates.len();
            }
            Err(error) => {
                out.push_str("\nCriterion estimate parsing failed for `");
                out.push_str(step.name);
                out.push_str("`: `");
                out.push_str(&error);
                out.push_str("`.\n");
            }
        }
    }
    if criterion_rows > 0 {
        append_format(&mut out, format_args!(
            "\nCriterion estimates are summarized in `summary.json` ({criterion_rows} rows across current-run steps).\n"
        ));
    }
    if steps.iter().any(|step| {
        step.name == "metal-decode-benchmark" && matches!(&step.status, StepStatus::Ran)
    }) {
        out.push_str("\nMetal decode benchmark rows are summarized in `summary.json` from `metal-decode-benchmark.out`.\n");
    }
    if steps.iter().any(|step| {
        step.name == "metal-encode-auto-routing" && matches!(&step.status, StepStatus::Ran)
    }) {
        out.push_str("\nMetal encode auto-routing rows are summarized in `summary.json` from `metal-encode-auto-routing.out`.\n");
    }
    if steps.iter().any(|step| {
        step.name == "metal-transcode-benchmark" && matches!(&step.status, StepStatus::Ran)
    }) {
        out.push_str("\nMetal transcode benchmark rows are summarized in `summary.json` from `metal-transcode-benchmark.out` and `metal-transcode-benchmark.err`.\n");
    }

    let path = options.out_dir.join("README.md");
    fs::write(&path, out).map_err(|err| format!("failed to write {}: {err}", path.display()))
}

fn append_format(out: &mut String, arguments: std::fmt::Arguments<'_>) {
    out.write_fmt(arguments)
        .expect("writing formatted text to a String cannot fail");
}
