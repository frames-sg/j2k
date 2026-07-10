use std::{fs, path::PathBuf};

use super::options::AdoptionBenchmarkOptions;
use super::parsing::{
    criterion_summary_json, read_metal_decode_summary, read_metal_encode_summary,
    read_metal_transcode_summary, read_tsv_metadata,
};
use super::runner::SCRUBBED_BENCH_ENV_VARS;
use super::support::unix_seconds;

#[derive(Debug)]
pub(super) struct AdoptionStep {
    pub(super) name: &'static str,
    pub(super) command: String,
    pub(super) stdout: PathBuf,
    pub(super) stderr: PathBuf,
    pub(super) criterion_root: Option<PathBuf>,
    pub(super) status: StepStatus,
}

#[derive(Debug)]
pub(super) enum StepStatus {
    Ran,
    Skipped { reason: String },
}

#[expect(
    clippy::too_many_lines,
    reason = "the benchmark summary writer keeps its machine-readable schema assembled in one place"
)]
pub(super) fn write_summary(
    options: &AdoptionBenchmarkOptions,
    steps: &[AdoptionStep],
) -> Result<(), String> {
    let cpu_fixture_metadata = read_tsv_metadata(
        &options.out_dir.join("cpu-fixture-compare.out"),
        &[
            "benchmark_mode",
            "publication_eligible",
            "publication_blockers",
            "benchmark_complete",
            "case_batch_sizes",
            "mixed_batch_sizes",
            "selected_cases",
            "external_case_count",
            "external_native_case_count",
            "external_materialized_case_count",
            "external_unique_input_count",
            "external_native_unique_input_count",
            "mixed_external_batch_group_count",
            "mixed_external_min_distinct_inputs",
            "mixed_external_max_distinct_inputs",
            "mixed_external_group_distinct_inputs",
            "generated_case_count",
            "mode_excluded_case_count",
            "skipped_comparators",
            "publication_gate_skipped_comparators",
            "build_profile",
            "debug_assertions",
            "git_revision",
            "git_dirty",
            "host_hardware",
            "openjpeg_version",
            "grok_version",
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
    .unwrap_or_else(|error| {
        serde_json::json!({
            "metadata_error": error,
        })
    });
    let cpu_encode_metadata = read_tsv_metadata(
        &options.out_dir.join("cpu-encode-compare.out"),
        &[
            "benchmark_mode",
            "publication_eligible",
            "publication_blockers",
            "benchmark_complete",
            "case_batch_sizes",
            "mixed_batch_sizes",
            "selected_encoders",
            "selected_cases",
            "external_case_count",
            "external_unique_input_count",
            "external_manifest_covered_case_count",
            "external_manifest_missing_case_count",
            "external_component_group_count",
            "external_dimension_count",
            "external_source_format_count",
            "mixed_external_batch_group_count",
            "mixed_external_min_distinct_inputs",
            "mixed_external_max_distinct_inputs",
            "mixed_external_group_distinct_inputs",
            "generated_case_count",
            "encode_manifest",
            "openjpeg_compress_available",
            "grok_compress_available",
            "build_profile",
            "debug_assertions",
            "git_revision",
            "git_dirty",
            "host_hardware",
            "openjpeg_version",
            "openjpeg_linked_library_version",
            "grok_version",
            "grok_linked_library_version",
            "kakadu_included",
            "kakadu_compress_available",
            "kakadu_compress_command",
            "kakadu_version",
        ],
    )
    .unwrap_or_else(|error| {
        serde_json::json!({
            "metadata_error": error,
        })
    });
    let criterion_estimates = criterion_summary_json(steps);
    let cuda_decode_metadata = read_tsv_metadata(
        &options.out_dir.join("cuda-htj2k-decode.out"),
        &[
            "j2k_cuda_decode_generated_included",
            "j2k_cuda_decode_batch_sizes",
            "j2k_cuda_decode_io_policy",
            "j2k_cuda_decode_input_dirs",
            "j2k_cuda_decode_manifest",
            "j2k_cuda_decode_case_count",
            "j2k_cuda_decode_external_case_count",
            "j2k_cuda_decode_external_fixture_count",
            "j2k_cuda_decode_external_skipped_non_htj2k_count",
            "j2k_cuda_decode_external_skipped_unsupported_shape_count",
            "j2k_cuda_decode_external_skipped_format_disabled_count",
        ],
    )
    .unwrap_or_else(|error| {
        serde_json::json!({
            "metadata_error": error,
        })
    });
    let metal_decode_summary =
        read_metal_decode_summary(&options.out_dir.join("metal-decode-benchmark.out"), steps);
    let metal_encode_summary = read_metal_encode_summary(
        &options.out_dir.join("metal-encode-auto-routing.out"),
        steps,
    );
    let metal_transcode_summary = read_metal_transcode_summary(
        &options.out_dir.join("metal-transcode-benchmark.out"),
        &options.out_dir.join("metal-transcode-benchmark.err"),
        steps,
    );
    let cuda_encode_metadata = read_tsv_metadata(
        &options.out_dir.join("cuda-htj2k-encode.out"),
        &[
            "j2k_cuda_encode_generated_host_input_included",
            "j2k_cuda_encode_io_policy",
            "j2k_cuda_encode_input_dirs",
            "j2k_cuda_encode_manifest",
            "j2k_cuda_encode_external_case_count",
            "j2k_cuda_encode_external_input_format",
            "j2k_cuda_encode_external_case_sources",
        ],
    )
    .unwrap_or_else(|error| {
        serde_json::json!({
            "metadata_error": error,
        })
    });
    let value = serde_json::json!({
        "version": 1,
        "created_unix_seconds": unix_seconds(),
        "mode": if options.quick { "quick" } else { "full" },
        "input_dirs": options.input_dirs,
        "manifest": options.manifest.as_ref().map(|path| path.display().to_string()),
        "encode_input_dirs": options.encode_input_dirs,
        "encode_manifest": options.encode_manifest.as_ref().map(|path| path.display().to_string()),
        "cuda_decode_batch_sizes": options.cuda_decode_batch_sizes,
        "include_generated": options.include_generated,
        "cuda_requested": options.cuda,
        "metal_requested": options.metal,
        "openjph_requested": options.openjph,
        "kakadu_requested": options.kakadu,
        "require_cuda": options.require_cuda,
        "require_metal": options.require_metal,
        "require_openjph": options.require_openjph,
        "require_kakadu": options.require_kakadu,
        "cpu_fixture_compare": cpu_fixture_metadata,
        "cpu_encode_compare": cpu_encode_metadata,
        "cuda_htj2k_decode": cuda_decode_metadata,
        "cuda_htj2k_encode": cuda_encode_metadata,
        "criterion": criterion_estimates,
        "metal_decode_benchmark": metal_decode_summary,
        "metal_encode_auto_routing": metal_encode_summary,
        "metal_transcode_benchmark": metal_transcode_summary,
        "steps": steps.iter().map(step_json).collect::<Vec<_>>(),
        "scrubbed_env_vars": SCRUBBED_BENCH_ENV_VARS,
        "fixture_comparability_scope": "cpu-fixture-compare uses external encoded fixtures and requires independently sourced native compressed J2K and HTJ2K fixtures for publishable decode claims; repo-materialized natural-image codestreams are useful workload diagnostics but do not satisfy native compressed codec coverage by themselves. Optional OpenJPH rows are opt-in HTJ2K/JPH-compatible CLI rows labeled by decode_method and are not part of the default J2K/OpenJPEG/Grok in-process matrix; optional Kakadu rows are opt-in CLI/file-output context rows labeled by decode_method or encode_method and are not part of the default J2K/OpenJPEG/Grok in-process matrix. cuda-htj2k-decode consumes the same external fixture dirs and manifest when --fixtures/--manifest are provided but measures only the supported HTJ2K subset and reports skipped fixture counts. metal-decode-benchmark consumes the same external fixture dirs and manifest when --fixtures/--manifest are provided, but currently publishes only raw-codestream Metal buffer rows; JP2/JPH wrapper rows are skipped until wrapper-specific strict Metal parity is claimed. cpu-encode-compare is classic lossless J2K-in-JP2 CLI throughput: source images are staged to identical PNM files before the run, but timed rows launch the CLI and include PNM read, JP2 write, and output-stat work; it is not filesystem-free codec timing and not an HTJ2K encode benchmark. cuda-htj2k-encode and metal-encode-auto-routing consume staged PGM/PPM source images from --encode-fixtures/--encode-manifest when supplied and label external host-input rows separately from generated component rows. Metal resident encode rows are HTJ2K lossless host-output comparisons only when packetization_used=true and codestream_assembly_used=true; resident_buffer_ms is GPU-pipeline context and not a direct CPU codec comparison. Metal transcode rows currently use generated same-geometry JPEG tile batches and are batch-route evidence only; they do not satisfy external corpus transcode adoption claims. CPU public API rows remain component microbenchmarks",
        "publication_note": "CPU fixture compare and CPU encode compare must both report publication_eligible=true, publication_blockers=none, and benchmark_complete=true before use in adoption claims; CPU decode publishability also requires independent native compressed classic J2K and HTJ2K coverage, not only codestreams generated by this repo. CPU encode rows are classic lossless JP2 only. CUDA decode hardware rows must be run with --require-cuda and the same pinned fixture manifest for supported-HTJ2K-subset claims; Metal decode hardware rows must be run with --require-metal and the same pinned fixture manifest before they are used for Metal decode speed claims. CUDA encode hardware rows must be run with --require-cuda and J2K_CUDA_ENCODE_MANIFEST-backed staged PNM sources before they are described as using the same encode source matrix; Metal encode hardware rows must be run with --require-metal and J2K_METAL_ENCODE_MANIFEST-backed staged PNM sources before they are described as using the same encode source matrix. Metal transcode rows must be run with --require-metal before same-geometry batch Metal speed claims; generated transcode rows must stay labeled as generated batch-route evidence. Use metal_readback_ms vs cpu_ms for host-observable Metal decode claims; use metal_resident_ms only for resident/no-readback context. Use resident_host_ms vs cpu_ms for resident Metal encode claims only on rows where packetization_used=true and codestream_assembly_used=true; keep j2k_metal_encode_auto_bench and resident_buffer_ms labeled separately."
    });
    let data = serde_json::to_string_pretty(&value)
        .map_err(|err| format!("failed to serialize adoption benchmark summary: {err}"))?;
    let path = options.out_dir.join("summary.json");
    fs::write(&path, format!("{data}\n"))
        .map_err(|err| format!("failed to write {}: {err}", path.display()))
}

pub(super) fn step_json(step: &AdoptionStep) -> serde_json::Value {
    match &step.status {
        StepStatus::Ran => serde_json::json!({
            "name": step.name,
            "status": "ran",
            "command": step.command,
            "stdout": step.stdout.display().to_string(),
            "stderr": step.stderr.display().to_string(),
            "criterion_root": step.criterion_root.as_ref().map(|path| path.display().to_string()),
        }),
        StepStatus::Skipped { reason } => serde_json::json!({
            "name": step.name,
            "status": "skipped",
            "reason": reason,
            "command": step.command,
            "stdout": step.stdout.display().to_string(),
            "stderr": step.stderr.display().to_string(),
            "criterion_root": step.criterion_root.as_ref().map(|path| path.display().to_string()),
        }),
    }
}
