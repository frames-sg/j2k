// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use super::{write_readme, write_summary, AdoptionBenchmarkOptions, AdoptionStep, StepStatus};

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

fn temp_dir(label: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "j2k-adoption-artifacts-{label}-{}-{}",
        std::process::id(),
        NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&root).expect("create adoption artifact test directory");
    root
}

fn options(root: &Path) -> AdoptionBenchmarkOptions {
    AdoptionBenchmarkOptions::parse(
        [
            "--out-dir".to_string(),
            root.display().to_string(),
            "--include-generated".to_string(),
            "--quick".to_string(),
            "--cuda".to_string(),
            "--metal".to_string(),
            "--openjph".to_string(),
            "--kakadu".to_string(),
        ]
        .into_iter(),
    )
    .expect("artifact options")
}

fn write_metadata_inputs(root: &Path) {
    fs::write(
        root.join("cpu-fixture-compare.out"),
        concat!(
            "benchmark_mode\tportable-native\n",
            "publication_eligible\ttrue\n",
            "publication_blockers\tnone\n",
            "benchmark_complete\ttrue\n",
            "case_batch_sizes\t1\n",
            "mixed_batch_sizes\t16\n",
            "external_unique_input_count\t24\n",
            "external_component_group_count\t2\n",
            "external_dimension_count\t3\n",
            "external_source_format_count\t2\n",
            "mixed_external_batch_group_count\t2\n",
            "mixed_external_min_distinct_inputs\t2\n",
            "mixed_external_max_distinct_inputs\t4\n",
            "mixed_external_group_distinct_inputs\tgray:2,rgb:4\n",
            "openjph_included\ttrue\n",
            "openjph_available\ttrue\n",
            "kakadu_included\ttrue\n",
            "kakadu_available\ttrue\n",
        ),
    )
    .expect("write CPU fixture metadata");
    fs::write(
        root.join("cpu-encode-compare.out"),
        concat!(
            "benchmark_mode\tportable-native\n",
            "publication_eligible\ttrue\n",
            "publication_blockers\tnone\n",
            "benchmark_complete\ttrue\n",
            "case_batch_sizes\t1\n",
            "mixed_batch_sizes\t16\n",
            "external_unique_input_count\t24\n",
            "mixed_external_batch_group_count\t2\n",
            "mixed_external_min_distinct_inputs\t2\n",
            "mixed_external_max_distinct_inputs\t4\n",
            "mixed_external_group_distinct_inputs\tgray:2,rgb:4\n",
            "kakadu_included\ttrue\n",
            "kakadu_compress_available\ttrue\n",
        ),
    )
    .expect("write CPU encode metadata");
    fs::write(
        root.join("cuda-htj2k-decode.out"),
        "j2k_cuda_decode_generated_included\ttrue\nj2k_cuda_decode_batch_sizes\t1,16\n",
    )
    .expect("write CUDA decode metadata");
    fs::write(
        root.join("cuda-htj2k-encode.out"),
        "j2k_cuda_encode_generated_host_input_included\ttrue\nj2k_cuda_encode_io_policy\tstaged-pnm\n",
    )
    .expect("write CUDA encode metadata");
    for name in [
        "metal-decode-benchmark.out",
        "metal-encode-auto-routing.out",
        "metal-transcode-benchmark.out",
        "metal-transcode-benchmark.err",
    ] {
        fs::write(root.join(name), []).expect("write empty Metal artifact");
    }
}

fn steps(root: &Path) -> Vec<AdoptionStep> {
    [
        ("cpu-fixture-compare", StepStatus::Ran),
        (
            "cuda-htj2k-decode",
            StepStatus::Skipped {
                reason: "hardware unavailable".to_string(),
            },
        ),
        ("metal-decode-benchmark", StepStatus::Ran),
        ("metal-encode-auto-routing", StepStatus::Ran),
        ("metal-transcode-benchmark", StepStatus::Ran),
    ]
    .into_iter()
    .map(|(name, status)| AdoptionStep {
        name,
        command: format!("cargo test -- {name}"),
        stdout: root.join(format!("{name}.out")),
        stderr: root.join(format!("{name}.err")),
        criterion_root: None,
        status,
    })
    .collect()
}

#[test]
fn artifact_writers_preserve_metadata_step_status_and_hardware_labels() {
    let root = temp_dir("complete");
    write_metadata_inputs(&root);
    let options = options(&root);
    let steps = steps(&root);

    write_summary(&options, &steps).expect("write summary artifact");
    write_readme(&options, &steps).expect("write README artifact");

    let summary: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(root.join("summary.json")).expect("read summary artifact"),
    )
    .expect("parse summary artifact");
    assert_eq!(summary["version"], 1);
    assert_eq!(summary["mode"], "quick");
    assert_eq!(summary["cuda_requested"], true);
    assert_eq!(summary["metal_requested"], true);
    assert_eq!(
        summary["cpu_fixture_compare"]["publication_eligible"],
        "true"
    );
    assert_eq!(
        summary["cpu_encode_compare"]["publication_blockers"],
        "none"
    );
    assert_eq!(summary["steps"][0]["status"], "ran");
    assert_eq!(summary["steps"][1]["status"], "skipped");
    assert_eq!(summary["steps"][1]["reason"], "hardware unavailable");
    assert!(summary["scrubbed_env_vars"].is_array());

    let readme = fs::read_to_string(root.join("README.md")).expect("read README artifact");
    assert!(readme.contains("Current CPU fixture status"));
    assert!(readme.contains("Current CPU encode status"));
    assert!(readme.contains("skipped: hardware unavailable"));
    assert!(readme.contains("Metal decode benchmark rows are summarized"));
    assert!(readme.contains("Metal encode auto-routing rows are summarized"));
    assert!(readme.contains("Metal transcode benchmark rows are summarized"));
}

#[test]
fn summary_records_missing_optional_inputs_as_metadata_errors() {
    let root = temp_dir("missing");
    let options = options(&root);

    write_summary(&options, &[]).expect("missing optional inputs remain reportable");

    let summary: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(root.join("summary.json")).expect("read summary artifact"),
    )
    .expect("parse summary artifact");
    assert!(summary["cpu_fixture_compare"]["metadata_error"].is_string());
    assert!(summary["cpu_encode_compare"]["metadata_error"].is_string());
    assert!(summary["cuda_htj2k_decode"]["metadata_error"].is_string());
    assert!(summary["cuda_htj2k_encode"]["metadata_error"].is_string());
    assert_eq!(summary["steps"], serde_json::json!([]));
}
