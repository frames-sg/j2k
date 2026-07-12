// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use super::{
    criterion_estimate_json, criterion_summary_json, parse_bool_field, parse_metal_auto_probe_line,
    parse_metal_stage_bench_line, parse_metal_transcode_profile_line, read_metal_decode_summary,
    read_metal_encode_summary, read_metal_transcode_summary, read_tsv_metadata,
};
use crate::{
    adoption_benchmark::summary::{AdoptionStep, StepStatus},
    perf_guard::BenchEstimate,
};

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "j2k-adoption-parsing-{name}-{}-{}",
            std::process::id(),
            NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).expect("create test directory");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove test directory");
    }
}

fn step(
    name: &'static str,
    stdout: PathBuf,
    stderr: PathBuf,
    criterion_root: Option<PathBuf>,
    status: StepStatus,
) -> AdoptionStep {
    AdoptionStep {
        name,
        command: "test command".to_string(),
        stdout,
        stderr,
        criterion_root,
        status,
    }
}

fn write_estimate(root: &Path, id: &str, median: f64) {
    let path = root.join(id).join("new/estimates.json");
    fs::create_dir_all(path.parent().expect("estimate parent")).expect("create estimate parent");
    fs::write(
        path,
        serde_json::json!({
            "median": {
                "point_estimate": median,
                "confidence_interval": {
                    "lower_bound": median - 1.0,
                    "upper_bound": median + 1.0,
                }
            }
        })
        .to_string(),
    )
    .expect("write estimate");
}

#[test]
fn criterion_summary_reports_only_completed_benchmark_roots() {
    let directory = TestDirectory::new("criterion");
    let valid_root = directory.path().join("valid");
    let invalid_root = directory.path().join("invalid");
    let missing_root = directory.path().join("missing");
    write_estimate(&valid_root, "group/case", 42.0);
    fs::create_dir_all(invalid_root.join("broken/new")).expect("create invalid criterion tree");
    fs::write(invalid_root.join("broken/new/estimates.json"), "not json")
        .expect("write invalid estimate");
    let steps = vec![
        step(
            "no-criterion",
            PathBuf::new(),
            PathBuf::new(),
            None,
            StepStatus::Ran,
        ),
        step(
            "skipped-criterion",
            PathBuf::new(),
            PathBuf::new(),
            Some(valid_root.clone()),
            StepStatus::Skipped {
                reason: "not requested".to_string(),
            },
        ),
        step(
            "missing-criterion",
            PathBuf::new(),
            PathBuf::new(),
            Some(missing_root),
            StepStatus::Ran,
        ),
        step(
            "valid-criterion",
            PathBuf::new(),
            PathBuf::new(),
            Some(valid_root),
            StepStatus::Ran,
        ),
        step(
            "invalid-criterion",
            PathBuf::new(),
            PathBuf::new(),
            Some(invalid_root),
            StepStatus::Ran,
        ),
    ];

    let summary = criterion_summary_json(&steps);

    assert_eq!(summary["count"], 1);
    assert_eq!(summary["estimates"][0]["id"], "group/case");
    assert_eq!(
        summary["steps"].as_array().expect("step summaries").len(),
        3
    );
    assert_eq!(summary["steps"][0]["note"], "no Criterion output produced");
    assert_eq!(summary["steps"][1]["count"], 1);
    assert!(summary["steps"][2]["error"]
        .as_str()
        .is_some_and(|error| error.contains("failed to parse")));
}

#[test]
fn criterion_estimate_json_preserves_confidence_interval() {
    let estimate = BenchEstimate {
        id: "group/case".to_string(),
        median_ns: 12.0,
        median_lower_ns: 10.0,
        median_upper_ns: 14.0,
    };

    let value = criterion_estimate_json(&estimate);

    assert_eq!(value["id"], "group/case");
    assert_eq!(value["median_ns"], 12.0);
    assert_eq!(value["median_lower_ns"], 10.0);
    assert_eq!(value["median_upper_ns"], 14.0);
}

#[test]
fn metal_decode_summary_reports_skips_and_unreadable_output() {
    let directory = TestDirectory::new("decode-errors");
    let output = directory.path().join("decode.out");
    let skipped = step(
        "metal-decode-benchmark",
        output.clone(),
        directory.path().join("decode.err"),
        None,
        StepStatus::Skipped {
            reason: "Metal unavailable".to_string(),
        },
    );
    let ran = step(
        "metal-decode-benchmark",
        output.clone(),
        directory.path().join("decode.err"),
        None,
        StepStatus::Ran,
    );

    let skipped_summary = read_metal_decode_summary(&output, &[skipped]);
    let error_summary = read_metal_decode_summary(&output, &[ran]);

    assert_eq!(skipped_summary["status"], "skipped");
    assert_eq!(skipped_summary["reason"], "Metal unavailable");
    assert_eq!(error_summary["status"], "error");
    assert!(error_summary["error"]
        .as_str()
        .is_some_and(|error| error.contains("failed to read Metal decode")));
}

#[test]
fn metal_encode_summary_reconciles_all_row_kinds_and_metadata() {
    let directory = TestDirectory::new("encode-rows");
    let output = directory.path().join("encode.out");
    fs::write(
        &output,
        concat!(
            "j2k_metal_encode_auto_bench mode=lossless codec=htj2k components=gray8 size=64x64 cpu_ms=1.0 auto_ms=skipped\n",
            "j2k_metal_encode_auto_probe mode=lossless codec=htj2k components=gray8 size=64x64 error=Metal unavailable\n",
            "j2k_metal_encode_stage_bench stage=forward_dwt97 size=64x64 cpu_ms=1.0 metal_ms=skipped\n",
            "j2k_metal_encode_batch_sizes\t1,16\n",
            "unrelated\tvalue\n",
        ),
    )
    .expect("write Metal encode output");
    let ran = step(
        "metal-encode-auto-routing",
        output.clone(),
        directory.path().join("encode.err"),
        None,
        StepStatus::Ran,
    );

    let summary = read_metal_encode_summary(&output, &[ran]);

    assert_eq!(summary["auto_bench_count"], 1);
    assert_eq!(summary["auto_probe_count"], 1);
    assert_eq!(summary["stage_bench_count"], 1);
    assert_eq!(summary["skipped_auto_bench_count"], 1);
    assert_eq!(summary["skipped_stage_bench_count"], 1);
    assert_eq!(summary["probe_error_count"], 1);
    assert_eq!(summary["metadata"]["j2k_metal_encode_batch_sizes"], "1,16");
}

#[test]
fn metal_encode_summary_reports_skips_and_unreadable_output() {
    let directory = TestDirectory::new("encode-errors");
    let output = directory.path().join("encode.out");
    let skipped = step(
        "metal-encode-auto-routing",
        output.clone(),
        directory.path().join("encode.err"),
        None,
        StepStatus::Skipped {
            reason: "Metal unavailable".to_string(),
        },
    );
    let ran = step(
        "metal-encode-auto-routing",
        output.clone(),
        directory.path().join("encode.err"),
        None,
        StepStatus::Ran,
    );

    let skipped_summary = read_metal_encode_summary(&output, &[skipped]);
    let error_summary = read_metal_encode_summary(&output, &[ran]);

    assert_eq!(skipped_summary["status"], "skipped");
    assert_eq!(skipped_summary["reason"], "Metal unavailable");
    assert_eq!(error_summary["status"], "error");
    assert!(error_summary["error"]
        .as_str()
        .is_some_and(|error| error.contains("failed to read Metal benchmark output")));
}

#[test]
fn metal_transcode_summary_fails_closed_for_each_missing_stream() {
    let directory = TestDirectory::new("transcode-errors");
    let stdout = directory.path().join("transcode.out");
    let stderr = directory.path().join("transcode.err");
    let skipped = step(
        "metal-transcode-benchmark",
        stdout.clone(),
        stderr.clone(),
        None,
        StepStatus::Skipped {
            reason: "Metal unavailable".to_string(),
        },
    );
    let ran = || {
        step(
            "metal-transcode-benchmark",
            stdout.clone(),
            stderr.clone(),
            None,
            StepStatus::Ran,
        )
    };

    let skipped_summary = read_metal_transcode_summary(&stdout, &stderr, &[skipped]);
    let stdout_error = read_metal_transcode_summary(&stdout, &stderr, &[ran()]);
    fs::write(&stdout, "criterion output\n").expect("write transcode stdout");
    let stderr_error = read_metal_transcode_summary(&stdout, &stderr, &[ran()]);

    assert_eq!(skipped_summary["status"], "skipped");
    assert_eq!(skipped_summary["reason"], "Metal unavailable");
    assert!(stdout_error["error"]
        .as_str()
        .is_some_and(|error| error.contains("failed to read Metal transcode benchmark stdout")));
    assert!(stderr_error["error"]
        .as_str()
        .is_some_and(|error| error.contains("failed to read Metal transcode benchmark stderr")));
}

#[test]
fn row_parsers_preserve_optional_suffixes_and_reject_wrong_profiles() {
    let probe = parse_metal_auto_probe_line(
        "j2k_metal_encode_auto_probe mode=lossy codec=htj2k components=gray8 size=64x64 error=Metal unavailable",
    )
    .expect("valid error probe");
    let stage = parse_metal_stage_bench_line(
        "j2k_metal_encode_stage_bench stage=forward_dwt97 size=64x64 cpu_ms=1.0 metal_ms=0.5 dispatch=one dispatch",
    )
    .expect("valid dispatch stage");

    assert_eq!(probe["error"], "Metal unavailable");
    assert_eq!(stage["dispatch"], "one dispatch");
    assert!(
        parse_metal_transcode_profile_line("j2k_profile codec=jpeg op=transcode_batch").is_none()
    );
    assert!(parse_metal_transcode_profile_line("not a profile row").is_none());
}

#[test]
fn primitive_parsers_reject_invalid_boolean_and_empty_metadata() {
    let fields = vec![("enabled".to_string(), "sometimes".to_string())];
    assert_eq!(parse_bool_field(&fields, "enabled"), None);

    let directory = TestDirectory::new("metadata");
    let path = directory.path().join("metadata.tsv");
    fs::write(&path, "malformed line\nunselected\tvalue\n").expect("write metadata input");

    let error = read_tsv_metadata(&path, &["selected"])
        .expect_err("metadata without selected keys must fail closed");
    assert!(error.contains("contained no fixture metadata"));
}
