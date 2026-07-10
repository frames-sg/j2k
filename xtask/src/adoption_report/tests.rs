use super::{adoption_report, publication_issues, read_tsv_table, AdoptionReportModel, TsvTable};
use crate::publication_gate::collect_publication_gate_issues;
use serde_json::json;
use std::{collections::BTreeMap, path::Path};

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
                    "output_bytes": 262_144
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
fn serialized_report_model_schema_is_stable() {
    let model = AdoptionReportModel {
        run_dir: "fixture/run".into(),
        summary: json!({
            "mode": "full",
            "nested": {"preserved": true}
        }),
        cpu_fixture_compare: TsvTable {
            headers: vec!["decoder".to_string(), "case".to_string()],
            rows: vec![BTreeMap::from([
                ("case".to_string(), "fixture-a".to_string()),
                ("decoder".to_string(), "j2k".to_string()),
            ])],
        },
        cpu_encode_compare: TsvTable {
            headers: vec!["encoder".to_string(), "case".to_string()],
            rows: vec![BTreeMap::from([
                ("case".to_string(), "fixture-a".to_string()),
                ("encoder".to_string(), "grok".to_string()),
            ])],
        },
        publication_issues: vec!["diagnostic-only".to_string()],
    };

    let serialized =
        serde_json::to_string_pretty(&model.serialized_schema()).expect("serialize model");
    let schema: serde_json::Value = serde_json::from_str(&serialized).expect("parse model");

    assert_eq!(
        serialized,
        include_str!("golden/report-model-schema.json").trim_end()
    );

    assert_eq!(
        schema,
        json!({
            "run_dir": "fixture/run",
            "summary": {
                "mode": "full",
                "nested": {"preserved": true}
            },
            "cpu_fixture_compare": {
                "headers": ["decoder", "case"],
                "rows": [{"case": "fixture-a", "decoder": "j2k"}]
            },
            "cpu_encode_compare": {
                "headers": ["encoder", "case"],
                "rows": [{"case": "fixture-a", "encoder": "grok"}]
            },
            "publication_issues": ["diagnostic-only"]
        })
    );
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

    let error = adoption_report(["--run-dir".to_string(), dir.display().to_string()].into_iter())
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
fn publication_refusal_precedes_missing_table_errors() {
    let dir = std::env::current_dir()
        .expect("current dir")
        .join("target")
        .join("adoption-report-error-order-test")
        .join(std::process::id().to_string());
    std::fs::create_dir_all(&dir).expect("create dir");
    std::fs::write(
        dir.join("summary.json"),
        serde_json::to_string_pretty(&json!({
            "mode": "quick",
            "include_generated": true,
            "cpu_fixture_compare": {
                "publication_eligible": "false",
                "publication_blockers": "generated-fixtures-included",
                "benchmark_complete": "true"
            },
            "cpu_encode_compare": {
                "publication_eligible": "false",
                "publication_blockers": "generated-fixtures-included",
                "benchmark_complete": "true"
            }
        }))
        .expect("summary json"),
    )
    .expect("write summary");

    let error = adoption_report(["--run-dir".to_string(), dir.display().to_string()].into_iter())
        .expect_err("publication refusal should happen before table collection");
    assert!(error.contains("not publishable"));
    assert!(!error.contains("cpu-fixture-compare.out"));

    let diagnostic_error = adoption_report(
        [
            "--run-dir".to_string(),
            dir.display().to_string(),
            "--allow-nonpublishable".to_string(),
        ]
        .into_iter(),
    )
    .expect_err("diagnostic rendering should continue to table collection");
    assert!(diagnostic_error.contains("cpu-fixture-compare.out"));
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
    let normalized_report = report.replace(&dir.display().to_string(), "$RUN_DIR");
    assert_eq!(
        normalized_report,
        include_str!("golden/minimal-diagnostic-report.md")
    );
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
    assert!(report.contains(
        "| external_mixed_encode | 16 | 200.000 | 140.000 | 220.000 | grok | 220.000 | 0.909x |"
    ));
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

#[test]
fn report_marks_clean_bundle_publishable_without_blocking_section() {
    let dir = std::env::current_dir()
        .expect("current dir")
        .join("target")
        .join("adoption-report-publishable-test")
        .join(std::process::id().to_string());
    std::fs::create_dir_all(&dir).expect("create dir");
    write_minimal_bundle(&dir, true);
    let out = dir.join("report.md");

    adoption_report(
        [
            "--run-dir".to_string(),
            dir.display().to_string(),
            "--out".to_string(),
            out.display().to_string(),
        ]
        .into_iter(),
    )
    .expect("publishable report should write");

    let report = std::fs::read_to_string(out).expect("read report");
    assert!(report.contains("Status: publishable for the requested benchmark scopes"));
    assert!(!report.contains("Blocking issues:"));
}

#[expect(
    clippy::too_many_lines,
    reason = "the fixture builds one complete minimal benchmark bundle for report tests"
)]
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
        "fixture_comparability_scope": "Fixture comparability scope is pinned for this report.",
        "publication_note": "Publication note preserves the recorded bundle context.",
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
                            "median_ns": 1_500_000.0,
                            "median_lower_ns": 1_400_000.0,
                            "median_upper_ns": 1_600_000.0
                        }
                    ]
                },
                {
                    "step": "cuda-htj2k-encode",
                    "count": 1,
                    "estimates": [
                        {
                            "id": "cuda_encode_external_rgb8",
                            "median_ns": 2_500_000.0,
                            "median_lower_ns": 2_400_000.0,
                            "median_upper_ns": 2_600_000.0
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
                    "output_bytes": 262_144
                }
            ]
        },
        "metal_encode_auto_routing": {
            "status": "ran",
                "auto_bench_count": 2,
                "skipped_auto_bench_count": 0,
                "probe_error_count": 0,
                "resident_bench_count": 1,
                "skipped_resident_bench_count": 0,
                "resident_verified_bench_count": 1,
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
                ],
                "resident_benches": [
                    {
                        "mode": "lossless_external",
                        "codec": "htj2k",
                        "components": "gray8",
                        "size": "512x512",
                        "batch_size": "16",
                        "packetization_used": true,
                        "codestream_assembly_used": true,
                        "cpu_ms": 10.0,
                        "hybrid_cpu_packet_ms": 6.0,
                        "resident_host_ms": 4.0,
                        "resident_buffer_ms": 3.0,
                        "host_readback_ms": 1.0
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
openjpeg\tcase_skipped\tportable-native\tskipped\texternal:case-a\tnatural-image\tj2k\tjp2\tfull\trgb8\t128x128\t1\t1024\tNA\tNA\tNA\t0\topenjpeg-unavailable\n\
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
openjpeg\tcase_skipped\tclassic-lossless-cli\tskipped\texternal:case-a\tnatural-image\tpng\t128x128\t1\t1024\tNA\tNA\tNA\t0\topenjpeg-compress-unavailable\n\
benchmark_complete\ttrue\n",
    )
    .expect("write encode output");
}
