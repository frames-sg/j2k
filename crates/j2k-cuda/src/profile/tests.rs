// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::BackendKind;
use std::{
    fs,
    io::ErrorKind,
    path::PathBuf,
    process,
    time::{SystemTime, UNIX_EPOCH},
};

use super::emit::{add_payload_resource_upload_us, build_profile_fields, finalize_decode_total_us};
use super::trace::{chrome_encode_trace_json, chrome_trace_json, write_trace_file};
use super::{CudaHtj2kDecodeProfileDetail, CudaHtj2kEncodeProfileReport, CudaHtj2kProfileReport};
use crate::SurfaceResidency;

#[test]
fn finalize_decode_total_us_includes_cpu_and_cuda_stages() {
    let mut report = decode_report();
    report.total_us = 3;
    report.detail.status_d2h_us = 11;

    finalize_decode_total_us(&mut report);

    assert_eq!(report.total_us, 66);
    assert_eq!(report.detail.stage_sum_us, 66);
}

#[test]
fn detailed_decode_profile_separates_wall_and_stage_sum() {
    let mut report = decode_report();
    report.detail.wall_total_us = 100;
    report.detail.table_upload_us = 13;
    report.detail.payload_upload_us = 17;
    report.detail.ht_dispatch_count = 2;
    finalize_decode_total_us(&mut report);

    assert_eq!(report.detail.wall_total_us, 100);
    assert_eq!(report.detail.stage_sum_us, report.total_us);
    assert_eq!(report.detail.ht_dispatch_count, 2);
}

#[test]
fn payload_resource_upload_detail_does_not_claim_job_status_split() {
    let mut report = CudaHtj2kProfileReport::default();

    add_payload_resource_upload_us(&mut report, 23);

    assert_eq!(report.h2d_us, 23);
    assert_eq!(report.detail.payload_upload_us, 23);
    assert_eq!(report.detail.job_upload_us, 0);
    assert_eq!(report.detail.status_d2h_us, 0);
    assert_eq!(report.detail.output_d2h_us, 0);
}

#[test]
fn profile_field_build_failure_is_diagnostic_only() {
    let fields = build_profile_fields::<1>("test_profile_fields", || {
        Err(j2k_profile::ProfileError::InvalidInput {
            what: "injected profile failure",
        })
    });

    assert!(fields.is_none());
}

#[test]
fn decode_trace_json_contains_ordered_stage_spans() {
    let mut report = decode_report();
    report.detail.status_d2h_us = 11;
    let trace = chrome_trace_json("decode", &report).expect("bounded decode trace");

    assert!(trace.starts_with("{\"traceEvents\":["));
    assert!(trace.contains("\"name\":\"parse\",\"cat\":\"decode\",\"ph\":\"X\""));
    assert!(trace.contains("\"name\":\"ht_cleanup\",\"cat\":\"decode\",\"ph\":\"X\""));
    assert!(trace.contains("\"name\":\"status_d2h\",\"cat\":\"decode\",\"ph\":\"X\""));
    assert!(trace.contains("\"name\":\"store\",\"cat\":\"decode\",\"ph\":\"X\""));
    assert!(trace.contains("\"ts\":0,\"dur\":1"));
    assert!(trace.contains("\"ts\":50,\"dur\":10"));
    assert!(trace.ends_with("]}"));
}

#[test]
fn decode_trace_json_does_not_advance_time_for_fused_refinement() {
    let mut report = decode_report();
    report.ht_refine_us = report.ht_cleanup_us;
    let trace = chrome_trace_json("decode", &report).expect("bounded decode trace");

    assert!(trace.contains("\"name\":\"ht_refine\",\"cat\":\"decode\",\"ph\":\"X\""));
    assert!(trace.contains("\"name\":\"ht_refine\",\"cat\":\"decode\",\"ph\":\"X\",\"pid\":1,\"tid\":1,\"ts\":10,\"dur\":5"));
    assert!(trace.contains("\"name\":\"dequant\",\"cat\":\"decode\",\"ph\":\"X\",\"pid\":1,\"tid\":1,\"ts\":15,\"dur\":7"));
}

#[test]
fn encode_trace_json_contains_ordered_stage_spans() {
    let report = CudaHtj2kEncodeProfileReport {
        deinterleave_us: 11,
        mct_us: 12,
        dwt_us: 13,
        quantize_us: 14,
        ht_encode_us: 15,
        packetize_us: 16,
        total_us: 81,
        input_bytes: 100,
        codestream_bytes: 50,
        block_count: 4,
        dispatch_count: 6,
        backend: BackendKind::Cuda,
    };
    let trace = chrome_encode_trace_json("encode", &report).expect("bounded encode trace");

    assert!(trace.starts_with("{\"traceEvents\":["));
    assert!(trace.contains("\"name\":\"deinterleave\",\"cat\":\"encode\",\"ph\":\"X\""));
    assert!(trace.contains("\"name\":\"ht_encode\",\"cat\":\"encode\",\"ph\":\"X\""));
    assert!(trace.contains("\"name\":\"packetize\",\"cat\":\"encode\",\"ph\":\"X\""));
    assert!(trace.contains("\"ts\":0,\"dur\":11"));
    assert!(trace.contains("\"ts\":65,\"dur\":16"));
    assert!(trace.ends_with("]}"));
}

#[test]
fn trace_categories_are_json_escaped_and_bounded() {
    let escaped = chrome_trace_json("decode\"\\\n", &decode_report()).expect("escaped trace");
    assert!(escaped.contains("\"cat\":\"decode\\\"\\\\\\n\""));
    assert!(!escaped.contains("decode\"\\\n"));

    let oversized = "x".repeat(j2k_profile::ProfileLimits::default().max_token_bytes() + 1);
    assert!(matches!(
        chrome_trace_json(&oversized, &decode_report()),
        Err(j2k_profile::ProfileError::LimitExceeded {
            what: "CUDA trace category",
            ..
        })
    ));
}

#[test]
fn trace_file_write_creates_new_file() {
    let path = unique_trace_path("create");
    let trace = "{\"traceEvents\":[]}";

    write_trace_file(&path, trace).expect("write new CUDA trace");

    assert_eq!(fs::read_to_string(&path).expect("read trace"), trace);
    let _ = fs::remove_file(path);
}

#[test]
fn trace_file_write_refuses_to_overwrite_existing_file() {
    let path = unique_trace_path("existing");
    fs::write(&path, "keep").expect("seed existing trace path");

    let error = write_trace_file(&path, "replace").expect_err("existing trace is rejected");

    assert_eq!(error.kind(), ErrorKind::AlreadyExists);
    assert_eq!(fs::read_to_string(&path).expect("read trace"), "keep");
    let _ = fs::remove_file(path);
}

fn decode_report() -> CudaHtj2kProfileReport {
    CudaHtj2kProfileReport {
        parse_us: 1,
        plan_us: 2,
        flatten_us: 3,
        h2d_us: 4,
        ht_cleanup_us: 5,
        ht_refine_us: 6,
        classic_tier1_us: 0,
        dequant_us: 7,
        idwt_us: 8,
        mct_us: 9,
        store_us: 10,
        total_us: 55,
        block_count: 1,
        classic_block_count: 0,
        ht_block_count: 1,
        payload_bytes: 2,
        dispatch_count: 3,
        residency: SurfaceResidency::CudaResidentDecode,
        detail: CudaHtj2kDecodeProfileDetail::default(),
    }
}

fn unique_trace_path(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before UNIX_EPOCH")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "j2k-cuda-profile-{label}-{}-{nanos}.json",
        process::id()
    ))
}
