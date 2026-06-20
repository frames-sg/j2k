// SPDX-License-Identifier: Apache-2.0

use core::fmt::Write as _;
use std::cell::RefCell;
use std::sync::OnceLock;
use std::time::Instant;

use j2k_core::BackendKind;
use j2k_profile::{profile_stage_mode_from_env, ProfileStageMode};

use crate::SurfaceResidency;

const PROFILE_ENV_VAR: &str = "J2K_PROFILE_STAGES";
const CUDA_TRACE_ENV_VAR: &str = "J2K_CUDA_TRACE";

thread_local! {
    static PROFILE_SUMMARY: RefCell<j2k_profile::ProfileSummary> =
        RefCell::new(j2k_profile::ProfileSummary::default().emit_on_drop());
}

/// Detailed route-overhead timings for strict CUDA HTJ2K decode.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CudaHtj2kDecodeProfileDetail {
    /// End-to-end profiled decode wall time.
    pub wall_total_us: u128,
    /// Sum of the reported decode stage timings.
    pub stage_sum_us: u128,
    /// CUDA table/resource upload time.
    pub table_upload_us: u128,
    /// CUDA compressed payload/resource upload time.
    ///
    /// This includes mixed resource upload calls that contain compressed
    /// payload bytes plus decode metadata. Metadata-only job upload is not
    /// split out until the CUDA runtime exposes separate timings.
    pub payload_upload_us: u128,
    /// CUDA decode job upload time, reserved as zero until split runtime timings exist.
    pub job_upload_us: u128,
    /// CUDA status download time, reserved as zero until split runtime timings exist.
    pub status_d2h_us: u128,
    /// CUDA output download time, reserved as zero until split runtime timings exist.
    pub output_d2h_us: u128,
    /// HT cleanup/refinement CUDA dispatch count.
    pub ht_dispatch_count: usize,
    /// Dequantization CUDA dispatch count.
    pub dequant_dispatch_count: usize,
    /// Inverse DWT CUDA dispatch count.
    pub idwt_dispatch_count: usize,
    /// Inverse MCT CUDA dispatch count.
    pub mct_dispatch_count: usize,
    /// Store/format conversion CUDA dispatch count.
    pub store_dispatch_count: usize,
}

/// Structured stage timings for a strict CUDA HTJ2K operation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CudaHtj2kProfileReport {
    /// CPU marker/box parse time.
    pub parse_us: u128,
    /// Native direct-plan construction time.
    pub plan_us: u128,
    /// Flat CUDA plan construction time.
    pub flatten_us: u128,
    /// Host-to-device upload time for payload and metadata.
    pub h2d_us: u128,
    /// HT cleanup kernel time.
    pub ht_cleanup_us: u128,
    /// HT refinement kernel time.
    pub ht_refine_us: u128,
    /// Dequantization kernel time.
    pub dequant_us: u128,
    /// Inverse DWT kernel time.
    pub idwt_us: u128,
    /// Inverse MCT kernel time.
    pub mct_us: u128,
    /// Store/format conversion kernel time.
    pub store_us: u128,
    /// Sum of measured decode stages.
    ///
    /// End-to-end wall time is reported in `detail.wall_total_us`.
    pub total_us: u128,
    /// Number of HTJ2K code blocks in the flat plan.
    pub block_count: usize,
    /// Number of compressed payload bytes uploaded to CUDA.
    pub payload_bytes: usize,
    /// Number of CUDA kernel dispatches.
    pub dispatch_count: usize,
    /// Surface residency represented by this profile.
    pub residency: SurfaceResidency,
    /// Detailed route-overhead profile for RCA.
    pub detail: CudaHtj2kDecodeProfileDetail,
}

impl CudaHtj2kProfileReport {
    /// Emit the report using `J2K_PROFILE_STAGES`, when enabled.
    pub fn emit(&self, path: &str) {
        emit_htj2k_profile_row(path, self);
        export_trace_if_requested(path, self);
    }
}

/// Structured stage timings for a strict CUDA HTJ2K encode operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaHtj2kEncodeProfileReport {
    /// Pixel deinterleave and level-shift CUDA stage time.
    pub deinterleave_us: u128,
    /// Forward MCT CUDA stage time.
    pub mct_us: u128,
    /// Forward DWT CUDA stage time.
    pub dwt_us: u128,
    /// Quantization CUDA stage time.
    pub quantize_us: u128,
    /// HTJ2K cleanup code-block encode CUDA stage time.
    pub ht_encode_us: u128,
    /// HTJ2K packetization CUDA stage time.
    pub packetize_us: u128,
    /// Total wall time for the measured encode call.
    pub total_us: u128,
    /// Input pixel byte count.
    pub input_bytes: usize,
    /// Output codestream byte count.
    pub codestream_bytes: usize,
    /// Number of HTJ2K code blocks encoded.
    pub block_count: usize,
    /// Number of CUDA kernel dispatches.
    pub dispatch_count: usize,
    /// Backend that satisfied the encode request.
    pub backend: BackendKind,
}

impl Default for CudaHtj2kEncodeProfileReport {
    fn default() -> Self {
        Self {
            deinterleave_us: 0,
            mct_us: 0,
            dwt_us: 0,
            quantize_us: 0,
            ht_encode_us: 0,
            packetize_us: 0,
            total_us: 0,
            input_bytes: 0,
            codestream_bytes: 0,
            block_count: 0,
            dispatch_count: 0,
            backend: BackendKind::Cpu,
        }
    }
}

impl CudaHtj2kEncodeProfileReport {
    /// Emit the report using `J2K_PROFILE_STAGES`, when enabled.
    pub fn emit(&self, path: &str) {
        emit_htj2k_encode_profile_row(path, self);
        export_encode_trace_if_requested(path, self);
    }
}

pub(crate) type ProfileInstant = Instant;

fn profile_stage_mode() -> ProfileStageMode {
    static MODE: OnceLock<ProfileStageMode> = OnceLock::new();
    *MODE.get_or_init(|| profile_stage_mode_from_env(PROFILE_ENV_VAR))
}

pub(crate) fn profile_stages_enabled() -> bool {
    profile_stage_mode() != ProfileStageMode::Disabled
}

pub(crate) fn profile_now(enabled: bool) -> Option<ProfileInstant> {
    enabled.then(Instant::now)
}

pub(crate) fn elapsed_us(start: Option<ProfileInstant>) -> u128 {
    start.map_or(0, |start| start.elapsed().as_micros())
}

#[cfg_attr(not(feature = "cuda-runtime"), allow(dead_code))]
pub(crate) fn add_payload_resource_upload_us(
    report: &mut CudaHtj2kProfileReport,
    elapsed_us: u128,
) {
    report.h2d_us = report.h2d_us.saturating_add(elapsed_us);
    report.detail.payload_upload_us = report.detail.payload_upload_us.saturating_add(elapsed_us);
}

#[cfg_attr(not(feature = "cuda-runtime"), allow(dead_code))]
pub(crate) fn finalize_decode_total_us(report: &mut CudaHtj2kProfileReport) {
    report.total_us = [
        report.parse_us,
        report.plan_us,
        report.flatten_us,
        report.h2d_us,
        report.ht_cleanup_us,
        report.ht_refine_us,
        report.dequant_us,
        report.idwt_us,
        report.mct_us,
        report.store_us,
    ]
    .into_iter()
    .fold(0u128, u128::saturating_add);
    report.detail.stage_sum_us = report.total_us;
}

pub(crate) fn emit_htj2k_profile_row(path: &str, report: &CudaHtj2kProfileReport) {
    let parse_us = report.parse_us.to_string();
    let plan_us = report.plan_us.to_string();
    let flatten_us = report.flatten_us.to_string();
    let h2d_us = report.h2d_us.to_string();
    let ht_cleanup_us = report.ht_cleanup_us.to_string();
    let ht_refine_us = report.ht_refine_us.to_string();
    let dequant_us = report.dequant_us.to_string();
    let idwt_us = report.idwt_us.to_string();
    let mct_us = report.mct_us.to_string();
    let store_us = report.store_us.to_string();
    let total_us = report.total_us.to_string();
    let block_count = report.block_count.to_string();
    let payload_bytes = report.payload_bytes.to_string();
    let dispatch_count = report.dispatch_count.to_string();
    let residency = format!("{:?}", report.residency);
    let wall_total_us = report.detail.wall_total_us.to_string();
    let stage_sum_us = report.detail.stage_sum_us.to_string();
    let table_upload_us = report.detail.table_upload_us.to_string();
    let payload_upload_us = report.detail.payload_upload_us.to_string();
    let job_upload_us = report.detail.job_upload_us.to_string();
    let status_d2h_us = report.detail.status_d2h_us.to_string();
    let output_d2h_us = report.detail.output_d2h_us.to_string();
    let ht_dispatch_count = report.detail.ht_dispatch_count.to_string();
    let dequant_dispatch_count = report.detail.dequant_dispatch_count.to_string();
    let idwt_dispatch_count = report.detail.idwt_dispatch_count.to_string();
    let mct_dispatch_count = report.detail.mct_dispatch_count.to_string();
    let store_dispatch_count = report.detail.store_dispatch_count.to_string();

    j2k_profile::emit_profile_row(
        profile_stage_mode(),
        &PROFILE_SUMMARY,
        "j2k",
        "cuda_htj2k",
        path,
        &[
            ("parse_us", parse_us.as_str()),
            ("plan_us", plan_us.as_str()),
            ("flatten_us", flatten_us.as_str()),
            ("h2d_us", h2d_us.as_str()),
            ("ht_cleanup_us", ht_cleanup_us.as_str()),
            ("ht_refine_us", ht_refine_us.as_str()),
            ("dequant_us", dequant_us.as_str()),
            ("idwt_us", idwt_us.as_str()),
            ("mct_us", mct_us.as_str()),
            ("store_us", store_us.as_str()),
            ("total_us", total_us.as_str()),
            ("block_count", block_count.as_str()),
            ("payload_bytes", payload_bytes.as_str()),
            ("dispatch_count", dispatch_count.as_str()),
            ("residency", residency.as_str()),
            ("wall_total_us", wall_total_us.as_str()),
            ("stage_sum_us", stage_sum_us.as_str()),
            ("table_upload_us", table_upload_us.as_str()),
            ("payload_upload_us", payload_upload_us.as_str()),
            ("job_upload_us", job_upload_us.as_str()),
            ("status_d2h_us", status_d2h_us.as_str()),
            ("output_d2h_us", output_d2h_us.as_str()),
            ("ht_dispatch_count", ht_dispatch_count.as_str()),
            ("dequant_dispatch_count", dequant_dispatch_count.as_str()),
            ("idwt_dispatch_count", idwt_dispatch_count.as_str()),
            ("mct_dispatch_count", mct_dispatch_count.as_str()),
            ("store_dispatch_count", store_dispatch_count.as_str()),
        ],
    );
}

pub(crate) fn emit_htj2k_encode_profile_row(path: &str, report: &CudaHtj2kEncodeProfileReport) {
    let deinterleave_us = report.deinterleave_us.to_string();
    let mct_us = report.mct_us.to_string();
    let dwt_us = report.dwt_us.to_string();
    let quantize_us = report.quantize_us.to_string();
    let ht_encode_us = report.ht_encode_us.to_string();
    let packetize_us = report.packetize_us.to_string();
    let total_us = report.total_us.to_string();
    let input_bytes = report.input_bytes.to_string();
    let codestream_bytes = report.codestream_bytes.to_string();
    let block_count = report.block_count.to_string();
    let dispatch_count = report.dispatch_count.to_string();
    let backend = format!("{:?}", report.backend);

    j2k_profile::emit_profile_row(
        profile_stage_mode(),
        &PROFILE_SUMMARY,
        "j2k",
        "cuda_htj2k_encode",
        path,
        &[
            ("deinterleave_us", deinterleave_us.as_str()),
            ("mct_us", mct_us.as_str()),
            ("dwt_us", dwt_us.as_str()),
            ("quantize_us", quantize_us.as_str()),
            ("ht_encode_us", ht_encode_us.as_str()),
            ("packetize_us", packetize_us.as_str()),
            ("total_us", total_us.as_str()),
            ("input_bytes", input_bytes.as_str()),
            ("codestream_bytes", codestream_bytes.as_str()),
            ("block_count", block_count.as_str()),
            ("dispatch_count", dispatch_count.as_str()),
            ("backend", backend.as_str()),
        ],
    );
}

fn export_trace_if_requested(path: &str, report: &CudaHtj2kProfileReport) {
    let Some(trace_path) = std::env::var_os(CUDA_TRACE_ENV_VAR) else {
        return;
    };
    let trace = chrome_trace_json(path, report);
    if let Err(error) = std::fs::write(&trace_path, trace) {
        std::eprintln!("j2k_profile codec=j2k op=cuda_htj2k_trace path=cuda error={error}");
    }
}

fn chrome_trace_json(path: &str, report: &CudaHtj2kProfileReport) -> String {
    let stages = [
        ("parse", report.parse_us),
        ("plan", report.plan_us),
        ("flatten", report.flatten_us),
        ("h2d", report.h2d_us),
        ("ht_cleanup", report.ht_cleanup_us),
        ("ht_refine", report.ht_refine_us),
        ("dequant", report.dequant_us),
        ("idwt", report.idwt_us),
        ("mct", report.mct_us),
        ("store", report.store_us),
    ];
    let mut trace = String::from("{\"traceEvents\":[");
    let mut ts = 0u128;
    for (index, (name, dur)) in stages.iter().enumerate() {
        if index != 0 {
            trace.push(',');
        }
        let event_ts = if *name == "ht_refine" {
            ts.saturating_sub(report.ht_cleanup_us)
        } else {
            ts
        };
        write!(
            trace,
            "{{\"name\":\"{name}\",\"cat\":\"{path}\",\"ph\":\"X\",\"pid\":1,\"tid\":1,\"ts\":{event_ts},\"dur\":{dur}}}"
        )
        .expect("writing trace JSON to String failed");
        if *name != "ht_refine" {
            ts = ts.saturating_add(*dur);
        }
    }
    trace.push_str("]}");
    trace
}

fn export_encode_trace_if_requested(path: &str, report: &CudaHtj2kEncodeProfileReport) {
    let Some(trace_path) = std::env::var_os(CUDA_TRACE_ENV_VAR) else {
        return;
    };
    let trace = chrome_encode_trace_json(path, report);
    if let Err(error) = std::fs::write(&trace_path, trace) {
        std::eprintln!("j2k_profile codec=j2k op=cuda_htj2k_encode_trace path=cuda error={error}");
    }
}

fn chrome_encode_trace_json(path: &str, report: &CudaHtj2kEncodeProfileReport) -> String {
    let stages = [
        ("deinterleave", report.deinterleave_us),
        ("mct", report.mct_us),
        ("dwt", report.dwt_us),
        ("quantize", report.quantize_us),
        ("ht_encode", report.ht_encode_us),
        ("packetize", report.packetize_us),
    ];
    let mut trace = String::from("{\"traceEvents\":[");
    let mut ts = 0u128;
    for (index, (name, dur)) in stages.iter().enumerate() {
        if index != 0 {
            trace.push(',');
        }
        write!(
            trace,
            "{{\"name\":\"{name}\",\"cat\":\"{path}\",\"ph\":\"X\",\"pid\":1,\"tid\":1,\"ts\":{ts},\"dur\":{dur}}}"
        )
        .expect("writing trace JSON to String failed");
        ts = ts.saturating_add(*dur);
    }
    trace.push_str("]}");
    trace
}

#[cfg(test)]
mod tests {
    use super::{
        add_payload_resource_upload_us, chrome_encode_trace_json, chrome_trace_json,
        finalize_decode_total_us, CudaHtj2kDecodeProfileDetail, CudaHtj2kEncodeProfileReport,
        CudaHtj2kProfileReport,
    };
    use j2k_core::BackendKind;

    use crate::SurfaceResidency;

    #[test]
    fn finalize_decode_total_us_includes_cpu_and_cuda_stages() {
        let mut report = CudaHtj2kProfileReport {
            parse_us: 1,
            plan_us: 2,
            flatten_us: 3,
            h2d_us: 4,
            ht_cleanup_us: 5,
            ht_refine_us: 6,
            dequant_us: 7,
            idwt_us: 8,
            mct_us: 9,
            store_us: 10,
            total_us: 3,
            block_count: 1,
            payload_bytes: 2,
            dispatch_count: 3,
            residency: SurfaceResidency::CudaResidentDecode,
            detail: CudaHtj2kDecodeProfileDetail::default(),
        };

        finalize_decode_total_us(&mut report);

        assert_eq!(report.total_us, 55);
        assert_eq!(report.detail.stage_sum_us, 55);
    }

    #[test]
    fn detailed_decode_profile_separates_wall_and_stage_sum() {
        let mut report = CudaHtj2kProfileReport {
            parse_us: 1,
            plan_us: 2,
            flatten_us: 3,
            h2d_us: 4,
            ht_cleanup_us: 5,
            ht_refine_us: 5,
            dequant_us: 6,
            idwt_us: 7,
            mct_us: 8,
            store_us: 9,
            total_us: 0,
            block_count: 10,
            payload_bytes: 11,
            dispatch_count: 12,
            residency: SurfaceResidency::CudaResidentDecode,
            detail: CudaHtj2kDecodeProfileDetail::default(),
        };
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
    fn decode_trace_json_contains_ordered_stage_spans() {
        let report = CudaHtj2kProfileReport {
            parse_us: 1,
            plan_us: 2,
            flatten_us: 3,
            h2d_us: 4,
            ht_cleanup_us: 5,
            ht_refine_us: 6,
            dequant_us: 7,
            idwt_us: 8,
            mct_us: 9,
            store_us: 10,
            total_us: 55,
            block_count: 1,
            payload_bytes: 2,
            dispatch_count: 3,
            residency: SurfaceResidency::CudaResidentDecode,
            detail: CudaHtj2kDecodeProfileDetail::default(),
        };

        let trace = chrome_trace_json("decode", &report);

        assert!(trace.starts_with("{\"traceEvents\":["));
        assert!(trace.contains("\"name\":\"parse\",\"cat\":\"decode\",\"ph\":\"X\""));
        assert!(trace.contains("\"name\":\"ht_cleanup\",\"cat\":\"decode\",\"ph\":\"X\""));
        assert!(trace.contains("\"name\":\"store\",\"cat\":\"decode\",\"ph\":\"X\""));
        assert!(trace.contains("\"ts\":0,\"dur\":1"));
        assert!(trace.contains("\"ts\":39,\"dur\":10"));
        assert!(trace.ends_with("]}"));
    }

    #[test]
    fn decode_trace_json_does_not_advance_time_for_fused_refinement() {
        let report = CudaHtj2kProfileReport {
            parse_us: 1,
            plan_us: 2,
            flatten_us: 3,
            h2d_us: 4,
            ht_cleanup_us: 5,
            ht_refine_us: 5,
            dequant_us: 6,
            idwt_us: 7,
            mct_us: 8,
            store_us: 9,
            total_us: 45,
            block_count: 1,
            payload_bytes: 2,
            dispatch_count: 3,
            residency: SurfaceResidency::CudaResidentDecode,
            detail: CudaHtj2kDecodeProfileDetail::default(),
        };

        let trace = chrome_trace_json("decode", &report);

        assert!(trace.contains("\"name\":\"ht_refine\",\"cat\":\"decode\",\"ph\":\"X\""));
        assert!(trace.contains("\"name\":\"ht_refine\",\"cat\":\"decode\",\"ph\":\"X\",\"pid\":1,\"tid\":1,\"ts\":10,\"dur\":5"));
        assert!(trace.contains("\"name\":\"dequant\",\"cat\":\"decode\",\"ph\":\"X\",\"pid\":1,\"tid\":1,\"ts\":15,\"dur\":6"));
        assert!(trace.contains("\"name\":\"store\",\"cat\":\"decode\",\"ph\":\"X\",\"pid\":1,\"tid\":1,\"ts\":36,\"dur\":9"));
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

        let trace = chrome_encode_trace_json("encode", &report);

        assert!(trace.starts_with("{\"traceEvents\":["));
        assert!(trace.contains("\"name\":\"deinterleave\",\"cat\":\"encode\",\"ph\":\"X\""));
        assert!(trace.contains("\"name\":\"ht_encode\",\"cat\":\"encode\",\"ph\":\"X\""));
        assert!(trace.contains("\"name\":\"packetize\",\"cat\":\"encode\",\"ph\":\"X\""));
        assert!(trace.contains("\"ts\":0,\"dur\":11"));
        assert!(trace.contains("\"ts\":65,\"dur\":16"));
        assert!(trace.ends_with("]}"));
    }
}
