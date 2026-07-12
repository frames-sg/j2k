// SPDX-License-Identifier: MIT OR Apache-2.0

use std::cell::RefCell;

use j2k_profile::{ProfileField, ProfileResult, ProfileStageMode, StageModeCache};

use super::CudaHtj2kEncodeProfileReport;
#[cfg(any(feature = "cuda-runtime", test))]
use super::CudaHtj2kProfileReport;

const PROFILE_ENV_VAR: &str = "J2K_PROFILE_STAGES";

thread_local! {
    static PROFILE_SUMMARY: RefCell<j2k_profile::ProfileSummary> =
        RefCell::new(j2k_profile::ProfileSummary::default().emit_on_drop());
}

fn profile_stage_mode() -> ProfileStageMode {
    static MODE: StageModeCache = StageModeCache::new();
    MODE.mode_from_env(PROFILE_ENV_VAR)
}

pub(crate) fn profile_stages_enabled() -> bool {
    profile_stage_mode() != ProfileStageMode::Disabled
}

pub(crate) fn emit_optional_gpu_route_fields<const N: usize>(
    operation: &'static str,
    build: impl FnOnce() -> ProfileResult<[ProfileField; N]>,
    emit: impl FnOnce([ProfileField; N]),
) {
    if !j2k_profile::gpu_route_profile_enabled() {
        return;
    }
    match build() {
        Ok(fields) => emit(fields),
        Err(error) => j2k_profile::emit_profile_error(operation, &error),
    }
}

#[cfg(any(feature = "cuda-runtime", test))]
pub(crate) fn add_payload_resource_upload_us(
    report: &mut CudaHtj2kProfileReport,
    elapsed_us: u128,
) {
    report.h2d_us = report.h2d_us.saturating_add(elapsed_us);
    report.detail.payload_upload_us = report.detail.payload_upload_us.saturating_add(elapsed_us);
}

#[cfg(any(feature = "cuda-runtime", test))]
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

#[cfg(feature = "cuda-runtime")]
pub(crate) fn emit_htj2k_profile_row(path: &str, report: &CudaHtj2kProfileReport) {
    emit_cuda_profile_fields("cuda_htj2k_decode_fields", "cuda_htj2k", path, || {
        Ok([
            ProfileField::metric("parse_us", report.parse_us)?,
            ProfileField::metric("plan_us", report.plan_us)?,
            ProfileField::metric("flatten_us", report.flatten_us)?,
            ProfileField::metric("h2d_us", report.h2d_us)?,
            ProfileField::metric("ht_cleanup_us", report.ht_cleanup_us)?,
            ProfileField::metric("ht_refine_us", report.ht_refine_us)?,
            ProfileField::metric("dequant_us", report.dequant_us)?,
            ProfileField::metric("idwt_us", report.idwt_us)?,
            ProfileField::metric("mct_us", report.mct_us)?,
            ProfileField::metric("store_us", report.store_us)?,
            ProfileField::metric("total_us", report.total_us)?,
            ProfileField::metric("block_count", report.block_count)?,
            ProfileField::metric("payload_bytes", report.payload_bytes)?,
            ProfileField::metric("dispatch_count", report.dispatch_count)?,
            ProfileField::label("residency", DebugValue(report.residency))?,
            ProfileField::metric("wall_total_us", report.detail.wall_total_us)?,
            ProfileField::metric("stage_sum_us", report.detail.stage_sum_us)?,
            ProfileField::metric("table_upload_us", report.detail.table_upload_us)?,
            ProfileField::metric("payload_upload_us", report.detail.payload_upload_us)?,
            ProfileField::metric("job_upload_us", report.detail.job_upload_us)?,
            ProfileField::metric("status_d2h_us", report.detail.status_d2h_us)?,
            ProfileField::metric("output_d2h_us", report.detail.output_d2h_us)?,
            ProfileField::metric("ht_dispatch_count", report.detail.ht_dispatch_count)?,
            ProfileField::metric(
                "dequant_dispatch_count",
                report.detail.dequant_dispatch_count,
            )?,
            ProfileField::metric("idwt_dispatch_count", report.detail.idwt_dispatch_count)?,
            ProfileField::metric("mct_dispatch_count", report.detail.mct_dispatch_count)?,
            ProfileField::metric("store_dispatch_count", report.detail.store_dispatch_count)?,
        ])
    });
}

pub(crate) fn emit_htj2k_encode_profile_row(path: &str, report: &CudaHtj2kEncodeProfileReport) {
    emit_cuda_profile_fields(
        "cuda_htj2k_encode_fields",
        "cuda_htj2k_encode",
        path,
        || {
            Ok([
                ProfileField::metric("deinterleave_us", report.deinterleave_us)?,
                ProfileField::metric("mct_us", report.mct_us)?,
                ProfileField::metric("dwt_us", report.dwt_us)?,
                ProfileField::metric("quantize_us", report.quantize_us)?,
                ProfileField::metric("ht_encode_us", report.ht_encode_us)?,
                ProfileField::metric("packetize_us", report.packetize_us)?,
                ProfileField::metric("total_us", report.total_us)?,
                ProfileField::metric("input_bytes", report.input_bytes)?,
                ProfileField::metric("codestream_bytes", report.codestream_bytes)?,
                ProfileField::metric("block_count", report.block_count)?,
                ProfileField::metric("dispatch_count", report.dispatch_count)?,
                ProfileField::label("backend", DebugValue(report.backend))?,
            ])
        },
    );
}

fn emit_cuda_profile_fields<const N: usize>(
    operation: &'static str,
    profile_operation: &'static str,
    path: &str,
    build: impl FnOnce() -> ProfileResult<[ProfileField; N]>,
) {
    let mode = profile_stage_mode();
    if mode == ProfileStageMode::Disabled {
        return;
    }
    let Some(fields) = build_profile_fields(operation, build) else {
        return;
    };
    j2k_profile::emit_profile_fields(
        mode,
        &PROFILE_SUMMARY,
        "j2k",
        profile_operation,
        path,
        &fields,
    );
}

pub(super) fn build_profile_fields<const N: usize>(
    operation: &'static str,
    build: impl FnOnce() -> ProfileResult<[ProfileField; N]>,
) -> Option<[ProfileField; N]> {
    match build() {
        Ok(fields) => Some(fields),
        Err(error) => {
            j2k_profile::emit_profile_error(operation, &error);
            None
        }
    }
}

struct DebugValue<T>(T);

impl<T: core::fmt::Debug> core::fmt::Display for DebugValue<T> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Debug::fmt(&self.0, formatter)
    }
}
