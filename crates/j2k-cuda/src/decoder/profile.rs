// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "cuda-runtime")]
use super::{
    CudaBufferPoolTakeTrace, CudaHtj2kProfileReport, SurfaceResidency, CUDA_IDWT_TRACE_ENV_VAR,
};

#[cfg(feature = "cuda-runtime")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct CudaIdwtBatchHostTraceRow {
    pub(super) component_count: usize,
    pub(super) step_count: usize,
    pub(super) output_alloc_us: u128,
    pub(super) target_build_us: u128,
    pub(super) enqueue_us: u128,
    pub(super) output_take_count: usize,
    pub(super) output_pool_reuse_count: usize,
    pub(super) output_pool_alloc_count: usize,
    pub(super) output_pool_scanned_count: usize,
    pub(super) output_pool_max_free_count: usize,
    pub(super) output_requested_bytes: usize,
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn format_cuda_idwt_batch_host_trace_row(
    row: CudaIdwtBatchHostTraceRow,
) -> j2k_profile::ProfileResult<String> {
    j2k_profile::format_profile_row_u128(
        "j2k",
        "cuda_idwt_batch_host",
        "decode",
        &[
            ("component_count", row.component_count as u128),
            ("step_count", row.step_count as u128),
            ("output_alloc_us", row.output_alloc_us),
            ("target_build_us", row.target_build_us),
            ("enqueue_us", row.enqueue_us),
            ("output_take_count", row.output_take_count as u128),
            (
                "output_pool_reuse_count",
                row.output_pool_reuse_count as u128,
            ),
            (
                "output_pool_alloc_count",
                row.output_pool_alloc_count as u128,
            ),
            (
                "output_pool_scanned_count",
                row.output_pool_scanned_count as u128,
            ),
            (
                "output_pool_max_free_count",
                row.output_pool_max_free_count as u128,
            ),
            ("output_requested_bytes", row.output_requested_bytes as u128),
        ],
    )
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn emit_cuda_idwt_batch_host_trace_row(row: CudaIdwtBatchHostTraceRow) {
    match format_cuda_idwt_batch_host_trace_row(row) {
        Ok(row) => j2k_profile::emit_profile_line(row),
        Err(error) => j2k_profile::emit_profile_error("cuda_idwt_batch_host_trace", &error),
    }
}

#[cfg(feature = "cuda-runtime")]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct CudaIdwtOutputPoolTraceTotals {
    pub(super) take_count: usize,
    pub(super) reuse_count: usize,
    pub(super) alloc_count: usize,
    pub(super) scanned_count: usize,
    pub(super) max_free_count: usize,
    pub(super) requested_bytes: usize,
}

#[cfg(feature = "cuda-runtime")]
impl CudaIdwtOutputPoolTraceTotals {
    pub(super) fn add_take(&mut self, trace: CudaBufferPoolTakeTrace) {
        self.take_count = self.take_count.saturating_add(1);
        if trace.reused {
            self.reuse_count = self.reuse_count.saturating_add(1);
        } else {
            self.alloc_count = self.alloc_count.saturating_add(1);
        }
        self.scanned_count = self.scanned_count.saturating_add(trace.scanned_count);
        self.max_free_count = self.max_free_count.max(trace.free_count_before);
        self.requested_bytes = self.requested_bytes.saturating_add(trace.requested_len);
    }
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn cuda_idwt_trace_enabled() -> bool {
    std::env::var_os(CUDA_IDWT_TRACE_ENV_VAR).is_some()
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn elapsed_host_us(start: Option<std::time::Instant>) -> u128 {
    start.map_or(0, |start| start.elapsed().as_micros())
}

#[cfg(feature = "cuda-runtime")]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct CudaDecodeStageTimings {
    pub(super) h2d: u128,
    pub(super) table_upload: u128,
    pub(super) payload_upload: u128,
    pub(super) job_upload: u128,
    pub(super) status_d2h: u128,
    pub(super) ht_cleanup: u128,
    pub(super) ht_refine: u128,
    pub(super) classic_tier1: u128,
    pub(super) dequant: u128,
    pub(super) ht_dispatch_count: usize,
    pub(super) classic_dispatch_count: usize,
    pub(super) idwt: u128,
    pub(super) dequant_dispatch_count: usize,
    pub(super) idwt_dispatch_count: usize,
}

#[cfg(feature = "cuda-runtime")]
impl CudaDecodeStageTimings {
    pub(super) fn add_to_report(self, report: &mut CudaHtj2kProfileReport) {
        report.h2d_us = report.h2d_us.saturating_add(self.h2d);
        report.detail.payload_upload_us = report
            .detail
            .payload_upload_us
            .saturating_add(self.payload_upload);
        report.detail.table_upload_us = report
            .detail
            .table_upload_us
            .saturating_add(self.table_upload);
        report.detail.job_upload_us = report.detail.job_upload_us.saturating_add(self.job_upload);
        report.detail.status_d2h_us = report.detail.status_d2h_us.saturating_add(self.status_d2h);
        report.ht_cleanup_us = report.ht_cleanup_us.saturating_add(self.ht_cleanup);
        report.ht_refine_us = report.ht_refine_us.saturating_add(self.ht_refine);
        report.classic_tier1_us = report.classic_tier1_us.saturating_add(self.classic_tier1);
        report.dequant_us = report.dequant_us.saturating_add(self.dequant);
        report.idwt_us = report.idwt_us.saturating_add(self.idwt);
        report.detail.ht_dispatch_count = report
            .detail
            .ht_dispatch_count
            .saturating_add(self.ht_dispatch_count);
        report.detail.classic_dispatch_count = report
            .detail
            .classic_dispatch_count
            .saturating_add(self.classic_dispatch_count);
        report.detail.dequant_dispatch_count = report
            .detail
            .dequant_dispatch_count
            .saturating_add(self.dequant_dispatch_count);
        report.detail.idwt_dispatch_count = report
            .detail
            .idwt_dispatch_count
            .saturating_add(self.idwt_dispatch_count);
    }
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn aggregate_decode_reports(
    reports: &[CudaHtj2kProfileReport],
) -> CudaHtj2kProfileReport {
    let mut aggregate = CudaHtj2kProfileReport {
        residency: SurfaceResidency::CudaResidentDecode,
        ..CudaHtj2kProfileReport::default()
    };
    for report in reports {
        add_decode_report(&mut aggregate, report);
    }
    aggregate
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn add_decode_report(
    aggregate: &mut CudaHtj2kProfileReport,
    report: &CudaHtj2kProfileReport,
) {
    aggregate.parse_us = aggregate.parse_us.saturating_add(report.parse_us);
    aggregate.plan_us = aggregate.plan_us.saturating_add(report.plan_us);
    aggregate.flatten_us = aggregate.flatten_us.saturating_add(report.flatten_us);
    aggregate.h2d_us = aggregate.h2d_us.saturating_add(report.h2d_us);
    aggregate.ht_cleanup_us = aggregate.ht_cleanup_us.saturating_add(report.ht_cleanup_us);
    aggregate.ht_refine_us = aggregate.ht_refine_us.saturating_add(report.ht_refine_us);
    aggregate.classic_tier1_us = aggregate
        .classic_tier1_us
        .saturating_add(report.classic_tier1_us);
    aggregate.dequant_us = aggregate.dequant_us.saturating_add(report.dequant_us);
    aggregate.idwt_us = aggregate.idwt_us.saturating_add(report.idwt_us);
    aggregate.mct_us = aggregate.mct_us.saturating_add(report.mct_us);
    aggregate.store_us = aggregate.store_us.saturating_add(report.store_us);
    aggregate.block_count = aggregate.block_count.saturating_add(report.block_count);
    aggregate.classic_block_count = aggregate
        .classic_block_count
        .saturating_add(report.classic_block_count);
    aggregate.ht_block_count = aggregate
        .ht_block_count
        .saturating_add(report.ht_block_count);
    aggregate.payload_bytes = aggregate.payload_bytes.saturating_add(report.payload_bytes);
    aggregate.dispatch_count = aggregate
        .dispatch_count
        .saturating_add(report.dispatch_count);
    aggregate.detail.table_upload_us = aggregate
        .detail
        .table_upload_us
        .saturating_add(report.detail.table_upload_us);
    aggregate.detail.payload_upload_us = aggregate
        .detail
        .payload_upload_us
        .saturating_add(report.detail.payload_upload_us);
    aggregate.detail.job_upload_us = aggregate
        .detail
        .job_upload_us
        .saturating_add(report.detail.job_upload_us);
    aggregate.detail.status_d2h_us = aggregate
        .detail
        .status_d2h_us
        .saturating_add(report.detail.status_d2h_us);
    aggregate.detail.output_d2h_us = aggregate
        .detail
        .output_d2h_us
        .saturating_add(report.detail.output_d2h_us);
    aggregate.detail.ht_dispatch_count = aggregate
        .detail
        .ht_dispatch_count
        .saturating_add(report.detail.ht_dispatch_count);
    aggregate.detail.classic_dispatch_count = aggregate
        .detail
        .classic_dispatch_count
        .saturating_add(report.detail.classic_dispatch_count);
    aggregate.detail.dequant_dispatch_count = aggregate
        .detail
        .dequant_dispatch_count
        .saturating_add(report.detail.dequant_dispatch_count);
    aggregate.detail.idwt_dispatch_count = aggregate
        .detail
        .idwt_dispatch_count
        .saturating_add(report.detail.idwt_dispatch_count);
    aggregate.detail.mct_dispatch_count = aggregate
        .detail
        .mct_dispatch_count
        .saturating_add(report.detail.mct_dispatch_count);
    aggregate.detail.store_dispatch_count = aggregate
        .detail
        .store_dispatch_count
        .saturating_add(report.detail.store_dispatch_count);
}
