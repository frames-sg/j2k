// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    profile, BackendKind, CudaDeviceBuffer, CudaExecutionStats, CudaHtj2kColorDecodePlans,
    CudaHtj2kProfileReport, CudaSurfaceStats, PixelFormat, Storage, Surface, SurfaceResidency,
};

pub(super) struct FinalizeColorSurfaceRequest {
    pub(super) fmt: PixelFormat,
    pub(super) color: CudaHtj2kColorDecodePlans,
    pub(super) surface_buffer: CudaDeviceBuffer,
    pub(super) dispatches: usize,
    pub(super) decode_dispatches: usize,
    pub(super) store_stats: CudaExecutionStats,
    pub(super) store_us: u128,
    pub(super) wall_started: Option<profile::ProfileInstant>,
    pub(super) emit_report: bool,
}

pub(super) fn finalize_color_surface(
    request: FinalizeColorSurfaceRequest,
) -> (Surface, CudaHtj2kProfileReport) {
    let FinalizeColorSurfaceRequest {
        fmt,
        mut color,
        surface_buffer,
        mut dispatches,
        mut decode_dispatches,
        store_stats,
        store_us,
        wall_started,
        emit_report,
    } = request;
    dispatches = dispatches.saturating_add(store_stats.kernel_dispatches());
    decode_dispatches = decode_dispatches.saturating_add(store_stats.decode_kernel_dispatches());
    color.report.dispatch_count = dispatches;
    color.report.store_us = color.report.store_us.saturating_add(store_us);
    color.report.detail.store_dispatch_count = color
        .report
        .detail
        .store_dispatch_count
        .saturating_add(store_stats.kernel_dispatches());
    color.report.detail.wall_total_us = profile::elapsed_us(wall_started);
    profile::finalize_decode_total_us(&mut color.report);
    if emit_report {
        color.report.emit("decode");
    }
    let surface = Surface {
        backend: BackendKind::Cuda,
        residency: SurfaceResidency::CudaResidentDecode,
        dimensions: color.dimensions,
        fmt,
        pitch_bytes: color.dimensions.0 as usize * fmt.bytes_per_pixel(),
        stats: CudaSurfaceStats {
            total: dispatches,
            copy: 0,
            decode: decode_dispatches,
        },
        storage: Storage::Cuda(surface_buffer),
    };
    (surface, color.report)
}
