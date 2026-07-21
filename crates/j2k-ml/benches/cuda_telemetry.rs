// SPDX-License-Identifier: MIT OR Apache-2.0

use std::time::{Duration, Instant};

use burn_core::tensor::backend::Backend;
use burn_cuda::Cuda;
use j2k_cuda::{CudaBatchDecoder, CudaSessionDiagnostics};
use j2k_ml::CudaBurnDecoder;

use super::Workload;

#[derive(Clone, Copy)]
pub(super) struct CudaTelemetryCase {
    route: &'static str,
    decode_mode: &'static str,
    input_mode: &'static str,
    workload: &'static str,
    request: &'static str,
    batch_size: u32,
    decoded_megapixels: f64,
}

impl CudaTelemetryCase {
    pub(super) fn new(
        route: &'static str,
        decode_mode: &'static str,
        workload: &Workload,
        request: &'static str,
        batch_size: usize,
        output_pixels: u32,
    ) -> Self {
        let batch_size = u32::try_from(batch_size).expect("benchmark batch size fits u32");
        Self {
            route,
            decode_mode,
            input_mode: workload.input_mode.label(),
            workload: workload.name,
            request,
            batch_size,
            decoded_megapixels: f64::from(output_pixels) * f64::from(batch_size) / 1_000_000.0,
        }
    }
}

pub(super) struct CudaTelemetryRow {
    case: CudaTelemetryCase,
    elapsed: Duration,
    before: CudaSessionDiagnostics,
    after: CudaSessionDiagnostics,
    consumer_host_synchronizations: u64,
}

pub(super) fn capture_codec_telemetry<T, E>(
    telemetry: &mut Vec<CudaTelemetryRow>,
    case: CudaTelemetryCase,
    decoder: &mut CudaBatchDecoder,
    decode: impl FnOnce(&mut CudaBatchDecoder) -> Result<T, E>,
) where
    E: core::fmt::Debug,
{
    let before = decoder.diagnostics().expect("pre-decode CUDA diagnostics");
    let start = Instant::now();
    let completed_batch = decode(decoder).expect("CUDA codec telemetry decode");
    let elapsed = start.elapsed();
    drop(completed_batch);
    let after = decoder.diagnostics().expect("post-decode CUDA diagnostics");
    telemetry.push(CudaTelemetryRow {
        case,
        elapsed,
        before,
        after,
        consumer_host_synchronizations: 0,
    });
}

pub(super) fn capture_burn_telemetry<T, E>(
    telemetry: &mut Vec<CudaTelemetryRow>,
    case: CudaTelemetryCase,
    decoder: &mut CudaBurnDecoder,
    decode: impl FnOnce(&mut CudaBurnDecoder) -> Result<T, E>,
) where
    E: core::fmt::Debug,
{
    let before = decoder
        .codec()
        .diagnostics()
        .expect("pre-decode CUDA Burn diagnostics");
    let start = Instant::now();
    let completed_batch = decode(decoder).expect("CUDA Burn telemetry decode");
    <Cuda as Backend>::sync(decoder.device()).expect("synchronize CUDA Burn telemetry completion");
    let elapsed = start.elapsed();
    drop(completed_batch);
    let after = decoder
        .codec()
        .diagnostics()
        .expect("post-decode CUDA Burn diagnostics");
    telemetry.push(CudaTelemetryRow {
        case,
        elapsed,
        before,
        after,
        consumer_host_synchronizations: 1,
    });
}

pub(super) fn print_cuda_telemetry(rows: &[CudaTelemetryRow]) {
    println!(
        "cuda_telemetry_v2,route,decode_mode,input_mode,workload,request,batch_size,probe_images_per_s,\
         probe_megapixels_per_s,h2d_ops,h2d_bytes,d2h_ops,d2h_bytes,status_d2h_ops,\
         status_d2h_bytes,codec_kernel_launches,runtime_device_allocations,\
         runtime_device_allocation_bytes,runtime_live_device_allocations_after,\
         runtime_live_device_bytes_after,session_peak_live_device_allocations_before,\
         session_peak_live_device_allocations_after,session_peak_live_device_bytes_before,\
         session_peak_live_device_bytes_after,codec_pool_retained_bytes_after,\
         session_pool_peak_retained_bytes_before,session_pool_peak_retained_bytes_after,\
         event_driver_allocations,event_reuses,\
         codec_event_host_syncs,codec_context_host_syncs,consumer_host_syncs"
    );
    for row in rows {
        let before = row.before.runtime.unwrap_or_default();
        let after = row
            .after
            .runtime
            .expect("a completed CUDA telemetry decode initializes the runtime");
        let elapsed_seconds = row.elapsed.as_secs_f64();
        let images_per_second = f64::from(row.case.batch_size) / elapsed_seconds;
        let megapixels_per_second = row.case.decoded_megapixels / elapsed_seconds;
        println!(
            concat!(
                "cuda_telemetry_v2,{},{},{},{},{},{},",
                "{:.3},{:.3},",
                "{},{},{},{},{},{},{},{},{},{},",
                "{},{},{},{},{},{},{},{},{},{},",
                "{},{},{}"
            ),
            row.case.route,
            row.case.decode_mode,
            row.case.input_mode,
            row.case.workload,
            row.case.request,
            row.case.batch_size,
            images_per_second,
            megapixels_per_second,
            delta(
                before.host_to_device_operations,
                after.host_to_device_operations
            ),
            delta(before.host_to_device_bytes, after.host_to_device_bytes),
            delta(
                before.device_to_host_operations,
                after.device_to_host_operations
            ),
            delta(before.device_to_host_bytes, after.device_to_host_bytes),
            delta(
                before.status_device_to_host_operations,
                after.status_device_to_host_operations
            ),
            delta(
                before.status_device_to_host_bytes,
                after.status_device_to_host_bytes
            ),
            delta(before.kernel_launches, after.kernel_launches),
            delta(
                before.device_allocation_operations,
                after.device_allocation_operations
            ),
            delta(
                before.device_allocation_bytes,
                after.device_allocation_bytes
            ),
            after.live_device_allocations,
            after.live_device_bytes,
            before.peak_live_device_allocations,
            after.peak_live_device_allocations,
            before.peak_live_device_bytes,
            after.peak_live_device_bytes,
            row.after.pools.retained_bytes(),
            row.before.pools.peak_retained_bytes_upper_bound(),
            row.after.pools.peak_retained_bytes_upper_bound(),
            delta(
                before.event_driver_allocations,
                after.event_driver_allocations
            ),
            delta(before.event_reuses, after.event_reuses),
            delta(
                before.event_host_synchronizations,
                after.event_host_synchronizations
            ),
            delta(
                before.context_host_synchronizations,
                after.context_host_synchronizations
            ),
            row.consumer_host_synchronizations,
        );
    }
}

const fn delta(before: u64, after: u64) -> u64 {
    after.saturating_sub(before)
}
