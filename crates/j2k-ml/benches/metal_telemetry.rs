// SPDX-License-Identifier: MIT OR Apache-2.0

use std::time::{Duration, Instant};

use j2k_metal::{MetalBatchDecoder, MetalBufferPoolsDiagnostics};
use j2k_ml::MetalBurnDecoder;

use super::Workload;

#[derive(Clone, Copy)]
pub(super) struct MetalTelemetryCase {
    route: &'static str,
    decode_mode: &'static str,
    input_mode: &'static str,
    workload: &'static str,
    request: &'static str,
    batch_size: usize,
    decoded_megapixels: f64,
    asserted_decoded_host_uploads: u64,
}

impl MetalTelemetryCase {
    pub(super) fn new(
        route: &'static str,
        decode_mode: &'static str,
        workload: &Workload,
        request: &'static str,
        batch_size: usize,
        output_pixels: u32,
        asserted_decoded_host_uploads: u64,
    ) -> Self {
        Self {
            route,
            decode_mode,
            input_mode: workload.input_mode.label(),
            workload: workload.name,
            request,
            batch_size,
            decoded_megapixels: f64::from(output_pixels)
                * f64::from(u32::try_from(batch_size).expect("benchmark batch size fits u32"))
                / 1_000_000.0,
            asserted_decoded_host_uploads,
        }
    }
}

#[derive(Clone, Copy)]
struct MetalTelemetrySnapshot {
    submissions: u64,
    pools: MetalBufferPoolsDiagnostics,
}

pub(super) struct MetalTelemetryRow {
    case: MetalTelemetryCase,
    elapsed: Duration,
    before: Option<MetalTelemetrySnapshot>,
    after: Option<MetalTelemetrySnapshot>,
    asserted_codec_group_waits: u64,
    asserted_consumer_host_synchronizations: u64,
}

fn metal_snapshot(decoder: &MetalBatchDecoder) -> MetalTelemetrySnapshot {
    MetalTelemetrySnapshot {
        submissions: decoder
            .submissions()
            .expect("Metal batch submission counter"),
        pools: decoder
            .backend_session()
            .buffer_pool_diagnostics()
            .expect("Metal telemetry buffer-pool diagnostics"),
    }
}

pub(super) fn capture_codec_telemetry<T, E>(
    telemetry: &mut Vec<MetalTelemetryRow>,
    case: MetalTelemetryCase,
    decoder: &mut MetalBatchDecoder,
    decode: impl FnOnce(&mut MetalBatchDecoder) -> Result<T, E>,
) where
    E: core::fmt::Debug,
{
    let before = metal_snapshot(decoder);
    let start = Instant::now();
    let completed = decode(decoder).expect("Metal codec telemetry decode");
    let elapsed = start.elapsed();
    drop(completed);
    let after = metal_snapshot(decoder);
    telemetry.push(MetalTelemetryRow {
        case,
        elapsed,
        before: Some(before),
        after: Some(after),
        asserted_codec_group_waits: 1,
        asserted_consumer_host_synchronizations: 0,
    });
}

pub(super) fn capture_burn_telemetry<T, E>(
    telemetry: &mut Vec<MetalTelemetryRow>,
    case: MetalTelemetryCase,
    decoder: &mut MetalBurnDecoder,
    decode: impl FnOnce(&mut MetalBurnDecoder) -> Result<T, E>,
) where
    E: core::fmt::Debug,
{
    let before = metal_snapshot(decoder.codec());
    let start = Instant::now();
    let completed = decode(decoder).expect("Metal Burn telemetry decode");
    let elapsed = start.elapsed();
    drop(completed);
    let after = metal_snapshot(decoder.codec());
    telemetry.push(MetalTelemetryRow {
        case,
        elapsed,
        before: Some(before),
        after: Some(after),
        asserted_codec_group_waits: 1,
        asserted_consumer_host_synchronizations: 0,
    });
}

pub(super) fn capture_staged_telemetry<T>(
    telemetry: &mut Vec<MetalTelemetryRow>,
    case: MetalTelemetryCase,
    decode: impl FnOnce() -> T,
) {
    let start = Instant::now();
    let completed = decode();
    let elapsed = start.elapsed();
    drop(completed);
    telemetry.push(MetalTelemetryRow {
        case,
        elapsed,
        before: None,
        after: None,
        asserted_codec_group_waits: 0,
        asserted_consumer_host_synchronizations: 1,
    });
}

pub(super) fn print_metal_telemetry(rows: &[MetalTelemetryRow]) {
    println!(
        "metal_telemetry_v2,route,decode_mode,input_mode,workload,request,batch_size,probe_images_per_s,\
         probe_megapixels_per_s,asserted_decoded_host_uploads,asserted_decoded_host_readbacks,\
         asserted_final_output_allocations,asserted_final_device_copies,measured_codec_group_launches,\
         asserted_codec_group_waits,asserted_consumer_host_syncs,private_cached_buffers_after,\
         shared_cached_buffers_after,private_cached_bytes_after,shared_cached_bytes_after,\
         session_pool_peak_cached_bytes_before,session_pool_peak_cached_bytes_after"
    );
    for row in rows {
        let elapsed_seconds = row.elapsed.as_secs_f64();
        let images_per_second =
            f64::from(u32::try_from(row.case.batch_size).expect("benchmark batch size fits u32"))
                / elapsed_seconds;
        let megapixels_per_second = row.case.decoded_megapixels / elapsed_seconds;
        let before = row.before.unwrap_or(MetalTelemetrySnapshot {
            submissions: 0,
            pools: MetalBufferPoolsDiagnostics::default(),
        });
        let after = row.after.unwrap_or(before);
        println!(
            concat!(
                "metal_telemetry_v2,{},{},{},{},{},{},",
                "{:.3},{:.3},{},{},{},{},{},{},{},",
                "{},{},{},{},{},{}"
            ),
            row.case.route,
            row.case.decode_mode,
            row.case.input_mode,
            row.case.workload,
            row.case.request,
            row.case.batch_size,
            images_per_second,
            megapixels_per_second,
            row.case.asserted_decoded_host_uploads,
            0,
            1,
            0,
            after.submissions.saturating_sub(before.submissions),
            row.asserted_codec_group_waits,
            row.asserted_consumer_host_synchronizations,
            after.pools.private.cached_buffers,
            after.pools.shared.cached_buffers,
            after.pools.private.cached_bytes,
            after.pools.shared.cached_bytes,
            before
                .pools
                .private
                .peak_cached_bytes
                .saturating_add(before.pools.shared.peak_cached_bytes),
            after
                .pools
                .private
                .peak_cached_bytes
                .saturating_add(after.pools.shared.peak_cached_bytes),
        );
    }
}
