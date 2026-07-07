// SPDX-License-Identifier: MIT OR Apache-2.0

//! Run a JPEG-to-HTJ2K transcode through the Metal route facade.
//!
//! Run with:
//! `cargo run -p j2k-transcode-metal --example jpeg_to_htj2k_route_report`

use j2k_core::BackendRequest;
use j2k_test_support::JPEG_GRAYSCALE_8X8;
use j2k_transcode::{JpegToHtj2kOptions, TranscodePipelineMap};
use j2k_transcode_metal::jpeg_to_htj2k_with_metal_route;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let routed = jpeg_to_htj2k_with_metal_route(
        JPEG_GRAYSCALE_8X8,
        &JpegToHtj2kOptions::lossless_53(),
        BackendRequest::Auto,
    )?;
    let fallback = routed
        .route
        .fallback_reason
        .map_or("none", |reason| reason.as_str());
    let transfer_bytes = routed
        .route
        .pipeline_map
        .stages
        .iter()
        .fold(0_u64, |total, stage| {
            total.saturating_add(stage.transfer_bytes)
        });

    println!(
        "request={:?} selected_transform={:?} output_backend={:?} fallback={} codestream_bytes={} transfer_bytes={}",
        routed.route.request,
        routed.route.selected_transform_backend,
        routed.route.output_backend,
        fallback,
        routed.encoded.codestream.len(),
        transfer_bytes
    );
    print_pipeline_map(&routed.route.pipeline_map);
    Ok(())
}

fn print_pipeline_map(map: &TranscodePipelineMap) {
    println!("jpeg_to_htj2k_pipeline_map");
    for stage in &map.stages {
        println!(
            "stage={} processor={} cpu_us={} metal_us={} transfer_us={} transfer_count={} transfer_bytes={} resident_handoffs={} dispatches={} fallback_jobs={} note={}",
            stage.stage,
            stage.processor,
            stage.cpu_us,
            stage.metal_us,
            stage.transfer_us,
            stage.transfer_count,
            stage.transfer_bytes,
            stage.resident_handoff_count,
            stage.dispatches,
            stage.fallback_jobs,
            stage.note,
        );
    }
    println!(
        "recommend_next_stage={} evidence_us={} evidence_dispatches={} reason={}",
        map.recommendation.stage,
        map.recommendation.evidence_us,
        map.recommendation.evidence_dispatches,
        map.recommendation.reason,
    );
}
