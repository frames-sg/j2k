// SPDX-License-Identifier: MIT OR Apache-2.0

//! Run a JPEG-to-HTJ2K transcode through the Metal route facade.
//!
//! Run with:
//! `cargo run -p j2k-transcode-metal --example jpeg_to_htj2k_route_report`

use j2k_core::BackendRequest;
use j2k_test_support::JPEG_GRAYSCALE_8X8;
use j2k_transcode::JpegToHtj2kOptions;
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
    print!("{}", routed.route.pipeline_map.debug_report());
    Ok(())
}
