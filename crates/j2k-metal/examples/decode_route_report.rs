// SPDX-License-Identifier: MIT OR Apache-2.0

//! Decode a generated HTJ2K codestream through the Metal adapter route-report
//! API and print Auto fallback plus strict Metal behavior.
//!
//! Run with:
//! `cargo run -p j2k-metal --example decode_route_report`

use j2k::{BackendRequest, Downscale, PixelFormat, Rect};
use j2k_core::DeviceSurface;
use j2k_metal::{J2kDecoder, MetalDecodeRequest};
use j2k_native::{encode_htj2k, EncodeOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let width = 16_u32;
    let height = 16_u32;
    let pixels: Vec<u8> = (0..width * height)
        .map(|value| ((value * 17 + value / 3) & 0xff) as u8)
        .collect();
    let codestream = encode_htj2k(
        &pixels,
        width,
        height,
        1,
        8,
        false,
        &EncodeOptions::default(),
    )?;
    let roi = Rect {
        x: 4,
        y: 4,
        w: 8,
        h: 8,
    };
    let scale = Downscale::Half;

    let mut auto_decoder = J2kDecoder::new(&codestream)?;
    let auto = auto_decoder.decode_request_to_device_with_report(
        MetalDecodeRequest::region_scaled(PixelFormat::Gray8, roi, scale, BackendRequest::Auto),
    )?;
    println!("auto_selected_backend={:?}", auto.report.selected_backend);
    println!("auto_residency={:?}", auto.report.surface_residency);
    println!(
        "auto_fallback_reason={}",
        auto.report.fallback_reason.unwrap_or("none")
    );
    println!("auto_output_bytes={}", auto.surface.byte_len());

    let mut strict_decoder = J2kDecoder::new(&codestream)?;
    match strict_decoder.decode_request_to_device_with_report(MetalDecodeRequest::region_scaled(
        PixelFormat::Gray8,
        roi,
        scale,
        BackendRequest::Metal,
    )) {
        Ok(strict) => {
            println!(
                "strict_metal_selected_backend={:?}",
                strict.report.selected_backend
            );
            println!(
                "strict_metal_residency={:?}",
                strict.report.surface_residency
            );
            println!("strict_metal_output_bytes={}", strict.surface.byte_len());
        }
        Err(error) => {
            println!("strict_metal_error={error}");
        }
    }

    Ok(())
}
