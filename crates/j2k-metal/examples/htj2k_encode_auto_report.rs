// SPDX-License-Identifier: MIT OR Apache-2.0

//! Encode generated RGB samples as lossless HTJ2K with the conservative Metal
//! Auto host-output route and print the final backend plus stage dispatches.
//!
//! Run with:
//! `cargo run -p j2k-metal --example htj2k_encode_auto_report`

use j2k::{
    encode_j2k_lossless_with_accelerator, BackendKind, J2kBlockCodingMode, J2kEncodeValidation,
    J2kLosslessEncodeOptions, J2kLosslessSamples,
};
use j2k_metal::MetalEncodeStageAccelerator;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let width = 768_u32;
    let height = 512_u32;
    let pixels = generated_rgb8(width, height);
    let samples = J2kLosslessSamples::new(&pixels, width, height, 3, 8, false)?;
    let options = J2kLosslessEncodeOptions::default()
        .with_accelerated_backend()
        .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
        .with_validation(J2kEncodeValidation::CpuRoundTrip);
    let mut accelerator = MetalEncodeStageAccelerator::for_auto_host_output();

    let encoded = encode_j2k_lossless_with_accelerator(
        samples,
        &options,
        BackendKind::Metal,
        &mut accelerator,
    )?;
    let dispatch = encoded.dispatch_report;

    println!("encoded_htj2k_bytes={}", encoded.codestream.len());
    println!("final_backend={:?}", encoded.backend);
    println!("stage_dispatch_total={}", dispatch.total());
    println!(
        "stage_dispatches deinterleave={} rct={} dwt53={} ht_code_block={} packetization={}",
        dispatch.deinterleave,
        dispatch.forward_rct,
        dispatch.forward_dwt53,
        dispatch.ht_code_block,
        dispatch.packetization
    );
    if encoded.backend == BackendKind::Cpu && dispatch.any() {
        println!(
            "route_note=Metal stages dispatched, but the final host-output encode contract stayed CPU because not every required stage was device-backed"
        );
    } else if encoded.backend == BackendKind::Metal {
        println!("route_note=Auto selected a fully device-backed encode contract");
    } else {
        println!("route_note=Auto stayed on CPU for this shape or host");
    }

    Ok(())
}

fn generated_rgb8(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 3 + y * 5) & 0xff) as u8);
            pixels.push(((x * 7 + y * 11) & 0xff) as u8);
            pixels.push(((x * 13 + y * 17) & 0xff) as u8);
        }
    }
    pixels
}
