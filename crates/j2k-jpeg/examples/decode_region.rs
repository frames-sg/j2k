// SPDX-License-Identifier: Apache-2.0

//! Decode a source-coordinate ROI from a JPEG tile.
//!
//! Run with:
//! `cargo run -p j2k-jpeg --example decode_region`

use j2k_jpeg::{Decoder, PixelFormat, Rect};
use j2k_test_support::JPEG_BASELINE_420_16X16;

const TILE: &[u8] = JPEG_BASELINE_420_16X16;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let decoder = Decoder::new(TILE)?;
    let roi = Rect {
        x: 4,
        y: 4,
        w: 8,
        h: 8,
    };
    let (rgb, outcome) = decoder.decode_region(PixelFormat::Rgb8, roi)?;

    println!(
        "decoded {}x{} ROI into {} RGB bytes",
        outcome.decoded.w,
        outcome.decoded.h,
        rgb.len()
    );
    Ok(())
}
