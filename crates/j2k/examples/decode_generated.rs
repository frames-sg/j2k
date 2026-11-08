// SPDX-License-Identifier: MIT OR Apache-2.0

//! Inspect, plan, allocate, and decode an exact-reduction region through the
//! public API, using a generated HTJ2K source so no external fixture is needed.
//!
//! Run with:
//! `cargo run -p j2k --example decode_generated`

use j2k::{J2kDecodeControl, J2kDecoder, PixelFormat, Rect};
use j2k_native::{encode_htj2k, EncodeOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let width = 16_u32;
    let height = 16_u32;
    let pixels: Vec<u8> = (0..width * height)
        .map(|value| u8::try_from(value).expect("16x16 fixture index fits u8"))
        .collect();
    let bytes = encode_htj2k(
        &pixels,
        width,
        height,
        1,
        8,
        false,
        &EncodeOptions {
            num_decomposition_levels: 3,
            ..EncodeOptions::default()
        },
    )?;

    let mut decoder = J2kDecoder::new(&bytes)?;
    let source_dimensions = decoder.info().dimensions;
    let available_reduction = decoder.max_reduction_levels()?;
    // Coordinates stay relative to the full-resolution source. The plan owns
    // the rounding and reference-grid origin math at the exact requested level.
    let roi = Rect {
        x: 3,
        y: 5,
        w: 9,
        h: 7,
    };
    let plan = decoder.region_decode_plan(PixelFormat::Gray8, roi, 1)?;
    let mut region = vec![0_u8; plan.buffer_bytes];
    let mut control = J2kDecodeControl::default();
    control.max_output_bytes = Some(plan.buffer_bytes);
    let outcome = decoder.decode_region_scaled_pow2_into_with_control(
        &mut region,
        plan.row_bytes,
        plan.format,
        roi,
        plan.reduction_levels,
        &control,
    )?;
    assert_eq!(outcome.decoded, plan.output_region);

    println!(
        "source {source_dimensions:?}, reductions 0..={available_reduction}; decoded {:?} into {} bytes ({:?})",
        outcome.decoded,
        region.len(),
        plan.decode.route,
    );
    Ok(())
}
