// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{full_rgb8_reference, scaled_dimensions, scaled_rect, JPEG};
use crate::decoder::tile::{
    decode_tiles_into, decode_tiles_into_with_options, decode_tiles_region_scaled_into,
    decode_tiles_region_scaled_into_with_options, decode_tiles_scaled_into,
    decode_tiles_scaled_into_with_options,
};
use crate::{
    DecodeOptions, Decoder, Downscale, PixelFormat, Rect, TileBatchOptions, TileDecodeJob,
    TileRegionScaledDecodeJob, TileScaledDecodeJob,
};

#[test]
fn batch_facades_preserve_single_job_output_and_order() {
    let (expected, stride) = full_rgb8_reference();
    let options = TileBatchOptions::default();
    let mut default_output = vec![0u8; expected.len()];
    let mut explicit_output = vec![0u8; expected.len()];

    let default_outcomes = decode_tiles_into(
        &mut [TileDecodeJob {
            input: JPEG,
            out: &mut default_output,
            stride,
        }],
        PixelFormat::Rgb8,
        options,
    )
    .expect("default batch facade");
    let explicit_outcomes = decode_tiles_into_with_options(
        &mut [TileDecodeJob {
            input: JPEG,
            out: &mut explicit_output,
            stride,
        }],
        PixelFormat::Rgb8,
        DecodeOptions::default(),
        options,
    )
    .expect("explicit batch facade");

    assert_eq!(default_output, expected);
    assert_eq!(explicit_output, expected);
    assert_eq!(default_outcomes.len(), 1);
    assert_eq!(explicit_outcomes, default_outcomes);
}

#[test]
fn scaled_batch_facades_preserve_single_job_output_and_order() {
    let scale = Downscale::Eighth;
    let dimensions = scaled_dimensions(scale);
    let stride = dimensions.0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let mut expected = vec![0u8; stride * dimensions.1 as usize];
    Decoder::new(JPEG)
        .expect("fixture decoder")
        .decode_scaled_into(&mut expected, stride, PixelFormat::Rgb8, scale)
        .expect("reference scaled decode");
    let options = TileBatchOptions::default();
    let mut default_output = vec![0u8; expected.len()];
    let mut explicit_output = vec![0u8; expected.len()];

    let default_outcomes = decode_tiles_scaled_into(
        &mut [TileScaledDecodeJob {
            input: JPEG,
            out: &mut default_output,
            stride,
            scale,
        }],
        PixelFormat::Rgb8,
        options,
    )
    .expect("default scaled batch facade");
    let explicit_outcomes = decode_tiles_scaled_into_with_options(
        &mut [TileScaledDecodeJob {
            input: JPEG,
            out: &mut explicit_output,
            stride,
            scale,
        }],
        PixelFormat::Rgb8,
        DecodeOptions::default(),
        options,
    )
    .expect("explicit scaled batch facade");

    assert_eq!(default_output, expected);
    assert_eq!(explicit_output, expected);
    assert_eq!(default_outcomes.len(), 1);
    assert_eq!(explicit_outcomes, default_outcomes);
}

#[test]
fn region_scaled_batch_facades_preserve_single_job_output_and_order() {
    let roi = Rect {
        x: 3,
        y: 4,
        w: 9,
        h: 7,
    };
    let scale = Downscale::Quarter;
    let output_rect = scaled_rect(roi, scale);
    let stride = output_rect.w as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let mut expected = vec![0u8; stride * output_rect.h as usize];
    Decoder::new(JPEG)
        .expect("fixture decoder")
        .decode_region_scaled_into(&mut expected, stride, PixelFormat::Rgb8, roi, scale)
        .expect("reference region-scaled decode");
    let options = TileBatchOptions::default();
    let mut default_output = vec![0u8; expected.len()];
    let mut explicit_output = vec![0u8; expected.len()];

    let default_outcomes = decode_tiles_region_scaled_into(
        &mut [TileRegionScaledDecodeJob {
            input: JPEG,
            out: &mut default_output,
            stride,
            roi: roi.into(),
            scale,
        }],
        PixelFormat::Rgb8,
        options,
    )
    .expect("default region-scaled batch facade");
    let explicit_outcomes = decode_tiles_region_scaled_into_with_options(
        &mut [TileRegionScaledDecodeJob {
            input: JPEG,
            out: &mut explicit_output,
            stride,
            roi: roi.into(),
            scale,
        }],
        PixelFormat::Rgb8,
        DecodeOptions::default(),
        options,
    )
    .expect("explicit region-scaled batch facade");

    assert_eq!(default_output, expected);
    assert_eq!(explicit_output, expected);
    assert_eq!(default_outcomes.len(), 1);
    assert_eq!(explicit_outcomes, default_outcomes);
}
