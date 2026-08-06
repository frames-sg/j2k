// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{DecodeSettings, J2kDecodeWarning, J2kDecoder, J2kError, J2kScratchPool};
use j2k_core::{Downscale, PixelFormat, Rect};
use j2k_native::{encode, EncodeOptions};
use j2k_test_support::wrap_jp2_codestream;

fn encode_rgb_fixture(
    width: u32,
    height: u32,
    decomposition_levels: u8,
    tile_size: Option<(u32, u32)>,
) -> Vec<u8> {
    let pixels = (0..height)
        .flat_map(|y| {
            (0..width).flat_map(move |x| [(x % 251) as u8, (y % 241) as u8, ((x + y) % 233) as u8])
        })
        .collect::<Vec<_>>();
    encode(
        &pixels,
        width,
        height,
        3,
        8,
        false,
        &EncodeOptions {
            reversible: true,
            num_decomposition_levels: decomposition_levels,
            tile_size,
            ..EncodeOptions::default()
        },
    )
    .expect("encode RGB fixture")
}

fn scaled_covering_pow2(rect: Rect, levels: u8) -> Rect {
    let denominator = 1_u32
        .checked_shl(u32::from(levels))
        .expect("test reduction is representable");
    let x_end = rect.x.saturating_add(rect.w);
    let y_end = rect.y.saturating_add(rect.h);
    let x0 = rect.x / denominator;
    let y0 = rect.y / denominator;
    let x1 = x_end.div_ceil(denominator);
    let y1 = y_end.div_ceil(denominator);
    Rect {
        x: x0,
        y: y0,
        w: x1.saturating_sub(x0),
        h: y1.saturating_sub(y0),
    }
}

fn crop_interleaved(
    pixels: &[u8],
    source_width: u32,
    bytes_per_pixel: usize,
    rect: Rect,
) -> Vec<u8> {
    let source_stride = source_width as usize * bytes_per_pixel;
    let row_bytes = rect.w as usize * bytes_per_pixel;
    let mut cropped = Vec::with_capacity(row_bytes * rect.h as usize);
    for row in rect.y as usize..(rect.y + rect.h) as usize {
        let start = row * source_stride + rect.x as usize * bytes_per_pixel;
        cropped.extend_from_slice(&pixels[start..start + row_bytes]);
    }
    cropped
}

fn decode_reduced_region(bytes: &[u8], roi: Rect, levels: u8) -> Result<(Vec<u8>, Rect), J2kError> {
    let format = PixelFormat::Rgb8;
    let scaled = scaled_covering_pow2(roi, levels);
    let stride = scaled.w as usize * format.bytes_per_pixel();
    let mut output = vec![0_u8; stride * scaled.h as usize];
    let mut decoder = J2kDecoder::new(bytes)?;
    let outcome = decoder.decode_region_scaled_pow2_into(
        &mut J2kScratchPool::new(),
        &mut output,
        stride,
        format,
        roi,
        levels,
    )?;
    assert_eq!(outcome.decoded, scaled);
    Ok((output, scaled))
}

#[test]
fn deep_scaled_region_matches_eighth_and_tiled_full_image_crops() {
    let (width, height) = (640_u32, 512_u32);
    let bytes = encode_rgb_fixture(width, height, 5, Some((256, 256)));
    let full = Rect {
        x: 0,
        y: 0,
        w: width,
        h: height,
    };
    let roi = Rect {
        x: 96,
        y: 64,
        w: 320,
        h: 288,
    };

    let (pow2_full, full_resolution_roi) =
        decode_reduced_region(&bytes, roi, 0).expect("zero-level decode");
    let format = PixelFormat::Rgb8;
    let full_stride = full_resolution_roi.w as usize * format.bytes_per_pixel();
    let mut ordinary_output = vec![0_u8; full_stride * full_resolution_roi.h as usize];
    J2kDecoder::new(&bytes)
        .expect("ordinary decoder")
        .decode_region_into(
            &mut J2kScratchPool::new(),
            &mut ordinary_output,
            full_stride,
            format,
            roi,
        )
        .expect("ordinary region decode");
    assert_eq!(pow2_full, ordinary_output);

    let (pow2_eighth, scaled_eighth) =
        decode_reduced_region(&bytes, roi, 3).expect("1/8 power-of-two decode");
    let enum_stride = scaled_eighth.w as usize * format.bytes_per_pixel();
    let mut enum_output = vec![0_u8; enum_stride * scaled_eighth.h as usize];
    J2kDecoder::new(&bytes)
        .expect("enum decoder")
        .decode_region_scaled_into(
            &mut J2kScratchPool::new(),
            &mut enum_output,
            enum_stride,
            format,
            roi,
            Downscale::Eighth,
        )
        .expect("1/8 enum decode");
    assert_eq!(pow2_eighth, enum_output);

    for levels in [4_u8, 5] {
        let (whole, whole_rect) =
            decode_reduced_region(&bytes, full, levels).expect("whole-image deep decode");
        let (region, region_rect) =
            decode_reduced_region(&bytes, roi, levels).expect("region deep decode");
        assert_eq!(
            region,
            crop_interleaved(&whole, whole_rect.w, format.bytes_per_pixel(), region_rect,),
            "1/{} region disagrees with the full-image crop",
            1_u32 << levels,
        );
    }
}

#[test]
fn deep_scaled_decode_honors_exact_level_for_a_skinny_image() {
    let (width, height) = (17_u32, 257_u32);
    let bytes = encode_rgb_fixture(width, height, 4, None);
    let full = Rect {
        x: 0,
        y: 0,
        w: width,
        h: height,
    };

    let (output, decoded) =
        decode_reduced_region(&bytes, full, 4).expect("exact 1/16 skinny decode");

    assert_eq!(
        decoded,
        Rect {
            x: 0,
            y: 0,
            w: 2,
            h: 17
        }
    );
    assert_eq!(output.len(), 34 * PixelFormat::Rgb8.bytes_per_pixel());
}

#[test]
fn deep_scaled_decode_rejects_ladder_and_shift_overflow() {
    let (width, height) = (64_u32, 64_u32);
    let bytes = encode_rgb_fixture(width, height, 5, None);
    let full = Rect {
        x: 0,
        y: 0,
        w: width,
        h: height,
    };
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let mut output = vec![0_u8; width as usize * height as usize * 3];

    for levels in [6_u8, 32] {
        let error = decoder
            .decode_region_scaled_pow2_into(
                &mut J2kScratchPool::new(),
                &mut output,
                width as usize * 3,
                PixelFormat::Rgb8,
                full,
                levels,
            )
            .expect_err("invalid reduction must be rejected");
        assert!(matches!(error, J2kError::Unsupported(_)));
    }
}

#[test]
fn deep_scaled_decode_preserves_lenient_recovery_warning() {
    let pixels = [3_u8, 9, 27, 81];
    let codestream = encode(
        &pixels,
        2,
        2,
        1,
        8,
        false,
        &EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        },
    )
    .expect("encode grayscale fixture");
    let mut jp2 = wrap_jp2_codestream(&codestream, 2, 2, 1, 8, 17);
    jp2.extend_from_slice(&[0, 0, 0, 16, b'x', b'm', b'l', b' ']);
    let mut decoder =
        J2kDecoder::new_with_settings(&jp2, DecodeSettings::lenient()).expect("lenient decoder");
    let mut output = [0_u8; 1];

    let outcome = decoder
        .decode_region_scaled_pow2_into(
            &mut J2kScratchPool::new(),
            &mut output,
            1,
            PixelFormat::Gray8,
            Rect {
                x: 0,
                y: 0,
                w: 2,
                h: 2,
            },
            1,
        )
        .expect("lenient exact reduction");

    assert_eq!(
        outcome.warnings,
        vec![J2kDecodeWarning::LenientMetadataRecovery]
    );
}

#[test]
fn native_components_at_reduction_match_the_packed_production_path() {
    let bytes = encode_rgb_fixture(65, 33, 4, None);
    let mut decoder = J2kDecoder::new(&bytes).expect("native component decoder");

    let native = decoder
        .decode_native_components_at_reduction(3)
        .expect("native 1/8 decode");

    assert_eq!(native.dimensions(), (9, 5));
    assert_eq!(native.planes().len(), 3);
    assert!(native.planes().iter().all(|plane| {
        plane.dimensions() == (9, 5)
            && plane.bit_depth() == 8
            && !plane.signed()
            && plane.bytes_per_sample() == 1
    }));

    let mut packed = vec![0_u8; 9 * 5 * 3];
    let mut packed_decoder = J2kDecoder::new(&bytes).expect("packed decoder");
    packed_decoder
        .decode_region_scaled_pow2_into(
            &mut J2kScratchPool::new(),
            &mut packed,
            9 * 3,
            PixelFormat::Rgb8,
            Rect {
                x: 0,
                y: 0,
                w: 65,
                h: 33,
            },
            3,
        )
        .expect("packed 1/8 decode");
    let interleaved = (0..45)
        .flat_map(|index| native.planes().iter().map(move |plane| plane.data()[index]))
        .collect::<Vec<_>>();
    assert_eq!(interleaved, packed);
}

#[test]
fn native_component_reduction_zero_delegates_and_excess_levels_fail() {
    let bytes = encode_rgb_fixture(32, 17, 3, None);
    let mut full_decoder = J2kDecoder::new(&bytes).expect("full decoder");
    let full = full_decoder
        .decode_native_components()
        .expect("full native decode");
    let mut zero_decoder = J2kDecoder::new(&bytes).expect("zero-level decoder");
    let zero = zero_decoder
        .decode_native_components_at_reduction(0)
        .expect("zero-level native decode");
    assert_eq!(zero, full);

    let error = zero_decoder
        .decode_native_components_at_reduction(4)
        .expect_err("reduction beyond the wavelet ladder must fail");
    assert!(matches!(error, J2kError::Unsupported(_)));
}
