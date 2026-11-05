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
fn downscale_deeper_than_the_ladder_is_rejected_instead_of_cropped() {
    // Two decomposition levels reach Quarter but not Eighth. A clamped decode
    // would have returned the top-left crop of the quarter-resolution image.
    let (width, height) = (64_u32, 48_u32);
    let full = Rect {
        x: 0,
        y: 0,
        w: width,
        h: height,
    };
    let stride = width as usize * 3;
    for tile_size in [None, Some((32, 32))] {
        let bytes = encode_rgb_fixture(width, height, 2, tile_size);
        let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
        let mut output = vec![0_u8; stride * height as usize];

        decoder
            .decode_scaled_into(
                &mut J2kScratchPool::new(),
                &mut output,
                stride,
                PixelFormat::Rgb8,
                Downscale::Quarter,
            )
            .expect("quarter resolution is on the ladder");
        let full_error = decoder
            .decode_scaled_into(
                &mut J2kScratchPool::new(),
                &mut output,
                stride,
                PixelFormat::Rgb8,
                Downscale::Eighth,
            )
            .expect_err("eighth resolution exceeds the ladder");
        let region_error = decoder
            .decode_region_scaled_into(
                &mut J2kScratchPool::new(),
                &mut output,
                stride,
                PixelFormat::Rgb8,
                full,
                Downscale::Eighth,
            )
            .expect_err("eighth resolution exceeds the ladder");
        for error in [full_error, region_error] {
            assert!(
                matches!(error, J2kError::Unsupported(_)),
                "tile={tile_size:?}: {error}"
            );
        }
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

/// An image whose reference-grid origin is odd at every resolution reduces
/// to `[ceil(origin / s), ceil(extent / s))` on each axis, narrower than
/// `ceil(size / s)`. Every reduced decode of it, whole or of a region and with
/// classic or HTJ2K coding, writes the samples `OpenJPEG` decodes at that
/// reduction, and a region covers the reduced samples whose reference-grid
/// footprint it overlaps.
#[test]
fn reductions_of_an_odd_origin_image_match_openjpeg() {
    use j2k_test_support::{
        OPENJPEG_ORIGIN_23X17_AT_3_5, OPENJPH_ORIGIN_23X17_AT_3_5, ORIGIN_REDUCED_RGB,
        ORIGIN_SOURCE_23X17_RGB,
    };
    let size = (23, 17);
    for (coding, fixture) in [
        ("classic", OPENJPEG_ORIGIN_23X17_AT_3_5),
        ("HTJ2K", OPENJPH_ORIGIN_23X17_AT_3_5),
    ] {
        let mut decoder = J2kDecoder::new(fixture).expect("decoder");
        assert_eq!(decoder.info().dimensions, size, "{coding}");
        let grid = decoder.reference_grid();
        assert_eq!(grid.image_origin, (3, 5), "{coding}");
        assert_eq!(grid.image_size, size, "{coding}");
        assert_eq!(grid.extent(), (26, 22), "{coding}");
        assert_eq!(
            J2kDecoder::inspect_reference_grid(fixture).expect("header-only grid"),
            grid,
            "{coding}"
        );
        let mut out = vec![0_u8; ORIGIN_SOURCE_23X17_RGB.len()];
        decoder
            .decode_into(&mut out, 23 * 3, PixelFormat::Rgb8)
            .expect("full-resolution decode");
        assert_eq!(out, ORIGIN_SOURCE_23X17_RGB, "{coding}");

        let scales = [Downscale::Half, Downscale::Quarter, Downscale::Eighth];
        for ((levels, scale), (dims, reference)) in (1_u8..).zip(scales).zip(ORIGIN_REDUCED_RGB) {
            let case = format!("{coding}, {levels} levels");
            assert_reduced_decodes_match(&mut decoder, levels, scale, dims, reference, &case);
        }
    }
}

fn assert_reduced_decodes_match(
    decoder: &mut J2kDecoder<'_>,
    levels: u8,
    scale: Downscale,
    dims: (u32, u32),
    reference: &[u8],
    case: &str,
) {
    let mut pool = J2kScratchPool::new();
    let stride = dims.0 as usize * 3;

    let native = decoder
        .decode_native_components_at_reduction(levels)
        .unwrap_or_else(|error| panic!("{case}: native components: {error:?}"));
    assert_eq!(native.dimensions(), dims, "{case}: native components");
    for (channel, plane) in native.planes().iter().enumerate() {
        let expected = reference
            .iter()
            .skip(channel)
            .step_by(3)
            .copied()
            .collect::<Vec<_>>();
        assert_eq!(plane.data(), expected, "{case}: native component {channel}");
    }

    let mut whole = vec![0_u8; reference.len()];
    let outcome = decoder
        .decode_scaled_into(&mut pool, &mut whole, stride, PixelFormat::Rgb8, scale)
        .unwrap_or_else(|error| panic!("{case}: scaled: {error:?}"));
    assert_eq!(outcome.decoded, Rect::full(dims), "{case}: scaled");
    assert_eq!(whole, reference, "{case}: scaled");

    // The whole image, and a region whose edges fall inside reduced samples
    // on every side.
    let interior = Rect {
        x: 4,
        y: 2,
        w: 13,
        h: 11,
    };
    for roi in [Rect::full((23, 17)), interior] {
        let covered = reference_grid_covering(roi, (3, 5), (23, 17), levels);
        let expected = crop_rgb(reference, dims.0 as usize, covered);
        let region_stride = covered.w as usize * 3;
        let mut region = vec![0_u8; expected.len()];
        let outcome = decoder
            .decode_region_scaled_pow2_into(
                &mut pool,
                &mut region,
                region_stride,
                PixelFormat::Rgb8,
                roi,
                levels,
            )
            .unwrap_or_else(|error| panic!("{case}, {roi:?}: pow2 region: {error:?}"));
        assert_eq!(outcome.decoded, covered, "{case}, {roi:?}: pow2 region");
        assert_eq!(region, expected, "{case}, {roi:?}: pow2 region");

        let mut region = vec![0_u8; expected.len()];
        let outcome = decoder
            .decode_region_scaled_into(
                &mut pool,
                &mut region,
                region_stride,
                PixelFormat::Rgb8,
                roi,
                scale,
            )
            .unwrap_or_else(|error| panic!("{case}, {roi:?}: scaled region: {error:?}"));
        assert_eq!(outcome.decoded, covered, "{case}, {roi:?}: scaled region");
        assert_eq!(region, expected, "{case}, {roi:?}: scaled region");
    }
}

/// The reduced samples, in reduced image coordinates, whose reference-grid
/// footprint `[k * s, (k + 1) * s)` overlaps image region `roi` of an image
/// of `size` at `origin`.
fn reference_grid_covering(roi: Rect, origin: (u32, u32), size: (u32, u32), levels: u8) -> Rect {
    let scale = 1_u32 << levels;
    let axis = |start: u32, length: u32, origin: u32, size: u32| {
        let first = origin.div_ceil(scale);
        let end = (origin + size).div_ceil(scale);
        let low = ((origin + start) / scale).max(first);
        let high = (origin + start + length).div_ceil(scale).min(end);
        (low - first, high - low)
    };
    let (x, w) = axis(roi.x, roi.w, origin.0, size.0);
    let (y, h) = axis(roi.y, roi.h, origin.1, size.1);
    Rect { x, y, w, h }
}

fn crop_rgb(samples: &[u8], width: usize, rect: Rect) -> Vec<u8> {
    (rect.y..rect.y + rect.h)
        .flat_map(|y| {
            let start = (y as usize * width + rect.x as usize) * 3;
            samples[start..start + rect.w as usize * 3].iter().copied()
        })
        .collect()
}
