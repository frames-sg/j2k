// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{
    DecodeSettings, J2kDecodeWarning, J2kDecoder, J2kError, J2kSampleInterpretation, J2kScratchPool,
};
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
        let plan = assert_region_plan_output(
            decoder,
            PixelFormat::Rgb8,
            roi,
            levels,
            covered,
            dims,
            &expected,
        );
        assert_eq!(plan.decode.route, j2k::J2kDecodeRoute::Planes);
        assert_eq!(
            plan.decode.blocker,
            Some(j2k::J2kStreamingBlocker::GridOffset)
        );
        let region_stride = covered.w as usize * 3;
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

/// Counts rows and requires them in top-to-bottom order.
struct OrderedRows(u32);

impl j2k_core::RowSink<u8> for OrderedRows {
    type Error = core::convert::Infallible;
    fn write_row(&mut self, y: u32, _row: &[u8]) -> Result<(), Self::Error> {
        assert_eq!(y, self.0);
        self.0 += 1;
        Ok(())
    }
}

fn assert_region_plan_output(
    decoder: &mut J2kDecoder<'_>,
    fmt: PixelFormat,
    roi: Rect,
    levels: u8,
    expected_region: Rect,
    reduced_dims: (u32, u32),
    expected: &[u8],
) -> j2k::J2kRegionDecodePlan {
    let plan = decoder
        .region_decode_plan(fmt, roi, levels)
        .unwrap_or_else(|error| panic!("{fmt:?}, {levels} levels, {roi:?}: plan: {error:?}"));
    assert_eq!(plan.reference_grid, decoder.reference_grid());
    assert_eq!(plan.source_region, roi);
    assert_eq!(plan.output_region, expected_region);
    assert_eq!(plan.reduced_image_dimensions, reduced_dims);
    assert_eq!(plan.reduction_levels, levels);
    assert_eq!(plan.format, fmt);
    assert_eq!(
        plan.sample_interpretation,
        if fmt.bytes_per_sample() == 1 {
            J2kSampleInterpretation::NormalizedU8
        } else {
            J2kSampleInterpretation::Native16
        }
    );
    let row_bytes = expected_region.w as usize * fmt.bytes_per_pixel();
    assert_eq!(plan.row_bytes, row_bytes);
    assert_eq!(plan.buffer_bytes, expected.len());
    assert_eq!(plan.buffer_len(row_bytes).unwrap(), expected.len());
    assert!(matches!(
        plan.buffer_len(row_bytes - 1),
        Err(J2kError::Buffer(
            j2k_core::BufferError::StrideTooSmall { .. }
        ))
    ));
    if expected_region.h > 1 {
        assert!(matches!(
            plan.buffer_len(usize::MAX),
            Err(J2kError::Buffer(_))
        ));
        let unaddressable_stride = isize::MAX as usize / (expected_region.h as usize - 1) + 1;
        assert!(matches!(
            plan.buffer_len(unaddressable_stride),
            Err(J2kError::Buffer(
                j2k_core::BufferError::AllocationTooLarge { .. }
            ))
        ));
    } else {
        assert_eq!(
            plan.buffer_len(usize::MAX).unwrap(),
            row_bytes,
            "one row needs no trailing stride padding"
        );
    }
    for padding in [0, 7] {
        let stride = row_bytes + padding;
        let span = stride * (expected_region.h as usize - 1) + row_bytes;
        assert_eq!(plan.buffer_len(stride).unwrap(), span);
        let mut out = vec![0xa5; span];
        let outcome = decoder
            .decode_region_scaled_pow2_into(
                &mut J2kScratchPool::new(),
                &mut out,
                stride,
                fmt,
                roi,
                levels,
            )
            .unwrap_or_else(|error| {
                panic!(
                    "{fmt:?}, {levels} levels, {roi:?}, stride {stride}: planned decode: {error:?}"
                )
            });
        assert_eq!(outcome.decoded, expected_region);
        for row in 0..expected_region.h as usize {
            assert_eq!(
                &out[row * stride..row * stride + row_bytes],
                &expected[row * row_bytes..(row + 1) * row_bytes]
            );
            if row + 1 < expected_region.h as usize {
                assert!(out[row * stride + row_bytes..(row + 1) * stride]
                    .iter()
                    .all(|&byte| byte == 0xa5));
            }
        }
    }
    plan
}

#[test]
fn region_plans_cover_deep_edge_regions_and_exact_padded_buffers() {
    let size = (65, 49);
    let bytes = encode_rgb_fixture(size.0, size.1, 4, Some((33, 25)));
    let mut decoder = J2kDecoder::new(&bytes).unwrap();
    assert_eq!(decoder.max_reduction_levels().unwrap(), 4);
    let full = Rect::full(size);
    let edge = Rect {
        x: 59,
        y: 43,
        w: 6,
        h: 6,
    };
    for levels in [0, 1, 3, 4] {
        let reduced = reference_grid_covering(full, (0, 0), size, levels);
        let dims = (reduced.w, reduced.h);
        for fmt in [
            PixelFormat::Rgb8,
            PixelFormat::Rgba8,
            PixelFormat::Rgb16,
            PixelFormat::Rgba16,
        ] {
            let stride = reduced.w as usize * fmt.bytes_per_pixel();
            let mut whole = vec![0; stride * reduced.h as usize];
            decoder
                .decode_region_scaled_pow2_into(
                    &mut J2kScratchPool::new(),
                    &mut whole,
                    stride,
                    fmt,
                    full,
                    levels,
                )
                .unwrap();
            for roi in [
                full,
                edge,
                Rect {
                    x: 64,
                    y: 48,
                    w: 1,
                    h: 1,
                },
            ] {
                let covered = reference_grid_covering(roi, (0, 0), size, levels);
                let expected = crop_interleaved(&whole, reduced.w, fmt.bytes_per_pixel(), covered);
                let plan = assert_region_plan_output(
                    &mut decoder,
                    fmt,
                    roi,
                    levels,
                    covered,
                    dims,
                    &expected,
                );
                if levels == 0 {
                    assert_eq!(plan.decode.route, j2k::J2kDecodeRoute::StreamedRows);
                    assert_eq!(plan.decode.blocker, None);
                }
            }
        }
    }
}

#[test]
fn region_plans_reject_invalid_empty_unavailable_and_unsupported_requests() {
    let flat = encode_rgb_fixture(8, 8, 0, None);
    let mut flat = J2kDecoder::new(&flat).unwrap();
    assert_eq!(flat.max_reduction_levels().unwrap(), 0);
    let flat_plan = flat
        .region_decode_plan(PixelFormat::Rgb8, Rect::full((8, 8)), 0)
        .unwrap();
    assert_eq!(flat_plan.decode.route, j2k::J2kDecodeRoute::Planes);
    assert!(flat_plan.decode.blocker.is_some());
    let size = (32, 24);
    let bytes = encode_rgb_fixture(size.0, size.1, 3, None);
    let mut decoder = J2kDecoder::new(&bytes).unwrap();
    assert_eq!(decoder.max_reduction_levels().unwrap(), 3);
    for roi in [
        Rect {
            x: 0,
            y: 0,
            w: 0,
            h: 1,
        },
        Rect {
            x: 0,
            y: 0,
            w: 1,
            h: 0,
        },
        Rect {
            x: 32,
            y: 0,
            w: 1,
            h: 1,
        },
    ] {
        assert!(matches!(
            decoder.region_decode_plan(PixelFormat::Rgb8, roi, 0),
            Err(J2kError::InvalidRegion { .. })
        ));
    }
    for levels in [4, 32, u8::MAX] {
        assert!(matches!(
            decoder.region_decode_plan(PixelFormat::Rgb8, Rect::full(size), levels),
            Err(J2kError::Unsupported(_))
        ));
    }
    for fmt in [PixelFormat::Gray8, PixelFormat::RgbI16] {
        assert!(matches!(
            decoder.region_decode_plan(fmt, Rect::full(size), 0),
            Err(J2kError::Unsupported(_))
        ));
    }
    let mut odd = J2kDecoder::new(j2k_test_support::OPENJPEG_ORIGIN_23X17_AT_3_5).unwrap();
    assert!(odd.max_reduction_levels().unwrap() >= 1);
    let empty = odd
        .region_decode_plan(
            PixelFormat::Rgb8,
            Rect {
                x: 0,
                y: 0,
                w: 1,
                h: 1,
            },
            1,
        )
        .unwrap_err();
    assert!(matches!(
        empty,
        J2kError::Unsupported(j2k_core::Unsupported {
            what: "region has no samples at the requested reduction"
        })
    ));
}

fn main_segment(bytes: &[u8], marker: u8) -> core::ops::Range<usize> {
    let start = bytes
        .windows(2)
        .position(|pair| pair == [0xff, marker])
        .expect("fixture marker");
    let length = usize::from(u16::from_be_bytes([bytes[start + 2], bytes[start + 3]]));
    start..start + 2 + length
}

fn component_coding_segment(cod: &[u8], component: u8, levels: u8) -> Vec<u8> {
    assert_eq!(cod[4] & 1, 0, "fixture uses default precincts");
    let mut body = vec![component, 0];
    body.extend_from_slice(&cod[9..]);
    body[2] = levels;
    let mut marker = vec![0xff, 0x53];
    marker.extend_from_slice(&u16::try_from(body.len() + 2).unwrap().to_be_bytes());
    marker.extend(body);
    marker
}

fn with_first_tile_headers(bytes: &[u8], markers: &[u8]) -> Vec<u8> {
    let mut bytes = bytes.to_vec();
    let sot = bytes
        .windows(2)
        .position(|pair| pair == [0xff, 0x90])
        .expect("SOT");
    let tile_length = u32::from_be_bytes(bytes[sot + 6..sot + 10].try_into().unwrap());
    assert_ne!(tile_length, 0);
    bytes[sot + 6..sot + 10]
        .copy_from_slice(&(tile_length + u32::try_from(markers.len()).unwrap()).to_be_bytes());
    assert_eq!(&bytes[sot + 12..sot + 14], &[0xff, 0x93]);
    bytes.splice(sot + 12..sot + 12, markers.iter().copied());
    bytes
}

#[test]
fn maximum_reduction_includes_main_and_tile_component_overrides() {
    // Only headers are queried here; the small entropy payload is never decoded
    // after changing the component ladders.
    let size = (32, 32);
    let base = encode_rgb_fixture(size.0, size.1, 4, Some((16, 16)));
    let cod = base[main_segment(&base, 0x52)].to_vec();
    assert_eq!(cod[9], 4);
    for (main_coc, tile_cod, tile_coc, expected) in [
        (None, None, None, 4),
        (Some(2), None, None, 2),
        (None, Some(2), None, 2),
        (None, None, Some(1), 1),
        (Some(3), Some(2), Some(1), 1),
        (None, Some(0), None, 0),
    ] {
        let mut bytes = base.clone();
        if let Some(levels) = main_coc {
            let at = main_segment(&bytes, 0x52).end;
            bytes.splice(at..at, component_coding_segment(&cod, 1, levels));
        }
        let mut tile_headers = Vec::new();
        if let Some(levels) = tile_cod {
            let mut override_cod = cod.clone();
            override_cod[9] = levels;
            tile_headers.extend(override_cod);
        }
        if let Some(levels) = tile_coc {
            tile_headers.extend(component_coding_segment(&cod, 2, levels));
        }
        bytes = with_first_tile_headers(&bytes, &tile_headers);
        let mut decoder = J2kDecoder::new(&bytes).expect("valid main header");
        assert_eq!(
            decoder.max_reduction_levels().unwrap(),
            expected,
            "main COC {main_coc:?}, tile COD {tile_cod:?}, tile COC {tile_coc:?}"
        );
        assert!(matches!(
            decoder.region_decode_plan(PixelFormat::Rgb8, Rect::full(size), expected + 1),
            Err(J2kError::Unsupported(_))
        ));
    }
    let malformed = with_first_tile_headers(&base, &[0xff, 0x52, 0, 3, 0]);
    let mut decoder = J2kDecoder::new(&malformed).expect("main header remains valid");
    assert!(
        decoder.max_reduction_levels().is_err(),
        "malformed tile COD must not produce a limit"
    );
    assert!(decoder
        .region_decode_plan(PixelFormat::Rgb8, Rect::full(size), 0)
        .is_err());
}

#[test]
fn maximum_reduction_uses_effective_tile_coding_over_main_defaults() {
    let size = (32, 32);
    let original = encode_rgb_fixture(size.0, size.1, 3, None);
    let lower_defaults = encode_rgb_fixture(size.0, size.1, 1, None);
    let mut tile_headers = original[main_segment(&original, 0x52)].to_vec();
    tile_headers.extend_from_slice(&original[main_segment(&original, 0x5c)]);
    let mut bytes = original.clone();
    for marker in [0x52, 0x5c] {
        let range = main_segment(&bytes, marker);
        bytes.splice(
            range,
            lower_defaults[main_segment(&lower_defaults, marker)]
                .iter()
                .copied(),
        );
    }
    let bytes = with_first_tile_headers(&bytes, &tile_headers);
    let expected = J2kDecoder::new(&original)
        .unwrap()
        .decode_native_components()
        .unwrap();
    let mut decoder = J2kDecoder::new(&bytes).expect("overridden main defaults");
    assert_eq!(
        decoder.decode_native_components().unwrap(),
        expected,
        "tile COD/QCD preserve the encoded image"
    );
    assert_eq!(
        decoder.max_reduction_levels().unwrap(),
        3,
        "the only tile overrides the main-header default ladder"
    );
    for levels in [2, 3] {
        let dims = (size.0 >> levels, size.1 >> levels);
        let stride = dims.0 as usize * 3;
        let mut reduced = vec![0; stride * dims.1 as usize];
        J2kDecoder::new(&original)
            .unwrap()
            .decode_region_scaled_pow2_into(
                &mut J2kScratchPool::new(),
                &mut reduced,
                stride,
                PixelFormat::Rgb8,
                Rect::full(size),
                levels,
            )
            .expect("original three-level reduced oracle");
        assert_region_plan_output(
            &mut decoder,
            PixelFormat::Rgb8,
            Rect::full(size),
            levels,
            Rect::full(dims),
            dims,
            &reduced,
        );
    }
}

#[test]
fn region_plan_native16_matches_signed_samples_and_plane_preference() {
    let size = (16, 12);
    let values = (0..192_i16)
        .map(|value| value * 17 - 1500)
        .collect::<Vec<_>>();
    let input = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let expected = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    let bytes = encode(
        &input,
        size.0,
        size.1,
        1,
        12,
        true,
        &EncodeOptions {
            num_decomposition_levels: 2,
            ..EncodeOptions::default()
        },
    )
    .unwrap();
    let mut decoder = J2kDecoder::new(&bytes).unwrap();
    let full = Rect::full(size);
    let plan = assert_region_plan_output(
        &mut decoder,
        PixelFormat::Gray16,
        full,
        0,
        full,
        size,
        &expected,
    );
    assert_eq!(plan.decode.route, j2k::J2kDecodeRoute::Planes);
    assert_eq!(
        plan.decode.blocker, None,
        "small Gray16 prefers planes despite row eligibility"
    );
}

#[test]
fn region_plans_reject_row_precision_even_when_another_blocker_comes_first() {
    let size = (4, 4);
    let samples = (0..48_u32)
        .flat_map(|value| (value * 100_003).to_le_bytes())
        .collect::<Vec<_>>();
    let codestream = encode(
        &samples,
        size.0,
        size.1,
        3,
        25,
        false,
        &EncodeOptions {
            num_decomposition_levels: 1,
            use_mct: false,
            ..EncodeOptions::default()
        },
    )
    .unwrap();
    let channels = [1, 2, 3].map(|index| j2k::J2kChannelDefinition {
        channel_index: index - 1,
        channel_type: j2k::J2kChannelType::Color,
        association: j2k::J2kChannelAssociation::Color { index },
    });
    let wrapped = j2k::wrap_j2k_codestream(
        &codestream,
        j2k::J2kFileWrapOptions::jp2()
            .with_color(j2k::J2kFileColorSpec::Enumerated(
                j2k_core::Colorspace::SRgb,
            ))
            .with_metadata(j2k::J2kFileBoxMetadata {
                palette: None,
                component_mappings: &[],
                channel_definitions: &channels,
            }),
    )
    .unwrap();
    for (label, bytes) in [
        ("precision", &codestream),
        ("channel definitions and precision", &wrapped),
    ] {
        let mut decoder = J2kDecoder::new(bytes).unwrap();
        let mut output = [0xa5; 48];
        let error = decoder
            .decode_region_scaled_pow2_into(
                &mut J2kScratchPool::new(),
                &mut output,
                12,
                PixelFormat::Rgb8,
                Rect::full(size),
                0,
            )
            .expect_err("component rows reject precision above 24 bits");
        assert!(matches!(error, J2kError::Unsupported(_)));
        assert_eq!(output, [0xa5; 48]);
        let plan = decoder.region_decode_plan(PixelFormat::Rgb8, Rect::full(size), 0);
        assert!(
            matches!(plan, Err(J2kError::Unsupported(_))),
            "{label}: unsupported row precision was admitted: {plan:?}"
        );
    }
}

#[test]
fn region_plans_reject_mixed_precision_when_native_plane_fallback_cannot_pack_it() {
    let size = (4, 4);
    let eight = (0..16_u8).collect::<Vec<_>>();
    let twelve = (0..16_u16)
        .flat_map(|value| (value * 97).to_le_bytes())
        .collect::<Vec<_>>();
    let sixteen = (0..16_u16)
        .flat_map(|value| (value * 2000).to_le_bytes())
        .collect::<Vec<_>>();
    let planes = [(&eight, 8), (&twelve, 12), (&sixteen, 16)].map(|(data, bit_depth)| {
        j2k_native::EncodeTypedComponentPlane {
            data,
            x_rsiz: 1,
            y_rsiz: 1,
            bit_depth,
            signed: false,
        }
    });
    let bytes = j2k_native::encode_typed_component_planes_53(
        &planes,
        size.0,
        size.1,
        &EncodeOptions {
            num_decomposition_levels: 1,
            use_mct: false,
            ..EncodeOptions::default()
        },
    )
    .unwrap();
    let mut decoder = J2kDecoder::new(&bytes).unwrap();
    let mut output = [0xa5; 96];
    let error = decoder
        .decode_region_scaled_pow2_into(
            &mut J2kScratchPool::new(),
            &mut output,
            24,
            PixelFormat::Rgb16,
            Rect::full(size),
            0,
        )
        .expect_err("native bitmap fallback requires uniform precision");
    assert!(matches!(error, J2kError::Unsupported(_)));
    assert_eq!(output, [0xa5; 96]);
    let plan = decoder.region_decode_plan(PixelFormat::Rgb16, Rect::full(size), 0);
    assert!(
        matches!(plan, Err(J2kError::Unsupported(_))),
        "unsupported native-plane layout was admitted: {plan:?}"
    );
}

#[test]
fn region_plans_reject_uniform_component_sampling_geometry_at_full_resolution() {
    let samples = (0..16_u8).map(|value| value * 13).collect::<Vec<_>>();
    let planes = [j2k_native::EncodeTypedComponentPlane {
        data: &samples,
        x_rsiz: 2,
        y_rsiz: 2,
        bit_depth: 8,
        signed: false,
    }];
    let bytes = j2k_native::encode_typed_component_planes_53(
        &planes,
        8,
        8,
        &EncodeOptions {
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        },
    )
    .unwrap();
    let native = j2k_native::Image::new(&bytes, &j2k_native::DecodeSettings::default())
        .unwrap()
        .decode_native()
        .unwrap();
    assert_eq!((native.width, native.height), (4, 4));
    assert_eq!(native.data, samples);

    let mut decoder = J2kDecoder::new(&bytes).unwrap();
    assert_eq!(decoder.info().dimensions, (8, 8));
    let mut output = [0xa5; 64];
    decoder
        .decode_region_scaled_pow2_into(
            &mut J2kScratchPool::new(),
            &mut output,
            8,
            PixelFormat::Gray8,
            Rect::full((8, 8)),
            0,
        )
        .expect_err("native component grid cannot fill the reference-grid region");
    assert_eq!(output, [0xa5; 64]);
    for levels in [0, 1] {
        let plan = decoder.region_decode_plan(PixelFormat::Gray8, Rect::full((8, 8)), levels);
        assert!(
            matches!(plan, Err(J2kError::Unsupported(_))),
            "level {levels}: contradictory component/reference geometry was admitted: {plan:?}"
        );
    }
}

#[test]
fn borrowed_decoder_reuses_exact_levels_and_bounds_preparation_retention() {
    let size = (65, 49);
    let bytes = encode_rgb_fixture(size.0, size.1, 4, None);
    let mut decoder = J2kDecoder::new(&bytes).unwrap();
    assert_eq!(decoder.max_reduction_levels().unwrap(), 4);
    let grid = decoder.reference_grid();
    let mut previous = 0;
    for (index, levels) in [4, 4, 2, 2, 0, 0].into_iter().enumerate() {
        let roi = if index % 2 == 0 {
            Rect::full(size)
        } else {
            Rect {
                x: 5,
                y: 7,
                w: 29,
                h: 23,
            }
        };
        let covered = reference_grid_covering(roi, (0, 0), size, levels);
        let stride = covered.w as usize * 3;
        let mut expected = vec![0; stride * covered.h as usize];
        J2kDecoder::new(&bytes)
            .unwrap()
            .decode_region_scaled_pow2_into(
                &mut J2kScratchPool::new(),
                &mut expected,
                stride,
                PixelFormat::Rgb8,
                roi,
                levels,
            )
            .unwrap();
        let before = decoder.preparation_cache_stats();
        let plan = decoder
            .region_decode_plan(PixelFormat::Rgb8, roi, levels)
            .unwrap();
        let mut output = vec![0; plan.buffer_bytes];
        decoder
            .decode_region_scaled_pow2_into(
                &mut J2kScratchPool::new(),
                &mut output,
                stride,
                PixelFormat::Rgb8,
                roi,
                levels,
            )
            .unwrap();
        let after = decoder.preparation_cache_stats();
        assert_eq!(output, expected);
        assert_eq!(after.misses - before.misses, u64::from(levels != previous));
        assert!(
            after.hits > before.hits,
            "plan and decode share the parsed exact-level image"
        );
        assert_eq!(after.occupied_slots, 1);
        assert_eq!(decoder.info().dimensions, size);
        assert_eq!(decoder.reference_grid(), grid);
        assert!(
            decoder.preparation_cache_retained_bytes().unwrap()
                <= decoder.preparation_cache_limit_bytes()
        );
        previous = levels;
    }
    let retained = decoder.preparation_cache_retained_bytes().unwrap();
    assert!(retained > 0);
    decoder.set_preparation_cache_limit_bytes(retained - 1);
    assert_eq!(decoder.preparation_cache_retained_bytes().unwrap(), 0);
    decoder.set_preparation_cache_limit_bytes(0);
    decoder.full_decode_plan(PixelFormat::Rgb8).unwrap();
    assert_eq!(decoder.preparation_cache_retained_bytes().unwrap(), 0);
    decoder.rows_decode_plan().unwrap();
    assert_eq!(decoder.preparation_cache_retained_bytes().unwrap(), 0);
    let components = decoder.decode_components().unwrap();
    assert_eq!(components.dimensions(), size);
    drop(components);
    assert_eq!(decoder.preparation_cache_retained_bytes().unwrap(), 0);
    let native = decoder.decode_native_components_at_reduction(4).unwrap();
    assert_eq!(native.dimensions(), (5, 4));
    assert_eq!(decoder.preparation_cache_retained_bytes().unwrap(), 0);
    let mut rows = OrderedRows(0);
    decoder
        .decode_rows_u8_bounded(&mut rows, j2k::J2kRowDecodeOptions::new(7))
        .unwrap();
    assert_eq!(rows.0, size.1);
    assert_eq!(decoder.preparation_cache_retained_bytes().unwrap(), 0);
}

#[test]
fn operation_controls_bound_padded_regions_and_do_not_poison_reuse() {
    use core::num::NonZeroUsize;
    use j2k::{J2kCancellationToken, J2kDecodeControl};
    let bytes = encode_rgb_fixture(128, 96, 4, None);
    let mut decoder = J2kDecoder::new(&bytes).unwrap();
    let roi = Rect {
        x: 7,
        y: 11,
        w: 99,
        h: 71,
    };
    let plan = decoder
        .region_decode_plan(PixelFormat::Rgb8, roi, 4)
        .unwrap();
    let stride = plan.row_bytes + 7;
    let span = plan.buffer_len(stride).unwrap();
    let mut output = vec![0xa5; span];
    let mut control = J2kDecodeControl::default();
    control.max_output_bytes = Some(span - 1);
    assert!(decoder
        .decode_region_scaled_pow2_into_with_control(
            &mut output,
            stride,
            PixelFormat::Rgb8,
            roi,
            4,
            &control
        )
        .is_err());
    assert!(output.iter().all(|byte| *byte == 0xa5));
    control.max_output_bytes = Some(span);
    control.max_working_memory_bytes = NonZeroUsize::new(1);
    assert!(matches!(
        decoder.decode_region_scaled_pow2_into_with_control(
            &mut output,
            stride,
            PixelFormat::Rgb8,
            roi,
            4,
            &control
        ),
        Err(J2kError::ResourceLimit { .. })
    ));
    assert!(output.iter().all(|byte| *byte == 0xa5));
    control.max_working_memory_bytes = NonZeroUsize::new(4 * 1024 * 1024);
    let token = J2kCancellationToken::new();
    token.cancel();
    control.cancellation = Some(token);
    assert!(matches!(
        decoder.region_decode_plan_with_control(PixelFormat::Rgb8, roi, 4, &control),
        Err(J2kError::Cancelled)
    ));
    control.cancellation = None;
    decoder
        .decode_region_scaled_pow2_into_with_control(
            &mut output,
            stride,
            PixelFormat::Rgb8,
            roi,
            4,
            &control,
        )
        .unwrap();
    let (expected, _) = decode_reduced_region(&bytes, roi, 4).unwrap();
    for (row, reference) in expected.chunks_exact(plan.row_bytes).enumerate() {
        assert_eq!(
            &output[row * stride..row * stride + plan.row_bytes],
            reference
        );
        if row + 1 < plan.output_region.h as usize {
            assert!(output[row * stride + plan.row_bytes..(row + 1) * stride]
                .iter()
                .all(|byte| *byte == 0xa5));
        }
    }
}

#[test]
fn controlled_construction_bounds_parsing_without_retaining_its_allowance() {
    use core::num::NonZeroUsize;
    use j2k::{J2kCancellationToken, J2kDecodeControl};
    let codestream = encode_rgb_fixture(128, 96, 4, Some((64, 48)));
    let jp2 = wrap_jp2_codestream(&codestream, 128, 96, 3, 8, 16);
    let full = Rect::full((128, 96));
    let reduced_rgb = |decoder: &mut J2kDecoder<'_>| {
        let plan = decoder
            .region_decode_plan(PixelFormat::Rgb8, full, 2)
            .unwrap();
        let mut output = vec![0_u8; plan.buffer_bytes];
        decoder
            .decode_region_scaled_pow2_into(
                &mut J2kScratchPool::new(),
                &mut output,
                plan.row_bytes,
                PixelFormat::Rgb8,
                full,
                2,
            )
            .unwrap();
        output
    };
    for (name, bytes) in [("codestream", &codestream[..]), ("jp2", &jp2[..])] {
        let mut expected = J2kDecoder::new(bytes).unwrap();
        let mut control = J2kDecodeControl::default();
        let mut decoder =
            J2kDecoder::new_with_control(bytes, DecodeSettings::default(), &control).unwrap();
        assert_eq!(decoder.info(), expected.info(), "{name}");
        assert_eq!(
            decoder.reference_grid(),
            expected.reference_grid(),
            "{name}"
        );
        assert_eq!(
            decoder.max_reduction_levels_with_control(&control).unwrap(),
            expected.max_reduction_levels().unwrap(),
            "{name}"
        );
        let srgb = decoder
            .srgb8_region_decode_plan_with_control(full, 2, &control)
            .unwrap();
        assert_eq!(
            srgb,
            expected.srgb8_region_decode_plan(full, 2).unwrap(),
            "{name}"
        );
        control.max_output_bytes = Some(srgb.buffer_bytes - 1);
        assert!(
            decoder
                .srgb8_region_decode_plan_with_control(full, 2, &control)
                .is_err(),
            "{name}"
        );
        control.max_output_bytes = None;

        let token = J2kCancellationToken::new();
        token.cancel();
        control.cancellation = Some(token);
        assert!(
            matches!(
                J2kDecoder::new_with_control(bytes, DecodeSettings::default(), &control),
                Err(J2kError::Cancelled)
            ),
            "{name}"
        );
        assert!(
            matches!(
                decoder.max_reduction_levels_with_control(&control),
                Err(J2kError::Cancelled)
            ),
            "{name}"
        );
        control.cancellation = None;
        control.max_working_memory_bytes = NonZeroUsize::new(1);
        assert!(
            matches!(
                J2kDecoder::new_with_control(bytes, DecodeSettings::default(), &control),
                Err(J2kError::ResourceLimit { .. })
            ),
            "{name}"
        );

        // The smallest allowance that parses constructs a decoder that later
        // operations use without inheriting that allowance.
        control.max_working_memory_bytes = NonZeroUsize::new(smallest_parsing_allowance(bytes));
        let mut bounded =
            J2kDecoder::new_with_control(bytes, DecodeSettings::default(), &control).unwrap();
        assert_eq!(
            reduced_rgb(&mut bounded),
            reduced_rgb(&mut expected),
            "{name}"
        );
    }
}

/// The smallest working limit under which `bytes` constructs a decoder.
fn smallest_parsing_allowance(bytes: &[u8]) -> usize {
    use core::num::NonZeroUsize;
    use j2k::J2kDecodeControl;
    let parses = |limit: usize| {
        let mut control = J2kDecodeControl::default();
        control.max_working_memory_bytes = NonZeroUsize::new(limit);
        J2kDecoder::new_with_control(bytes, DecodeSettings::default(), &control).is_ok()
    };
    let (mut low, mut high) = (1, 1 << 20);
    assert!(parses(high));
    while low < high {
        let middle = low + (high - low) / 2;
        if parses(middle) {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    assert!(!parses(high - 1));
    high
}
