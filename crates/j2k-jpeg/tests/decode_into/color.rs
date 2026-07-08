// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn decode_region_into_rgb8_scaled_crops_constant_app14_rgb_pixels() {
    let bytes = rgb_app14_8x8_jpeg();
    let dec = Decoder::new(&bytes).unwrap();
    let roi = Rect {
        x: 2,
        y: 2,
        w: 4,
        h: 4,
    };
    let mut buf = vec![0u8; 2 * 2 * 3];
    let outcome = dec
        .decode_region_scaled_into(&mut buf, 2 * 3, PixelFormat::Rgb8, roi, Downscale::Half)
        .unwrap();
    assert_eq!(outcome.decoded, roi);
    let mut expected = Vec::with_capacity(buf.len());
    for _ in 0..4 {
        expected.extend_from_slice(&[200, 20, 10]);
    }
    assert_eq!(buf, expected);
}

#[test]
fn decode_into_rgb8_converts_cmyk_and_ycck() {
    for bytes in [cmyk_8x8_jpeg(), ycck_8x8_jpeg()] {
        let dec = Decoder::new(&bytes).expect("four-component baseline JPEG should construct");
        let (w, h) = dec.info().dimensions;
        let mut buf = vec![0u8; (w * h * 3) as usize];

        dec.decode_into(&mut buf, (w * 3) as usize, PixelFormat::Rgb8)
            .expect("CMYK/YCCK to RGB8 decode should succeed");

        assert_eq!(buf, four_component_8x8_rgb());
    }
}

#[test]
fn decode_into_rgba8_converts_cmyk_and_ycck_with_alpha() {
    for bytes in [cmyk_8x8_jpeg(), ycck_8x8_jpeg()] {
        let dec = Decoder::new(&bytes).expect("four-component baseline JPEG should construct");
        let (w, h) = dec.info().dimensions;
        let mut buf = vec![0u8; (w * h * 4) as usize];

        dec.decode_into(&mut buf, (w * 4) as usize, PixelFormat::Rgba8)
            .expect("CMYK/YCCK to RGBA8 decode should succeed");

        for pixel in buf.chunks_exact(4) {
            assert_eq!(pixel, &[64, 64, 64, 255]);
        }
    }
}

#[test]
fn decode_region_into_rgb8_crops_cmyk_and_ycck() {
    let roi = Rect {
        x: 2,
        y: 1,
        w: 5,
        h: 4,
    };
    let expected = crop_rgb(&four_component_8x8_rgb(), 8, roi);

    for bytes in [cmyk_8x8_jpeg(), ycck_8x8_jpeg()] {
        let dec = Decoder::new(&bytes).expect("four-component baseline JPEG should construct");
        let mut buf = vec![0u8; roi.w as usize * roi.h as usize * 3];

        let outcome = dec
            .decode_region_into(&mut buf, roi.w as usize * 3, PixelFormat::Rgb8, roi)
            .expect("CMYK/YCCK RGB8 ROI decode should succeed");

        assert_eq!(outcome.decoded, roi);
        assert_eq!(buf, expected);
    }
}

#[test]
fn decode_scaled_into_rgb8_projects_cmyk_and_ycck() {
    let expected = project_scaled_rgb(
        &four_component_8x8_rgb(),
        8,
        8,
        Rect {
            x: 0,
            y: 0,
            w: 4,
            h: 4,
        },
        2,
    );

    for bytes in [cmyk_8x8_jpeg(), ycck_8x8_jpeg()] {
        let dec = Decoder::new(&bytes).expect("four-component baseline JPEG should construct");
        let mut buf = vec![0u8; expected.len()];

        let outcome = dec
            .decode_scaled_into(&mut buf, 4 * 3, PixelFormat::Rgb8, Downscale::Half)
            .expect("CMYK/YCCK RGB8 scaled decode should succeed");

        assert_eq!(outcome.decoded, Rect::full((8, 8)));
        assert_eq!(buf, expected);
    }
}

#[test]
fn decode_scaled_into_rgba8_projects_cmyk_and_ycck() {
    let expected = rgb8_to_rgba8(
        &project_scaled_rgb(
            &four_component_8x8_rgb(),
            8,
            8,
            Rect {
                x: 0,
                y: 0,
                w: 4,
                h: 4,
            },
            2,
        ),
        255,
    );

    for bytes in [cmyk_8x8_jpeg(), ycck_8x8_jpeg()] {
        let dec = Decoder::new(&bytes).expect("four-component baseline JPEG should construct");
        let mut buf = vec![0u8; expected.len()];

        let outcome = dec
            .decode_scaled_into(&mut buf, 4 * 4, PixelFormat::Rgba8, Downscale::Half)
            .expect("CMYK/YCCK RGBA8 scaled decode should succeed");

        assert_eq!(outcome.decoded, Rect::full((8, 8)));
        assert_eq!(buf, expected);
    }
}

#[test]
fn decode_region_scaled_into_rgb8_projects_cmyk_and_ycck_with_padding() {
    let roi = Rect {
        x: 1,
        y: 1,
        w: 6,
        h: 6,
    };
    let scaled_roi = scaled_rect_covering_for_test(roi, 2);
    let expected = project_scaled_rgb(&four_component_8x8_rgb(), 8, 8, scaled_roi, 2);
    let row_bytes = scaled_roi.w as usize * 3;
    let stride = row_bytes + 5;

    for bytes in [cmyk_8x8_jpeg(), ycck_8x8_jpeg()] {
        let dec = Decoder::new(&bytes).expect("four-component baseline JPEG should construct");
        let mut buf = vec![0xaau8; stride * scaled_roi.h as usize];

        let outcome = dec
            .decode_region_scaled_into(&mut buf, stride, PixelFormat::Rgb8, roi, Downscale::Half)
            .expect("CMYK/YCCK RGB8 region-scaled decode should succeed");

        assert_eq!(outcome.decoded, roi);
        for (row, expected_row) in buf
            .chunks_exact(stride)
            .zip(expected.chunks_exact(row_bytes))
        {
            assert_eq!(&row[..row_bytes], expected_row);
            assert_eq!(&row[row_bytes..], &[0xaa; 5]);
        }
    }
}

#[test]
fn decode_region_scaled_into_rgba8_projects_cmyk_and_ycck_with_padding() {
    let roi = Rect {
        x: 1,
        y: 1,
        w: 6,
        h: 6,
    };
    let scaled_roi = scaled_rect_covering_for_test(roi, 2);
    let expected = rgb8_to_rgba8(
        &project_scaled_rgb(&four_component_8x8_rgb(), 8, 8, scaled_roi, 2),
        255,
    );
    let row_bytes = scaled_roi.w as usize * 4;
    let stride = row_bytes + 4;

    for bytes in [cmyk_8x8_jpeg(), ycck_8x8_jpeg()] {
        let dec = Decoder::new(&bytes).expect("four-component baseline JPEG should construct");
        let mut buf = vec![0xaau8; stride * scaled_roi.h as usize];

        let outcome = dec
            .decode_region_scaled_into(&mut buf, stride, PixelFormat::Rgba8, roi, Downscale::Half)
            .expect("CMYK/YCCK RGBA8 region-scaled decode should succeed");

        assert_eq!(outcome.decoded, roi);
        for (row, expected_row) in buf
            .chunks_exact(stride)
            .zip(expected.chunks_exact(row_bytes))
        {
            assert_eq!(&row[..row_bytes], expected_row);
            assert_eq!(&row[row_bytes..], &[0xaa; 4]);
        }
    }
}

#[test]
fn decode_subsampled_cmyk_ycck_full_and_region_scaled_outputs() {
    for (bytes, expected, width, height, label) in [
        (
            cmyk_16x8_422_jpeg(),
            four_component_16x8_rgb(),
            16,
            8,
            "CMYK 4:2:2",
        ),
        (
            ycck_16x8_422_jpeg(),
            four_component_16x8_rgb(),
            16,
            8,
            "YCCK 4:2:2",
        ),
        (
            cmyk_16x16_420_jpeg(),
            four_component_16x16_rgb(),
            16,
            16,
            "CMYK 4:2:0",
        ),
        (
            ycck_16x16_420_jpeg(),
            four_component_16x16_rgb(),
            16,
            16,
            "YCCK 4:2:0",
        ),
    ] {
        let dec = Decoder::new(&bytes)
            .unwrap_or_else(|err| panic!("{label} four-component JPEG should construct: {err}"));
        let mut full = vec![0u8; expected.len()];

        let outcome = dec
            .decode_into(
                &mut full,
                width * PixelFormat::Rgb8.bytes_per_pixel(),
                PixelFormat::Rgb8,
            )
            .unwrap_or_else(|err| panic!("{label} full RGB8 decode should succeed: {err}"));

        assert_eq!(
            outcome.decoded,
            Rect::full((width as u32, height as u32)),
            "{label}"
        );
        assert_eq!(full, expected, "{label}");

        let roi = Rect {
            x: width as u32 / 4,
            y: height as u32 / 4,
            w: width as u32 / 2,
            h: height as u32 / 2,
        };
        let scaled_roi = scaled_rect_covering_for_test(roi, 2);
        let row_bytes = scaled_roi.w as usize * PixelFormat::Rgba8.bytes_per_pixel();
        let stride = row_bytes + 4;
        let expected_rgba = rgb8_to_rgba8(
            &project_scaled_rgb(&expected, width as u32, height as u32, scaled_roi, 2),
            255,
        );
        let mut region = vec![0xaau8; stride * scaled_roi.h as usize];

        let outcome = dec
            .decode_region_scaled_into(
                &mut region,
                stride,
                PixelFormat::Rgba8,
                roi,
                Downscale::Half,
            )
            .unwrap_or_else(|err| {
                panic!("{label} region-scaled RGBA8 decode should succeed: {err}")
            });

        assert_eq!(outcome.decoded, roi, "{label}");
        for (row, expected_row) in region
            .chunks_exact(stride)
            .zip(expected_rgba.chunks_exact(row_bytes))
        {
            assert_eq!(&row[..row_bytes], expected_row, "{label}");
            assert_eq!(&row[row_bytes..], &[0xaa; 4], "{label}");
        }
    }
}

#[test]
fn decode_12bit_cmyk_ycck_full_roi_scaled_and_region_scaled_outputs() {
    for (bytes, expected_full, width, height, label) in [
        (
            extended_12bit_cmyk_8x8_jpeg(),
            four_component_12bit_8x8_rgb16(),
            8,
            8,
            "12-bit CMYK 4:4:4",
        ),
        (
            extended_12bit_ycck_8x8_jpeg(),
            four_component_12bit_8x8_rgb16(),
            8,
            8,
            "12-bit YCCK 4:4:4",
        ),
        (
            extended_12bit_cmyk_restart_16x8_jpeg(),
            four_component_12bit_16x8_rgb16(),
            16,
            8,
            "restart-coded 12-bit CMYK 4:4:4",
        ),
        (
            extended_12bit_ycck_restart_16x8_jpeg(),
            four_component_12bit_16x8_rgb16(),
            16,
            8,
            "restart-coded 12-bit YCCK 4:4:4",
        ),
        (
            extended_12bit_cmyk_16x8_422_jpeg(),
            four_component_12bit_16x8_rgb16(),
            16,
            8,
            "12-bit CMYK 4:2:2",
        ),
        (
            extended_12bit_ycck_16x8_422_jpeg(),
            four_component_12bit_16x8_rgb16(),
            16,
            8,
            "12-bit YCCK 4:2:2",
        ),
        (
            extended_12bit_cmyk_422_restart_32x8_jpeg(),
            four_component_12bit_32x8_rgb16(),
            32,
            8,
            "restart-coded 12-bit CMYK 4:2:2",
        ),
        (
            extended_12bit_ycck_422_restart_32x8_jpeg(),
            four_component_12bit_32x8_rgb16(),
            32,
            8,
            "restart-coded 12-bit YCCK 4:2:2",
        ),
        (
            extended_12bit_cmyk_16x16_420_jpeg(),
            four_component_12bit_16x16_rgb16(),
            16,
            16,
            "12-bit CMYK 4:2:0",
        ),
        (
            extended_12bit_ycck_16x16_420_jpeg(),
            four_component_12bit_16x16_rgb16(),
            16,
            16,
            "12-bit YCCK 4:2:0",
        ),
        (
            extended_12bit_cmyk_420_restart_32x16_jpeg(),
            four_component_12bit_32x16_rgb16(),
            32,
            16,
            "restart-coded 12-bit CMYK 4:2:0",
        ),
        (
            extended_12bit_ycck_420_restart_32x16_jpeg(),
            four_component_12bit_32x16_rgb16(),
            32,
            16,
            "restart-coded 12-bit YCCK 4:2:0",
        ),
        (
            progressive_12bit_cmyk_8x8_jpeg(),
            four_component_12bit_8x8_rgb16(),
            8,
            8,
            "progressive 12-bit CMYK 4:4:4",
        ),
        (
            progressive_12bit_ycck_8x8_jpeg(),
            four_component_12bit_8x8_rgb16(),
            8,
            8,
            "progressive 12-bit YCCK 4:4:4",
        ),
        (
            progressive_12bit_cmyk_restart_16x8_jpeg(),
            four_component_12bit_16x8_rgb16(),
            16,
            8,
            "restart-coded progressive 12-bit CMYK 4:4:4",
        ),
        (
            progressive_12bit_ycck_restart_16x8_jpeg(),
            four_component_12bit_16x8_rgb16(),
            16,
            8,
            "restart-coded progressive 12-bit YCCK 4:4:4",
        ),
        (
            progressive_12bit_cmyk_16x8_422_jpeg(),
            four_component_12bit_16x8_rgb16(),
            16,
            8,
            "progressive 12-bit CMYK 4:2:2",
        ),
        (
            progressive_12bit_ycck_16x8_422_jpeg(),
            four_component_12bit_16x8_rgb16(),
            16,
            8,
            "progressive 12-bit YCCK 4:2:2",
        ),
        (
            progressive_12bit_cmyk_422_restart_32x8_jpeg(),
            four_component_12bit_32x8_rgb16(),
            32,
            8,
            "restart-coded progressive 12-bit CMYK 4:2:2",
        ),
        (
            progressive_12bit_ycck_422_restart_32x8_jpeg(),
            four_component_12bit_32x8_rgb16(),
            32,
            8,
            "restart-coded progressive 12-bit YCCK 4:2:2",
        ),
        (
            progressive_12bit_cmyk_16x16_420_jpeg(),
            four_component_12bit_16x16_rgb16(),
            16,
            16,
            "progressive 12-bit CMYK 4:2:0",
        ),
        (
            progressive_12bit_ycck_16x16_420_jpeg(),
            four_component_12bit_16x16_rgb16(),
            16,
            16,
            "progressive 12-bit YCCK 4:2:0",
        ),
        (
            progressive_12bit_cmyk_420_restart_32x16_jpeg(),
            four_component_12bit_32x16_rgb16(),
            32,
            16,
            "restart-coded progressive 12-bit CMYK 4:2:0",
        ),
        (
            progressive_12bit_ycck_420_restart_32x16_jpeg(),
            four_component_12bit_32x16_rgb16(),
            32,
            16,
            "restart-coded progressive 12-bit YCCK 4:2:0",
        ),
    ] {
        let dec = Decoder::new(&bytes)
            .unwrap_or_else(|err| panic!("{label} decoder should construct: {err}"));
        let full_rect = Rect::full((width, height));

        let mut full =
            vec![0u8; width as usize * height as usize * PixelFormat::Rgb16.bytes_per_pixel()];
        let outcome = dec
            .decode_into(
                &mut full,
                width as usize * PixelFormat::Rgb16.bytes_per_pixel(),
                PixelFormat::Rgb16,
            )
            .unwrap_or_else(|err| panic!("{label} RGB16 full decode should succeed: {err}"));
        assert_eq!(outcome.decoded, full_rect, "{label}");
        assert_eq!(full, expected_full, "{label}");

        let roi = Rect {
            x: 1,
            y: 2,
            w: 5,
            h: 4,
        };
        let expected_roi = crop_rgb16_bytes(&expected_full, width as usize, roi);
        let mut roi_buf = vec![0u8; roi.w as usize * roi.h as usize * 6];
        let outcome = dec
            .decode_region_into(
                &mut roi_buf,
                roi.w as usize * PixelFormat::Rgb16.bytes_per_pixel(),
                PixelFormat::Rgb16,
                roi,
            )
            .unwrap_or_else(|err| panic!("{label} RGB16 ROI decode should succeed: {err}"));
        assert_eq!(outcome.decoded, roi, "{label}");
        assert_eq!(roi_buf, expected_roi, "{label}");

        let scaled_rect = scaled_rect_covering_for_test(full_rect, 2);
        let scaled_row_bytes = scaled_rect.w as usize * PixelFormat::Rgba16.bytes_per_pixel();
        let scaled_stride = scaled_row_bytes + 8;
        let expected_scaled = rgb16_to_rgba16(
            &expected_scaled_rgb16_pixels(&expected_full, width as usize, full_rect, 2),
            u16::MAX,
        );
        let mut scaled = vec![0xaau8; scaled_stride * scaled_rect.h as usize];
        let outcome = dec
            .decode_scaled_into(
                &mut scaled,
                scaled_stride,
                PixelFormat::Rgba16,
                Downscale::Half,
            )
            .unwrap_or_else(|err| panic!("{label} RGBA16 scaled decode should succeed: {err}"));
        assert_eq!(outcome.decoded, full_rect, "{label}");
        assert_padded_rgba16_rows(
            &scaled,
            scaled_stride,
            scaled_rect.w as usize,
            &expected_scaled,
        );

        let region_scaled = scaled_rect_covering_for_test(roi, 2);
        let region_scaled_row_bytes =
            region_scaled.w as usize * PixelFormat::Rgba16.bytes_per_pixel();
        let region_scaled_stride = region_scaled_row_bytes + 8;
        let expected_region_scaled = rgb16_to_rgba16(
            &expected_scaled_rgb16_pixels(&expected_full, width as usize, roi, 2),
            u16::MAX,
        );
        let mut region_scaled_buf = vec![0xaau8; region_scaled_stride * region_scaled.h as usize];
        let outcome = dec
            .decode_region_scaled_into(
                &mut region_scaled_buf,
                region_scaled_stride,
                PixelFormat::Rgba16,
                roi,
                Downscale::Half,
            )
            .unwrap_or_else(|err| {
                panic!("{label} RGBA16 region-scaled decode should succeed: {err}")
            });
        assert_eq!(outcome.decoded, roi, "{label}");
        assert_padded_rgba16_rows(
            &region_scaled_buf,
            region_scaled_stride,
            region_scaled.w as usize,
            &expected_region_scaled,
        );
    }
}

#[test]
fn decode_12bit_cmyk_ycck_nonconstant_full_and_region_scaled_outputs() {
    for (bytes, expected_full, label) in [
        (
            extended_12bit_cmyk_nonconstant_8x8_jpeg(),
            four_component_12bit_8x8_cmyk_nonconstant_rgb16(),
            "12-bit extended CMYK non-constant",
        ),
        (
            extended_12bit_ycck_nonconstant_8x8_jpeg(),
            four_component_12bit_8x8_ycck_nonconstant_rgb16(),
            "12-bit extended YCCK non-constant",
        ),
        (
            progressive_12bit_cmyk_nonconstant_8x8_jpeg(),
            four_component_12bit_8x8_cmyk_nonconstant_rgb16(),
            "12-bit progressive CMYK non-constant",
        ),
        (
            progressive_12bit_ycck_nonconstant_8x8_jpeg(),
            four_component_12bit_8x8_ycck_nonconstant_rgb16(),
            "12-bit progressive YCCK non-constant",
        ),
    ] {
        let dec = Decoder::new(&bytes)
            .unwrap_or_else(|err| panic!("{label} decoder should construct: {err}"));
        let mut full = vec![0u8; expected_full.len()];

        dec.decode_into(&mut full, 8 * 6, PixelFormat::Rgb16)
            .unwrap_or_else(|err| panic!("{label} full RGB16 decode should succeed: {err}"));

        assert_eq!(full, expected_full, "{label}");

        let roi = Rect {
            x: 1,
            y: 1,
            w: 6,
            h: 6,
        };
        let region_scaled = scaled_rect_covering_for_test(roi, 2);
        let row_bytes = region_scaled.w as usize * PixelFormat::Rgba16.bytes_per_pixel();
        let stride = row_bytes + 8;
        let expected_region_scaled = rgb16_to_rgba16(
            &expected_scaled_rgb16_pixels(&expected_full, 8, roi, 2),
            u16::MAX,
        );
        let mut region_scaled_buf = vec![0xaau8; stride * region_scaled.h as usize];

        let outcome = dec
            .decode_region_scaled_into(
                &mut region_scaled_buf,
                stride,
                PixelFormat::Rgba16,
                roi,
                Downscale::Half,
            )
            .unwrap_or_else(|err| {
                panic!("{label} region-scaled RGBA16 decode should succeed: {err}")
            });

        assert_eq!(outcome.decoded, roi, "{label}");
        assert_padded_rgba16_rows(
            &region_scaled_buf,
            stride,
            region_scaled.w as usize,
            &expected_region_scaled,
        );
    }
}

#[test]
fn decode_nonleading_max_four_component_sampling_uses_generic_upsample() {
    for (bytes, label) in [
        (cmyk_16x8_nonleading_max_422_jpeg(), "non-leading-max CMYK"),
        (ycck_16x8_nonleading_max_422_jpeg(), "non-leading-max YCCK"),
    ] {
        let expected = four_component_16x8_rgb();
        let dec = Decoder::new(&bytes)
            .unwrap_or_else(|err| panic!("{label} sampling should use generic upsample: {err}"));
        let mut full = vec![0u8; expected.len()];

        dec.decode_into(&mut full, 16 * 3, PixelFormat::Rgb8)
            .unwrap_or_else(|err| panic!("{label} full decode should succeed: {err}"));

        assert_eq!(full, expected, "{label}");

        let roi = Rect {
            x: 4,
            y: 2,
            w: 8,
            h: 4,
        };
        let scaled_roi = scaled_rect_covering_for_test(roi, 2);
        let row_bytes = scaled_roi.w as usize * PixelFormat::Rgba8.bytes_per_pixel();
        let stride = row_bytes + 4;
        let expected_rgba =
            rgb8_to_rgba8(&project_scaled_rgb(&expected, 16, 8, scaled_roi, 2), 255);
        let mut region = vec![0xaau8; stride * scaled_roi.h as usize];

        let outcome = dec
            .decode_region_scaled_into(
                &mut region,
                stride,
                PixelFormat::Rgba8,
                roi,
                Downscale::Half,
            )
            .unwrap_or_else(|err| panic!("{label} region-scaled decode should succeed: {err}"));

        assert_eq!(outcome.decoded, roi, "{label}");
        for (row, expected_row) in region
            .chunks_exact(stride)
            .zip(expected_rgba.chunks_exact(row_bytes))
        {
            assert_eq!(&row[..row_bytes], expected_row, "{label}");
            assert_eq!(&row[row_bytes..], &[0xaa; 4], "{label}");
        }
    }
}

#[test]
fn decoder_new_rejects_malformed_four_component_sampling_shape() {
    let input = malformed_cmyk_nondivisible_sampling_jpeg();
    let Err(err) = Decoder::new(&input) else {
        panic!("malformed four-component sampling should reject construction");
    };

    assert!(
        matches!(
            err,
            JpegError::NotImplemented {
                sof: SofKind::Baseline8
            }
        ),
        "{err}"
    );
}

#[test]
fn decode_region_into_rgba8_crops_cmyk_and_ycck_with_alpha() {
    let roi = Rect {
        x: 3,
        y: 2,
        w: 3,
        h: 4,
    };
    let stride = roi.w as usize * 4 + 4;

    for bytes in [cmyk_8x8_jpeg(), ycck_8x8_jpeg()] {
        let dec = Decoder::new(&bytes).expect("four-component baseline JPEG should construct");
        let mut buf = vec![0xaau8; stride * roi.h as usize];

        let outcome = dec
            .decode_region_into(&mut buf, stride, PixelFormat::Rgba8, roi)
            .expect("CMYK/YCCK RGBA8 ROI decode should succeed");

        assert_eq!(outcome.decoded, roi);
        for row in buf.chunks_exact(stride) {
            for pixel in row[..roi.w as usize * 4].chunks_exact(4) {
                assert_eq!(pixel, &[64, 64, 64, 255]);
            }
            assert_eq!(&row[roi.w as usize * 4..], &[0xaa; 4]);
        }
    }
}

#[test]
fn decoder_new_rejects_invalid_sequential_scan_parameters() {
    let mut bytes = minimal_baseline_420_jpeg();
    let sos = bytes
        .windows(2)
        .position(|w| w == [0xff, 0xda])
        .expect("fixture SOS");
    bytes[sos + 2 + 2 + 1 + 3 * 2] = 1;

    let err = Decoder::new(&bytes).expect_err("baseline Ss=1 must be rejected");
    assert!(matches!(
        err,
        JpegError::InvalidScanParameters {
            ss: 1,
            se: 63,
            ah: 0,
            al: 0,
            ..
        }
    ));
}
