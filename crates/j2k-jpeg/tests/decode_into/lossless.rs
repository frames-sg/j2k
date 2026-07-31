// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

struct LosslessFullCase {
    label: &'static str,
    fixture: fn(u8) -> Vec<u8>,
    expected: Vec<u8>,
    format: PixelFormat,
    restart: bool,
}

fn assert_lossless_full_case(case: &LosslessFullCase) {
    for predictor in 1..=7 {
        let bytes = (case.fixture)(predictor);
        let decoder = Decoder::new(&bytes).unwrap_or_else(|error| {
            panic!(
                "{} predictor-{predictor} decoder must construct: {error}",
                case.label
            )
        });
        if case.restart {
            assert_eq!(
                decoder.info().restart_interval,
                Some(3),
                "{} predictor-{predictor}",
                case.label
            );
        }
        let (width, height) = decoder.info().dimensions;
        let stride = width as usize * case.format.bytes_per_pixel();
        let mut output = vec![0u8; stride * height as usize];

        let outcome = decoder
            .decode_into(&mut output, stride, case.format)
            .unwrap_or_else(|error| {
                panic!(
                    "{} predictor-{predictor} decode must succeed: {error}",
                    case.label
                )
            });

        assert_eq!(
            outcome.decoded,
            Rect::full((width, height)),
            "{} predictor-{predictor}",
            case.label
        );
        assert_eq!(
            output, case.expected,
            "{} predictor-{predictor}",
            case.label
        );
    }
}

#[test]
fn decode_into_8bit_lossless_predictor_matrix_matches_fixtures() {
    let ycbcr = lossless_ycbcr_3x3_rgb8();
    for case in [
        LosslessFullCase {
            label: "lossless 8-bit grayscale",
            fixture: lossless_predictor_grayscale_3x3_jpeg,
            expected: LOSSLESS_GRAYSCALE_3X3_PIXELS.to_vec(),
            format: PixelFormat::Gray8,
            restart: false,
        },
        LosslessFullCase {
            label: "restart-coded lossless 8-bit grayscale",
            fixture: lossless_restart_predictor_grayscale_3x3_jpeg,
            expected: LOSSLESS_GRAYSCALE_3X3_PIXELS.to_vec(),
            format: PixelFormat::Gray8,
            restart: true,
        },
        LosslessFullCase {
            label: "lossless 8-bit APP14 RGB",
            fixture: lossless_predictor_rgb_3x3_jpeg,
            expected: LOSSLESS_RGB_3X3_PIXELS.to_vec(),
            format: PixelFormat::Rgb8,
            restart: false,
        },
        LosslessFullCase {
            label: "restart-coded lossless 8-bit APP14 RGB",
            fixture: lossless_restart_predictor_rgb_3x3_jpeg,
            expected: LOSSLESS_RGB_3X3_PIXELS.to_vec(),
            format: PixelFormat::Rgb8,
            restart: true,
        },
        LosslessFullCase {
            label: "lossless 8-bit YCbCr",
            fixture: lossless_predictor_ycbcr_3x3_jpeg,
            expected: ycbcr.clone(),
            format: PixelFormat::Rgb8,
            restart: false,
        },
        LosslessFullCase {
            label: "restart-coded lossless 8-bit YCbCr",
            fixture: lossless_restart_predictor_ycbcr_3x3_jpeg,
            expected: ycbcr,
            format: PixelFormat::Rgb8,
            restart: true,
        },
    ] {
        assert_lossless_full_case(&case);
    }
}

#[test]
fn decode_into_rgba8_accepts_lossless_color_common_predictors() {
    for predictor in 1..=7 {
        for (bytes, expected_rgb, label) in [
            (
                lossless_predictor_rgb_3x3_jpeg(predictor),
                LOSSLESS_RGB_3X3_PIXELS.to_vec(),
                "APP14 RGB",
            ),
            (
                lossless_predictor_ycbcr_3x3_jpeg(predictor),
                lossless_ycbcr_3x3_rgb8(),
                "YCbCr",
            ),
        ] {
            let dec = Decoder::new(&bytes).unwrap_or_else(|err| {
                panic!("lossless predictor-{predictor} {label} JPEG must construct: {err}")
            });
            let (w, h) = dec.info().dimensions;
            let stride = w as usize * PixelFormat::Rgba8.bytes_per_pixel();
            let mut buf = vec![0u8; stride * h as usize];

            let outcome = dec
                .decode_into(&mut buf, stride, PixelFormat::Rgba8)
                .unwrap_or_else(|err| {
                    panic!(
                        "lossless predictor-{predictor} {label} RGBA8 decode must succeed: {err}"
                    )
                });

            assert_eq!(outcome.decoded, Rect::full((w, h)));
            assert_eq!(
                buf,
                rgb8_to_rgba8(&expected_rgb, 255),
                "{label} predictor {predictor}"
            );
        }
    }
}

#[test]
fn decode_region_into_rgba8_crops_restart_coded_lossless_color() {
    let roi = Rect {
        x: 1,
        y: 1,
        w: 2,
        h: 2,
    };
    let stride = roi.w as usize * PixelFormat::Rgba8.bytes_per_pixel() + 4;

    for (bytes, expected_rgb, label) in [
        (
            lossless_restart_predictor_rgb_3x3_jpeg(4),
            LOSSLESS_RGB_3X3_PIXELS.to_vec(),
            "APP14 RGB",
        ),
        (
            lossless_restart_predictor_ycbcr_3x3_jpeg(4),
            lossless_ycbcr_3x3_rgb8(),
            "YCbCr",
        ),
    ] {
        let dec = Decoder::new(&bytes).unwrap_or_else(|err| {
            panic!("restart-coded lossless {label} JPEG must construct: {err}")
        });
        let expected = rgb8_to_rgba8(&crop_rgb(&expected_rgb, 3, roi), 255);
        let row_bytes = roi.w as usize * PixelFormat::Rgba8.bytes_per_pixel();
        let mut buf = vec![0xaau8; stride * roi.h as usize];

        let outcome = dec
            .decode_region_into(&mut buf, stride, PixelFormat::Rgba8, roi)
            .unwrap_or_else(|err| {
                panic!("restart-coded lossless {label} RGBA8 ROI decode must succeed: {err}")
            });

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
fn decode_scaled_into_rgba8_projects_lossless_color() {
    let scaled = Rect {
        x: 0,
        y: 0,
        w: 2,
        h: 2,
    };

    for (bytes, expected_rgb, label) in [
        (
            lossless_predictor_rgb_3x3_jpeg(5),
            LOSSLESS_RGB_3X3_PIXELS.to_vec(),
            "APP14 RGB",
        ),
        (
            lossless_predictor_ycbcr_3x3_jpeg(5),
            lossless_ycbcr_3x3_rgb8(),
            "YCbCr",
        ),
    ] {
        let dec = Decoder::new(&bytes)
            .unwrap_or_else(|err| panic!("lossless {label} JPEG must construct: {err}"));
        let stride = scaled.w as usize * PixelFormat::Rgba8.bytes_per_pixel();
        let mut buf = vec![0u8; stride * scaled.h as usize];
        let expected = rgb8_to_rgba8(&project_scaled_rgb(&expected_rgb, 3, 3, scaled, 2), 255);

        let outcome = dec
            .decode_scaled_into(&mut buf, stride, PixelFormat::Rgba8, Downscale::Half)
            .unwrap_or_else(|err| {
                panic!("lossless {label} RGBA8 scaled decode must succeed: {err}")
            });

        assert_eq!(outcome.decoded, Rect::full((3, 3)));
        assert_eq!(buf, expected);
    }
}

#[test]
fn decode_region_scaled_into_rgba8_projects_restart_coded_lossless_color() {
    let roi = Rect {
        x: 1,
        y: 1,
        w: 2,
        h: 2,
    };
    let scaled_roi = scaled_rect_covering_for_test(roi, 2);
    let row_bytes = scaled_roi.w as usize * PixelFormat::Rgba8.bytes_per_pixel();
    let stride = row_bytes + 4;

    for (bytes, expected_rgb, label) in [
        (
            lossless_restart_predictor_rgb_3x3_jpeg(5),
            LOSSLESS_RGB_3X3_PIXELS.to_vec(),
            "APP14 RGB",
        ),
        (
            lossless_restart_predictor_ycbcr_3x3_jpeg(5),
            lossless_ycbcr_3x3_rgb8(),
            "YCbCr",
        ),
    ] {
        let dec = Decoder::new(&bytes).unwrap_or_else(|err| {
            panic!("restart-coded lossless {label} JPEG must construct: {err}")
        });
        let expected = rgb8_to_rgba8(&project_scaled_rgb(&expected_rgb, 3, 3, scaled_roi, 2), 255);
        let mut buf = vec![0xaau8; stride * scaled_roi.h as usize];

        let outcome = dec
            .decode_region_scaled_into(&mut buf, stride, PixelFormat::Rgba8, roi, Downscale::Half)
            .unwrap_or_else(|err| {
                panic!(
                    "restart-coded lossless {label} RGBA8 region-scaled decode must succeed: {err}"
                )
            });

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
fn decode_into_16bit_lossless_predictor_matrix_matches_fixtures() {
    let rgb = rgb16_samples_to_le_bytes(&LOSSLESS_RGB_16BIT_3X3_PIXELS);
    let ycbcr = lossless_ycbcr_16bit_3x3_rgb16();
    let gray = rgb16_samples_to_le_bytes(&LOSSLESS_GRAYSCALE_16BIT_3X3_PIXELS);
    for case in [
        LosslessFullCase {
            label: "lossless 16-bit APP14 RGB",
            fixture: lossless_predictor_rgb_16bit_3x3_jpeg,
            expected: rgb.clone(),
            format: PixelFormat::Rgb16,
            restart: false,
        },
        LosslessFullCase {
            label: "restart-coded lossless 16-bit APP14 RGB",
            fixture: lossless_restart_predictor_rgb_16bit_3x3_jpeg,
            expected: rgb,
            format: PixelFormat::Rgb16,
            restart: true,
        },
        LosslessFullCase {
            label: "lossless 16-bit YCbCr",
            fixture: lossless_predictor_ycbcr_16bit_3x3_jpeg,
            expected: ycbcr.clone(),
            format: PixelFormat::Rgb16,
            restart: false,
        },
        LosslessFullCase {
            label: "restart-coded lossless 16-bit YCbCr",
            fixture: lossless_restart_predictor_ycbcr_16bit_3x3_jpeg,
            expected: ycbcr,
            format: PixelFormat::Rgb16,
            restart: true,
        },
        LosslessFullCase {
            label: "lossless 16-bit grayscale",
            fixture: lossless_predictor_grayscale_16bit_3x3_jpeg,
            expected: gray.clone(),
            format: PixelFormat::Gray16,
            restart: false,
        },
        LosslessFullCase {
            label: "restart-coded lossless 16-bit grayscale",
            fixture: lossless_restart_predictor_grayscale_16bit_3x3_jpeg,
            expected: gray,
            format: PixelFormat::Gray16,
            restart: true,
        },
    ] {
        assert_lossless_full_case(&case);
    }
}

#[test]
fn decode_8bit_lossless_422_color_full_roi_scaled_and_region_scaled_outputs() {
    let cases = [
        (
            "APP14 RGB",
            lossless_rgb_8bit_422_4x2_jpeg(4),
            lossless_rgb_8bit_422_4x2_rgb8(),
        ),
        (
            "APP14 RGB restart",
            lossless_rgb_8bit_422_restart_4x2_jpeg(4),
            lossless_rgb_8bit_422_4x2_rgb8(),
        ),
        (
            "YCbCr",
            lossless_ycbcr_8bit_422_4x2_jpeg(4),
            lossless_ycbcr_8bit_422_4x2_rgb8(),
        ),
        (
            "YCbCr restart",
            lossless_ycbcr_8bit_422_restart_4x2_jpeg(4),
            lossless_ycbcr_8bit_422_4x2_rgb8(),
        ),
    ];

    for (label, bytes, expected_full) in cases {
        assert_8bit_lossless_sampled_color_decode(
            label,
            &bytes,
            &expected_full,
            (4, 2),
            &[(2, 1), (1, 1), (1, 1)],
            Rect {
                x: 1,
                y: 0,
                w: 2,
                h: 2,
            },
        );
    }
}

#[test]
fn decode_8bit_lossless_420_color_full_roi_scaled_and_region_scaled_outputs() {
    let cases = [
        (
            "APP14 RGB",
            lossless_rgb_8bit_420_4x4_jpeg(4),
            lossless_rgb_8bit_420_4x4_rgb8(),
        ),
        (
            "APP14 RGB restart",
            lossless_rgb_8bit_420_restart_4x4_jpeg(4),
            lossless_rgb_8bit_420_4x4_rgb8(),
        ),
        (
            "YCbCr",
            lossless_ycbcr_8bit_420_4x4_jpeg(4),
            lossless_ycbcr_8bit_420_4x4_rgb8(),
        ),
        (
            "YCbCr restart",
            lossless_ycbcr_8bit_420_restart_4x4_jpeg(4),
            lossless_ycbcr_8bit_420_4x4_rgb8(),
        ),
    ];

    for (label, bytes, expected_full) in cases {
        assert_8bit_lossless_sampled_color_decode(
            label,
            &bytes,
            &expected_full,
            (4, 4),
            &[(2, 2), (1, 1), (1, 1)],
            Rect {
                x: 1,
                y: 1,
                w: 2,
                h: 2,
            },
        );
    }
}

fn assert_8bit_lossless_sampled_color_decode(
    label: &str,
    bytes: &[u8],
    expected_full: &[u8],
    dimensions: (u32, u32),
    sampling: &[(u8, u8)],
    roi: Rect,
) {
    let dec = Decoder::new(bytes)
        .unwrap_or_else(|err| panic!("8-bit lossless sampled {label} JPEG must construct: {err}"));
    let (w, h) = dec.info().dimensions;
    assert_eq!((w, h), dimensions, "{label}");
    assert_eq!(dec.info().sampling.components(), sampling, "{label}");

    let stride = w as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let mut full = vec![0u8; stride * h as usize];
    let outcome = dec
        .decode_into(&mut full, stride, PixelFormat::Rgb8)
        .unwrap_or_else(|err| panic!("8-bit lossless sampled {label} full decode: {err}"));
    assert_eq!(outcome.decoded, Rect::full((w, h)), "{label}");
    assert_eq!(full, expected_full, "{label}");

    let roi_stride = roi.w as usize * PixelFormat::Rgb8.bytes_per_pixel() + 3;
    let mut roi_buf = vec![0xaau8; roi_stride * roi.h as usize];
    let expected_roi = crop_rgb(expected_full, w, roi);
    let outcome = dec
        .decode_region_into(&mut roi_buf, roi_stride, PixelFormat::Rgb8, roi)
        .unwrap_or_else(|err| panic!("8-bit lossless sampled {label} ROI decode: {err}"));
    assert_eq!(outcome.decoded, roi, "{label}");
    let roi_row_bytes = roi.w as usize * PixelFormat::Rgb8.bytes_per_pixel();
    for (row, expected_row) in roi_buf
        .chunks_exact(roi_stride)
        .zip(expected_roi.chunks_exact(roi_row_bytes))
    {
        assert_eq!(&row[..roi_row_bytes], expected_row, "{label}");
        assert_eq!(&row[roi_row_bytes..], &[0xaa; 3], "{label}");
    }

    let scaled = scaled_rect_covering_for_test(Rect::full((w, h)), 2);
    let scaled_stride = scaled.w as usize * PixelFormat::Rgba8.bytes_per_pixel();
    let mut scaled_buf = vec![0u8; scaled_stride * scaled.h as usize];
    let expected_scaled = rgb8_to_rgba8(&project_scaled_rgb(expected_full, w, h, scaled, 2), 255);
    let outcome = dec
        .decode_scaled_into(
            &mut scaled_buf,
            scaled_stride,
            PixelFormat::Rgba8,
            Downscale::Half,
        )
        .unwrap_or_else(|err| panic!("8-bit lossless sampled {label} scaled decode: {err}"));
    assert_eq!(outcome.decoded, Rect::full((w, h)), "{label}");
    assert_eq!(scaled_buf, expected_scaled, "{label}");

    let scaled_roi = scaled_rect_covering_for_test(roi, 2);
    let scaled_roi_stride = scaled_roi.w as usize * PixelFormat::Rgba8.bytes_per_pixel() + 4;
    let scaled_roi_row_bytes = scaled_roi.w as usize * PixelFormat::Rgba8.bytes_per_pixel();
    let mut scaled_roi_buf = vec![0xaau8; scaled_roi_stride * scaled_roi.h as usize];
    let expected_scaled_roi =
        rgb8_to_rgba8(&project_scaled_rgb(expected_full, w, h, scaled_roi, 2), 255);
    let outcome = dec
        .decode_region_scaled_into(
            &mut scaled_roi_buf,
            scaled_roi_stride,
            PixelFormat::Rgba8,
            roi,
            Downscale::Half,
        )
        .unwrap_or_else(|err| panic!("8-bit lossless sampled {label} region-scaled decode: {err}"));
    assert_eq!(outcome.decoded, roi, "{label}");
    for (row, expected_row) in scaled_roi_buf
        .chunks_exact(scaled_roi_stride)
        .zip(expected_scaled_roi.chunks_exact(scaled_roi_row_bytes))
    {
        assert_eq!(&row[..scaled_roi_row_bytes], expected_row, "{label}");
        assert_eq!(&row[scaled_roi_row_bytes..], &[0xaa; 4], "{label}");
    }
}

#[test]
fn decode_16bit_lossless_422_color_full_roi_scaled_and_region_scaled_outputs() {
    for (label, bytes, expected_full) in [
        (
            "APP14 RGB",
            lossless_rgb_16bit_422_4x2_jpeg(4),
            lossless_rgb_16bit_422_4x2_rgb16(),
        ),
        (
            "APP14 RGB restart",
            lossless_rgb_16bit_422_restart_4x2_jpeg(4),
            lossless_rgb_16bit_422_4x2_rgb16(),
        ),
        (
            "YCbCr",
            lossless_ycbcr_16bit_422_4x2_jpeg(4),
            lossless_ycbcr_16bit_422_4x2_rgb16(),
        ),
        (
            "YCbCr restart",
            lossless_ycbcr_16bit_422_restart_4x2_jpeg(4),
            lossless_ycbcr_16bit_422_4x2_rgb16(),
        ),
    ] {
        assert_16bit_lossless_sampled_color_decode(
            label,
            &bytes,
            &expected_full,
            (4, 2),
            &[(2, 1), (1, 1), (1, 1)],
            Rect {
                x: 1,
                y: 0,
                w: 2,
                h: 2,
            },
        );
    }
}

#[test]
fn decode_16bit_lossless_420_color_full_roi_scaled_and_region_scaled_outputs() {
    for (label, bytes, expected_full) in [
        (
            "APP14 RGB",
            lossless_rgb_16bit_420_4x4_jpeg(4),
            lossless_rgb_16bit_420_4x4_rgb16(),
        ),
        (
            "APP14 RGB restart",
            lossless_rgb_16bit_420_restart_4x4_jpeg(4),
            lossless_rgb_16bit_420_4x4_rgb16(),
        ),
        (
            "YCbCr",
            lossless_ycbcr_16bit_420_4x4_jpeg(4),
            lossless_ycbcr_16bit_420_4x4_rgb16(),
        ),
        (
            "YCbCr restart",
            lossless_ycbcr_16bit_420_restart_4x4_jpeg(4),
            lossless_ycbcr_16bit_420_4x4_rgb16(),
        ),
    ] {
        assert_16bit_lossless_sampled_color_decode(
            label,
            &bytes,
            &expected_full,
            (4, 4),
            &[(2, 2), (1, 1), (1, 1)],
            Rect {
                x: 1,
                y: 1,
                w: 2,
                h: 2,
            },
        );
    }
}

fn assert_16bit_lossless_sampled_color_decode(
    label: &str,
    bytes: &[u8],
    expected_full: &[u8],
    dimensions: (u32, u32),
    sampling: &[(u8, u8)],
    roi: Rect,
) {
    let dec = Decoder::new(bytes)
        .unwrap_or_else(|err| panic!("16-bit lossless sampled {label} JPEG must construct: {err}"));
    let (w, h) = dec.info().dimensions;
    assert_eq!((w, h), dimensions, "{label}");
    assert_eq!(dec.info().sampling.components(), sampling, "{label}");
    assert_eq!(
        dec.info().sampling.max_h,
        sampling.iter().map(|&(h, _)| h).max().unwrap(),
        "{label}"
    );
    assert_eq!(
        dec.info().sampling.max_v,
        sampling.iter().map(|&(_, v)| v).max().unwrap(),
        "{label}"
    );

    let stride = w as usize * PixelFormat::Rgb16.bytes_per_pixel();
    let mut full = vec![0u8; stride * h as usize];
    let outcome = dec
        .decode_into(&mut full, stride, PixelFormat::Rgb16)
        .unwrap_or_else(|err| panic!("16-bit lossless sampled {label} full decode: {err}"));
    assert_eq!(outcome.decoded, Rect::full((w, h)), "{label}");
    assert_rgb16_image_eq(&full, expected_full, w as usize);

    let roi_stride = roi.w as usize * PixelFormat::Rgb16.bytes_per_pixel() + 4;
    let mut roi_buf = vec![0xaau8; roi_stride * roi.h as usize];
    let expected_roi = crop_rgb16_bytes(expected_full, w as usize, roi);
    let outcome = dec
        .decode_region_into(&mut roi_buf, roi_stride, PixelFormat::Rgb16, roi)
        .unwrap_or_else(|err| panic!("16-bit lossless sampled {label} ROI decode: {err}"));
    assert_eq!(outcome.decoded, roi, "{label}");
    let roi_row_bytes = roi.w as usize * PixelFormat::Rgb16.bytes_per_pixel();
    for (row, expected_row) in roi_buf
        .chunks_exact(roi_stride)
        .zip(expected_roi.chunks_exact(roi_row_bytes))
    {
        assert_eq!(&row[..roi_row_bytes], expected_row, "{label}");
        assert_eq!(&row[roi_row_bytes..], &[0xaa; 4], "{label}");
    }

    let scaled = scaled_rect_covering_for_test(Rect::full((w, h)), 2);
    let scaled_stride = scaled.w as usize * PixelFormat::Rgba16.bytes_per_pixel();
    let mut scaled_buf = vec![0u8; scaled_stride * scaled.h as usize];
    let expected_scaled = rgb16_to_rgba16(
        &expected_scaled_rgb16_pixels(expected_full, w as usize, Rect::full((w, h)), 2),
        u16::MAX,
    );
    let outcome = dec
        .decode_scaled_into(
            &mut scaled_buf,
            scaled_stride,
            PixelFormat::Rgba16,
            Downscale::Half,
        )
        .unwrap_or_else(|err| panic!("16-bit lossless sampled {label} scaled decode: {err}"));
    assert_eq!(outcome.decoded, Rect::full((w, h)), "{label}");
    assert_eq!(scaled_buf, expected_scaled, "{label}");

    let scaled_roi = scaled_rect_covering_for_test(roi, 2);
    let scaled_roi_stride = scaled_roi.w as usize * PixelFormat::Rgba16.bytes_per_pixel() + 8;
    let scaled_roi_row_bytes = scaled_roi.w as usize * PixelFormat::Rgba16.bytes_per_pixel();
    let mut scaled_roi_buf = vec![0xaau8; scaled_roi_stride * scaled_roi.h as usize];
    let expected_scaled_roi = rgb16_to_rgba16(
        &expected_scaled_rgb16_pixels(expected_full, w as usize, roi, 2),
        u16::MAX,
    );
    let outcome = dec
        .decode_region_scaled_into(
            &mut scaled_roi_buf,
            scaled_roi_stride,
            PixelFormat::Rgba16,
            roi,
            Downscale::Half,
        )
        .unwrap_or_else(|err| {
            panic!("16-bit lossless sampled {label} region-scaled decode: {err}")
        });
    assert_eq!(outcome.decoded, roi, "{label}");
    for (row, expected_row) in scaled_roi_buf
        .chunks_exact(scaled_roi_stride)
        .zip(expected_scaled_roi.chunks_exact(scaled_roi_row_bytes))
    {
        assert_eq!(&row[..scaled_roi_row_bytes], expected_row, "{label}");
        assert_eq!(&row[scaled_roi_row_bytes..], &[0xaa; 8], "{label}");
    }
}

#[test]
fn decode_into_rgba16_accepts_lossless_color_common_predictors() {
    for predictor in 1..=7 {
        for (bytes, expected_rgb, label) in [
            (
                lossless_predictor_rgb_16bit_3x3_jpeg(predictor),
                rgb16_samples_to_le_bytes(&LOSSLESS_RGB_16BIT_3X3_PIXELS),
                "APP14 RGB",
            ),
            (
                lossless_predictor_ycbcr_16bit_3x3_jpeg(predictor),
                lossless_ycbcr_16bit_3x3_rgb16(),
                "YCbCr",
            ),
        ] {
            let dec = Decoder::new(&bytes).unwrap_or_else(|err| {
                panic!("lossless 16-bit predictor-{predictor} {label} JPEG must construct: {err}")
            });
            let (w, h) = dec.info().dimensions;
            let stride = w as usize * PixelFormat::Rgba16.bytes_per_pixel();
            let mut buf = vec![0u8; stride * h as usize];

            let outcome = dec
                .decode_into(&mut buf, stride, PixelFormat::Rgba16)
                .unwrap_or_else(|err| {
                    panic!(
                        "lossless 16-bit predictor-{predictor} {label} RGBA16 decode must succeed: {err}"
                    )
                });

            assert_eq!(outcome.decoded, Rect::full((w, h)));
            assert_eq!(
                buf,
                rgb16_to_rgba16(&expected_rgb, u16::MAX),
                "{label} predictor {predictor}"
            );
        }
    }
}

#[test]
fn decode_region_into_rgba16_crops_restart_coded_lossless_color() {
    let roi = Rect {
        x: 1,
        y: 1,
        w: 2,
        h: 2,
    };
    let row_bytes = roi.w as usize * PixelFormat::Rgba16.bytes_per_pixel();
    let stride = row_bytes + 8;

    for (bytes, expected_rgb, label) in [
        (
            lossless_restart_predictor_rgb_16bit_3x3_jpeg(4),
            rgb16_samples_to_le_bytes(&LOSSLESS_RGB_16BIT_3X3_PIXELS),
            "APP14 RGB",
        ),
        (
            lossless_restart_predictor_ycbcr_16bit_3x3_jpeg(4),
            lossless_ycbcr_16bit_3x3_rgb16(),
            "YCbCr",
        ),
    ] {
        let dec = Decoder::new(&bytes).unwrap_or_else(|err| {
            panic!("restart-coded lossless 16-bit {label} JPEG must construct: {err}")
        });
        let cropped = crop_rgb16_bytes(&expected_rgb, 3, roi);
        let expected = rgb16_to_rgba16(&cropped, u16::MAX);
        let mut buf = vec![0xaau8; stride * roi.h as usize];

        let outcome = dec
            .decode_region_into(&mut buf, stride, PixelFormat::Rgba16, roi)
            .unwrap_or_else(|err| {
                panic!(
                    "restart-coded lossless 16-bit {label} RGBA16 ROI decode must succeed: {err}"
                )
            });

        assert_eq!(outcome.decoded, roi);
        for (row, expected_row) in buf
            .chunks_exact(stride)
            .zip(expected.chunks_exact(row_bytes))
        {
            assert_eq!(&row[..row_bytes], expected_row);
            assert_eq!(&row[row_bytes..], &[0xaa; 8]);
        }
    }
}

#[test]
fn decode_scaled_into_rgba16_projects_lossless_color() {
    let roi = Rect::full((3, 3));
    let scaled = scaled_rect_covering_for_test(roi, 2);

    for (bytes, expected_rgb, label) in [
        (
            lossless_predictor_rgb_16bit_3x3_jpeg(5),
            rgb16_samples_to_le_bytes(&LOSSLESS_RGB_16BIT_3X3_PIXELS),
            "APP14 RGB",
        ),
        (
            lossless_predictor_ycbcr_16bit_3x3_jpeg(5),
            lossless_ycbcr_16bit_3x3_rgb16(),
            "YCbCr",
        ),
    ] {
        let dec = Decoder::new(&bytes)
            .unwrap_or_else(|err| panic!("lossless 16-bit {label} JPEG must construct: {err}"));
        let stride = scaled.w as usize * PixelFormat::Rgba16.bytes_per_pixel();
        let mut buf = vec![0u8; stride * scaled.h as usize];
        let expected = rgb16_to_rgba16(
            &expected_scaled_rgb16_pixels(&expected_rgb, 3, roi, 2),
            u16::MAX,
        );

        let outcome = dec
            .decode_scaled_into(&mut buf, stride, PixelFormat::Rgba16, Downscale::Half)
            .unwrap_or_else(|err| {
                panic!("lossless 16-bit {label} RGBA16 scaled decode must succeed: {err}")
            });

        assert_eq!(outcome.decoded, roi);
        assert_eq!(buf, expected);
    }
}

#[test]
fn decode_region_scaled_into_rgba16_projects_restart_coded_lossless_color() {
    let roi = Rect {
        x: 1,
        y: 1,
        w: 2,
        h: 2,
    };
    let scaled_roi = scaled_rect_covering_for_test(roi, 2);
    let row_bytes = scaled_roi.w as usize * PixelFormat::Rgba16.bytes_per_pixel();
    let stride = row_bytes + 8;

    for (bytes, expected_rgb, label) in [
        (
            lossless_restart_predictor_rgb_16bit_3x3_jpeg(5),
            rgb16_samples_to_le_bytes(&LOSSLESS_RGB_16BIT_3X3_PIXELS),
            "APP14 RGB",
        ),
        (
            lossless_restart_predictor_ycbcr_16bit_3x3_jpeg(5),
            lossless_ycbcr_16bit_3x3_rgb16(),
            "YCbCr",
        ),
    ] {
        let dec = Decoder::new(&bytes).unwrap_or_else(|err| {
            panic!("restart-coded lossless 16-bit {label} JPEG must construct: {err}")
        });
        let expected = rgb16_to_rgba16(
            &expected_scaled_rgb16_pixels(&expected_rgb, 3, roi, 2),
            u16::MAX,
        );
        let mut buf = vec![0xaau8; stride * scaled_roi.h as usize];

        let outcome = dec
            .decode_region_scaled_into(&mut buf, stride, PixelFormat::Rgba16, roi, Downscale::Half)
            .unwrap_or_else(|err| {
                panic!(
                    "restart-coded lossless 16-bit {label} RGBA16 region-scaled decode must succeed: {err}"
                )
            });

        assert_eq!(outcome.decoded, roi);
        for (row, expected_row) in buf
            .chunks_exact(stride)
            .zip(expected.chunks_exact(row_bytes))
        {
            assert_eq!(&row[..row_bytes], expected_row);
            assert_eq!(&row[row_bytes..], &[0xaa; 8]);
        }
    }
}

#[test]
fn decode_region_into_gray8_crops_lossless_grayscale_common_predictors() {
    let roi = Rect {
        x: 1,
        y: 1,
        w: 2,
        h: 2,
    };
    let expected = crop_gray(&LOSSLESS_GRAYSCALE_3X3_PIXELS, 3, roi);
    for predictor in 1..=7 {
        let bytes = lossless_predictor_grayscale_3x3_jpeg(predictor);
        let dec = Decoder::new(&bytes).unwrap_or_else(|err| {
            panic!("lossless predictor-{predictor} grayscale JPEG must construct: {err}")
        });
        let stride = roi.w as usize + 2;
        let mut buf = vec![0xaau8; stride * roi.h as usize];

        let outcome = dec
            .decode_region_into(&mut buf, stride, PixelFormat::Gray8, roi)
            .unwrap_or_else(|err| {
                panic!("lossless predictor-{predictor} grayscale ROI decode must succeed: {err}")
            });

        assert_eq!(outcome.decoded, roi);
        for (row, expected_row) in buf
            .chunks_exact(stride)
            .zip(expected.chunks_exact(roi.w as usize))
        {
            assert_eq!(&row[..roi.w as usize], expected_row);
            assert_eq!(&row[roi.w as usize..], &[0xaa; 2]);
        }
    }
}

#[test]
fn decode_scaled_into_gray8_projects_lossless_grayscale_common_predictors() {
    let scale = Downscale::Half;
    let scaled_w = 2;
    let scaled_h = 2;
    let expected = project_scaled_gray(
        &LOSSLESS_GRAYSCALE_3X3_PIXELS,
        3,
        3,
        Rect {
            x: 0,
            y: 0,
            w: scaled_w,
            h: scaled_h,
        },
        2,
    );
    for predictor in 1..=7 {
        let bytes = lossless_predictor_grayscale_3x3_jpeg(predictor);
        let dec = Decoder::new(&bytes).unwrap_or_else(|err| {
            panic!("lossless predictor-{predictor} grayscale JPEG must construct: {err}")
        });
        let stride = scaled_w as usize + 2;
        let mut buf = vec![0xaau8; stride * scaled_h as usize];

        let outcome = dec
            .decode_scaled_into(&mut buf, stride, PixelFormat::Gray8, scale)
            .unwrap_or_else(|err| {
                panic!("lossless predictor-{predictor} grayscale scaled decode must succeed: {err}")
            });

        assert_eq!(outcome.decoded, Rect::full(dec.info().dimensions));
        for (row, expected_row) in buf
            .chunks_exact(stride)
            .zip(expected.chunks_exact(scaled_w as usize))
        {
            assert_eq!(&row[..scaled_w as usize], expected_row);
            assert_eq!(&row[scaled_w as usize..], &[0xaa; 2]);
        }
    }
}

#[test]
fn decode_region_scaled_into_gray8_projects_lossless_grayscale_common_predictors() {
    let roi = Rect {
        x: 1,
        y: 1,
        w: 2,
        h: 2,
    };
    let scaled_roi = scaled_rect_covering_for_test(roi, 2);
    let expected = project_scaled_gray(&LOSSLESS_GRAYSCALE_3X3_PIXELS, 3, 3, scaled_roi, 2);
    for predictor in 1..=7 {
        let bytes = lossless_predictor_grayscale_3x3_jpeg(predictor);
        let dec = Decoder::new(&bytes).unwrap_or_else(|err| {
            panic!("lossless predictor-{predictor} grayscale JPEG must construct: {err}")
        });
        let stride = scaled_roi.w as usize + 2;
        let mut buf = vec![0xaau8; stride * scaled_roi.h as usize];

        let outcome = dec
            .decode_region_scaled_into(&mut buf, stride, PixelFormat::Gray8, roi, Downscale::Half)
            .unwrap_or_else(|err| {
                panic!(
                    "lossless predictor-{predictor} grayscale region-scaled decode must succeed: {err}"
                )
            });

        assert_eq!(outcome.decoded, roi);
        for (row, expected_row) in buf
            .chunks_exact(stride)
            .zip(expected.chunks_exact(scaled_roi.w as usize))
        {
            assert_eq!(&row[..scaled_roi.w as usize], expected_row);
            assert_eq!(&row[scaled_roi.w as usize..], &[0xaa; 2]);
        }
    }
}

#[test]
fn decode_region_scaled_into_gray8_projects_restart_coded_lossless_grayscale() {
    let roi = Rect {
        x: 1,
        y: 1,
        w: 2,
        h: 2,
    };
    let scaled_roi = scaled_rect_covering_for_test(roi, 2);
    let expected = project_scaled_gray(&LOSSLESS_GRAYSCALE_3X3_PIXELS, 3, 3, scaled_roi, 2);
    let bytes = lossless_restart_predictor_grayscale_3x3_jpeg(1);
    let dec = Decoder::new(&bytes).expect("restart-coded lossless grayscale JPEG must construct");
    let stride = scaled_roi.w as usize + 2;
    let mut buf = vec![0xaau8; stride * scaled_roi.h as usize];

    let outcome = dec
        .decode_region_scaled_into(&mut buf, stride, PixelFormat::Gray8, roi, Downscale::Half)
        .expect("restart-coded lossless grayscale region-scaled decode must succeed");

    assert_eq!(outcome.decoded, roi);
    for (row, expected_row) in buf
        .chunks_exact(stride)
        .zip(expected.chunks_exact(scaled_roi.w as usize))
    {
        assert_eq!(&row[..scaled_roi.w as usize], expected_row);
        assert_eq!(&row[scaled_roi.w as usize..], &[0xaa; 2]);
    }
}

#[test]
fn decode_region_scaled_into_rgb8_projects_lossless_color_matrix() {
    let roi = Rect {
        x: 1,
        y: 1,
        w: 2,
        h: 2,
    };
    let scaled_roi = scaled_rect_covering_for_test(roi, 2);
    let stride = scaled_roi.w as usize * PixelFormat::Rgb8.bytes_per_pixel() + 3;
    let row_bytes = scaled_roi.w as usize * PixelFormat::Rgb8.bytes_per_pixel();

    for (label, bytes, full) in [
        (
            "APP14 RGB",
            lossless_predictor_rgb_3x3_jpeg(1),
            LOSSLESS_RGB_3X3_PIXELS.to_vec(),
        ),
        (
            "restart-coded APP14 RGB",
            lossless_restart_predictor_rgb_3x3_jpeg(1),
            LOSSLESS_RGB_3X3_PIXELS.to_vec(),
        ),
        (
            "YCbCr",
            lossless_predictor_ycbcr_3x3_jpeg(1),
            lossless_ycbcr_3x3_rgb8(),
        ),
    ] {
        let dec = Decoder::new(&bytes)
            .unwrap_or_else(|error| panic!("lossless {label} JPEG must construct: {error}"));
        let expected = project_scaled_rgb(&full, 3, 3, scaled_roi, 2);
        let mut buf = vec![0xaau8; stride * scaled_roi.h as usize];

        let outcome = dec
            .decode_region_scaled_into(&mut buf, stride, PixelFormat::Rgb8, roi, Downscale::Half)
            .unwrap_or_else(|error| {
                panic!("lossless {label} region-scaled decode must succeed: {error}")
            });

        assert_eq!(outcome.decoded, roi, "{label}");
        for (row, expected_row) in buf
            .chunks_exact(stride)
            .zip(expected.chunks_exact(row_bytes))
        {
            assert_eq!(&row[..row_bytes], expected_row, "{label}");
            assert_eq!(&row[row_bytes..], &[0xaa; 3], "{label}");
        }
    }
}

#[test]
fn decode_region_scaled_into_rgb16_projects_lossless_color_matrix() {
    let roi = Rect {
        x: 1,
        y: 1,
        w: 2,
        h: 2,
    };
    let scaled_roi = scaled_rect_covering_for_test(roi, 2);
    let stride = scaled_roi.w as usize * PixelFormat::Rgb16.bytes_per_pixel() + 6;

    for (label, bytes, full) in [
        (
            "APP14 RGB",
            lossless_predictor_rgb_16bit_3x3_jpeg(1),
            rgb16_samples_to_le_bytes(&LOSSLESS_RGB_16BIT_3X3_PIXELS),
        ),
        (
            "YCbCr",
            lossless_predictor_ycbcr_16bit_3x3_jpeg(1),
            lossless_ycbcr_16bit_3x3_rgb16(),
        ),
    ] {
        let dec = Decoder::new(&bytes)
            .unwrap_or_else(|error| panic!("lossless 16-bit {label} JPEG must construct: {error}"));
        let expected = expected_scaled_rgb16_pixels(&full, 3, roi, 2);
        let mut buf = vec![0xaau8; stride * scaled_roi.h as usize];

        let outcome = dec
            .decode_region_scaled_into(&mut buf, stride, PixelFormat::Rgb16, roi, Downscale::Half)
            .unwrap_or_else(|error| {
                panic!("lossless 16-bit {label} region-scaled decode must succeed: {error}")
            });

        assert_eq!(outcome.decoded, roi, "{label}");
        assert_padded_rgb16_rows(&buf, stride, scaled_roi.w as usize, &expected);
    }
}

#[test]
fn decode_region_into_gray16_crops_lossless_16bit_grayscale_common_predictors() {
    let roi = Rect {
        x: 1,
        y: 1,
        w: 2,
        h: 2,
    };
    let expected = crop_gray16(&LOSSLESS_GRAYSCALE_16BIT_3X3_PIXELS, 3, roi);
    for predictor in 1..=7 {
        let bytes = lossless_predictor_grayscale_16bit_3x3_jpeg(predictor);
        let dec = Decoder::new(&bytes).unwrap_or_else(|err| {
            panic!("lossless 16-bit predictor-{predictor} grayscale JPEG must construct: {err}")
        });
        let stride = roi.w as usize * PixelFormat::Gray16.bytes_per_pixel() + 4;
        let mut buf = vec![0xaau8; stride * roi.h as usize];

        let outcome = dec
            .decode_region_into(&mut buf, stride, PixelFormat::Gray16, roi)
            .unwrap_or_else(|err| {
                panic!(
                    "lossless 16-bit predictor-{predictor} Gray16 ROI decode must succeed: {err}"
                )
            });

        assert_eq!(outcome.decoded, roi);
        assert_gray16_rows_with_padding(&buf, stride, roi.w, &expected, 4, predictor);
    }
}

#[test]
fn decode_scaled_into_gray16_projects_lossless_16bit_grayscale_common_predictors() {
    let scaled_w = 2;
    let scaled_h = 2;
    let expected = project_scaled_gray16(
        &LOSSLESS_GRAYSCALE_16BIT_3X3_PIXELS,
        3,
        3,
        Rect {
            x: 0,
            y: 0,
            w: scaled_w,
            h: scaled_h,
        },
        2,
    );
    for predictor in 1..=7 {
        let bytes = lossless_predictor_grayscale_16bit_3x3_jpeg(predictor);
        let dec = Decoder::new(&bytes).unwrap_or_else(|err| {
            panic!("lossless 16-bit predictor-{predictor} grayscale JPEG must construct: {err}")
        });
        let stride = scaled_w as usize * PixelFormat::Gray16.bytes_per_pixel() + 4;
        let mut buf = vec![0xaau8; stride * scaled_h as usize];

        let outcome = dec
            .decode_scaled_into(&mut buf, stride, PixelFormat::Gray16, Downscale::Half)
            .unwrap_or_else(|err| {
                panic!("lossless 16-bit predictor-{predictor} Gray16 scaled decode must succeed: {err}")
            });

        assert_eq!(outcome.decoded, Rect::full(dec.info().dimensions));
        assert_gray16_rows_with_padding(&buf, stride, scaled_w, &expected, 4, predictor);
    }
}

#[test]
fn decode_region_scaled_into_gray16_projects_lossless_16bit_grayscale_common_predictors() {
    let roi = Rect {
        x: 1,
        y: 1,
        w: 2,
        h: 2,
    };
    let scaled_roi = scaled_rect_covering_for_test(roi, 2);
    let expected = project_scaled_gray16(&LOSSLESS_GRAYSCALE_16BIT_3X3_PIXELS, 3, 3, scaled_roi, 2);
    for predictor in 1..=7 {
        let bytes = lossless_predictor_grayscale_16bit_3x3_jpeg(predictor);
        let dec = Decoder::new(&bytes).unwrap_or_else(|err| {
            panic!("lossless 16-bit predictor-{predictor} grayscale JPEG must construct: {err}")
        });
        let stride = scaled_roi.w as usize * PixelFormat::Gray16.bytes_per_pixel() + 4;
        let mut buf = vec![0xaau8; stride * scaled_roi.h as usize];

        let outcome = dec
            .decode_region_scaled_into(
                &mut buf,
                stride,
                PixelFormat::Gray16,
                roi,
                Downscale::Half,
            )
            .unwrap_or_else(|err| {
                panic!(
                    "lossless 16-bit predictor-{predictor} Gray16 region-scaled decode must succeed: {err}"
                )
            });

        assert_eq!(outcome.decoded, roi);
        assert_gray16_rows_with_padding(&buf, stride, scaled_roi.w, &expected, 4, predictor);
    }
}
