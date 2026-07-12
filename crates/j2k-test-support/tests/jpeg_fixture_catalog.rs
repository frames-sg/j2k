// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_test_support::{
    baseline_420_restart_32x16_jpeg, baseline_420_restart_32x16_rgb, baseline_422_16x8_jpeg,
    baseline_422_16x8_rgb, baseline_444_8x8_jpeg, baseline_444_8x8_rgb, cmyk_16x16_420_jpeg,
    cmyk_16x8_422_jpeg, cmyk_16x8_nonleading_max_422_jpeg, cmyk_8x8_jpeg,
    extended_12bit_cmyk_16x16_420_jpeg, extended_12bit_cmyk_16x8_422_jpeg,
    extended_12bit_cmyk_420_restart_32x16_jpeg, extended_12bit_cmyk_422_restart_32x8_jpeg,
    extended_12bit_cmyk_8x8_jpeg, extended_12bit_cmyk_nonconstant_8x8_jpeg,
    extended_12bit_cmyk_restart_16x8_jpeg, extended_12bit_grayscale_8x8_jpeg,
    extended_12bit_grayscale_restart_16x8_jpeg, extended_12bit_rgb_32x32_rgb16,
    extended_12bit_rgb_32x8_rgb16, extended_12bit_rgb_420_32x32_jpeg,
    extended_12bit_rgb_422_32x8_jpeg, extended_12bit_rgb_8x8_jpeg, extended_12bit_rgb_8x8_rgb16,
    extended_12bit_rgb_restart_16x8_jpeg, extended_12bit_rgb_restart_16x8_rgb16,
    extended_12bit_ycbcr_420_32x32_jpeg, extended_12bit_ycbcr_420_32x32_rgb16,
    extended_12bit_ycbcr_420_restart_32x32_jpeg, extended_12bit_ycbcr_420_restart_32x32_rgb16,
    extended_12bit_ycbcr_422_32x8_jpeg, extended_12bit_ycbcr_422_32x8_rgb16,
    extended_12bit_ycbcr_422_restart_32x8_jpeg, extended_12bit_ycbcr_422_restart_32x8_rgb16,
    extended_12bit_ycbcr_8x8_jpeg, extended_12bit_ycbcr_8x8_rgb16,
    extended_12bit_ycbcr_restart_16x8_jpeg, extended_12bit_ycbcr_restart_16x8_rgb16,
    extended_12bit_ycck_16x16_420_jpeg, extended_12bit_ycck_16x8_422_jpeg,
    extended_12bit_ycck_420_restart_32x16_jpeg, extended_12bit_ycck_422_restart_32x8_jpeg,
    extended_12bit_ycck_8x8_jpeg, extended_12bit_ycck_nonconstant_8x8_jpeg,
    extended_12bit_ycck_restart_16x8_jpeg, four_component_12bit_16x16_rgb16,
    four_component_12bit_16x8_rgb16, four_component_12bit_32x16_rgb16,
    four_component_12bit_32x8_rgb16, four_component_12bit_8x8_cmyk_nonconstant_rgb16,
    four_component_12bit_8x8_rgb16, four_component_12bit_8x8_ycck_nonconstant_rgb16,
    four_component_16x16_rgb, four_component_16x8_rgb, four_component_8x8_rgb, grayscale_8x8_jpeg,
    lossless_predictor_grayscale_16bit_3x3_jpeg, lossless_predictor_grayscale_3x3_jpeg,
    lossless_predictor_rgb_16bit_3x3_jpeg, lossless_predictor_rgb_3x3_jpeg,
    lossless_predictor_ycbcr_16bit_3x3_jpeg, lossless_predictor_ycbcr_3x3_jpeg,
    lossless_restart_predictor_grayscale_16bit_3x3_jpeg,
    lossless_restart_predictor_grayscale_3x3_jpeg, lossless_restart_predictor_rgb_16bit_3x3_jpeg,
    lossless_restart_predictor_rgb_3x3_jpeg, lossless_restart_predictor_ycbcr_16bit_3x3_jpeg,
    lossless_restart_predictor_ycbcr_3x3_jpeg, lossless_rgb_16bit_420_4x4_jpeg,
    lossless_rgb_16bit_420_4x4_rgb16, lossless_rgb_16bit_420_restart_4x4_jpeg,
    lossless_rgb_16bit_422_4x2_jpeg, lossless_rgb_16bit_422_4x2_rgb16,
    lossless_rgb_16bit_422_restart_4x2_jpeg, lossless_rgb_8bit_420_4x4_jpeg,
    lossless_rgb_8bit_420_4x4_rgb8, lossless_rgb_8bit_420_restart_4x4_jpeg,
    lossless_rgb_8bit_422_4x2_jpeg, lossless_rgb_8bit_422_4x2_rgb8,
    lossless_rgb_8bit_422_restart_4x2_jpeg, lossless_ycbcr_16bit_3x3_rgb16,
    lossless_ycbcr_16bit_420_4x4_jpeg, lossless_ycbcr_16bit_420_4x4_rgb16,
    lossless_ycbcr_16bit_420_restart_4x4_jpeg, lossless_ycbcr_16bit_422_3x3_jpeg,
    lossless_ycbcr_16bit_422_4x2_jpeg, lossless_ycbcr_16bit_422_4x2_rgb16,
    lossless_ycbcr_16bit_422_restart_4x2_jpeg, lossless_ycbcr_3x3_rgb8,
    lossless_ycbcr_8bit_420_4x4_jpeg, lossless_ycbcr_8bit_420_4x4_rgb8,
    lossless_ycbcr_8bit_420_restart_4x4_jpeg, lossless_ycbcr_8bit_422_4x2_jpeg,
    lossless_ycbcr_8bit_422_4x2_rgb8, lossless_ycbcr_8bit_422_restart_4x2_jpeg,
    malformed_cmyk_nondivisible_sampling_jpeg, minimal_baseline_420_jpeg,
    progressive_12bit_cmyk_16x16_420_jpeg, progressive_12bit_cmyk_16x8_422_jpeg,
    progressive_12bit_cmyk_420_restart_32x16_jpeg, progressive_12bit_cmyk_422_restart_32x8_jpeg,
    progressive_12bit_cmyk_8x8_jpeg, progressive_12bit_cmyk_nonconstant_8x8_jpeg,
    progressive_12bit_cmyk_restart_16x8_jpeg, progressive_12bit_grayscale_8x8_jpeg,
    progressive_12bit_rgb_420_32x32_jpeg, progressive_12bit_rgb_422_32x8_jpeg,
    progressive_12bit_rgb_8x8_jpeg, progressive_12bit_ycbcr_420_32x32_jpeg,
    progressive_12bit_ycbcr_422_32x8_jpeg, progressive_12bit_ycbcr_8x8_jpeg,
    progressive_12bit_ycck_16x16_420_jpeg, progressive_12bit_ycck_16x8_422_jpeg,
    progressive_12bit_ycck_420_restart_32x16_jpeg, progressive_12bit_ycck_422_restart_32x8_jpeg,
    progressive_12bit_ycck_8x8_jpeg, progressive_12bit_ycck_nonconstant_8x8_jpeg,
    progressive_12bit_ycck_restart_16x8_jpeg, progressive_8x8_jpeg, rgb_app14_8x8_jpeg,
    rgb_app14_8x8_rgb, ycck_16x16_420_jpeg, ycck_16x8_422_jpeg, ycck_16x8_nonleading_max_422_jpeg,
    ycck_8x8_jpeg,
};

#[derive(Clone, Debug, Eq, PartialEq)]
struct Frame {
    marker: u8,
    precision: u8,
    width: u16,
    height: u16,
    components: u8,
    sampling: Vec<u8>,
}

fn frame(jpeg: &[u8]) -> Frame {
    assert!(
        jpeg.starts_with(&[0xff, 0xd8]),
        "fixture must start with SOI"
    );
    assert!(jpeg.ends_with(&[0xff, 0xd9]), "fixture must end with EOI");

    let offset = jpeg
        .windows(2)
        .position(|window| window[0] == 0xff && matches!(window[1], 0xc0..=0xc3))
        .expect("fixture must contain a supported SOF marker");
    let components = jpeg[offset + 9];
    let sampling = (0..usize::from(components))
        .map(|component| jpeg[offset + 11 + component * 3])
        .collect();

    Frame {
        marker: jpeg[offset + 1],
        precision: jpeg[offset + 4],
        height: u16::from_be_bytes([jpeg[offset + 5], jpeg[offset + 6]]),
        width: u16::from_be_bytes([jpeg[offset + 7], jpeg[offset + 8]]),
        components,
        sampling,
    }
}

fn assert_frame(jpeg: &[u8], expected: &Frame) {
    assert_eq!(frame(jpeg), *expected);
    assert!(
        jpeg.windows(2).any(|window| window == [0xff, 0xda]),
        "fixture must contain a scan"
    );
}

fn expected(
    marker: u8,
    precision: u8,
    width: u16,
    height: u16,
    components: u8,
    sampling: &[u8],
) -> Frame {
    Frame {
        marker,
        precision,
        width,
        height,
        components,
        sampling: sampling.to_vec(),
    }
}

fn has_marker(jpeg: &[u8], marker: u8) -> bool {
    jpeg.windows(2).any(|window| window == [0xff, marker])
}

fn app14_transform(jpeg: &[u8]) -> Option<u8> {
    jpeg.windows(5)
        .position(|window| window == b"Adobe")
        .map(|offset| jpeg[offset + 11])
}

#[test]
fn extended_and_progressive_fixture_headers_match_their_contracts() {
    assert_extended_sequential_catalog();
    assert_progressive_catalog();
    assert_extended_restart_catalog();
}

fn assert_extended_sequential_catalog() {
    let sequential = [
        (extended_12bit_grayscale_8x8_jpeg(), 8, 8, 1, vec![0x11]),
        (
            extended_12bit_grayscale_restart_16x8_jpeg(),
            16,
            8,
            1,
            vec![0x11],
        ),
        (extended_12bit_rgb_8x8_jpeg(), 8, 8, 3, vec![0x11; 3]),
        (
            extended_12bit_rgb_restart_16x8_jpeg(),
            16,
            8,
            3,
            vec![0x11; 3],
        ),
        (
            extended_12bit_rgb_422_32x8_jpeg(),
            32,
            8,
            3,
            vec![0x21, 0x11, 0x11],
        ),
        (
            extended_12bit_rgb_420_32x32_jpeg(),
            32,
            32,
            3,
            vec![0x22, 0x11, 0x11],
        ),
        (extended_12bit_ycbcr_8x8_jpeg(), 8, 8, 3, vec![0x11; 3]),
        (
            extended_12bit_ycbcr_restart_16x8_jpeg(),
            16,
            8,
            3,
            vec![0x11; 3],
        ),
        (
            extended_12bit_ycbcr_422_32x8_jpeg(),
            32,
            8,
            3,
            vec![0x21, 0x11, 0x11],
        ),
        (
            extended_12bit_ycbcr_422_restart_32x8_jpeg(),
            32,
            8,
            3,
            vec![0x21, 0x11, 0x11],
        ),
        (
            extended_12bit_ycbcr_420_32x32_jpeg(),
            32,
            32,
            3,
            vec![0x22, 0x11, 0x11],
        ),
        (
            extended_12bit_ycbcr_420_restart_32x32_jpeg(),
            32,
            32,
            3,
            vec![0x22, 0x11, 0x11],
        ),
    ];
    for (jpeg, width, height, components, sampling) in sequential {
        let expected = expected(0xc1, 12, width, height, components, &sampling);
        assert_frame(&jpeg, &expected);
    }
}

fn assert_progressive_catalog() {
    let progressive = [
        (progressive_12bit_grayscale_8x8_jpeg(), 8, 8, 1, vec![0x11]),
        (progressive_12bit_rgb_8x8_jpeg(), 8, 8, 3, vec![0x11; 3]),
        (
            progressive_12bit_rgb_422_32x8_jpeg(),
            32,
            8,
            3,
            vec![0x21, 0x11, 0x11],
        ),
        (
            progressive_12bit_rgb_420_32x32_jpeg(),
            32,
            32,
            3,
            vec![0x22, 0x11, 0x11],
        ),
        (progressive_12bit_ycbcr_8x8_jpeg(), 8, 8, 3, vec![0x11; 3]),
        (
            progressive_12bit_ycbcr_422_32x8_jpeg(),
            32,
            8,
            3,
            vec![0x21, 0x11, 0x11],
        ),
        (
            progressive_12bit_ycbcr_420_32x32_jpeg(),
            32,
            32,
            3,
            vec![0x22, 0x11, 0x11],
        ),
    ];
    for (jpeg, width, height, components, sampling) in progressive {
        let expected = expected(0xc2, 12, width, height, components, &sampling);
        assert_frame(&jpeg, &expected);
        assert!(
            jpeg.windows(2)
                .filter(|window| *window == [0xff, 0xda])
                .count()
                >= 1
        );
    }
}

fn assert_extended_restart_catalog() {
    for restart_fixture in [
        extended_12bit_grayscale_restart_16x8_jpeg(),
        extended_12bit_rgb_restart_16x8_jpeg(),
        extended_12bit_ycbcr_restart_16x8_jpeg(),
        extended_12bit_ycbcr_422_restart_32x8_jpeg(),
        extended_12bit_ycbcr_420_restart_32x32_jpeg(),
    ] {
        assert!(has_marker(&restart_fixture, 0xdd));
        assert!(has_marker(&restart_fixture, 0xd0));
    }
}

#[test]
fn lossless_fixture_catalog_covers_predictors_sampling_and_restarts() {
    for predictor in 1..=7 {
        let fixtures = [
            lossless_predictor_grayscale_3x3_jpeg(predictor),
            lossless_predictor_rgb_3x3_jpeg(predictor),
            lossless_predictor_ycbcr_3x3_jpeg(predictor),
            lossless_predictor_grayscale_16bit_3x3_jpeg(predictor),
            lossless_predictor_rgb_16bit_3x3_jpeg(predictor),
            lossless_predictor_ycbcr_16bit_3x3_jpeg(predictor),
        ];
        for jpeg in fixtures {
            assert_eq!(frame(&jpeg).marker, 0xc3);
            let scan = jpeg
                .windows(2)
                .position(|window| window == [0xff, 0xda])
                .expect("lossless fixture has SOS");
            let components = usize::from(jpeg[scan + 4]);
            assert_eq!(jpeg[scan + 5 + components * 2], predictor);
        }
    }

    let restarted = [
        lossless_restart_predictor_grayscale_3x3_jpeg(4),
        lossless_restart_predictor_rgb_3x3_jpeg(4),
        lossless_restart_predictor_ycbcr_3x3_jpeg(4),
        lossless_restart_predictor_grayscale_16bit_3x3_jpeg(4),
        lossless_restart_predictor_rgb_16bit_3x3_jpeg(4),
        lossless_restart_predictor_ycbcr_16bit_3x3_jpeg(4),
    ];
    for jpeg in restarted {
        assert_eq!(frame(&jpeg).marker, 0xc3);
        assert!(has_marker(&jpeg, 0xdd));
        assert!(has_marker(&jpeg, 0xd0));
    }

    let subsampled = [
        lossless_rgb_8bit_422_4x2_jpeg(1),
        lossless_rgb_8bit_422_restart_4x2_jpeg(7),
        lossless_rgb_16bit_422_4x2_jpeg(1),
        lossless_rgb_16bit_422_restart_4x2_jpeg(7),
        lossless_rgb_8bit_420_4x4_jpeg(1),
        lossless_rgb_8bit_420_restart_4x4_jpeg(7),
        lossless_rgb_16bit_420_4x4_jpeg(1),
        lossless_rgb_16bit_420_restart_4x4_jpeg(7),
        lossless_ycbcr_8bit_422_4x2_jpeg(1),
        lossless_ycbcr_8bit_422_restart_4x2_jpeg(7),
        lossless_ycbcr_16bit_422_3x3_jpeg(),
        lossless_ycbcr_16bit_422_4x2_jpeg(1),
        lossless_ycbcr_16bit_422_restart_4x2_jpeg(7),
        lossless_ycbcr_8bit_420_4x4_jpeg(1),
        lossless_ycbcr_8bit_420_restart_4x4_jpeg(7),
        lossless_ycbcr_16bit_420_4x4_jpeg(1),
        lossless_ycbcr_16bit_420_restart_4x4_jpeg(7),
    ];
    for jpeg in subsampled {
        let parsed = frame(&jpeg);
        assert_eq!(parsed.marker, 0xc3);
        assert_eq!(parsed.components, 3);
        assert!(matches!(
            parsed.sampling.as_slice(),
            [0x21 | 0x22, 0x11, 0x11]
        ));
    }
}

#[test]
fn four_component_fixture_catalog_preserves_color_transform_and_geometry() {
    assert_cmyk_fixture_catalog();
    assert_ycck_fixture_catalog();
    assert_four_component_progressive_catalog();
}

fn assert_cmyk_fixture_catalog() {
    let cmyk = [
        (cmyk_8x8_jpeg(), 0xc0, 8, 8, 8, vec![0x11; 4]),
        (
            extended_12bit_cmyk_8x8_jpeg(),
            0xc1,
            12,
            8,
            8,
            vec![0x11; 4],
        ),
        (
            extended_12bit_cmyk_nonconstant_8x8_jpeg(),
            0xc1,
            12,
            8,
            8,
            vec![0x11; 4],
        ),
        (
            extended_12bit_cmyk_restart_16x8_jpeg(),
            0xc1,
            12,
            16,
            8,
            vec![0x11; 4],
        ),
        (
            extended_12bit_cmyk_16x8_422_jpeg(),
            0xc1,
            12,
            16,
            8,
            vec![0x21, 0x11, 0x11, 0x11],
        ),
        (
            extended_12bit_cmyk_422_restart_32x8_jpeg(),
            0xc1,
            12,
            32,
            8,
            vec![0x21, 0x11, 0x11, 0x11],
        ),
        (
            extended_12bit_cmyk_16x16_420_jpeg(),
            0xc1,
            12,
            16,
            16,
            vec![0x22, 0x11, 0x11, 0x11],
        ),
        (
            extended_12bit_cmyk_420_restart_32x16_jpeg(),
            0xc1,
            12,
            32,
            16,
            vec![0x22, 0x11, 0x11, 0x11],
        ),
        (
            cmyk_16x8_422_jpeg(),
            0xc0,
            8,
            16,
            8,
            vec![0x21, 0x11, 0x11, 0x11],
        ),
        (
            cmyk_16x16_420_jpeg(),
            0xc0,
            8,
            16,
            16,
            vec![0x22, 0x11, 0x11, 0x11],
        ),
        (
            cmyk_16x8_nonleading_max_422_jpeg(),
            0xc0,
            8,
            16,
            8,
            vec![0x11, 0x21, 0x11, 0x11],
        ),
        (
            malformed_cmyk_nondivisible_sampling_jpeg(),
            0xc0,
            8,
            24,
            8,
            vec![0x31, 0x21, 0x11, 0x11],
        ),
    ];
    for (jpeg, marker, precision, width, height, sampling) in cmyk {
        assert_eq!(app14_transform(&jpeg), Some(0));
        let expected = expected(marker, precision, width, height, 4, &sampling);
        assert_frame(&jpeg, &expected);
    }
}

fn assert_ycck_fixture_catalog() {
    let ycck = [
        ycck_8x8_jpeg(),
        extended_12bit_ycck_8x8_jpeg(),
        extended_12bit_ycck_nonconstant_8x8_jpeg(),
        extended_12bit_ycck_restart_16x8_jpeg(),
        extended_12bit_ycck_16x8_422_jpeg(),
        extended_12bit_ycck_422_restart_32x8_jpeg(),
        extended_12bit_ycck_16x16_420_jpeg(),
        extended_12bit_ycck_420_restart_32x16_jpeg(),
        ycck_16x8_422_jpeg(),
        ycck_16x8_nonleading_max_422_jpeg(),
        ycck_16x16_420_jpeg(),
    ];
    for jpeg in ycck {
        assert_eq!(app14_transform(&jpeg), Some(2));
        assert_eq!(frame(&jpeg).components, 4);
    }
}

fn assert_four_component_progressive_catalog() {
    let progressive = [
        progressive_12bit_cmyk_8x8_jpeg(),
        progressive_12bit_cmyk_nonconstant_8x8_jpeg(),
        progressive_12bit_cmyk_restart_16x8_jpeg(),
        progressive_12bit_cmyk_16x8_422_jpeg(),
        progressive_12bit_cmyk_422_restart_32x8_jpeg(),
        progressive_12bit_cmyk_16x16_420_jpeg(),
        progressive_12bit_cmyk_420_restart_32x16_jpeg(),
        progressive_12bit_ycck_8x8_jpeg(),
        progressive_12bit_ycck_nonconstant_8x8_jpeg(),
        progressive_12bit_ycck_restart_16x8_jpeg(),
        progressive_12bit_ycck_16x8_422_jpeg(),
        progressive_12bit_ycck_422_restart_32x8_jpeg(),
        progressive_12bit_ycck_16x16_420_jpeg(),
        progressive_12bit_ycck_420_restart_32x16_jpeg(),
    ];
    for jpeg in progressive {
        let parsed = frame(&jpeg);
        assert_eq!(parsed.marker, 0xc2);
        assert_eq!(parsed.precision, 12);
        assert_eq!(parsed.components, 4);
    }
}

#[test]
fn reference_pixels_and_stored_fixtures_have_exact_shapes() {
    let rgb8 = [
        (lossless_rgb_8bit_422_4x2_rgb8(), 4 * 2 * 3),
        (lossless_rgb_8bit_420_4x4_rgb8(), 4 * 4 * 3),
        (lossless_ycbcr_3x3_rgb8(), 3 * 3 * 3),
        (lossless_ycbcr_8bit_422_4x2_rgb8(), 4 * 2 * 3),
        (lossless_ycbcr_8bit_420_4x4_rgb8(), 4 * 4 * 3),
        (four_component_8x8_rgb(), 8 * 8 * 3),
        (four_component_16x8_rgb(), 16 * 8 * 3),
        (four_component_16x16_rgb(), 16 * 16 * 3),
        (baseline_444_8x8_rgb(), 8 * 8 * 3),
        (baseline_422_16x8_rgb(), 16 * 8 * 3),
        (baseline_420_restart_32x16_rgb(), 32 * 16 * 3),
        (rgb_app14_8x8_rgb(), 8 * 8 * 3),
    ];
    for (pixels, expected_len) in rgb8 {
        assert_eq!(pixels.len(), expected_len);
    }

    let rgb16 = [
        (extended_12bit_rgb_8x8_rgb16(), 8 * 8 * 3 * 2),
        (extended_12bit_rgb_restart_16x8_rgb16(), 16 * 8 * 3 * 2),
        (extended_12bit_rgb_32x8_rgb16(), 32 * 8 * 3 * 2),
        (extended_12bit_rgb_32x32_rgb16(), 32 * 32 * 3 * 2),
        (extended_12bit_ycbcr_8x8_rgb16(), 8 * 8 * 3 * 2),
        (extended_12bit_ycbcr_restart_16x8_rgb16(), 16 * 8 * 3 * 2),
        (extended_12bit_ycbcr_422_32x8_rgb16(), 32 * 8 * 3 * 2),
        (
            extended_12bit_ycbcr_422_restart_32x8_rgb16(),
            32 * 8 * 3 * 2,
        ),
        (extended_12bit_ycbcr_420_32x32_rgb16(), 32 * 32 * 3 * 2),
        (
            extended_12bit_ycbcr_420_restart_32x32_rgb16(),
            32 * 32 * 3 * 2,
        ),
        (lossless_rgb_16bit_422_4x2_rgb16(), 4 * 2 * 3 * 2),
        (lossless_rgb_16bit_420_4x4_rgb16(), 4 * 4 * 3 * 2),
        (lossless_ycbcr_16bit_422_4x2_rgb16(), 4 * 2 * 3 * 2),
        (lossless_ycbcr_16bit_420_4x4_rgb16(), 4 * 4 * 3 * 2),
        (lossless_ycbcr_16bit_3x3_rgb16(), 3 * 3 * 3 * 2),
        (four_component_12bit_8x8_rgb16(), 8 * 8 * 3 * 2),
        (
            four_component_12bit_8x8_cmyk_nonconstant_rgb16(),
            8 * 8 * 3 * 2,
        ),
        (
            four_component_12bit_8x8_ycck_nonconstant_rgb16(),
            8 * 8 * 3 * 2,
        ),
        (four_component_12bit_16x8_rgb16(), 16 * 8 * 3 * 2),
        (four_component_12bit_32x8_rgb16(), 32 * 8 * 3 * 2),
        (four_component_12bit_16x16_rgb16(), 16 * 16 * 3 * 2),
        (four_component_12bit_32x16_rgb16(), 32 * 16 * 3 * 2),
    ];
    for (pixels, expected_len) in rgb16 {
        assert_eq!(pixels.len(), expected_len);
    }

    assert_eq!(
        frame(&minimal_baseline_420_jpeg()),
        expected(0xc0, 8, 16, 16, 3, &[0x22, 0x11, 0x11])
    );
    assert_eq!(
        frame(&grayscale_8x8_jpeg()),
        expected(0xc0, 8, 8, 8, 1, &[0x11])
    );
    assert_eq!(frame(&baseline_444_8x8_jpeg()).sampling, [0x11; 3]);
    assert_eq!(
        frame(&baseline_422_16x8_jpeg()).sampling,
        [0x21, 0x11, 0x11]
    );
    assert!(has_marker(&baseline_420_restart_32x16_jpeg(), 0xdd));
    assert_eq!(app14_transform(&rgb_app14_8x8_jpeg()), Some(0));
    assert_eq!(
        progressive_8x8_jpeg()
            .windows(2)
            .filter(|window| *window == [0xff, 0xda])
            .count(),
        10
    );
}
