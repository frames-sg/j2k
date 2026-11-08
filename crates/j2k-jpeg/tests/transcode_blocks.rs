// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_test_support as fixtures;

use j2k_jpeg::transcode::{
    estimate_dct_windows, extract_dct_blocks, extract_dct_windows, idct_islow_block, lossless_crop,
    DctExtractOptions, JpegDctCodingMode, LosslessCropOptions, McuRect,
};
use j2k_jpeg::{
    encode_jpeg_baseline, rewrite_sof_dimensions, JpegEncodeOptions, JpegError, JpegSamples,
    JpegSubsampling, Rect,
};

#[test]
fn huge_sequential_dct_planes_are_rejected_before_entropy_for_both_extract_modes() {
    let rewritten = rewrite_sof_dimensions(&fixtures::grayscale_8x8_jpeg(), (65_500, 65_500))
        .expect("rewrite grayscale SOF dimensions");

    for options in [
        DctExtractOptions::default(),
        DctExtractOptions::dequantized_only(),
    ] {
        let error = extract_dct_blocks(&rewritten, options)
            .expect_err("huge sequential DCT workspace must be rejected");
        assert!(
            matches!(error, JpegError::MemoryCapExceeded { requested, cap } if requested > cap),
            "unexpected huge sequential extraction error: {error:?}"
        );
    }
}

#[test]
fn estimate_rejects_the_same_unsupported_dct_frame_as_extraction() {
    let bytes = fixtures::extended_12bit_grayscale_8x8_jpeg();
    let error = DctExtractOptions::quantized_only()
        .estimate(&bytes)
        .expect_err("12-bit extended extraction remains unsupported");
    assert!(matches!(error, JpegError::NotImplemented { .. }));
}

#[test]
fn extraction_rejects_out_of_grid_windows_as_input_errors() {
    let bytes = fixtures::grayscale_8x8_jpeg();
    assert!(matches!(
        extract_dct_windows(
            &bytes,
            &[McuRect {
                x: 1,
                y: 0,
                width: 1,
                height: 1,
            }],
            DctExtractOptions::quantized_only(),
        ),
        Err(JpegError::DctWindowOutOfBounds { .. })
    ));
}

#[test]
fn repeated_windows_are_rejected_by_the_aggregate_working_set_before_entropy() {
    let bytes = rewrite_sof_dimensions(&fixtures::minimal_baseline_420_jpeg(), (4096, 4096))
        .expect("rewrite large 4:2:0 dimensions");
    let windows = [McuRect {
        x: 0,
        y: 0,
        width: 256,
        height: 256,
    }; 11];
    let error = extract_dct_windows(&bytes, &windows, DctExtractOptions::quantized_only())
        .expect_err("aggregate retained windows must exceed the shared cap");
    assert!(matches!(error, JpegError::MemoryCapExceeded { requested, cap } if requested > cap));
}

/// A one-MCU window of a 16384 x 16384 4:2:0 baseline header is estimated as
/// the small streamed window it is: the whole-image quantized matrix (about
/// 768 MiB) that this route never allocates does not decide admission.
/// Whole-image estimation and crops above the caller's limit are still refused.
#[test]
fn a_one_mcu_window_of_a_huge_baseline_header_is_estimated_as_streamed() {
    let bytes = rewrite_sof_dimensions(&fixtures::minimal_baseline_420_jpeg(), (16_384, 16_384))
        .expect("rewrite huge 4:2:0 dimensions");
    let window = [McuRect {
        x: 0,
        y: 0,
        width: 1,
        height: 1,
    }];
    let estimate = estimate_dct_windows(&bytes, &window, DctExtractOptions::quantized_only())
        .expect("streamed one-MCU window estimate");
    // One 4:2:0 MCU holds six quantized blocks of 128 bytes.
    assert!(estimate.retained_bytes >= 6 * 128, "{estimate:?}");
    assert!(estimate.peak_bytes < 64 * 1024, "{estimate:?}");

    let error = DctExtractOptions::quantized_only()
        .estimate(&bytes)
        .expect_err("whole-image coefficients exceed the shared cap");
    assert!(matches!(error, JpegError::MemoryCapExceeded { requested, cap } if requested > cap));

    let mut options = LosslessCropOptions::default();
    options.max_working_bytes = estimate.peak_bytes - 1;
    let error = lossless_crop(
        &bytes,
        Rect {
            x: 0,
            y: 0,
            w: 16,
            h: 16,
        },
        options,
    )
    .expect_err("a crop above its working limit is refused before entropy decoding");
    assert!(
        matches!(
            error,
            j2k_jpeg::transcode::LosslessCropError::Decode(JpegError::MemoryCapExceeded { .. })
        ),
        "{error:?}"
    );
}

#[test]
fn extracts_grayscale_dct_blocks() {
    let image = extract_dct_blocks(
        &fixtures::grayscale_8x8_jpeg(),
        DctExtractOptions::default(),
    )
    .expect("extract grayscale DCT blocks");

    assert_eq!((image.width, image.height), (8, 8));
    assert_eq!(image.components.len(), 1);
    assert_component(&image.components[0], (8, 8), (1, 1), (1, 1), 1);
}

#[test]
fn exposes_quantized_and_dequantized_natural_order_blocks() {
    let image = extract_dct_blocks(
        &fixtures::baseline_444_8x8_jpeg(),
        DctExtractOptions::default(),
    )
    .expect("extract 4:4:4 DCT blocks");
    let component = &image.components[0];

    assert_eq!(
        component.quantized_blocks.len(),
        component.dequantized_blocks.len()
    );
    assert!(component
        .quantized_blocks
        .iter()
        .any(|block| block.iter().any(|&coefficient| coefficient != 0)));
    assert_ne!(
        component.quantized_blocks[0],
        component.dequantized_blocks[0]
    );

    for (quantized, dequantized) in component
        .quantized_blocks
        .iter()
        .zip(component.dequantized_blocks.iter())
    {
        for (zigzag_idx, &natural_idx) in JPEG_ZIGZAG.iter().enumerate() {
            let expected =
                i32::from(quantized[natural_idx]) * i32::from(component.quant_table[zigzag_idx]);
            assert_eq!(
                i32::from(dequantized[natural_idx]),
                expected,
                "dequantized coefficient at natural index {natural_idx}"
            );
        }
    }
}

#[test]
fn dequantized_only_extraction_omits_quantized_blocks() {
    let default_image = extract_dct_blocks(
        &fixtures::baseline_422_16x8_jpeg(),
        DctExtractOptions::default(),
    )
    .expect("extract default DCT blocks");
    let dequantized_only = extract_dct_blocks(
        &fixtures::baseline_422_16x8_jpeg(),
        DctExtractOptions::dequantized_only(),
    )
    .expect("extract dequantized-only DCT blocks");

    assert_eq!(
        dequantized_only.components.len(),
        default_image.components.len()
    );
    for (actual, expected) in dequantized_only
        .components
        .iter()
        .zip(default_image.components.iter())
    {
        assert_eq!(actual.quantized_blocks, [] as [[i16; 64]; 0]);
        assert_eq!(actual.dequantized_blocks, expected.dequantized_blocks);
        assert_eq!(actual.block_cols, expected.block_cols);
        assert_eq!(actual.block_rows, expected.block_rows);
    }
}

#[test]
fn quantized_only_extraction_omits_dequantized_blocks() {
    for bytes in [
        fixtures::baseline_422_16x8_jpeg(),
        fixtures::progressive_8x8_jpeg(),
    ] {
        let expected = extract_dct_blocks(&bytes, DctExtractOptions::default())
            .expect("extract default DCT blocks");
        let actual = extract_dct_blocks(&bytes, DctExtractOptions::quantized_only())
            .expect("extract quantized-only DCT blocks");
        for (actual, expected) in actual.components.iter().zip(&expected.components) {
            assert_eq!(actual.quantized_blocks, expected.quantized_blocks);
            assert_eq!(actual.dequantized_blocks, Vec::<[i16; 64]>::new());
        }
    }
}

#[test]
fn baseline_windows_match_full_quantized_subgrids_and_estimate_their_storage() {
    let bytes = fixtures::baseline_420_restart_32x16_jpeg();
    let windows = [
        McuRect {
            x: 0,
            y: 0,
            width: 1,
            height: 1,
        },
        McuRect {
            x: 1,
            y: 0,
            width: 1,
            height: 1,
        },
    ];
    let full = extract_dct_blocks(&bytes, DctExtractOptions::quantized_only())
        .expect("extract full coefficient grid");
    let extracted = extract_dct_windows(&bytes, &windows, DctExtractOptions::quantized_only())
        .expect("extract coefficient windows");
    assert_eq!(extracted.len(), windows.len());
    for (window_index, image) in extracted.iter().enumerate() {
        assert_eq!((image.width, image.height), (16, 16));
        for (component, source) in image.components.iter().zip(&full.components) {
            let h = u32::from(component.h_samp);
            let expected = (0..component.block_rows)
                .flat_map(|row| {
                    let start = (row * source.block_cols + windows[window_index].x * h) as usize;
                    source.quantized_blocks[start..start + component.block_cols as usize].iter()
                })
                .copied()
                .collect::<Vec<_>>();
            assert_eq!(component.quantized_blocks, expected);
            assert_eq!(component.dequantized_blocks, Vec::<[i16; 64]>::new());
        }
    }
    let estimate = estimate_dct_windows(&bytes, &windows[..1], DctExtractOptions::quantized_only())
        .expect("estimate window storage");
    assert!(estimate.retained_bytes >= 6 * 64 * 2);
    assert!(estimate.peak_bytes >= estimate.retained_bytes);
}

#[test]
fn multiscan_window_at_the_bottom_right_matches_the_full_grid() {
    let case = fixtures::scaled_matrix_cases()
        .into_iter()
        .find(|case| case.coding == "multiscan" && case.layout == "444" && case.precision == 8)
        .expect("8-bit 4:4:4 multi-scan fixture");
    let full = extract_dct_blocks(case.jpeg, DctExtractOptions::quantized_only())
        .expect("extract full multi-scan coefficients");
    let max_h = full
        .components
        .iter()
        .map(|component| component.h_samp)
        .max()
        .unwrap();
    let max_v = full
        .components
        .iter()
        .map(|component| component.v_samp)
        .max()
        .unwrap();
    let mcu_cols = full.width.div_ceil(u32::from(max_h) * 8);
    let mcu_rows = full.height.div_ceil(u32::from(max_v) * 8);
    let window = McuRect {
        x: mcu_cols - 1,
        y: mcu_rows - 1,
        width: 1,
        height: 1,
    };
    let actual = extract_dct_windows(case.jpeg, &[window], DctExtractOptions::quantized_only())
        .expect("extract multi-scan edge window")
        .pop()
        .unwrap();
    for (actual, source) in actual.components.iter().zip(&full.components) {
        let block_x = window.x * u32::from(source.h_samp);
        let block_y = window.y * u32::from(source.v_samp);
        let expected = (0..actual.block_rows)
            .flat_map(|row| {
                let start = ((block_y + row) * source.block_cols + block_x) as usize;
                source.quantized_blocks[start..start + actual.block_cols as usize].iter()
            })
            .copied()
            .collect::<Vec<_>>();
        assert_eq!(actual.quantized_blocks, expected);
    }
}

#[test]
fn progressive_odd_bottom_right_window_matches_native_component_grids() {
    let case = fixtures::scaled_matrix_cases()
        .into_iter()
        .find(|case| case.coding == "progressive" && case.layout == "420" && case.precision == 8)
        .expect("8-bit 4:2:0 progressive fixture");
    let full = extract_dct_blocks(case.jpeg, DctExtractOptions::quantized_only())
        .expect("extract full progressive coefficients");
    let max_h = full
        .components
        .iter()
        .map(|component| component.h_samp)
        .max()
        .unwrap();
    let max_v = full
        .components
        .iter()
        .map(|component| component.v_samp)
        .max()
        .unwrap();
    let mcu_cols = full.width.div_ceil(u32::from(max_h) * 8);
    let mcu_rows = full.height.div_ceil(u32::from(max_v) * 8);
    let window = McuRect {
        x: mcu_cols - 1,
        y: mcu_rows - 1,
        width: 1,
        height: 1,
    };
    let actual = extract_dct_windows(case.jpeg, &[window], DctExtractOptions::quantized_only())
        .expect("extract progressive edge window")
        .pop()
        .unwrap();
    for (actual, source) in actual.components.iter().zip(&full.components) {
        let block_x = window.x * u32::from(source.h_samp);
        let block_y = window.y * u32::from(source.v_samp);
        let expected = (0..actual.block_rows)
            .flat_map(|row| {
                let start = ((block_y + row) * source.block_cols + block_x) as usize;
                source.quantized_blocks[start..start + actual.block_cols as usize].iter()
            })
            .copied()
            .collect::<Vec<_>>();
        assert_eq!(actual.quantized_blocks, expected);
    }
}

#[test]
fn lossless_crop_preserves_progressive_coefficients_and_source_signalling() {
    let bytes = fixtures::progressive_8x8_jpeg();
    let estimate = estimate_dct_windows(
        &bytes,
        &[McuRect {
            x: 0,
            y: 0,
            width: 1,
            height: 1,
        }],
        DctExtractOptions::quantized_only(),
    )
    .expect("estimate progressive crop");
    assert!(estimate.peak_bytes > estimate.retained_bytes);
    let crop = lossless_crop(
        &bytes,
        Rect {
            x: 2,
            y: 3,
            w: 4,
            h: 4,
        },
        LosslessCropOptions::default(),
    )
    .expect("crop progressive JPEG");
    assert_eq!((crop.width, crop.height), (6, 7));
    assert_eq!(
        crop.region,
        Rect {
            x: 2,
            y: 3,
            w: 4,
            h: 4
        }
    );
    let expected = extract_dct_blocks(&bytes, DctExtractOptions::quantized_only())
        .expect("extract source coefficients");
    let actual = extract_dct_blocks(&crop.data, DctExtractOptions::quantized_only())
        .expect("extract crop coefficients");
    for (actual, expected) in actual.components.iter().zip(&expected.components) {
        assert_eq!(actual.quantized_blocks, expected.quantized_blocks);
    }

    let rgb = fixtures::rgb_app14_8x8_jpeg();
    let rgb_crop = lossless_crop(&rgb, Rect::full((8, 8)), LosslessCropOptions::default())
        .expect("crop APP14 RGB JPEG");
    assert!(rgb_crop
        .data
        .windows(7)
        .any(|window| window == b"\xff\xee\0\x0eAdo"));
    assert_eq!(
        j2k_jpeg::Decoder::inspect(&rgb_crop.data)
            .unwrap()
            .color_space,
        j2k_jpeg::ColorSpace::Rgb
    );
}

#[test]
fn lossless_crop_uses_the_requested_far_edge_in_its_sof() {
    let pixels = (0..64_u32 * 64)
        .map(|index| u8::try_from(index * 17 % 256).expect("byte sample"))
        .collect::<Vec<_>>();
    let source = encode_jpeg_baseline(
        JpegSamples::Gray8 {
            data: &pixels,
            width: 64,
            height: 64,
        },
        JpegEncodeOptions {
            subsampling: JpegSubsampling::Gray,
            ..JpegEncodeOptions::default()
        },
    )
    .expect("encode crop fixture");
    let crop = lossless_crop(
        &source.data,
        Rect {
            x: 5,
            y: 6,
            w: 19,
            h: 11,
        },
        LosslessCropOptions::default(),
    )
    .expect("crop unaligned rectangle");
    assert_eq!((crop.width, crop.height), (24, 17));
    assert_eq!(
        crop.region,
        Rect {
            x: 5,
            y: 6,
            w: 19,
            h: 11
        }
    );
    assert_eq!(
        j2k_jpeg::Decoder::inspect(&crop.data).unwrap().dimensions,
        (24, 17)
    );
}

#[test]
fn extracts_ycbcr_444_dct_blocks() {
    let image = extract_dct_blocks(
        &fixtures::baseline_444_8x8_jpeg(),
        DctExtractOptions::default(),
    )
    .expect("extract 4:4:4 DCT blocks");

    assert_eq!((image.width, image.height), (8, 8));
    assert_eq!(image.components.len(), 3);
    for component in &image.components {
        assert_component(component, (8, 8), (1, 1), (1, 1), 1);
    }
}

#[test]
fn extracts_ycbcr_422_dct_blocks_at_native_component_resolution() {
    let image = extract_dct_blocks(
        &fixtures::baseline_422_16x8_jpeg(),
        DctExtractOptions::default(),
    )
    .expect("extract 4:2:2 DCT blocks");

    assert_eq!((image.width, image.height), (16, 8));
    assert_eq!(image.components.len(), 3);
    assert_component(&image.components[0], (16, 8), (2, 1), (2, 1), 2);
    assert_component(&image.components[1], (8, 8), (1, 1), (1, 1), 1);
    assert_component(&image.components[2], (8, 8), (1, 1), (1, 1), 1);
}

#[test]
fn extracts_ycbcr_420_dct_blocks_at_native_component_resolution() {
    let image = extract_dct_blocks(
        &fixtures::minimal_baseline_420_jpeg(),
        DctExtractOptions::default(),
    )
    .expect("extract 4:2:0 DCT blocks");

    assert_eq!((image.width, image.height), (16, 16));
    assert_eq!(image.components.len(), 3);
    assert_component(&image.components[0], (16, 16), (2, 2), (2, 2), 4);
    assert_component(&image.components[1], (8, 8), (1, 1), (1, 1), 1);
    assert_component(&image.components[2], (8, 8), (1, 1), (1, 1), 1);
}

#[test]
fn extracts_restart_coded_ycbcr_420_blocks_and_restart_metadata() {
    let image = extract_dct_blocks(
        &fixtures::baseline_420_restart_32x16_jpeg(),
        DctExtractOptions::default(),
    )
    .expect("extract restart-coded 4:2:0 DCT blocks");

    assert_eq!((image.width, image.height), (32, 16));
    assert_eq!(
        image.restart_index.as_ref().map(|idx| idx.interval_mcus),
        Some(2)
    );
    assert_component(&image.components[0], (32, 16), (2, 2), (4, 2), 8);
    assert_component(&image.components[1], (16, 8), (1, 1), (2, 1), 2);
    assert_component(&image.components[2], (16, 8), (1, 1), (2, 1), 2);
}

#[test]
fn extracts_progressive_ycbcr_420_dct_blocks_at_native_component_resolution() {
    let image = extract_dct_blocks(
        &fixtures::progressive_8x8_jpeg(),
        DctExtractOptions::default(),
    )
    .expect("extract progressive 4:2:0 DCT blocks");

    assert_eq!((image.width, image.height), (8, 8));
    assert_eq!(image.coding_mode, JpegDctCodingMode::Progressive);
    assert_eq!(image.scan_count, 10);
    assert_eq!(image.components.len(), 3);
    assert_component(&image.components[0], (8, 8), (2, 2), (2, 2), 4);
    assert_component(&image.components[1], (4, 4), (1, 1), (1, 1), 1);
    assert_component(&image.components[2], (4, 4), (1, 1), (1, 1), 1);
    assert!(image.restart_index.is_none());
}

#[test]
fn progressive_dequantized_only_extraction_omits_quantized_blocks() {
    let bytes = fixtures::progressive_8x8_jpeg();
    let default_image = extract_dct_blocks(&bytes, DctExtractOptions::default())
        .expect("extract default progressive DCT blocks");
    let dequantized_only = extract_dct_blocks(&bytes, DctExtractOptions::dequantized_only())
        .expect("extract dequantized-only progressive DCT blocks");

    assert_eq!(
        dequantized_only.components.len(),
        default_image.components.len()
    );
    for (actual, expected) in dequantized_only
        .components
        .iter()
        .zip(default_image.components.iter())
    {
        assert_eq!(actual.quantized_blocks, [] as [[i16; 64]; 0]);
        assert_eq!(actual.dequantized_blocks, expected.dequantized_blocks);
    }
}

#[test]
fn exposes_scalar_islow_block_idct_for_transcode_oracles() {
    let image = extract_dct_blocks(
        &fixtures::grayscale_8x8_jpeg(),
        DctExtractOptions::default(),
    )
    .expect("extract grayscale DCT blocks");

    let samples = idct_islow_block(&image.components[0].dequantized_blocks[0]);

    assert_eq!(samples.len(), 64);
    assert!(samples.iter().any(|&sample| sample != 128));
}

fn assert_component(
    component: &j2k_jpeg::transcode::JpegDctComponent,
    sample_dimensions: (u32, u32),
    sampling: (u8, u8),
    block_grid: (u32, u32),
    block_count: usize,
) {
    assert_eq!(
        (component.width, component.height),
        sample_dimensions,
        "component dimensions"
    );
    assert_eq!((component.h_samp, component.v_samp), sampling, "sampling");
    assert_eq!(
        (component.block_cols, component.block_rows),
        block_grid,
        "block grid"
    );
    assert_eq!(component.dequantized_blocks.len(), block_count);
    assert_eq!(component.quantized_blocks.len(), block_count);
    assert!(component
        .dequantized_blocks
        .iter()
        .any(|block| block.iter().any(|&coefficient| coefficient != 0)));
    assert!(component
        .quantized_blocks
        .iter()
        .any(|block| block.iter().any(|&coefficient| coefficient != 0)));
}

const JPEG_ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];
