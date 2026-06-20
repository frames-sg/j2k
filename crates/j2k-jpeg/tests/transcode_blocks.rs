// SPDX-License-Identifier: Apache-2.0

use j2k_test_support as fixtures;

use j2k_jpeg::transcode::{
    extract_dct_blocks, idct_islow_block, DctExtractOptions, JpegDctCodingMode,
};

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
        assert!(actual.quantized_blocks.is_empty());
        assert_eq!(actual.dequantized_blocks, expected.dequantized_blocks);
        assert_eq!(actual.block_cols, expected.block_cols);
        assert_eq!(actual.block_rows, expected.block_rows);
    }
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
