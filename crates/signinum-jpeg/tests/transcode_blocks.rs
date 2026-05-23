// SPDX-License-Identifier: Apache-2.0

mod fixtures;

use signinum_jpeg::transcode::{extract_dct_blocks, idct_islow_block, DctExtractOptions};
use signinum_jpeg::{JpegError, SofKind};

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
fn rejects_progressive_jpeg_for_dct_extraction() {
    let err = extract_dct_blocks(
        &fixtures::progressive_8x8_jpeg(),
        DctExtractOptions::default(),
    )
    .expect_err("progressive is out of scope");

    assert!(matches!(
        err,
        JpegError::NotImplemented {
            sof: SofKind::Progressive8
        }
    ));
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
    component: &signinum_jpeg::transcode::JpegDctComponent,
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
    assert!(component
        .dequantized_blocks
        .iter()
        .any(|block| block.iter().any(|&coefficient| coefficient != 0)));
}
