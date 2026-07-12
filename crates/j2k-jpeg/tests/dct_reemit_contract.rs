// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_jpeg::transcode::{
    encode_baseline_dct_image, extract_dct_blocks, DctExtractOptions, JpegDctCodingMode,
    JpegDctImage, JpegDctImageError,
};
use j2k_jpeg::{ColorSpace, JpegEncodeError, RestartIndex};
use j2k_test_support as fixtures;

fn grayscale_image() -> JpegDctImage {
    extract_dct_blocks(
        &fixtures::grayscale_8x8_jpeg(),
        DctExtractOptions::default(),
    )
    .expect("extract grayscale baseline DCT image")
}

fn color_image() -> JpegDctImage {
    extract_dct_blocks(
        &fixtures::minimal_baseline_420_jpeg(),
        DctExtractOptions::default(),
    )
    .expect("extract color baseline DCT image")
}

fn invalid_reason(image: &JpegDctImage) -> JpegDctImageError {
    match encode_baseline_dct_image(image) {
        Err(JpegEncodeError::InvalidDctImage { reason }) => reason,
        Err(other) => panic!("expected typed invalid DCT image, got {other:?}"),
        Ok(_) => panic!("invalid DCT image unexpectedly encoded"),
    }
}

#[test]
fn dct_reemission_rejects_mode_dimensions_component_count_and_order() {
    let mut image = grayscale_image();
    image.coding_mode = JpegDctCodingMode::Progressive;
    assert!(matches!(
        invalid_reason(&image),
        JpegDctImageError::UnsupportedCodingMode {
            actual: JpegDctCodingMode::Progressive
        }
    ));

    let mut image = grayscale_image();
    image.width = 0;
    assert!(matches!(
        invalid_reason(&image),
        JpegDctImageError::EmptyDimensions {
            width: 0,
            height: 8
        }
    ));

    let mut image = grayscale_image();
    image.height = u32::from(u16::MAX) + 1;
    assert!(matches!(
        invalid_reason(&image),
        JpegDctImageError::DimensionsTooLarge { .. }
    ));

    let mut image = color_image();
    image.components.truncate(2);
    assert!(matches!(
        invalid_reason(&image),
        JpegDctImageError::UnsupportedComponentCount { actual: 2 }
    ));

    let mut image = color_image();
    image.components[1].component_index = 0;
    assert!(matches!(
        invalid_reason(&image),
        JpegDctImageError::ComponentOrderMismatch {
            position: 1,
            component_index: 0
        }
    ));
}

#[test]
fn dct_reemission_rejects_each_invalid_sampling_factor_and_mcu_sum() {
    for (h_samp, v_samp) in [(0, 1), (1, 0), (5, 1), (1, 5)] {
        let mut image = color_image();
        image.components[1].h_samp = h_samp;
        image.components[1].v_samp = v_samp;
        assert!(matches!(
            invalid_reason(&image),
            JpegDctImageError::SamplingFactorOutOfRange {
                component_index: 1,
                h_samp: actual_h,
                v_samp: actual_v,
            } if actual_h == h_samp && actual_v == v_samp
        ));
    }

    let mut image = grayscale_image();
    image.components[0].h_samp = 2;
    assert!(matches!(
        invalid_reason(&image),
        JpegDctImageError::UnsupportedGrayscaleSampling {
            h_samp: 2,
            v_samp: 1
        }
    ));

    let mut image = color_image();
    (image.components[0].h_samp, image.components[0].v_samp) = (4, 2);
    (image.components[1].h_samp, image.components[1].v_samp) = (1, 2);
    (image.components[2].h_samp, image.components[2].v_samp) = (1, 1);
    assert!(matches!(
        invalid_reason(&image),
        JpegDctImageError::TooManyBlocksPerMcu { blocks_per_mcu: 11 }
    ));
}

#[test]
fn dct_reemission_rejects_grid_and_quantized_block_count_mismatches() {
    let mut image = color_image();
    image.components[0].block_cols += 1;
    assert!(matches!(
        invalid_reason(&image),
        JpegDctImageError::BlockGridMismatch {
            component_index: 0,
            ..
        }
    ));

    let mut image = color_image();
    image.components[0].quantized_blocks.pop();
    assert!(matches!(
        invalid_reason(&image),
        JpegDctImageError::QuantizedBlockCountMismatch {
            component_index: 0,
            actual: 3,
            expected: 4,
        }
    ));
}

#[test]
fn dct_reemission_rejects_nonbaseline_coefficient_categories() {
    let mut image = grayscale_image();
    image.components[0].quantized_blocks[0][0] = i16::MAX;
    assert!(matches!(
        invalid_reason(&image),
        JpegDctImageError::DcMagnitudeCategoryOutOfRange {
            component_index: 0,
            block_index: 0,
            difference: 32_767,
            category: 15,
        }
    ));

    let mut image = grayscale_image();
    image.components[0].quantized_blocks[0] = [0; 64];
    image.components[0].quantized_blocks[0][12] = i16::MIN;
    assert!(matches!(
        invalid_reason(&image),
        JpegDctImageError::AcMagnitudeCategoryOutOfRange {
            component_index: 0,
            block_index: 0,
            coefficient_index: 12,
            value: i16::MIN,
            category: 16,
        }
    ));
}

#[test]
fn dct_reemission_accepts_baseline_coefficient_category_boundaries() {
    let mut image = grayscale_image();
    image.components[0].quantized_blocks[0] = [0; 64];
    image.components[0].quantized_blocks[0][0] = 2_047;
    image.components[0].quantized_blocks[0][1] = 1_023;

    let encoded = encode_baseline_dct_image(&image).expect("encode baseline category boundaries");
    let decoded = extract_dct_blocks(&encoded, DctExtractOptions::default())
        .expect("decode baseline category boundaries");
    assert_eq!(
        decoded.components[0].quantized_blocks,
        image.components[0].quantized_blocks
    );
}

#[test]
fn dct_reemission_rejects_nonbaseline_and_unshared_quantization_tables() {
    for value in [0, 256] {
        let mut image = grayscale_image();
        image.components[0].quant_table[7] = value;
        assert!(matches!(
            invalid_reason(&image),
            JpegDctImageError::QuantizationValueOutOfRange {
                component_index: 0,
                zigzag_index: 7,
                value: actual,
            } if actual == value
        ));
    }

    let mut image = color_image();
    image.components[2].quant_table[0] = image.components[2].quant_table[0].saturating_add(1);
    assert!(matches!(
        invalid_reason(&image),
        JpegDctImageError::ChromaQuantizationTableMismatch
    ));
}

#[test]
fn dct_reemission_ignores_metadata_that_does_not_affect_baseline_output() {
    let source = color_image();
    let expected = encode_baseline_dct_image(&source).expect("encode source DCT image");
    let mut ignored = source;
    ignored.color_space = ColorSpace::Cmyk;
    ignored.scan_count = u16::MAX;
    ignored.restart_index = Some(RestartIndex {
        scan_data_offset: usize::MAX,
        interval_mcus: u32::MAX,
        segments: Vec::new(),
    });
    for component in &mut ignored.components {
        component.width = u32::MAX;
        component.height = 0;
        component.dequantized_blocks.clear();
    }

    let actual = encode_baseline_dct_image(&ignored).expect("ignored metadata stays accepted");
    assert_eq!(actual, expected);
}
