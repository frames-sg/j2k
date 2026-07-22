// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{num::NonZeroUsize, sync::Arc};

use j2k::{
    prepare_batch, wrap_j2k_codestream, BatchDecodeOptions, BatchDecoder, BatchItemError,
    BatchLayout, CpuBatchDecoder, CpuBatchSamples, DecodeRequest, Downscale, EncodedImage,
    J2kComponentMapping, J2kComponentMappingType, J2kFileBoxMetadata, J2kFileColorSpec,
    J2kFileWrapOptions, J2kPaletteColumn, J2kPaletteMetadata, NonRepresentableReason, Rect,
};
use j2k_core::Colorspace;
use j2k_native::{encode, EncodeOptions};

use super::fixtures::{
    four_component_fixture, htj2k_gray8_fixture, rewrite_component_descriptor, rgb8_fixture,
    signed_gray16_fixture,
};

#[test]
fn shared_batch_decoder_interface_prepares_and_decodes_owned_inputs() {
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        workers: NonZeroUsize::new(1),
        ..BatchDecodeOptions::default()
    };
    let input = EncodedImage::full(Arc::from(htj2k_gray8_fixture(3, 2)));
    let mut decoder = CpuBatchDecoder::new(options);

    let prepared = <CpuBatchDecoder as BatchDecoder>::prepare_batch(&decoder, vec![input.clone()])
        .expect("prepare through shared interface");
    assert_eq!(prepared.groups()[0].options().layout, BatchLayout::Nhwc);

    let prepared_output =
        <CpuBatchDecoder as BatchDecoder>::decode_prepared(&mut decoder, &prepared)
            .expect("decode prepared through shared interface");
    let one_shot_output =
        <CpuBatchDecoder as BatchDecoder>::decode_batch(&mut decoder, vec![input])
            .expect("decode owned input through shared interface");
    assert_eq!(prepared_output.groups(), one_shot_output.groups());
}

#[test]
fn prepared_batch_keeps_indexed_preflight_errors_and_decodes_other_inputs() {
    let valid = Arc::<[u8]>::from(htj2k_gray8_fixture(3, 2));
    let inputs = vec![
        EncodedImage::full(Arc::clone(&valid)),
        EncodedImage::full(Arc::<[u8]>::from([0_u8, 1, 2, 3])),
        EncodedImage::new(
            valid,
            DecodeRequest::Region {
                roi: Rect {
                    x: 1,
                    y: 0,
                    w: 2,
                    h: 2,
                },
            },
        ),
    ];
    let options = BatchDecodeOptions::default();
    let prepared = prepare_batch(inputs, options).expect("prepare partial batch");

    assert_eq!(prepared.errors().len(), 1);
    assert_eq!(prepared.errors()[0].index, 1);
    assert!(matches!(
        prepared.errors()[0].source,
        BatchItemError::Codec { .. }
    ));

    let mut session = CpuBatchDecoder::new(options);
    let result = session
        .decode_prepared(&prepared)
        .expect("decode valid prepared inputs");
    assert_eq!(result.groups().len(), 2);
    assert_eq!(result.errors().len(), 1);
    assert_eq!(result.errors()[0].index, 1);
}

#[test]
fn cpu_execution_failure_compacts_and_preserves_successful_group_members() {
    let valid = htj2k_gray8_fixture(4, 4);
    let valid_prepared = prepare_batch(
        vec![EncodedImage::full(Arc::from(valid.clone()))],
        BatchDecodeOptions::default(),
    )
    .expect("prepare valid HT corruption source");
    let cleanup = valid_prepared.groups()[0].images()[0]
        .htj2k_plan()
        .and_then(|plan| plan.payload(0))
        .expect("first HT cleanup payload")
        .cleanup;
    let cleanup_end = cleanup.end().expect("cleanup payload end");
    let mut corrupted = valid.clone();
    corrupted[cleanup.offset..cleanup_end].fill(0xff);
    let options = BatchDecodeOptions::default();
    let prepared = prepare_batch(
        vec![
            EncodedImage::full(Arc::<[u8]>::from(valid.clone())),
            EncodedImage::full(Arc::<[u8]>::from(corrupted)),
            EncodedImage::full(Arc::<[u8]>::from(valid)),
        ],
        options,
    )
    .expect("prepare header-valid group");

    assert!(prepared.errors().is_empty());
    assert_eq!(prepared.groups().len(), 1);
    let mut session = CpuBatchDecoder::new(options);
    let result = session
        .decode_prepared(&prepared)
        .expect("complete batch execution");

    assert_eq!(result.groups().len(), 1);
    assert_eq!(result.groups()[0].source_indices(), &[0, 2]);
    let CpuBatchSamples::U8(samples) = result.groups()[0].samples() else {
        panic!("expected Gray8 samples")
    };
    assert_eq!(samples.len(), 2 * 4 * 4);
    let expected = (0_u8..16).collect::<Vec<_>>();
    assert_eq!(&samples[..expected.len()], &expected);
    assert_eq!(&samples[expected.len()..], &expected);
    assert_eq!(result.errors().len(), 1);
    assert_eq!(result.errors()[0].index, 1);
}

#[test]
fn cpu_batch_preserves_signed_i16_samples_in_nchw_layout() {
    let first = [-300_i16, -1, 0, 300];
    let second = [511_i16, -512, 12, -12];
    let inputs = vec![
        EncodedImage::full(Arc::<[u8]>::from(signed_gray16_fixture(&first, 2, 2))),
        EncodedImage::full(Arc::<[u8]>::from(signed_gray16_fixture(&second, 2, 2))),
    ];
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nchw,
        ..BatchDecodeOptions::default()
    };
    let mut session = CpuBatchDecoder::new(options);
    let result = session.decode(inputs).expect("decode signed batch");

    assert!(result.errors().is_empty());
    assert_eq!(result.groups().len(), 1);
    assert_eq!(result.groups()[0].source_indices(), &[0, 1]);
    let CpuBatchSamples::I16(samples) = result.groups()[0].samples() else {
        panic!("expected i16 samples")
    };
    assert_eq!(samples, &[first, second].concat());
}

#[test]
fn reduced_and_region_reduced_requests_group_by_decoded_geometry() {
    let bytes = Arc::<[u8]>::from(htj2k_gray8_fixture(7, 5));
    let prepared = prepare_batch(
        vec![
            EncodedImage::new(
                Arc::clone(&bytes),
                DecodeRequest::Reduced {
                    scale: Downscale::Half,
                },
            ),
            EncodedImage::new(
                bytes,
                DecodeRequest::RegionReduced {
                    roi: Rect {
                        x: 1,
                        y: 1,
                        w: 5,
                        h: 3,
                    },
                    scale: Downscale::Half,
                },
            ),
        ],
        BatchDecodeOptions::default(),
    )
    .expect("prepare reduced requests");

    assert!(prepared.errors().is_empty());
    assert_eq!(prepared.groups().len(), 2);
    assert_eq!(prepared.groups()[0].info().dimensions, (4, 3));
    assert_eq!(prepared.groups()[1].info().dimensions, (3, 2));

    let mut session = CpuBatchDecoder::new(BatchDecodeOptions::default());
    let result = session
        .decode_prepared(&prepared)
        .expect("decode reduced requests");
    assert!(result.errors().is_empty());
    assert_eq!(result.groups()[0].samples().len(), 12);
    assert_eq!(result.groups()[1].samples().len(), 6);
}

#[test]
fn prepare_reports_nonrepresentable_subsampled_mixed_and_high_precision_inputs() {
    let mut subsampled = rgb8_fixture();
    let siz_marker = subsampled
        .windows(2)
        .position(|marker| marker == [0xff, 0x51])
        .expect("SIZ marker");
    subsampled[siz_marker + 44] = 2;

    let mut mixed = rgb8_fixture();
    rewrite_component_descriptor(&mut mixed, 1, 11);

    let mut high_precision = rgb8_fixture();
    for component in 0..3 {
        rewrite_component_descriptor(&mut high_precision, component, 16);
    }

    let prepared = prepare_batch(
        vec![subsampled, mixed, high_precision]
            .into_iter()
            .map(|bytes| EncodedImage::full(Arc::<[u8]>::from(bytes)))
            .collect(),
        BatchDecodeOptions::default(),
    )
    .expect("prepare rejected profiles");

    assert!(prepared.groups().is_empty());
    assert_eq!(prepared.errors().len(), 3);
    let reasons = prepared
        .errors()
        .iter()
        .map(|error| match error.source {
            BatchItemError::NonRepresentableBatchOutput { reason } => reason,
            _ => panic!("expected representability error"),
        })
        .collect::<Vec<_>>();
    assert_eq!(
        reasons,
        [
            NonRepresentableReason::ComponentSubsampling,
            NonRepresentableReason::MixedPrecision,
            NonRepresentableReason::PrecisionAboveSixteen,
        ]
    );
}

#[test]
fn prepare_rejects_raw_four_component_data_without_explicit_alpha_metadata() {
    let prepared = prepare_batch(
        vec![EncodedImage::full(Arc::from(four_component_fixture()))],
        BatchDecodeOptions::default(),
    )
    .expect("prepare raw four-component input");

    assert!(prepared.groups().is_empty());
    assert_eq!(prepared.errors().len(), 1);
    assert!(matches!(
        prepared.errors()[0].source,
        BatchItemError::NonRepresentableBatchOutput {
            reason: NonRepresentableReason::UnsupportedColor
        }
    ));
}

#[test]
fn prepare_rejects_palette_mapped_wrapper_before_pixel_decode() {
    let codestream = encode(
        &[0_u8, 1, 1, 0],
        2,
        2,
        1,
        8,
        false,
        &EncodeOptions::default(),
    )
    .expect("encode palette indices");
    let palette = J2kPaletteMetadata {
        columns: vec![
            J2kPaletteColumn {
                bit_depth: 8,
                signed: false,
            },
            J2kPaletteColumn {
                bit_depth: 8,
                signed: false,
            },
            J2kPaletteColumn {
                bit_depth: 8,
                signed: false,
            },
        ],
        entries: vec![vec![10, 20, 30], vec![200, 210, 220]],
    };
    let mappings = [0_u8, 1, 2].map(|column| J2kComponentMapping {
        component_index: 0,
        mapping_type: J2kComponentMappingType::Palette { column },
    });
    let wrapped = wrap_j2k_codestream(
        &codestream,
        J2kFileWrapOptions::jp2()
            .with_color(J2kFileColorSpec::Enumerated(Colorspace::SRgb))
            .with_metadata(J2kFileBoxMetadata {
                palette: Some(&palette),
                component_mappings: &mappings,
                channel_definitions: &[],
            }),
    )
    .expect("wrap palette image");

    let prepared = prepare_batch(
        vec![EncodedImage::full(Arc::from(wrapped))],
        BatchDecodeOptions::default(),
    )
    .expect("prepare palette image");

    assert!(prepared.groups().is_empty());
    assert!(matches!(
        prepared.errors()[0].source,
        BatchItemError::NonRepresentableBatchOutput {
            reason: NonRepresentableReason::UnsupportedColor
        }
    ));
}
