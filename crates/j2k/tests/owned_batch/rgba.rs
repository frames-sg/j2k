// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{num::NonZeroUsize, sync::Arc};

use j2k::{
    prepare_batch, wrap_j2k_codestream, BatchAlpha, BatchDecodeOptions, BatchItemError,
    BatchLayout, CpuBatchDecoder, CpuBatchSamples, DecodeRequest, Downscale, EncodedImage,
    J2kChannelAssociation, J2kChannelDefinition, J2kChannelType, J2kFileBoxMetadata,
    J2kFileColorSpec, J2kFileWrapOptions, NativeSampleType, NonRepresentableReason,
    PreparationDepth, Rect,
};
use j2k_core::Colorspace;
use j2k_test_support::{
    generated_htj2k_rgba_fixture, Htj2kRgbaAlpha, Htj2kRgbaSampleProfile, Htj2kRgbaSamples,
};

use super::fixtures::{four_component_fixture, rgb8_fixture, wrap_rgba_jph};
use super::oracles::{apply_batch_layout, native_request_oracle};
use super::payload_plan::native_prepared_plan;

#[test]
fn prepare_accepts_identity_rgb_cdef_with_explicit_alpha() {
    let codestream = four_component_fixture();
    let channels = [
        J2kChannelDefinition {
            channel_index: 0,
            channel_type: J2kChannelType::Color,
            association: J2kChannelAssociation::Color { index: 1 },
        },
        J2kChannelDefinition {
            channel_index: 1,
            channel_type: J2kChannelType::Color,
            association: J2kChannelAssociation::Color { index: 2 },
        },
        J2kChannelDefinition {
            channel_index: 2,
            channel_type: J2kChannelType::Color,
            association: J2kChannelAssociation::Color { index: 3 },
        },
        J2kChannelDefinition {
            channel_index: 3,
            channel_type: J2kChannelType::Opacity,
            association: J2kChannelAssociation::WholeImage,
        },
    ];
    let wrapped = wrap_j2k_codestream(
        &codestream,
        J2kFileWrapOptions::jp2()
            .with_color(J2kFileColorSpec::Enumerated(Colorspace::SRgb))
            .with_metadata(J2kFileBoxMetadata {
                palette: None,
                component_mappings: &[],
                channel_definitions: &channels,
            }),
    )
    .expect("wrap explicit RGBA image");

    let prepared = prepare_batch(
        vec![EncodedImage::full(Arc::from(wrapped))],
        BatchDecodeOptions::default(),
    )
    .expect("prepare explicit RGBA image");

    assert!(prepared.errors().is_empty());
    assert_eq!(prepared.groups()[0].info().color.channels(), 4);
    assert_eq!(prepared.groups()[0].info().alpha, BatchAlpha::Straight);
    let prepared_image = &prepared.groups()[0].images()[0];
    assert_eq!(
        prepared_image.preparation_depth(),
        PreparationDepth::ClassicOffsetPlan
    );
    assert!(prepared_image
        .classic_plan()
        .is_some_and(j2k::PreparedClassicPlan::is_rgba));

    let mut session = CpuBatchDecoder::new(BatchDecodeOptions::default());
    let output = session
        .decode_prepared(&prepared)
        .expect("decode explicit RGBA image");
    assert!(output.errors().is_empty());
    let CpuBatchSamples::U8(samples) = output.groups()[0].samples() else {
        panic!("expected native RGBA8 samples")
    };
    let expected = (0_u8..4)
        .flat_map(|channel| (0_u8..16).map(move |pixel| pixel * 4 + channel))
        .collect::<Vec<_>>();
    assert_eq!(samples, &expected);
    assert_eq!(session.workspace_stats().decode_calls(), 0);
    assert_eq!(session.workspace_stats().prepared_plan_decode_calls(), 1);
}

#[test]
fn prepare_preserves_and_groups_straight_and_premultiplied_alpha_separately() {
    let codestream = four_component_fixture();
    let rgba_definitions = |alpha_type| {
        [
            J2kChannelDefinition {
                channel_index: 0,
                channel_type: J2kChannelType::Color,
                association: J2kChannelAssociation::Color { index: 1 },
            },
            J2kChannelDefinition {
                channel_index: 1,
                channel_type: J2kChannelType::Color,
                association: J2kChannelAssociation::Color { index: 2 },
            },
            J2kChannelDefinition {
                channel_index: 2,
                channel_type: J2kChannelType::Color,
                association: J2kChannelAssociation::Color { index: 3 },
            },
            J2kChannelDefinition {
                channel_index: 3,
                channel_type: alpha_type,
                association: J2kChannelAssociation::WholeImage,
            },
        ]
    };
    let wrap = |alpha_type| {
        let definitions = rgba_definitions(alpha_type);
        wrap_j2k_codestream(
            &codestream,
            J2kFileWrapOptions::jp2()
                .with_color(J2kFileColorSpec::Enumerated(Colorspace::SRgb))
                .with_metadata(J2kFileBoxMetadata {
                    palette: None,
                    component_mappings: &[],
                    channel_definitions: &definitions,
                }),
        )
        .expect("wrap RGBA image")
    };

    let prepared = prepare_batch(
        vec![
            EncodedImage::full(Arc::from(wrap(J2kChannelType::Opacity))),
            EncodedImage::full(Arc::from(wrap(J2kChannelType::PremultipliedOpacity))),
        ],
        BatchDecodeOptions::default(),
    )
    .expect("prepare alpha interpretations");

    assert!(prepared.errors().is_empty());
    assert_eq!(prepared.groups().len(), 2);
    assert_eq!(prepared.groups()[0].source_indices(), &[0]);
    assert_eq!(prepared.groups()[0].info().alpha, BatchAlpha::Straight);
    assert_eq!(prepared.groups()[1].source_indices(), &[1]);
    assert_eq!(prepared.groups()[1].info().alpha, BatchAlpha::Premultiplied);
}

#[test]
fn prepared_htj2k_rgba_preserves_alpha_semantics_and_avoids_reparse() {
    let fixture =
        generated_htj2k_rgba_fixture(Htj2kRgbaSampleProfile::U8Rct, Htj2kRgbaAlpha::Straight);
    let codestream = fixture.encoded;
    let options = BatchDecodeOptions {
        workers: NonZeroUsize::new(1),
        layout: BatchLayout::Nchw,
        ..BatchDecodeOptions::default()
    };
    let prepared = prepare_batch(
        vec![
            EncodedImage::full(Arc::from(wrap_rgba_jph(
                &codestream,
                Htj2kRgbaAlpha::Straight,
            ))),
            EncodedImage::full(Arc::from(wrap_rgba_jph(
                &codestream,
                Htj2kRgbaAlpha::Premultiplied,
            ))),
        ],
        options,
    )
    .expect("prepare retained RGBA HTJ2K plans");

    assert!(prepared.errors().is_empty());
    assert_eq!(prepared.groups().len(), 2);
    assert_eq!(prepared.groups()[0].info().alpha, BatchAlpha::Straight);
    assert_eq!(prepared.groups()[1].info().alpha, BatchAlpha::Premultiplied);
    assert!(prepared.groups().iter().all(|group| {
        let image = &group.images()[0];
        image.preparation_depth() == PreparationDepth::Htj2kOffsetPlan
            && image.htj2k_plan().is_some_and(|plan| {
                plan.is_rgba()
                    && !plan.is_grayscale()
                    && !plan.is_color()
                    && native_prepared_plan(plan)
                        .rgba_geometry()
                        .is_some_and(|geometry| {
                            geometry.component_plans.len() == 4
                                && geometry.bit_depths == [8; 4]
                                && geometry.mct
                        })
            })
    }));

    let mut decoder = CpuBatchDecoder::new(options);
    let first = decoder
        .decode_prepared(&prepared)
        .expect("first parse-free RGBA decode");
    let first_stats = decoder.workspace_stats();
    let second = decoder
        .decode_prepared(&prepared)
        .expect("second parse-free RGBA decode");
    let second_stats = decoder.workspace_stats();

    assert!(first.errors().is_empty());
    assert_eq!(first.groups(), second.groups());
    let Htj2kRgbaSamples::U8(source_samples) = fixture.samples else {
        panic!("shared U8 RGBA fixture must retain U8 source samples")
    };
    let expected_source = CpuBatchSamples::U8(apply_batch_layout(
        source_samples,
        fixture.width as usize * fixture.height as usize,
        4,
        options.layout,
    ));
    for group in first.groups() {
        let source_index = group.source_indices()[0];
        let prepared_image = prepared
            .groups()
            .iter()
            .flat_map(j2k::PreparedBatchGroup::images)
            .find(|image| image.source_index() == source_index)
            .expect("prepared RGBA source");
        assert_eq!(
            group.samples(),
            &native_request_oracle(prepared_image, options.layout)
        );
        assert_eq!(group.samples(), &expected_source);
    }
    assert_eq!(first_stats.prepared_plan_decode_calls(), 2);
    assert_eq!(second_stats.prepared_plan_decode_calls(), 4);
    assert_eq!(first_stats.decode_calls(), 0);
    assert_eq!(second_stats.decode_calls(), 0);
    assert!(first_stats.retained_prepared_plan_ht_workspace_bytes() > 0);
    assert_eq!(
        second_stats.retained_prepared_plan_ht_workspace_bytes(),
        first_stats.retained_prepared_plan_ht_workspace_bytes()
    );
}

#[test]
fn prepared_htj2k_rgba_supports_native_types_requests_and_layouts_exactly() {
    let profiles = [
        (Htj2kRgbaSampleProfile::U8Rct, NativeSampleType::U8),
        (Htj2kRgbaSampleProfile::U12, NativeSampleType::U16),
        (Htj2kRgbaSampleProfile::I16, NativeSampleType::I16),
    ];
    let roi = Rect {
        x: 1,
        y: 2,
        w: 5,
        h: 4,
    };
    let requests = [
        DecodeRequest::Full,
        DecodeRequest::Region { roi },
        DecodeRequest::Reduced {
            scale: Downscale::Half,
        },
        DecodeRequest::RegionReduced {
            roi,
            scale: Downscale::Half,
        },
    ];

    for (profile, sample_type) in profiles {
        let fixture = generated_htj2k_rgba_fixture(profile, Htj2kRgbaAlpha::Straight);
        let bit_depth = fixture.bit_depth;
        let signed = fixture.signed;
        let encoded = Arc::<[u8]>::from(wrap_rgba_jph(&fixture.encoded, fixture.alpha));
        for layout in [BatchLayout::Nchw, BatchLayout::Nhwc] {
            let options = BatchDecodeOptions {
                workers: NonZeroUsize::new(1),
                layout,
                ..BatchDecodeOptions::default()
            };
            let prepared = prepare_batch(
                requests
                    .into_iter()
                    .map(|request| EncodedImage::new(Arc::clone(&encoded), request))
                    .collect(),
                options,
            )
            .expect("prepare RGBA request matrix");
            assert!(
                prepared.errors().is_empty(),
                "{bit_depth}-bit signed={signed} {layout:?}: {:?}",
                prepared.errors()
            );
            assert!(prepared.groups().iter().all(|group| {
                group.info().sample_type == sample_type
                    && group.info().color.channels() == 4
                    && group.images().iter().all(|image| {
                        image.preparation_depth() == PreparationDepth::Htj2kOffsetPlan
                            && image.htj2k_plan().is_some()
                    })
            }));

            let mut decoder = CpuBatchDecoder::new(options);
            let result = decoder
                .decode_prepared(&prepared)
                .expect("decode RGBA request matrix");
            assert!(result.errors().is_empty());
            assert_eq!(decoder.workspace_stats().decode_calls(), 0);
            assert_eq!(decoder.workspace_stats().prepared_plan_decode_calls(), 4);

            for source_index in 0..requests.len() {
                let prepared_image = prepared
                    .groups()
                    .iter()
                    .flat_map(j2k::PreparedBatchGroup::images)
                    .find(|image| image.source_index() == source_index)
                    .expect("prepared RGBA request source");
                let group = result
                    .groups()
                    .iter()
                    .find(|group| group.source_indices() == [source_index])
                    .expect("decoded RGBA request group");
                assert_eq!(group.samples().sample_type(), sample_type);
                assert_eq!(
                    group.samples(),
                    &native_request_oracle(prepared_image, layout),
                    "{bit_depth}-bit signed={signed} {layout:?} source={source_index}"
                );
            }
        }
    }
}

#[test]
fn prepare_rejects_icc_tagged_and_reordering_cdef_outputs() {
    let rgb = rgb8_fixture();
    let icc = wrap_j2k_codestream(
        &rgb,
        J2kFileWrapOptions::jp2().with_color(J2kFileColorSpec::IccProfile(b"test-profile")),
    )
    .expect("wrap ICC-tagged RGB");
    let reordered = [
        J2kChannelDefinition {
            channel_index: 0,
            channel_type: J2kChannelType::Color,
            association: J2kChannelAssociation::Color { index: 3 },
        },
        J2kChannelDefinition {
            channel_index: 1,
            channel_type: J2kChannelType::Color,
            association: J2kChannelAssociation::Color { index: 2 },
        },
        J2kChannelDefinition {
            channel_index: 2,
            channel_type: J2kChannelType::Color,
            association: J2kChannelAssociation::Color { index: 1 },
        },
    ];
    let cdef = wrap_j2k_codestream(
        &rgb,
        J2kFileWrapOptions::jp2()
            .with_color(J2kFileColorSpec::Enumerated(Colorspace::SRgb))
            .with_metadata(J2kFileBoxMetadata {
                palette: None,
                component_mappings: &[],
                channel_definitions: &reordered,
            }),
    )
    .expect("wrap reordered CDEF RGB");

    let prepared = prepare_batch(
        vec![icc, cdef]
            .into_iter()
            .map(|bytes| EncodedImage::full(Arc::from(bytes)))
            .collect(),
        BatchDecodeOptions::default(),
    )
    .expect("prepare unsupported color metadata");

    assert!(prepared.groups().is_empty());
    assert_eq!(prepared.errors().len(), 2);
    assert!(prepared.errors().iter().all(|error| matches!(
        error.source,
        BatchItemError::NonRepresentableBatchOutput {
            reason: NonRepresentableReason::UnsupportedColor
        }
    )));
}
