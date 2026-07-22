#![cfg(target_os = "macos")]

use std::sync::Arc;

use j2k::{
    wrap_j2k_codestream, BatchAlpha, BatchColor, BatchDecodeOptions, BatchLayout, CpuBatchDecoder,
    CpuBatchSamples, DecodeRequest, EncodedImage, J2kChannelAssociation, J2kChannelDefinition,
    J2kChannelType, J2kFileBoxMetadata, J2kFileColorSpec, J2kFileWrapOptions, NativeSampleType,
};
use j2k_core::{Colorspace, DeviceSurface, Downscale, PixelFormat, Rect};
use j2k_metal::{MetalBatchDecoder, MetalImageDestination, MetalImageLayout};
use j2k_test_support::{
    generated_htj2k_rgba_fixture, Htj2kRgbaAlpha, Htj2kRgbaFixture, Htj2kRgbaSampleProfile,
    Htj2kRgbaSamples,
};

fn should_run_metal_runtime() -> bool {
    j2k_test_support::metal_runtime_gate(module_path!())
}

fn rgba_format(sample_type: NativeSampleType) -> PixelFormat {
    match sample_type {
        NativeSampleType::U8 => PixelFormat::Rgba8,
        NativeSampleType::U16 => PixelFormat::Rgba16,
        NativeSampleType::I16 => PixelFormat::RgbaI16,
        _ => panic!("unsupported RGBA sample type"),
    }
}

fn assert_native_samples(actual: &[u8], expected: &CpuBatchSamples) {
    match expected {
        CpuBatchSamples::U8(expected) => assert_eq!(actual, expected),
        CpuBatchSamples::U16(expected) => {
            let actual = actual
                .chunks_exact(2)
                .map(|sample| u16::from_le_bytes([sample[0], sample[1]]))
                .collect::<Vec<_>>();
            assert_eq!(&actual, expected);
        }
        CpuBatchSamples::I16(expected) => {
            let actual = actual
                .chunks_exact(2)
                .map(|sample| i16::from_le_bytes([sample[0], sample[1]]))
                .collect::<Vec<_>>();
            assert_eq!(&actual, expected);
        }
        _ => panic!("unsupported CPU RGBA oracle type"),
    }
}

fn wrap_rgba_jph(codestream: &[u8], alpha: Htj2kRgbaAlpha) -> Vec<u8> {
    wrap_rgba_container(
        codestream,
        alpha,
        J2kFileWrapOptions::jph(),
        "wrap explicit HTJ2K RGBA image",
    )
}

fn wrap_rgba_container(
    codestream: &[u8],
    alpha: Htj2kRgbaAlpha,
    options: J2kFileWrapOptions<'_>,
    context: &str,
) -> Vec<u8> {
    let alpha_type = match alpha {
        Htj2kRgbaAlpha::Straight => J2kChannelType::Opacity,
        Htj2kRgbaAlpha::Premultiplied => J2kChannelType::PremultipliedOpacity,
    };
    let channel_definitions = [
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
    ];
    wrap_j2k_codestream(
        codestream,
        options
            .with_color(J2kFileColorSpec::Enumerated(Colorspace::SRgb))
            .with_metadata(J2kFileBoxMetadata {
                palette: None,
                component_mappings: &[],
                channel_definitions: &channel_definitions,
            }),
    )
    .unwrap_or_else(|error| panic!("{context}: {error}"))
}

fn wrap_classic_rgba_jp2(fixture: &Htj2kRgbaFixture) -> Vec<u8> {
    let pixels = match &fixture.samples {
        Htj2kRgbaSamples::U8(samples) => samples.clone(),
        Htj2kRgbaSamples::U16(samples) => samples
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect(),
        Htj2kRgbaSamples::I16(samples) => samples
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect(),
    };
    let codestream = j2k_native::encode(
        &pixels,
        fixture.width,
        fixture.height,
        4,
        fixture.bit_depth,
        fixture.signed,
        &j2k_native::EncodeOptions {
            reversible: true,
            num_decomposition_levels: 2,
            use_mct: fixture.use_mct,
            ..j2k_native::EncodeOptions::default()
        },
    )
    .expect("encode classic RGBA fixture");
    wrap_rgba_container(
        &codestream,
        fixture.alpha,
        J2kFileWrapOptions::jp2(),
        "wrap classic RGBA image",
    )
}

fn assert_rgba_encoding(codec: &str, encoded: &Arc<[u8]>, requests: &[DecodeRequest]) {
    for layout in [BatchLayout::Nchw, BatchLayout::Nhwc] {
        let options = BatchDecodeOptions {
            layout,
            ..BatchDecodeOptions::default()
        };
        let inputs = requests
            .iter()
            .copied()
            .map(|request| EncodedImage::new(encoded.clone(), request))
            .collect::<Vec<_>>();
        let mut cpu = CpuBatchDecoder::new(options);
        let expected = cpu.decode(inputs.clone()).expect("CPU RGBA oracle");
        assert!(
            expected.errors().is_empty(),
            "{codec} CPU RGBA oracle errors: {:?}",
            expected.errors()
        );

        let mut decoder = MetalBatchDecoder::system_default_with_options(options)
            .expect("persistent Metal decoder");
        let prepared = decoder
            .prepare(inputs)
            .expect("prepare Metal RGBA request matrix");
        assert!(
            prepared.errors().is_empty(),
            "{codec} prepare errors: {:?}",
            prepared.errors()
        );

        for group in prepared.groups() {
            assert_eq!(group.info().color, BatchColor::Rgba);
            assert_eq!(group.info().alpha, BatchAlpha::Straight);
            let expected_group = expected
                .groups()
                .iter()
                .find(|expected| expected.source_indices() == group.source_indices())
                .expect("matching CPU RGBA group");
            let fmt = rgba_format(group.info().sample_type);
            let bytes_per_sample = fmt.bytes_per_sample();
            let (width, height) = group.info().dimensions;
            let row_bytes = width as usize * 4 * bytes_per_sample;
            let image_bytes = row_bytes * height as usize;
            let output_bytes = image_bytes * group.images().len();
            let buffer = j2k_metal_support::checked_shared_buffer_for_len::<u8>(
                decoder.backend_session().device(),
                output_bytes + 8,
            )
            .expect("RGBA destination buffer");
            let destination_layout = MetalImageLayout::new_batch(
                4,
                (width, height),
                row_bytes,
                fmt,
                group.images().len(),
                image_bytes,
            )
            .expect("RGBA destination layout");
            // SAFETY: this fresh output range remains exclusively retained by
            // the pending codec submission until completion.
            let destination = unsafe {
                MetalImageDestination::from_exclusive_buffer(buffer.clone(), destination_layout)
                    .expect("RGBA destination")
            };
            decoder
                .submit_prepared_group_into(group, destination)
                .expect("submit prepared RGBA group")
                .wait()
                .expect("complete prepared RGBA group");

            // SAFETY: codec completion released exclusive output access.
            let actual = unsafe {
                j2k_metal_support::checked_buffer_read_vec::<u8>(&buffer, 4, output_bytes)
                    .expect("RGBA output samples")
            };
            assert_native_samples(&actual, expected_group.samples());
        }
    }
}

#[test]
fn prepared_rgba_matches_cpu_for_codecs_native_types_requests_and_layouts() {
    if !should_run_metal_runtime() {
        return;
    }

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

    for profile in [
        Htj2kRgbaSampleProfile::U8Rct,
        Htj2kRgbaSampleProfile::U12,
        Htj2kRgbaSampleProfile::I16,
    ] {
        let fixture = generated_htj2k_rgba_fixture(profile, Htj2kRgbaAlpha::Straight);
        let htj2k = Arc::<[u8]>::from(wrap_rgba_jph(&fixture.encoded, fixture.alpha));
        let classic = Arc::<[u8]>::from(wrap_classic_rgba_jp2(&fixture));
        assert_rgba_encoding("HTJ2K", &htj2k, &requests);
        assert_rgba_encoding("classic JPEG 2000", &classic, &requests);
    }
}

#[test]
fn prepared_htj2k_rgba_nhwc_resident_group_is_exact_and_uses_one_allocation() {
    if !should_run_metal_runtime() {
        return;
    }

    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let fixture =
        generated_htj2k_rgba_fixture(Htj2kRgbaSampleProfile::U8Rct, Htj2kRgbaAlpha::Straight);
    let encoded = Arc::<[u8]>::from(wrap_rgba_jph(&fixture.encoded, fixture.alpha));
    let inputs = vec![
        EncodedImage::full(encoded.clone()),
        EncodedImage::full(encoded),
    ];
    let mut cpu = CpuBatchDecoder::new(options);
    let expected = cpu
        .decode(inputs.clone())
        .expect("CPU resident RGBA oracle");
    let CpuBatchSamples::U8(expected) = expected.groups()[0].samples() else {
        panic!("U8 RGBA fixture must use U8 batch storage")
    };

    let mut decoder =
        MetalBatchDecoder::system_default_with_options(options).expect("persistent Metal decoder");
    let prepared = decoder
        .prepare(inputs)
        .expect("prepare resident RGBA group");
    let result = decoder
        .decode_prepared(&prepared)
        .expect("decode resident RGBA group");
    assert!(result.errors().is_empty());
    assert!(result.group_errors().is_empty());
    assert_eq!(result.groups().len(), 1);
    let group = &result.groups()[0];
    assert_eq!(group.info().color, BatchColor::Rgba);
    assert_eq!(group.info().layout, BatchLayout::Nhwc);
    assert_eq!(group.surfaces().len(), 2);
    let image_bytes = expected.len() / group.surfaces().len();
    for (index, surface) in group.surfaces().iter().enumerate() {
        assert_eq!(surface.pixel_format(), PixelFormat::Rgba8);
        assert_eq!(
            surface.as_bytes().expect("resident RGBA bytes").as_ref(),
            &expected[index * image_bytes..(index + 1) * image_bytes]
        );
    }
    let first_range = group.surfaces()[0]
        .memory_range()
        .expect("first resident RGBA range");
    let second_range = group.surfaces()[1]
        .memory_range()
        .expect("second resident RGBA range");
    assert_eq!(first_range.allocation, second_range.allocation);
    assert_eq!(first_range.offset, 0);
    assert_eq!(second_range.offset, image_bytes);
    assert_eq!(first_range.len, image_bytes);
    assert_eq!(second_range.len, image_bytes);
}

#[test]
fn prepared_htj2k_rgba_nchw_resident_group_is_exact_without_surface_mislabeling() {
    if !should_run_metal_runtime() {
        return;
    }

    let options = BatchDecodeOptions {
        layout: BatchLayout::Nchw,
        ..BatchDecodeOptions::default()
    };
    let fixture =
        generated_htj2k_rgba_fixture(Htj2kRgbaSampleProfile::U8Rct, Htj2kRgbaAlpha::Straight);
    let encoded = Arc::<[u8]>::from(wrap_rgba_jph(&fixture.encoded, fixture.alpha));
    let inputs = vec![
        EncodedImage::full(encoded.clone()),
        EncodedImage::full(encoded),
    ];
    let mut cpu = CpuBatchDecoder::new(options);
    let expected = cpu
        .decode(inputs.clone())
        .expect("CPU NCHW resident RGBA oracle");

    let mut decoder =
        MetalBatchDecoder::system_default_with_options(options).expect("persistent Metal decoder");
    let prepared = decoder
        .prepare(inputs)
        .expect("prepare NCHW resident RGBA group");
    let result = decoder
        .decode_prepared(&prepared)
        .expect("decode NCHW resident RGBA group");
    let (mut groups, errors, group_errors) = result.into_parts();
    assert!(errors.is_empty());
    assert!(group_errors.is_empty());
    assert_eq!(groups.len(), 1);
    let group = groups.pop().expect("one NCHW RGBA group");
    let resident = group
        .resident_batch()
        .expect("completed RGBA group has resident Metal storage")
        .clone();
    let (info, source_indices, decoded_rects, warnings, surfaces) = group.into_parts();
    assert_eq!(info.color, BatchColor::Rgba);
    assert_eq!(info.layout, BatchLayout::Nchw);
    assert_eq!(source_indices, [0, 1]);
    assert_eq!(decoded_rects.len(), 2);
    assert_eq!(warnings.len(), 2);
    assert!(
        surfaces.is_empty(),
        "planar RGBA bytes must not be mislabeled as interleaved Surface values"
    );
    assert_eq!(resident.image_count(), 2);
    assert_eq!(resident.image_stride_bytes(), resident.byte_len() / 2);
    // SAFETY: the group is visible only after codec completion, and this test
    // performs one readback without submitting a writer or retaining the handle.
    let actual = unsafe {
        j2k_metal_support::checked_buffer_read_vec::<u8>(
            resident.metal_buffer(),
            resident.byte_offset(),
            resident.byte_len(),
        )
        .expect("read consumed dense resident RGBA batch")
    };
    assert_native_samples(&actual, expected.groups()[0].samples());
}
