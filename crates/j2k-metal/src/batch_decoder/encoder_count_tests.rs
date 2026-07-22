// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k::{
    wrap_j2k_codestream, BatchDecodeOptions, CpuBatchDecoder, CpuBatchSamples, EncodedImage,
    J2kChannelAssociation, J2kChannelDefinition, J2kChannelType, J2kFileBoxMetadata,
    J2kFileColorSpec, J2kFileWrapOptions,
};
use j2k_core::{Colorspace, PixelFormat};
use j2k_metal_support::{MetalImageDestination, MetalImageLayout};
use j2k_native::{encode, encode_htj2k, EncodeOptions};

use super::MetalBatchDecoder;

#[derive(Clone, Copy, Debug)]
enum FixtureColor {
    Gray,
    Rgb,
    Rgba,
}

#[derive(Clone, Copy, Debug)]
enum FixtureRoute {
    Classic,
    Ht,
}

impl FixtureColor {
    fn channels(self) -> usize {
        match self {
            Self::Gray => 1,
            Self::Rgb => 3,
            Self::Rgba => 4,
        }
    }

    fn format(self) -> PixelFormat {
        match self {
            Self::Gray => PixelFormat::Gray8,
            Self::Rgb => PixelFormat::Rgb8,
            Self::Rgba => PixelFormat::Rgba8,
        }
    }
}

fn distinct_j2k(route: FixtureRoute, color: FixtureColor, seed: u8) -> Arc<[u8]> {
    const WIDTH: u32 = 8;
    const HEIGHT: u32 = 8;

    let channels = color.channels();
    let mut pixels = Vec::with_capacity(WIDTH as usize * HEIGHT as usize * channels);
    for y in 0..HEIGHT {
        let y = u8::try_from(y).expect("fixture height fits u8");
        for x in 0..WIDTH {
            let x = u8::try_from(x).expect("fixture width fits u8");
            for channel in 0..channels {
                let channel = u8::try_from(channel).expect("fixture channel count fits u8");
                pixels.push(
                    x.wrapping_mul(17)
                        .wrapping_add(y.wrapping_mul(29))
                        .wrapping_add(channel.wrapping_mul(37))
                        .wrapping_add(seed),
                );
            }
        }
    }
    let channels = u16::try_from(channels).expect("fixture channel count fits u16");
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 2,
        use_mct: matches!(color, FixtureColor::Rgb),
        ..EncodeOptions::default()
    };
    let codestream = match route {
        FixtureRoute::Classic => encode(&pixels, WIDTH, HEIGHT, channels, 8, false, &options),
        FixtureRoute::Ht => encode_htj2k(&pixels, WIDTH, HEIGHT, channels, 8, false, &options),
    }
    .expect("encode structural J2K fixture");
    if !matches!(color, FixtureColor::Rgba) {
        return Arc::from(codestream);
    }

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
            channel_type: J2kChannelType::Opacity,
            association: J2kChannelAssociation::WholeImage,
        },
    ];
    Arc::from(
        wrap_j2k_codestream(
            &codestream,
            match route {
                FixtureRoute::Classic => J2kFileWrapOptions::jp2(),
                FixtureRoute::Ht => J2kFileWrapOptions::jph(),
            }
            .with_color(J2kFileColorSpec::Enumerated(Colorspace::SRgb))
            .with_metadata(J2kFileBoxMetadata {
                palette: None,
                component_mappings: &[],
                channel_definitions: &channel_definitions,
            }),
        )
        .expect("wrap structural HTJ2K RGBA fixture"),
    )
}

fn assert_external_group_uses_one_command_buffer_and_encoder(
    route: FixtureRoute,
    color: FixtureColor,
    count: usize,
) -> Option<(usize, usize)> {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return None;
    }

    let options = BatchDecodeOptions::default();
    let mut decoder =
        MetalBatchDecoder::system_default_with_options(options).expect("persistent Metal decoder");
    let encoded = (0..count)
        .map(|index| {
            distinct_j2k(
                route,
                color,
                u8::try_from(index).expect("structural batch index fits u8"),
            )
        })
        .collect::<Vec<_>>();
    let inputs = encoded
        .iter()
        .cloned()
        .map(EncodedImage::full)
        .collect::<Vec<_>>();
    let mut cpu = CpuBatchDecoder::new(options);
    let expected = cpu
        .decode(inputs.clone())
        .expect("CPU structural group oracle");
    assert!(expected.errors().is_empty(), "{:?}", expected.errors());
    assert_eq!(expected.groups().len(), 1);
    let prepared = decoder.prepare(inputs).expect("prepare structural group");
    assert!(prepared.errors().is_empty(), "{:?}", prepared.errors());
    assert_eq!(prepared.groups().len(), 1);
    let group = &prepared.groups()[0];
    let (width, height) = group.info().dimensions;
    let format = color.format();
    let row_bytes = width as usize * format.bytes_per_pixel();
    let image_bytes = row_bytes * height as usize;
    let output = j2k_metal_support::checked_shared_buffer_for_len::<u8>(
        decoder.backend_session().device(),
        image_bytes * count,
    )
    .expect("structural destination allocation");
    let layout =
        MetalImageLayout::new_batch(0, (width, height), row_bytes, format, count, image_bytes)
            .expect("structural destination layout");
    // SAFETY: the fresh allocation has one logical writer; the retained raw
    // handle is not read until completion releases the destination guard.
    let destination = unsafe {
        MetalImageDestination::from_exclusive_buffer(output.clone(), layout)
            .expect("structural destination")
    };

    crate::compute::reset_metal_command_buffers_for_test();
    crate::compute::reset_metal_compute_encoders_for_test();
    let completion = decoder
        .submit_prepared_group_into(group, destination)
        .expect("submit structural group")
        .wait()
        .expect("complete structural group");
    assert_eq!(completion.decoded_rects().len(), count);

    // SAFETY: group completion released the destination's exclusive write
    // guard, so the retained shared allocation is now host-readable.
    let actual = unsafe {
        j2k_metal_support::checked_buffer_read_vec::<u8>(&output, 0, image_bytes * count)
            .expect("completed structural destination bytes")
    };
    let expected_group = &expected.groups()[0];
    assert_eq!(expected_group.source_indices(), group.source_indices());
    let CpuBatchSamples::U8(expected) = expected_group.samples() else {
        panic!("8-bit structural fixtures must have a U8 CPU oracle")
    };
    assert_eq!(
        actual,
        expected.as_slice(),
        "{route:?} {color:?} batch {count} must retain exact source-order CPU parity"
    );

    Some((
        crate::compute::metal_command_buffers_for_test(),
        crate::compute::metal_compute_encoders_for_test(),
    ))
}

#[test]
fn external_groups_use_one_producer_command_buffer_and_compute_encoder() {
    let mut counts = Vec::new();
    for route in [FixtureRoute::Classic, FixtureRoute::Ht] {
        for count in [1, 8] {
            for color in [FixtureColor::Gray, FixtureColor::Rgb, FixtureColor::Rgba] {
                if let Some(actual) =
                    assert_external_group_uses_one_command_buffer_and_encoder(route, color, count)
                {
                    counts.push(actual);
                }
            }
        }
    }
    if !counts.is_empty() {
        assert_eq!(
            counts,
            vec![(1, 1); 12],
            "classic and HT Gray/RGB/RGBA batch 1 and 8 must each use one producer command buffer and one compute encoder"
        );
    }
}
