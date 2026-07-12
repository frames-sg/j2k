// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::jp2::cdef::ChannelDefinitionBox;
use crate::jp2::colr::{CieLab, ColorSpace as NativeColorSpace};
use crate::jp2::pclr::{PaletteBox, PaletteColumn};
use crate::{encode, EncodeOptions};
use alloc::vec;

fn gray_fixture() -> (Vec<u8>, Vec<u8>) {
    let samples = (0_u8..64).map(|value| value * 3).collect::<Vec<_>>();
    let encoded = encode(
        &samples,
        8,
        8,
        1,
        8,
        false,
        &EncodeOptions {
            num_decomposition_levels: 0,
            reversible: true,
            ..EncodeOptions::default()
        },
    )
    .expect("reversible grayscale fixture encodes");
    (samples, encoded)
}

fn expected_crop(samples: &[u8], roi: (u32, u32, u32, u32)) -> Vec<u8> {
    let (x, y, width, height) = roi;
    let mut cropped = Vec::new();
    for row in y as usize..(y + height) as usize {
        let start = row * 8 + x as usize;
        cropped.extend_from_slice(&samples[start..start + width as usize]);
    }
    cropped
}

#[derive(Default)]
struct DecliningHtDecoder;

impl HtCodeBlockDecoder for DecliningHtDecoder {}

#[test]
fn image_output_entrypoints_preserve_pixels_regions_and_metadata() {
    let (samples, encoded) = gray_fixture();
    let settings = DecodeSettings::strict();
    let image = Image::new(&encoded, &settings).expect("fixture parses");

    assert_eq!((image.width(), image.height()), (8, 8));
    assert_eq!(image.original_bit_depth(), 8);
    assert!(!image.has_alpha());
    assert!(matches!(image.color_space(), ColorSpace::Gray));
    assert!(image.supports_direct_device_plane_reuse());
    assert_eq!(image.decode().expect("owned byte decode"), samples);

    let mut context = DecoderContext::default();
    let bitmap = image
        .decode_with_context(&mut context)
        .expect("context byte decode");
    assert_eq!(bitmap.data, samples);
    assert_eq!((bitmap.width, bitmap.height), (8, 8));
    assert_eq!(bitmap.original_bit_depth, 8);
    assert!(!bitmap.has_alpha);
    assert!(matches!(bitmap.color_space, ColorSpace::Gray));

    let mut into = vec![0; samples.len()];
    image
        .decode_into(&mut into, &mut context)
        .expect("caller-owned output decode");
    assert_eq!(into, samples);

    let roi = (2, 3, 3, 2);
    let expected = expected_crop(&samples, roi);
    let region = image.decode_region(roi).expect("owned region decode");
    assert_eq!(region.data, expected);
    assert_eq!((region.width, region.height), (3, 2));
    assert_eq!(region.original_bit_depth, 8);

    let mut region_context = DecoderContext::default();
    let region = image
        .decode_region_with_context(roi, &mut region_context)
        .expect("context region decode");
    assert_eq!(region.data, expected);

    let mut component_context = DecoderContext::default();
    {
        let components = image
            .decode_components_with_context(&mut component_context)
            .expect("borrowed component decode");
        assert_eq!(components.dimensions(), (8, 8));
        assert_eq!(components.planes().len(), 1);
        assert_eq!(components.planes()[0].dimensions(), (8, 8));
        assert_eq!(components.planes()[0].bit_depth(), 8);
        assert_eq!(components.planes()[0].sampling(), (1, 1));
    }

    let mut region_component_context = DecoderContext::default();
    let components = image
        .decode_region_components_with_context(roi, &mut region_component_context)
        .expect("borrowed region component decode");
    assert_eq!(components.dimensions(), (3, 2));
    assert_eq!(components.planes()[0].dimensions(), (3, 2));
    let expected_samples = expected.iter().copied().map(f32::from).collect::<Vec<_>>();
    assert_eq!(components.planes()[0].samples(), expected_samples);

    let mut ht_context = DecoderContext::default();
    let mut ht_decoder = DecliningHtDecoder;
    let ht_components = image
        .decode_region_components_with_ht_decoder(&mut ht_context, roi, &mut ht_decoder)
        .expect("declined HT hook uses scalar region decode");
    assert_eq!(ht_components.planes()[0].samples(), expected_samples);
}

#[test]
fn direct_device_plane_reuse_rejects_each_host_postprocess_requirement() {
    let (_, encoded) = gray_fixture();
    let mut image = Image::new(&encoded, &DecodeSettings::strict()).expect("fixture parses");
    assert!(image.supports_direct_device_plane_reuse());

    image.boxes.channel_definition = Some(ChannelDefinitionBox {
        channel_definitions: Vec::new(),
    });
    assert!(!image.supports_direct_device_plane_reuse());
    image.boxes.channel_definition = None;

    image.boxes.palette = Some(PaletteBox {
        entries: vec![vec![0]],
        columns: vec![PaletteColumn {
            bit_depth: 8,
            signed: false,
        }],
    });
    assert!(!image.supports_direct_device_plane_reuse());
    image.settings.resolve_palette_indices = false;
    assert!(image.supports_direct_device_plane_reuse());
    image.boxes.palette = None;

    image.boxes.color_specifications[0].color_space =
        NativeColorSpace::Enumerated(EnumeratedColorspace::Sycc);
    assert!(!image.supports_direct_device_plane_reuse());
    image.boxes.color_specifications[0].color_space =
        NativeColorSpace::Enumerated(EnumeratedColorspace::CieLab(CieLab {
            rl: None,
            ol: None,
            ra: None,
            oa: None,
            rb: None,
            ob: None,
        }));
    assert!(!image.supports_direct_device_plane_reuse());
    image.boxes.color_specifications[0].color_space =
        NativeColorSpace::Enumerated(EnumeratedColorspace::Greyscale);
    assert!(image.supports_direct_device_plane_reuse());
}

#[test]
fn retained_baseline_zero_and_nonzero_paths_preserve_parse_metadata() {
    let (_, encoded) = gray_fixture();
    let settings = DecodeSettings::strict();
    let ordinary = Image::new_with_retained_baseline(&encoded, &settings, 0)
        .expect("zero retained baseline delegates to ordinary parse");
    let retained = Image::new_with_retained_baseline(&encoded, &settings, 1)
        .expect("small retained baseline parses");

    assert_eq!(ordinary.width(), retained.width());
    assert_eq!(ordinary.height(), retained.height());
    assert_eq!(ordinary.original_bit_depth(), retained.original_bit_depth());
    assert_eq!(ordinary.has_alpha(), retained.has_alpha());
    assert_eq!(
        ordinary.color_space().num_channels(),
        retained.color_space().num_channels()
    );
}
