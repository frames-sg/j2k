// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    encode, DecodeSettings, DecodedComponents, DecoderContext, EncodeOptions,
    EncodeProgressionOrder, Image,
};

fn assert_planes_equal(a: &DecodedComponents<'_>, b: &DecodedComponents<'_>) {
    assert_eq!(a.dimensions, b.dimensions);
    assert_eq!(a.planes.len(), b.planes.len());
    for (a, b) in a.planes.iter().zip(&b.planes) {
        assert_eq!(a.dimensions, b.dimensions);
        assert_eq!(
            (a.bit_depth, a.signed, a.sampling),
            (b.bit_depth, b.signed, b.sampling)
        );
        assert!(a
            .samples
            .iter()
            .map(|v| v.to_bits())
            .eq(b.samples.iter().map(|v| v.to_bits())));
    }
}

fn options(reversible: bool, block: (u8, u8), markers: bool) -> EncodeOptions {
    EncodeOptions {
        reversible,
        progression_order: EncodeProgressionOrder::Rpcl,
        num_layers: 3,
        num_decomposition_levels: 3,
        code_block_width_exp: block.0,
        code_block_height_exp: block.1,
        precinct_exponents: vec![((block.0 + 3).max(5), (block.1 + 3).max(5)); 4],
        write_sop: markers,
        write_eph: markers,
        ..EncodeOptions::default()
    }
}

fn fixture(options: &EncodeOptions) -> Vec<u8> {
    let pixels: Vec<u8> = (0..257 * 193 * 3)
        .map(|i| ((i * 71 + i / 37) & 255) as u8)
        .collect();
    encode(&pixels, 257, 193, 3, 8, false, options).unwrap()
}

fn images(bytes: &[u8], reduction: u8) -> (Image<'_>, Image<'_>) {
    let mut baseline =
        Image::new_with_reduction(bytes, &DecodeSettings::default(), reduction).unwrap();
    baseline.packet_cache = crate::image::packet_directory::PacketDirectoryCache::disabled();
    let image = Image::new_with_reduction(bytes, &DecodeSettings::default(), reduction).unwrap();
    (baseline, image)
}

#[test]
fn packet_directory_accelerates_existing_calls_with_fresh_contexts() {
    let bytes = fixture(&options(false, (0, 0), true));
    let (baseline, image) = images(&bytes, 0);
    let roi = (220, 160, 16, 16);
    let metadata = image.retained_allocation_bytes().unwrap();
    for call in 0..3 {
        let mut a = DecoderContext::default();
        let mut b = DecoderContext::default();
        let expected = baseline
            .decode_region_components_for_integer_output_with_context(roi, &mut b)
            .unwrap();
        let actual = image
            .decode_region_components_for_integer_output_with_context(roi, &mut a)
            .unwrap();
        assert_planes_equal(&actual, &expected);
        if call == 0 {
            assert_eq!(image.retained_allocation_bytes().unwrap(), metadata);
            assert_eq!(a.tile_decode_context.debug_counters.directory_skips, 0);
        } else {
            assert!(image.retained_allocation_bytes().unwrap() > metadata);
            assert!(a.tile_decode_context.debug_counters.directory_skips > 0);
        }
        assert_eq!(
            a.tile_decode_context.debug_counters.decoded_code_blocks,
            b.tile_decode_context.debug_counters.decoded_code_blocks
        );
    }
}

#[test]
fn packet_directory_preserves_planes_across_transforms_reductions_and_blocks() {
    for (reversible, block, markers, ht) in [
        (true, (0, 0), false, false),
        (false, (3, 1), true, false),
        (false, (1, 4), false, false),
        (true, (2, 2), true, true),
    ] {
        let mut opt = options(reversible, block, markers);
        opt.use_ht_block_coding = ht;
        let bytes = fixture(&opt);
        for reduction in [0, 1, 3] {
            let (baseline, image) = images(&bytes, reduction);
            let (w, h) = (image.width(), image.height());
            for roi in [
                (w / 2, h / 2, (w / 4).max(1), (h / 4).max(1)),
                (w - 1, h - 1, 1, 1),
                (0, 0, w, h),
            ] {
                let mut a = DecoderContext::default();
                let mut b = DecoderContext::default();
                let expected = baseline
                    .decode_region_components_for_integer_output_with_context(roi, &mut b)
                    .unwrap();
                let actual = image
                    .decode_region_components_for_integer_output_with_context(roi, &mut a)
                    .unwrap();
                assert_planes_equal(&actual, &expected);
                let ac = a.tile_decode_context.debug_counters;
                let bc = b.tile_decode_context.debug_counters;
                assert_eq!(ac.decoded_code_blocks, bc.decoded_code_blocks);
                assert_eq!(ac.skipped_code_blocks, bc.skipped_code_blocks);
                assert_eq!(ac.built_layer_records, bc.built_layer_records);
                assert_eq!(
                    image.decode_region(roi).unwrap().data,
                    baseline.decode_region(roi).unwrap().data
                );
            }
        }
    }
}

#[test]
fn packet_directory_shares_existing_image_across_concurrent_contexts() {
    let bytes = fixture(&options(false, (0, 0), true));
    let (baseline, image) = images(&bytes, 0);
    let roi = (220, 160, 16, 16);
    for _ in 0..2 {
        image
            .decode_region_components_with_context(roi, &mut DecoderContext::default())
            .unwrap();
    }
    let retained = image.retained_allocation_bytes().unwrap();
    let check = || {
        let mut a = DecoderContext::default();
        let mut b = DecoderContext::default();
        let expected = baseline
            .decode_region_components_with_context(roi, &mut b)
            .unwrap();
        let actual = image
            .decode_region_components_with_context(roi, &mut a)
            .unwrap();
        assert_planes_equal(&actual, &expected);
        let ac = a.tile_decode_context.debug_counters;
        let bc = b.tile_decode_context.debug_counters;
        assert!(ac.directory_skips > 0);
        assert_eq!(ac.packet_headers + ac.directory_skips, bc.packet_headers);
    };
    std::thread::scope(|scope| {
        let tasks: Vec<_> = (0..4).map(|_| scope.spawn(&check)).collect();
        for task in tasks {
            task.join().unwrap();
        }
    });
    assert_eq!(image.retained_allocation_bytes().unwrap(), retained);
}

#[test]
fn packet_directory_falls_back_for_unsupported_layouts_and_small_caps() {
    for case in 0..8 {
        let mut opt = options(true, (2, 2), false);
        match case {
            0 => opt.write_plt = true,
            1 => opt.write_ppm = true,
            2 => opt.write_ppt = true,
            3 => opt.tile_part_packet_limit = Some(7),
            4 => opt.progression_order = EncodeProgressionOrder::Lrcp,
            5 => opt.tile_size = Some((100, 80)),
            _ => (),
        }
        let mut bytes = fixture(&opt);
        if case == 7 {
            let sot = bytes
                .windows(2)
                .position(|pair| pair == [0xff, 0x90])
                .unwrap();
            bytes.splice(sot..sot, [0xff, 0x5f, 0, 9, 0, 0, 0, 3, 4, 3, 2]);
        }
        let (baseline, mut image) = images(&bytes, 0);
        if case == 6 {
            image.packet_cache =
                crate::image::packet_directory::PacketDirectoryCache::with_limit(8);
        }
        let retained = image.retained_allocation_bytes().unwrap();
        for _ in 0..3 {
            let mut a = DecoderContext::default();
            let mut b = DecoderContext::default();
            let expected = baseline
                .decode_region_components_with_context((150, 120, 12, 17), &mut b)
                .unwrap();
            let actual = image
                .decode_region_components_with_context((150, 120, 12, 17), &mut a)
                .unwrap();
            assert_planes_equal(&actual, &expected);
            assert_eq!(a.tile_decode_context.debug_counters.directory_skips, 0);
        }
        assert_eq!(image.retained_allocation_bytes().unwrap(), retained);
    }
}

#[test]
fn packet_directory_preserves_errors_early_stop_and_context_reuse() {
    let valid = fixture(&options(false, (0, 0), true));
    let (baseline, _) = images(&valid, 0);
    let mut context = DecoderContext::default();
    baseline
        .decode_region_components_with_context((0, 0, 257, 193), &mut context)
        .unwrap();
    let last = context.tile_decode_context.debug_counters.packet_headers - 1;
    for at_packet in [0, last] {
        let mut bytes = valid.clone();
        let positions: Vec<_> = bytes
            .windows(2)
            .enumerate()
            .filter_map(|(i, pair)| (pair == [0xff, 0x92]).then_some(i))
            .collect();
        bytes[positions[at_packet] + 1] = 0x93;
        for settings in [DecodeSettings::strict(), DecodeSettings::lenient()] {
            let mut baseline = Image::new(&bytes, &settings).unwrap();
            baseline.packet_cache =
                crate::image::packet_directory::PacketDirectoryCache::disabled();
            let image = Image::new(&bytes, &settings).unwrap();
            let mut a = DecoderContext::default();
            let mut b = DecoderContext::default();
            for _ in 0..3 {
                assert_eq!(
                    image
                        .decode_region_components_with_context((0, 0, 257, 193), &mut a)
                        .err()
                        .unwrap(),
                    baseline
                        .decode_region_components_with_context((0, 0, 257, 193), &mut b)
                        .err()
                        .unwrap()
                );
                if at_packet == last {
                    let expected = baseline
                        .decode_region_components_with_context((0, 0, 1, 1), &mut b)
                        .unwrap();
                    let actual = image
                        .decode_region_components_with_context((0, 0, 1, 1), &mut a)
                        .unwrap();
                    assert_planes_equal(&actual, &expected);
                }
            }
        }
    }
}

#[test]
fn packet_directory_matches_independent_openjpeg_at_a_nonzero_origin() {
    let bytes = include_bytes!("../../fixtures/packet_directory/openjpeg_rgb_origin5x3_rpcl.j2k");
    for reduction in [0, 1, 3] {
        let (baseline, image) = images(bytes, reduction);
        let (w, h) = (image.width(), image.height());
        for roi in [
            (w / 2, h / 3, w / 4, h / 3),
            (w - 1, h - 1, 1, 1),
            (0, 0, w, h),
        ] {
            let mut a = DecoderContext::default();
            let mut b = DecoderContext::default();
            let expected = baseline
                .decode_region_components_for_integer_output_with_context(roi, &mut b)
                .unwrap();
            let actual = image
                .decode_region_components_for_integer_output_with_context(roi, &mut a)
                .unwrap();
            assert_planes_equal(&actual, &expected);
        }
    }
}

#[test]
fn packet_directory_preserves_truncated_body_and_tile_errors() {
    let pixels: Vec<_> = (0..1024).map(|i| ((i * 31 + i / 7) & 255) as u8).collect();
    let mut bytes = encode(
        &pixels,
        32,
        32,
        1,
        8,
        false,
        &EncodeOptions {
            num_decomposition_levels: 0,
            progression_order: EncodeProgressionOrder::Rpcl,
            ..EncodeOptions::default()
        },
    )
    .unwrap();
    let sot = bytes
        .windows(2)
        .position(|pair| pair == [0xff, 0x90])
        .unwrap();
    bytes.truncate(bytes.len() - 10);
    let part_length = u32::try_from(bytes.len() - sot).unwrap();
    bytes[sot + 6..sot + 10].copy_from_slice(&part_length.to_be_bytes());
    bytes.extend_from_slice(&[0xff, 0xd9]);
    let (baseline, image) = images(&bytes, 0);
    let mut a = DecoderContext::default();
    let mut b = DecoderContext::default();
    for _ in 0..3 {
        assert_eq!(
            image
                .decode_region_components_with_context((0, 0, 32, 32), &mut a)
                .err()
                .unwrap(),
            baseline
                .decode_region_components_with_context((0, 0, 32, 32), &mut b)
                .err()
                .unwrap()
        );
    }
    let (baseline, image) = images(&bytes[..bytes.len() - 2], 0);
    for _ in 0..3 {
        assert_eq!(
            image
                .decode_region_components_with_context((0, 0, 32, 32), &mut a)
                .err()
                .unwrap(),
            baseline
                .decode_region_components_with_context((0, 0, 32, 32), &mut b)
                .err()
                .unwrap()
        );
    }
}
