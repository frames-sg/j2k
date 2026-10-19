// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_native::{encode, inspect_j2k_codestream_header, DecodeSettings, EncodeOptions, Image};

fn sample(x: u32, y: u32) -> u8 {
    u8::try_from((x * 13 + y * 29 + (x / 127) * 7) % 251).expect("sample below 251")
}

fn source(width: u32, height: u32) -> Vec<u8> {
    (0..height)
        .flat_map(|y| (0..width).map(move |x| sample(x, y)))
        .collect()
}

#[test]
fn openjpeg_long_images_decode_whole_and_far_edge_regions() {
    for (bytes, width, height) in [
        (
            include_bytes!("../fixtures/large_geometry/openjpeg_65537x4.j2k").as_slice(),
            65_537_u32,
            4_u32,
        ),
        (
            include_bytes!("../fixtures/large_geometry/openjpeg_4x65537.j2k").as_slice(),
            4_u32,
            65_537_u32,
        ),
    ] {
        let header = inspect_j2k_codestream_header(bytes).expect("inspect long image");
        assert_eq!(header.dimensions, (width, height));
        let image = Image::new(bytes, &DecodeSettings::strict()).expect("open long image");
        let decoded = image
            .decode_native()
            .expect("decode long image within budget");
        assert_eq!(decoded.data, source(width, height));

        let (roi_width, roi_height) = (width.min(17), height.min(13));
        let (x0, y0) = (width - roi_width, height - roi_height);
        let region = image
            .decode_native_region((x0, y0, roi_width, roi_height))
            .expect("decode region at far edge");
        let expected: Vec<_> = (y0..height)
            .flat_map(|y| (x0..width).map(move |x| sample(x, y)))
            .collect();
        assert_eq!((region.width, region.height), (roi_width, roi_height));
        assert_eq!(region.data, expected);

        let reduced = Image::new(
            bytes,
            &DecodeSettings {
                target_resolution: Some((width.div_ceil(2), height.div_ceil(2))),
                ..DecodeSettings::strict()
            },
        )
        .expect("open reduced long image");
        let whole = reduced.decode_native().expect("decode reduced long image");
        assert_eq!(
            (whole.width, whole.height),
            (width.div_ceil(2), height.div_ceil(2))
        );
        let last = reduced
            .decode_native_region((whole.width - 1, whole.height - 1, 1, 1))
            .expect("decode reduced far corner");
        assert_eq!(last.data, whole.data[whole.data.len() - 1..]);
    }
}

fn translate_siz(bytes: &mut [u8], origin: u32, tile_extent: u32) {
    let siz = bytes
        .windows(2)
        .position(|marker| marker == [0xFF, 0x51])
        .unwrap();
    for offset in [6, 10] {
        bytes[siz + offset..siz + offset + 4].copy_from_slice(&u32::MAX.to_be_bytes());
    }
    for offset in [14, 18, 30, 34] {
        bytes[siz + offset..siz + offset + 4].copy_from_slice(&origin.to_be_bytes());
    }
    for offset in [22, 26] {
        bytes[siz + offset..siz + offset + 4].copy_from_slice(&tile_extent.to_be_bytes());
    }
}

#[test]
fn image_at_u32_limit_decodes_clipped_precincts_and_code_blocks() {
    let width = 63;
    let pixels = source(width, width);
    // This translation is a multiple of 64, so both wavelet phase and the
    // single code-block's shape remain unchanged. The enclosing precinct and
    // code-block grids end at 2^32, one beyond the image's exclusive endpoint.
    let origin = u32::MAX - width;
    for levels in [0, 1] {
        for ht in [false, true] {
            let options = EncodeOptions {
                reversible: true,
                num_decomposition_levels: levels,
                use_ht_block_coding: ht,
                ..EncodeOptions::default()
            };
            let mut bytes = encode(&pixels, width, width, 1, 8, false, &options).unwrap();
            translate_siz(&mut bytes, origin, width);
            let header = inspect_j2k_codestream_header(&bytes).expect("inspect high-origin image");
            assert_eq!(header.dimensions, (width, width));
            let image =
                Image::new(&bytes, &DecodeSettings::strict()).expect("open high-origin image");
            let decoded = image.decode_native().expect("decode high-origin image");
            assert_eq!(decoded.data, pixels, "levels={levels}, HT={ht}");
            let corner = image
                .decode_native_region((width - 3, width - 2, 3, 2))
                .expect("decode high-origin far corner");
            let expected: Vec<_> = (width - 2..width)
                .flat_map(|y| (width - 3..width).map(move |x| sample(x, y)))
                .collect();
            assert_eq!(corner.data, expected);
        }
    }
}

#[test]
fn first_tile_may_extend_past_the_reference_grid() {
    let pixels = source(63, 63);
    let options = EncodeOptions {
        num_decomposition_levels: 0,
        ..EncodeOptions::default()
    };
    let mut bytes = encode(&pixels, 63, 63, 1, 8, false, &options).unwrap();
    translate_siz(&mut bytes, u32::MAX - 63, 64);
    let header = inspect_j2k_codestream_header(&bytes).expect("tile extends past image");
    assert_eq!(header.dimensions, (63, 63));
    let decoded = Image::new(&bytes, &DecodeSettings::strict())
        .and_then(|image| image.decode_native())
        .expect("decode clipped first tile");
    assert_eq!(decoded.data, pixels);
}
