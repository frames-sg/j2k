// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

fn irreversible_fixture(route: CodingRoute, components: usize) -> Arc<[u8]> {
    const WIDTH: u32 = 33;
    const HEIGHT: u32 = 19;
    let pixels = (0..WIDTH * HEIGHT * u32::try_from(components).expect("components"))
        .map(|index| ((index * 31 + index / 5 + 17) & 0xff) as u8)
        .collect::<Vec<_>>();
    let options = EncodeOptions {
        reversible: false,
        num_decomposition_levels: 2,
        use_mct: false,
        ..EncodeOptions::default()
    };
    Arc::from(match route {
        CodingRoute::Classic => encode(
            &pixels,
            WIDTH,
            HEIGHT,
            u16::try_from(components).expect("components"),
            8,
            false,
            &options,
        )
        .expect("encode classic 9/7 fixture"),
        CodingRoute::Htj2k => encode_htj2k(
            &pixels,
            WIDTH,
            HEIGHT,
            u16::try_from(components).expect("components"),
            8,
            false,
            &options,
        )
        .expect("encode HTJ2K 9/7 fixture"),
    })
}

fn irreversible_gray_with_coc_override() -> Arc<[u8]> {
    let mut encoded = irreversible_fixture(CodingRoute::Classic, 1).to_vec();
    let cod = encoded
        .windows(2)
        .position(|marker| marker == [0xff, 0x52])
        .expect("COD marker");
    let segment_length = usize::from(u16::from_be_bytes([encoded[cod + 2], encoded[cod + 3]]));
    let segment_end = cod + 2 + segment_length;
    let scod = encoded[cod + 4];
    let component_parameters = encoded[cod + 9..segment_end].to_vec();
    assert_eq!(component_parameters.last(), Some(&0), "fixture uses 9/7");

    encoded[cod + 13] = 1;
    let coc_length = 2 + 1 + 1 + component_parameters.len();
    let mut coc = Vec::with_capacity(2 + coc_length);
    coc.extend_from_slice(&[0xff, 0x53]);
    coc.extend_from_slice(
        &u16::try_from(coc_length)
            .expect("small COC segment")
            .to_be_bytes(),
    );
    coc.push(0);
    coc.push(scod);
    coc.extend_from_slice(&component_parameters);
    encoded.splice(segment_end..segment_end, coc);
    Arc::from(encoded)
}

fn ht_gray_with_coc_block_coding_override() -> Arc<[u8]> {
    let mut encoded = irreversible_fixture(CodingRoute::Htj2k, 1).to_vec();
    let cod = encoded
        .windows(2)
        .position(|marker| marker == [0xff, 0x52])
        .expect("COD marker");
    let segment_length = usize::from(u16::from_be_bytes([encoded[cod + 2], encoded[cod + 3]]));
    let segment_end = cod + 2 + segment_length;
    let scod = encoded[cod + 4];
    let component_parameters = encoded[cod + 9..segment_end].to_vec();
    assert_ne!(
        component_parameters[3] & 0x40,
        0,
        "fixture uses HT block coding"
    );

    encoded[cod + 12] &= !0x40;
    let coc_length = 2 + 1 + 1 + component_parameters.len();
    let mut coc = Vec::with_capacity(2 + coc_length);
    coc.extend_from_slice(&[0xff, 0x53]);
    coc.extend_from_slice(
        &u16::try_from(coc_length)
            .expect("small COC segment")
            .to_be_bytes(),
    );
    coc.push(0);
    coc.push(scod);
    coc.extend_from_slice(&component_parameters);
    encoded.splice(segment_end..segment_end, coc);
    Arc::from(encoded)
}

fn irreversible_gray_with_tile_cod_override() -> Arc<[u8]> {
    let mut encoded = irreversible_fixture(CodingRoute::Classic, 1).to_vec();
    let cod = encoded
        .windows(2)
        .position(|marker| marker == [0xff, 0x52])
        .expect("COD marker");
    let cod_length = usize::from(u16::from_be_bytes([encoded[cod + 2], encoded[cod + 3]]));
    let cod_end = cod + 2 + cod_length;
    let tile_cod = encoded[cod..cod_end].to_vec();
    assert_eq!(tile_cod.last(), Some(&0), "fixture uses 9/7");
    encoded[cod + 13] = 1;

    let sot = encoded
        .windows(2)
        .position(|marker| marker == [0xff, 0x90])
        .expect("SOT marker");
    let psot = u32::from_be_bytes([
        encoded[sot + 6],
        encoded[sot + 7],
        encoded[sot + 8],
        encoded[sot + 9],
    ]);
    let updated_psot = psot
        .checked_add(u32::try_from(tile_cod.len()).expect("small tile COD"))
        .expect("tile-part length");
    encoded[sot + 6..sot + 10].copy_from_slice(&updated_psot.to_be_bytes());
    let tile_header_start = sot + 12;
    assert_eq!(
        &encoded[tile_header_start..tile_header_start + 2],
        &[0xff, 0x93],
        "generated fixture has an empty tile header"
    );
    encoded.splice(tile_header_start..tile_header_start, tile_cod);
    Arc::from(encoded)
}

#[test]
fn batch_metadata_uses_component_coding_overrides_from_the_execution_plan() {
    let prepared = prepare_batch(
        vec![EncodedImage::full(irreversible_gray_with_coc_override())],
        BatchDecodeOptions::default(),
    )
    .expect("prepare COC override fixture");

    assert!(prepared.errors().is_empty(), "{:?}", prepared.errors());
    assert_eq!(prepared.groups().len(), 1);
    let info = prepared.groups()[0].info();
    assert_eq!(info.route, BatchCodecRoute::Classic);
    assert_eq!(info.transform, BatchWaveletTransform::Irreversible97);
    assert_eq!(
        info.transfer_syntax,
        CompressedTransferSyntax::Jpeg2000Lossy
    );
}

#[test]
fn batch_route_uses_component_block_coding_overrides_from_the_execution_plan() {
    let prepared = prepare_batch(
        vec![EncodedImage::full(ht_gray_with_coc_block_coding_override())],
        BatchDecodeOptions::default(),
    )
    .expect("prepare HT COC override fixture");

    assert!(prepared.errors().is_empty(), "{:?}", prepared.errors());
    assert_eq!(prepared.groups().len(), 1);
    let info = prepared.groups()[0].info();
    assert_eq!(info.route, BatchCodecRoute::Htj2k);
    assert_eq!(info.transform, BatchWaveletTransform::Irreversible97);
    assert_eq!(
        info.transfer_syntax,
        CompressedTransferSyntax::HtJpeg2000Lossy
    );
}

#[test]
fn batch_metadata_uses_tile_coding_overrides_from_the_execution_plan() {
    let prepared = prepare_batch(
        vec![EncodedImage::full(
            irreversible_gray_with_tile_cod_override(),
        )],
        BatchDecodeOptions::default(),
    )
    .expect("prepare tile COD override fixture");

    assert!(prepared.errors().is_empty(), "{:?}", prepared.errors());
    assert_eq!(prepared.groups().len(), 1);
    let info = prepared.groups()[0].info();
    assert_eq!(info.route, BatchCodecRoute::Classic);
    assert_eq!(info.transform, BatchWaveletTransform::Irreversible97);
    assert_eq!(
        info.transfer_syntax,
        CompressedTransferSyntax::Jpeg2000Lossy
    );
}

#[test]
fn irreversible_97_batches_agree_with_scalar_reconstruction_within_one_lsb() {
    const WIDTH: usize = 33;
    const HEIGHT: usize = 19;
    let fixtures = [
        (irreversible_fixture(CodingRoute::Classic, 1), 1),
        (irreversible_fixture(CodingRoute::Classic, 3), 3),
        (irreversible_fixture(CodingRoute::Htj2k, 1), 1),
        (irreversible_fixture(CodingRoute::Htj2k, 3), 3),
    ];
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let mut decoder = CpuBatchDecoder::new(options);
    let result = decoder
        .decode(
            fixtures
                .iter()
                .map(|(bytes, _)| EncodedImage::full(Arc::clone(bytes)))
                .collect(),
        )
        .expect("decode irreversible batch");

    assert!(
        result.errors().is_empty(),
        "9/7 errors: {:?}",
        result.errors()
    );
    for (source_index, (encoded, components)) in fixtures.iter().enumerate() {
        let group = result
            .groups()
            .iter()
            .find(|group| group.source_indices() == [source_index])
            .expect("9/7 output group");
        assert_eq!(
            group.info().transform,
            BatchWaveletTransform::Irreversible97
        );
        let CpuBatchSamples::U8(batch) = group.samples() else {
            panic!("8-bit 9/7 group must retain u8 samples")
        };
        let mut scalar = J2kDecoder::new(encoded).expect("scalar 9/7 decoder");
        let mut oracle = vec![0_u8; WIDTH * HEIGHT * components];
        scalar
            .decode_into(
                &mut oracle,
                WIDTH * components,
                if *components == 1 {
                    PixelFormat::Gray8
                } else {
                    PixelFormat::Rgb8
                },
            )
            .expect("scalar 9/7 oracle");
        assert_eq!(batch.len(), oracle.len());
        assert!(
            batch
                .iter()
                .zip(&oracle)
                .all(|(batch, oracle)| batch.abs_diff(*oracle) <= 1),
            "source {source_index}: batch reconstruction differs by more than one LSB"
        );
    }
}
