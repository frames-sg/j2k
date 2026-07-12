// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::j2c::encode::NativeEncodeRetainedInput;
use crate::{DecodeSettings, Image};

fn high_bit_plane() -> Vec<u8> {
    let mut data = Vec::new();
    data.try_reserve_exact(4 * 4 * 4)
        .expect("small typed high-bit fixture");
    for index in 0..16_u32 {
        let sample = index * 1_048_583 + 19;
        data.extend_from_slice(&sample.to_le_bytes());
    }
    data
}

#[test]
fn high_bit_multitile_ppt_round_trips_without_legacy_writer_or_split_clones() {
    let data = high_bit_plane();
    let planes = [EncodeTypedComponentPlane {
        data: &data,
        x_rsiz: 1,
        y_rsiz: 1,
        bit_depth: 25,
        signed: false,
    }];
    let options = EncodeOptions {
        num_decomposition_levels: 1,
        reversible: true,
        tile_size: Some((2, 2)),
        tile_part_packet_limit: Some(1),
        write_plt: true,
        write_ppt: true,
        ..EncodeOptions::default()
    };
    let session = NativeEncodeSession::try_with_cap(
        NativeEncodeRetainedInput::none(),
        crate::DEFAULT_MAX_CODEC_BYTES,
    )
    .expect("typed high-bit test session");
    let codestream = encode_typed_component_planes_53_i64(&planes, 4, 4, &options, &session)
        .expect("typed high-bit multi-tile encode");
    assert!(codestream.windows(2).any(|marker| marker == [0xFF, 0x58]));
    assert!(codestream.windows(2).any(|marker| marker == [0xFF, 0x61]));

    let decoded = Image::new(&codestream, &DecodeSettings::default())
        .expect("typed high-bit codestream parse")
        .decode_native()
        .expect("typed high-bit multi-tile decode");
    assert_eq!((decoded.width, decoded.height), (4, 4));
    assert_eq!(decoded.bit_depth, 25);
    assert_eq!(decoded.data, data);
}
