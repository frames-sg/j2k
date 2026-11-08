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

#[test]
fn common_option_errors_precede_typed_i64_capability_checks() {
    let data = high_bit_plane();
    let planes = [EncodeTypedComponentPlane {
        data: &data,
        x_rsiz: 1,
        y_rsiz: 1,
        bit_depth: 25,
        signed: false,
    }];
    let options = EncodeOptions {
        num_layers: 0,
        irreversible_quantization_profile: Some(Vec::new()),
        ..EncodeOptions::default()
    };
    let expected = crate::EncodeError::InvalidInput {
        what: "quality layer count must be non-zero",
    };
    assert_eq!(
        crate::encode(&data, 4, 4, 1, 25, false, &options),
        Err(expected)
    );
    assert_eq!(
        crate::encode_typed_component_planes_53(&planes, 4, 4, &options),
        Err(crate::EncodeError::InvalidInput {
            what: "quality layer count must be non-zero",
        }),
    );
}

#[test]
fn prepared_packet_options_preserve_normal_and_high_bit_routes() {
    use crate::{EncodeProgressionOrder, J2kEncodeStageAccelerator, J2kPacketizationEncodeJob};

    #[derive(Default)]
    struct PacketizationProbe(usize);
    impl J2kEncodeStageAccelerator for PacketizationProbe {
        fn encode_packetization(
            &mut self,
            _job: J2kPacketizationEncodeJob<'_>,
        ) -> crate::J2kEncodeStageResult<Option<Vec<u8>>> {
            self.0 += 1;
            Ok(None)
        }
    }

    let normal = (0_u8..16)
        .map(|value| value.wrapping_mul(17))
        .collect::<Vec<_>>();
    let high_bit = high_bit_plane();
    let planes = [EncodeTypedComponentPlane {
        data: &high_bit,
        x_rsiz: 1,
        y_rsiz: 1,
        bit_depth: 25,
        signed: false,
    }];
    for ht in [false, true] {
        for (order, layers, sop_eph, ppm, ppt, tile_parts) in [
            (EncodeProgressionOrder::Lrcp, 1, false, false, false, None),
            (EncodeProgressionOrder::Rlcp, 3, true, false, false, None),
            (EncodeProgressionOrder::Pcrl, 1, false, false, true, Some(1)),
            (EncodeProgressionOrder::Cprl, 3, false, true, false, Some(2)),
        ] {
            let options = EncodeOptions {
                num_decomposition_levels: 1,
                use_ht_block_coding: ht,
                progression_order: order,
                num_layers: layers,
                write_sop: sop_eph,
                write_eph: sop_eph,
                write_ppm: ppm,
                write_ppt: ppt,
                tile_part_packet_limit: tile_parts,
                ..EncodeOptions::default()
            };
            for (samples, precision) in [(&normal, 8), (&high_bit, 25)] {
                let expected = crate::encode(samples, 4, 4, 1, precision, false, &options).unwrap();
                let mut accelerator = PacketizationProbe::default();
                let actual = crate::encode_with_accelerator(
                    samples,
                    4,
                    4,
                    1,
                    precision,
                    false,
                    &options,
                    &mut accelerator,
                )
                .unwrap();
                assert_eq!(
                    actual, expected,
                    "{order:?}, {layers} layers, HT={ht}, {precision} bits"
                );
                assert_eq!(
                    accelerator.0,
                    usize::from(layers == 1 && !sop_eph && !ppm && !ppt && tile_parts.is_none())
                );
                let decoded = Image::new(&actual, &DecodeSettings::strict())
                    .unwrap()
                    .decode_native()
                    .unwrap();
                assert_eq!(decoded.data, *samples);
            }
            let typed = crate::encode_typed_component_planes_53(&planes, 4, 4, &options).unwrap();
            let decoded = Image::new(&typed, &DecodeSettings::strict())
                .unwrap()
                .decode_native()
                .unwrap();
            assert_eq!(decoded.data, high_bit);
            for (enabled, marker) in [(sop_eph, 0x91), (sop_eph, 0x92), (ppm, 0x60), (ppt, 0x61)] {
                if enabled {
                    assert!(typed.windows(2).any(|bytes| bytes == [0xff, marker]));
                }
            }
        }
    }
}
