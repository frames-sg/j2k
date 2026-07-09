// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::*;

fn codestream_fingerprint(codestream: &[u8]) -> (usize, u64) {
    let hash = codestream
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
        });
    (codestream.len(), hash)
}

#[test]
fn reference_codestreams_remain_byte_exact() {
    let rgb_pixels = (0..8 * 8 * 3)
        .map(|index| u8::try_from((index * 29 + 17) & 0xff).expect("sample fits u8"))
        .collect::<Vec<_>>();
    let classic = encode(
        &rgb_pixels,
        8,
        8,
        3,
        8,
        false,
        &EncodeOptions {
            num_decomposition_levels: 1,
            reversible: true,
            use_mct: true,
            ..EncodeOptions::default()
        },
    )
    .expect("classic reference encode");

    let gray_pixels = (0..8 * 8)
        .map(|index| u8::try_from((index * 37 + 11) & 0xff).expect("sample fits u8"))
        .collect::<Vec<_>>();
    let ht = encode_htj2k(
        &gray_pixels,
        8,
        8,
        1,
        8,
        false,
        &EncodeOptions {
            num_decomposition_levels: 1,
            reversible: false,
            guard_bits: 2,
            ..EncodeOptions::default()
        },
    )
    .expect("HT reference encode");

    let mut high_bit_pixels = Vec::with_capacity(4 * 4 * 4);
    for index in 0..4 * 4_u32 {
        high_bit_pixels.extend_from_slice(&(index * 1_048_583 + 19).to_le_bytes());
    }
    let high_bit = encode(
        &high_bit_pixels,
        4,
        4,
        1,
        25,
        false,
        &EncodeOptions {
            num_decomposition_levels: 1,
            reversible: true,
            ..EncodeOptions::default()
        },
    )
    .expect("high-bit reference encode");

    assert_eq!(
        [
            codestream_fingerprint(&classic),
            codestream_fingerprint(&ht),
            codestream_fingerprint(&high_bit),
        ],
        [
            (296, 12_923_494_458_190_091_223),
            (204, 3_789_367_123_908_416_719),
            (122, 15_606_374_678_349_145_608),
        ]
    );
}

#[test]
fn ht_accelerator_hooks_keep_pipeline_order() {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Stage {
        WholeTile,
        Deinterleave,
        ForwardDwt,
        Packetization,
    }

    #[derive(Default)]
    struct RecordingAccelerator {
        stages: Vec<Stage>,
    }

    impl crate::J2kEncodeStageAccelerator for RecordingAccelerator {
        fn encode_htj2k_tile(
            &mut self,
            _job: crate::J2kHtj2kTileEncodeJob<'_>,
        ) -> core::result::Result<Option<Vec<u8>>, &'static str> {
            self.stages.push(Stage::WholeTile);
            Ok(None)
        }

        fn encode_deinterleave(
            &mut self,
            _job: crate::J2kDeinterleaveToF32Job<'_>,
        ) -> core::result::Result<Option<Vec<Vec<f32>>>, &'static str> {
            self.stages.push(Stage::Deinterleave);
            Ok(None)
        }

        fn encode_forward_dwt53(
            &mut self,
            _job: crate::J2kForwardDwt53Job<'_>,
        ) -> core::result::Result<Option<crate::J2kForwardDwt53Output>, &'static str> {
            self.stages.push(Stage::ForwardDwt);
            Ok(None)
        }

        fn encode_packetization(
            &mut self,
            _job: crate::J2kPacketizationEncodeJob<'_>,
        ) -> core::result::Result<Option<Vec<u8>>, &'static str> {
            self.stages.push(Stage::Packetization);
            Ok(None)
        }
    }

    let pixels = (0..8 * 8)
        .map(|index| u8::try_from((index * 41 + 3) & 0xff).expect("sample fits u8"))
        .collect::<Vec<_>>();
    let options = EncodeOptions {
        num_decomposition_levels: 1,
        reversible: true,
        use_ht_block_coding: true,
        ..EncodeOptions::default()
    };
    let expected = encode(&pixels, 8, 8, 1, 8, false, &options).expect("CPU reference encode");
    let mut accelerator = RecordingAccelerator::default();
    let actual = encode_with_accelerator(&pixels, 8, 8, 1, 8, false, &options, &mut accelerator)
        .expect("fallback accelerator encode");

    assert_eq!(actual, expected);
    assert_eq!(
        accelerator.stages,
        [
            Stage::WholeTile,
            Stage::Deinterleave,
            Stage::ForwardDwt,
            Stage::Packetization,
        ]
    );
}
