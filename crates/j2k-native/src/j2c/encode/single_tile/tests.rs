// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::{vec, vec::Vec};
use std::panic::catch_unwind;

use super::super::*;

mod resident_errors;
mod whole_tile;

fn codestream_fingerprint(codestream: &[u8]) -> (usize, u64) {
    let hash = codestream
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
        });
    (codestream.len(), hash)
}

#[test]
fn public_encode_distinguishes_malformed_and_unsupported_requests() {
    let malformed = encode(&[], 0, 1, 1, 8, false, &EncodeOptions::default())
        .expect_err("zero-width encode must fail");
    assert_eq!(
        malformed,
        crate::EncodeError::InvalidInput {
            what: "invalid dimensions"
        }
    );

    let sampled_multitile = encode(
        &[0; 16],
        4,
        4,
        1,
        8,
        false,
        &EncodeOptions {
            component_sampling: Some(vec![(2, 1)]),
            tile_size: Some((2, 2)),
            ..EncodeOptions::default()
        },
    )
    .expect_err("sampled multi-tile encode is not implemented");
    assert_eq!(
        sampled_multitile,
        crate::EncodeError::Unsupported {
            what: "multi-tile encode with component sampling is not implemented"
        }
    );

    let high_bit_sampled = encode(
        &[0; 4],
        1,
        1,
        1,
        25,
        false,
        &EncodeOptions {
            reversible: true,
            component_sampling: Some(vec![(2, 1)]),
            ..EncodeOptions::default()
        },
    )
    .expect_err("exact high-bit raw encode requires full-resolution components");
    assert_eq!(
        high_bit_sampled,
        crate::EncodeError::Unsupported {
            what: "25-38 bit encode currently requires full-resolution components"
        }
    );
}

#[test]
fn public_code_block_exponents_fail_typed_without_panicking() {
    let oversized_width = catch_unwind(|| {
        encode(
            &[0; 4],
            2,
            2,
            1,
            8,
            false,
            &EncodeOptions {
                code_block_width_exp: u8::MAX,
                ..EncodeOptions::default()
            },
        )
    })
    .expect("u8::MAX code-block exponent must not panic")
    .expect_err("u8::MAX code-block exponent must fail");
    assert_eq!(
        oversized_width,
        crate::EncodeError::InvalidInput {
            what: "code-block width exponent exceeds supported range"
        }
    );

    let high_bit_data = [0_u8; 4];
    let high_bit_planes = [EncodeTypedComponentPlane {
        data: &high_bit_data,
        x_rsiz: 1,
        y_rsiz: 1,
        bit_depth: 25,
        signed: false,
    }];
    let oversized_height = catch_unwind(|| {
        encode_typed_component_planes_53(
            &high_bit_planes,
            1,
            1,
            &EncodeOptions {
                code_block_height_exp: 30,
                ..EncodeOptions::default()
            },
        )
    })
    .expect("large typed code-block exponent must not panic")
    .expect_err("large typed code-block exponent must fail");
    assert_eq!(
        oversized_height,
        crate::EncodeError::InvalidInput {
            what: "code-block height exponent exceeds supported range"
        }
    );

    let oversized_area = catch_unwind(|| {
        encode(
            &[0; 4],
            2,
            2,
            1,
            8,
            false,
            &EncodeOptions {
                code_block_width_exp: 5,
                code_block_height_exp: 4,
                ..EncodeOptions::default()
            },
        )
    })
    .expect("oversized code-block area must not panic")
    .expect_err("oversized code-block area must fail");
    assert_eq!(
        oversized_area,
        crate::EncodeError::InvalidInput {
            what: "code-block dimensions exceed JPEG 2000 Part 1 area limit"
        }
    );
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
        ) -> crate::J2kEncodeStageResult<Option<Vec<u8>>> {
            self.stages.push(Stage::WholeTile);
            Ok(None)
        }

        fn encode_deinterleave(
            &mut self,
            _job: crate::J2kDeinterleaveToF32Job<'_>,
        ) -> crate::J2kEncodeStageResult<Option<Vec<Vec<f32>>>> {
            self.stages.push(Stage::Deinterleave);
            Ok(None)
        }

        fn encode_forward_dwt53(
            &mut self,
            _job: crate::J2kForwardDwt53Job<'_>,
        ) -> crate::J2kEncodeStageResult<Option<crate::J2kForwardDwt53Output>> {
            self.stages.push(Stage::ForwardDwt);
            Ok(None)
        }

        fn encode_packetization(
            &mut self,
            _job: crate::J2kPacketizationEncodeJob<'_>,
        ) -> crate::J2kEncodeStageResult<Option<Vec<u8>>> {
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
