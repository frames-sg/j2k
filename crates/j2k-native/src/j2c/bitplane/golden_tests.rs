// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::{string::String, vec::Vec};

use super::super::bitplane_encode;
use super::facade::decode_code_block_segments_validated_with_observer;
use super::observer::J2kDecodeObserver;
use super::{
    decode_code_block_segments_validated, decode_code_block_segments_validated_profiled,
    BitPlaneDecodeContext, Coefficient, J2kBlockDecodeStats,
};
use crate::error::{DecodeError, DecodingError};
use crate::j2c::build::SubBandType;
use crate::j2c::codestream::CodeBlockStyle;
use crate::J2kCodeBlockSegment;

fn generated_coefficients(width: u32, height: u32, seed: u32) -> Vec<i32> {
    let mut coefficients = Vec::with_capacity(width as usize * height as usize);
    let mut state = seed ^ 0x9e37_79b9;
    for idx in 0..width * height {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let value = ((state >> 16) & 0x01ff) as i32 - 255;
        coefficients.push(if (idx + seed).is_multiple_of(11) {
            0
        } else {
            value
        });
    }
    coefficients
}

fn hash_bytes<'a>(chunks: impl IntoIterator<Item = &'a [u8]>) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for chunk in chunks {
        for byte in chunk {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

fn segment_hash(segments: &[J2kCodeBlockSegment]) -> u64 {
    let bytes = segments
        .iter()
        .flat_map(|segment| {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&segment.data_offset.to_le_bytes());
            bytes.extend_from_slice(&segment.data_length.to_le_bytes());
            bytes.push(segment.start_coding_pass);
            bytes.push(segment.end_coding_pass);
            bytes.push(u8::from(segment.use_arithmetic));
            bytes
        })
        .collect::<Vec<_>>();
    hash_bytes([bytes.as_slice()])
}

fn coefficient_hash(coefficients: &[i32]) -> u64 {
    let bytes = coefficients
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    hash_bytes([bytes.as_slice()])
}

#[derive(Default)]
struct PassObserver(String);

impl J2kDecodeObserver for PassObserver {
    fn add_sigprop_us(&mut self, _: Option<crate::profile::ProfileInstant>) {
        self.0.push('S');
    }

    fn add_magref_us(&mut self, _: Option<crate::profile::ProfileInstant>) {
        self.0.push('M');
    }

    fn add_cleanup_us(&mut self, _: Option<crate::profile::ProfileInstant>) {
        self.0.push('C');
    }

    fn add_bypass_us(&mut self, _: Option<crate::profile::ProfileInstant>) {
        self.0.push('B');
    }
}

struct ExpectedStyleGolden {
    name: &'static str,
    data_len: usize,
    missing_bitplanes: u8,
    coding_passes: u8,
    segment_count: usize,
    data_hash: u64,
    segment_hash: u64,
    coefficient_hash: u64,
    pass_order: &'static str,
}

#[test]
fn classic_style_pass_and_segment_goldens_are_bit_exact() {
    let width = 13u32;
    let height = 9u32;
    let total_bitplanes = 10u8;
    let cases = [
        (
            ExpectedStyleGolden {
                name: "normal_ll",
                data_len: 138,
                missing_bitplanes: 2,
                coding_passes: 22,
                segment_count: 1,
                data_hash: 0xB117_89F2_DC85_9393,
                segment_hash: 0xD443_AE1A_4BC1_E110,
                coefficient_hash: 0x4846_FAD8_71AE_2D4A,
                pass_order: "CSMCSMCSMCSMCSMCSMCSMC",
            },
            CodeBlockStyle::default(),
            SubBandType::LowLow,
            0x5100u32,
        ),
        (
            ExpectedStyleGolden {
                name: "bypass_lh",
                data_len: 146,
                missing_bitplanes: 2,
                coding_passes: 22,
                segment_count: 9,
                data_hash: 0xBF45_3854_26E5_30D1,
                segment_hash: 0x7EEB_406D_6994_5230,
                coefficient_hash: 0xC779_1B2B_276C_7033,
                pass_order: "CSMCSMCSMCBBCBBCBBCBBC",
            },
            CodeBlockStyle {
                selective_arithmetic_coding_bypass: true,
                ..CodeBlockStyle::default()
            },
            SubBandType::LowHigh,
            0x5200,
        ),
        (
            ExpectedStyleGolden {
                name: "term_reset_hl",
                data_len: 174,
                missing_bitplanes: 1,
                coding_passes: 25,
                segment_count: 25,
                data_hash: 0x9DC9_47B0_7CD5_5AE0,
                segment_hash: 0x6957_4F58_B7C3_4764,
                coefficient_hash: 0xB4FE_5BFF_E52E_6151,
                pass_order: "CSMCSMCSMCSMCSMCSMCSMCSMC",
            },
            CodeBlockStyle {
                termination_on_each_pass: true,
                reset_context_probabilities: true,
                ..CodeBlockStyle::default()
            },
            SubBandType::HighLow,
            0x5300,
        ),
        (
            ExpectedStyleGolden {
                name: "segmentation_hh",
                data_len: 142,
                missing_bitplanes: 2,
                coding_passes: 22,
                segment_count: 1,
                data_hash: 0x9D0B_C2D6_2186_AC44,
                segment_hash: 0xBF14_EFF1_0489_AEEC,
                coefficient_hash: 0x762A_A534_6BCD_308A,
                pass_order: "CSMCSMCSMCSMCSMCSMCSMC",
            },
            CodeBlockStyle {
                segmentation_symbols: true,
                ..CodeBlockStyle::default()
            },
            SubBandType::HighHigh,
            0x5400,
        ),
        (
            ExpectedStyleGolden {
                name: "vcausal_ll",
                data_len: 134,
                missing_bitplanes: 2,
                coding_passes: 22,
                segment_count: 1,
                data_hash: 0xD263_5BD4_0A01_06D0,
                segment_hash: 0x9481_139E_75EB_1814,
                coefficient_hash: 0xB515_9929_9126_C857,
                pass_order: "CSMCSMCSMCSMCSMCSMCSMC",
            },
            CodeBlockStyle {
                vertically_causal_context: true,
                ..CodeBlockStyle::default()
            },
            SubBandType::LowLow,
            0x5500,
        ),
    ];

    for (expected, style, subband, seed) in cases {
        let coefficients = generated_coefficients(width, height, seed);
        let encoded = bitplane_encode::encode_code_block_segments_with_style(
            &coefficients,
            width,
            height,
            subband,
            total_bitplanes,
            &style,
        );
        let segments = encoded
            .segments
            .iter()
            .map(|segment| J2kCodeBlockSegment {
                data_offset: segment.data_offset,
                data_length: segment.data_length,
                start_coding_pass: segment.start_coding_pass,
                end_coding_pass: segment.end_coding_pass,
                use_arithmetic: segment.use_arithmetic,
            })
            .collect::<Vec<_>>();
        let mut ctx = BitPlaneDecodeContext::default();
        let mut observer = PassObserver::default();

        decode_code_block_segments_validated_with_observer(
            &encoded.data,
            &segments,
            width,
            height,
            encoded.num_zero_bitplanes,
            encoded.num_coding_passes,
            total_bitplanes,
            subband,
            &style,
            true,
            &mut ctx,
            &mut observer,
        )
        .expect("decode style golden");

        let decoded = ctx
            .coefficient_rows()
            .flat_map(|row| row.iter().map(Coefficient::get))
            .collect::<Vec<_>>();
        assert_eq!(decoded, coefficients, "{} coefficients", expected.name);
        assert_eq!(
            encoded.data.len(),
            expected.data_len,
            "{} len",
            expected.name
        );
        assert_eq!(
            encoded.num_zero_bitplanes, expected.missing_bitplanes,
            "{} missing bitplanes",
            expected.name
        );
        assert_eq!(
            encoded.num_coding_passes, expected.coding_passes,
            "{} coding passes",
            expected.name
        );
        assert_eq!(
            segments.len(),
            expected.segment_count,
            "{} segments",
            expected.name
        );
        assert_eq!(
            hash_bytes([encoded.data.as_slice()]),
            expected.data_hash,
            "{} data hash",
            expected.name
        );
        assert_eq!(
            segment_hash(&segments),
            expected.segment_hash,
            "{} segment hash",
            expected.name
        );
        assert_eq!(
            coefficient_hash(&decoded),
            expected.coefficient_hash,
            "{} coefficient hash",
            expected.name
        );
        assert_eq!(
            observer.0, expected.pass_order,
            "{} pass order",
            expected.name
        );
    }
}

#[test]
fn disabled_profile_counters_and_output_match_unprofiled_decode() {
    let width = 9u32;
    let height = 7u32;
    let total_bitplanes = 10u8;
    let coefficients = generated_coefficients(width, height, 0xA510);
    let style = CodeBlockStyle::default();
    let encoded = bitplane_encode::encode_code_block_segments_with_style(
        &coefficients,
        width,
        height,
        SubBandType::LowLow,
        total_bitplanes,
        &style,
    );
    let segments = encoded
        .segments
        .iter()
        .map(|segment| J2kCodeBlockSegment {
            data_offset: segment.data_offset,
            data_length: segment.data_length,
            start_coding_pass: segment.start_coding_pass,
            end_coding_pass: segment.end_coding_pass,
            use_arithmetic: segment.use_arithmetic,
        })
        .collect::<Vec<_>>();
    let mut ctx = BitPlaneDecodeContext::default();
    let mut stats = J2kBlockDecodeStats::default();

    decode_code_block_segments_validated_profiled(
        &encoded.data,
        &segments,
        width,
        height,
        encoded.num_zero_bitplanes,
        encoded.num_coding_passes,
        total_bitplanes,
        SubBandType::LowLow,
        &style,
        true,
        &mut ctx,
        &mut stats,
        false,
    )
    .expect("profile-disabled decode");

    let decoded = ctx
        .coefficient_rows()
        .flat_map(|row| row.iter().map(Coefficient::get))
        .collect::<Vec<_>>();
    assert_eq!(decoded, coefficients);
    assert_eq!(stats, J2kBlockDecodeStats::default());
}

#[test]
fn strict_metadata_and_segment_failures_keep_exact_error_classes() {
    let style = CodeBlockStyle::default();
    let mut ctx = BitPlaneDecodeContext::default();

    let error = decode_code_block_segments_validated(
        &[],
        &[],
        2,
        2,
        2,
        1,
        1,
        SubBandType::LowLow,
        &style,
        true,
        &mut ctx,
    )
    .expect_err("missing bitplanes above total must fail");
    assert_eq!(
        error,
        DecodeError::Decoding(DecodingError::InvalidBitplaneCount)
    );

    let error = decode_code_block_segments_validated(
        &[],
        &[],
        2,
        2,
        0,
        2,
        1,
        SubBandType::LowLow,
        &style,
        true,
        &mut ctx,
    )
    .expect_err("passes above available bitplanes must fail");
    assert_eq!(
        error,
        DecodeError::Decoding(DecodingError::TooManyCodingPasses)
    );

    let discontinuous = [J2kCodeBlockSegment {
        data_offset: 0,
        data_length: 0,
        start_coding_pass: 1,
        end_coding_pass: 1,
        use_arithmetic: true,
    }];
    let error = decode_code_block_segments_validated(
        &[],
        &discontinuous,
        2,
        2,
        0,
        1,
        1,
        SubBandType::LowLow,
        &style,
        true,
        &mut ctx,
    )
    .expect_err("discontinuous segment order must fail");
    assert_eq!(
        error,
        DecodeError::Decoding(DecodingError::CodeBlockDecodeFailure)
    );
    assert!(ctx
        .coefficient_rows()
        .flatten()
        .all(|coefficient| coefficient.get_i64() == 0));
}

#[test]
fn bitplane_modules_remain_focused_without_suppression_shortcuts() {
    const ROOT: &str = include_str!("../bitplane.rs");
    const MODULES: [(&str, &str, usize); 7] = [
        ("arithmetic", include_str!("arithmetic.rs"), 320),
        ("bypass", include_str!("bypass.rs"), 280),
        ("context", include_str!("context.rs"), 210),
        ("facade", include_str!("facade.rs"), 170),
        ("observer", include_str!("observer.rs"), 100),
        ("schedule", include_str!("schedule.rs"), 350),
        ("state", include_str!("state.rs"), 500),
    ];

    assert!(ROOT.lines().count() <= 35, "bitplane root regrew");
    for (name, source, line_cap) in MODULES {
        assert!(
            source.lines().count() <= line_cap,
            "{name}.rs exceeded its focused-module line cap"
        );
        assert!(!source.contains("include!"), "{name}.rs uses include!");
        assert!(
            !source.contains("allow(unused"),
            "{name}.rs suppresses unused-code diagnostics"
        );
        assert!(
            !source.contains("allow(clippy::too_many_lines"),
            "{name}.rs suppresses the god-function lint"
        );
    }
}
