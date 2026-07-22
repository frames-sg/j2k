// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k::{
    encode_j2k_lossless, prepare_batch, wrap_j2k_codestream, BatchCodecRoute, BatchDecodeOptions,
    BatchLayout, BatchWaveletTransform, CompressedPayloadKind, CompressedTransferSyntax,
    CpuBatchDecoder, CpuBatchGroup, CpuBatchSamples, DecodeRequest, Downscale, EncodedImage,
    J2kDecoder, J2kFileWrapOptions, J2kLosslessEncodeOptions, J2kLosslessSamples, J2kScratchPool,
    PixelFormat, Rect,
};
use j2k_native::{encode, encode_htj2k, EncodeOptions};
use j2k_test_support::{openjph_batch_fixtures, OpenJphBatchFixture};

#[derive(Clone, Copy, Debug)]
enum CodingRoute {
    Classic,
    Htj2k,
}

#[derive(Debug)]
enum NativeOracle {
    U8(Vec<u8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
}

fn fixture_oracle(fixture: OpenJphBatchFixture) -> NativeOracle {
    if fixture.signed {
        if fixture.precision <= 8 {
            NativeOracle::I16(
                fixture
                    .oracle
                    .iter()
                    .map(|sample| i16::from(i8::from_ne_bytes([*sample])))
                    .collect(),
            )
        } else {
            NativeOracle::I16(
                fixture
                    .oracle
                    .chunks_exact(2)
                    .map(|sample| i16::from_le_bytes([sample[0], sample[1]]))
                    .collect(),
            )
        }
    } else if fixture.precision <= 8 {
        NativeOracle::U8(fixture.oracle.to_vec())
    } else {
        NativeOracle::U16(
            fixture
                .oracle
                .chunks_exact(2)
                .map(|sample| u16::from_le_bytes([sample[0], sample[1]]))
                .collect(),
        )
    }
}

#[test]
fn independent_openjph_batch_matrix_preserves_native_samples_and_metadata() {
    let fixtures = openjph_batch_fixtures();
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let mut decoder = CpuBatchDecoder::new(options);
    let result = decoder
        .decode(
            fixtures
                .iter()
                .map(|fixture| EncodedImage::full(Arc::from(fixture.encoded)))
                .collect(),
        )
        .expect("decode independent OpenJPH batch matrix");

    assert!(
        result.errors().is_empty(),
        "OpenJPH matrix errors: {:?}",
        result.errors()
    );
    assert_eq!(result.groups().len(), fixtures.len());
    for (source_index, fixture) in fixtures.iter().copied().enumerate() {
        let group = result
            .groups()
            .iter()
            .find(|group| group.source_indices() == [source_index])
            .unwrap_or_else(|| panic!("{}: output group", fixture.name));
        assert_eq!(
            group.info().route,
            BatchCodecRoute::Htj2k,
            "{}",
            fixture.name
        );
        assert_eq!(
            group.info().dimensions,
            (fixture.width, fixture.height),
            "{}",
            fixture.name
        );
        assert_eq!(
            group.info().precision,
            fixture.precision,
            "{}",
            fixture.name
        );
        assert_eq!(group.info().signed, fixture.signed, "{}", fixture.name);
        assert_eq!(
            group.info().color.channels(),
            fixture.components,
            "{}",
            fixture.name
        );
        assert_eq!(
            group.info().transform,
            if fixture.reversible {
                BatchWaveletTransform::Reversible53
            } else {
                BatchWaveletTransform::Irreversible97
            },
            "{}",
            fixture.name
        );
        assert_eq!(
            group.info().payload_kind,
            if fixture.jph {
                CompressedPayloadKind::JphFile
            } else {
                CompressedPayloadKind::Jpeg2000Codestream
            },
            "{}",
            fixture.name
        );

        let expected = fixture_oracle(fixture);
        if fixture.reversible {
            expected.assert_samples(group.samples(), fixture.name);
        } else {
            expected.assert_within_one_lsb(group.samples(), fixture.name);
        }
    }
}

#[test]
fn independent_openjph_signed_rgb_component_planes_preserve_negative_samples() {
    let fixture = openjph_batch_fixtures()
        .iter()
        .find(|fixture| fixture.name == "openjph-rgb-s8-53-raw")
        .expect("signed RGB8 OpenJPH fixture");
    let NativeOracle::I16(expected) = fixture_oracle(*fixture) else {
        panic!("signed RGB8 fixture oracle must be i16")
    };
    let mut decoder = J2kDecoder::new(fixture.encoded).expect("OpenJPH component decoder");
    let components = decoder
        .decode_components()
        .expect("decode OpenJPH component planes");

    assert_eq!(components.planes().len(), 3);
    for (channel, plane) in components.planes().iter().enumerate() {
        assert!(plane.signed());
        let expected = expected
            .iter()
            .skip(channel)
            .step_by(3)
            .copied()
            .collect::<Vec<_>>();
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the decoded fixture plane is bounded by its asserted signed 8-bit metadata"
        )]
        let actual = plane
            .samples()
            .iter()
            .map(|sample| sample.round() as i16)
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "component {channel}");
    }
}

impl NativeOracle {
    fn assert_samples(&self, actual: &CpuBatchSamples, name: &str) {
        match (self, actual) {
            (Self::U8(expected), CpuBatchSamples::U8(actual)) => {
                assert_eq!(actual, expected, "{name}: native u8 samples");
            }
            (Self::U16(expected), CpuBatchSamples::U16(actual)) => {
                assert_eq!(actual, expected, "{name}: native u16 samples");
            }
            (Self::I16(expected), CpuBatchSamples::I16(actual)) => {
                assert_eq!(actual, expected, "{name}: native i16 samples");
            }
            _ => panic!("{name}: batch returned the wrong native sample owner"),
        }
    }

    fn assert_within_one_lsb(&self, actual: &CpuBatchSamples, name: &str) {
        let (lengths_match, (max_difference, max_index, actual_at_max, expected_at_max)) =
            match (self, actual) {
                (Self::U8(expected), CpuBatchSamples::U8(actual)) => (
                    actual.len() == expected.len(),
                    actual
                        .iter()
                        .zip(expected)
                        .enumerate()
                        .map(|(index, (actual, expected))| {
                            (
                                u16::from(actual.abs_diff(*expected)),
                                index,
                                i32::from(*actual),
                                i32::from(*expected),
                            )
                        })
                        .max()
                        .unwrap_or((0, 0, 0, 0)),
                ),
                (Self::U16(expected), CpuBatchSamples::U16(actual)) => (
                    actual.len() == expected.len(),
                    actual
                        .iter()
                        .zip(expected)
                        .enumerate()
                        .map(|(index, (actual, expected))| {
                            (
                                actual.abs_diff(*expected),
                                index,
                                i32::from(*actual),
                                i32::from(*expected),
                            )
                        })
                        .max()
                        .unwrap_or((0, 0, 0, 0)),
                ),
                (Self::I16(expected), CpuBatchSamples::I16(actual)) => (
                    actual.len() == expected.len(),
                    actual
                        .iter()
                        .zip(expected)
                        .enumerate()
                        .map(|(index, (actual, expected))| {
                            (
                                actual.abs_diff(*expected),
                                index,
                                i32::from(*actual),
                                i32::from(*expected),
                            )
                        })
                        .max()
                        .unwrap_or((0, 0, 0, 0)),
                ),
                _ => (false, (u16::MAX, 0, 0, 0)),
            };
        assert!(
            lengths_match && max_difference <= 1,
            "{name}: irreversible reconstruction has max OpenJPH difference {max_difference} LSB at sample {max_index}: codec={actual_at_max}, OpenJPH={expected_at_max}; or returned the wrong native sample owner"
        );
    }
}

#[derive(Debug)]
struct NativeCase {
    name: String,
    encoded: Arc<[u8]>,
    components: usize,
    precision: u8,
    signed: bool,
    route: BatchCodecRoute,
    oracle: NativeOracle,
}

fn encode_case(route: CodingRoute, components: usize, precision: u8, signed: bool) -> NativeCase {
    const WIDTH: u32 = 5;
    const HEIGHT: u32 = 3;
    let sample_count = WIDTH as usize * HEIGHT as usize * components;
    let (bytes, oracle) = if signed {
        let modulus = 1_i32 << precision;
        let midpoint = modulus / 2;
        let samples = (0..sample_count)
            .map(|index| {
                let raw = (i32::try_from(index).expect("small fixture index") * 83 + 19) % modulus;
                i16::try_from(raw - midpoint).expect("fixture stays in i16")
            })
            .collect::<Vec<_>>();
        let bytes = if precision <= 8 {
            samples
                .iter()
                .map(|sample| sample.to_le_bytes()[0])
                .collect()
        } else {
            samples
                .iter()
                .flat_map(|sample| sample.to_le_bytes())
                .collect()
        };
        (bytes, NativeOracle::I16(samples))
    } else if precision <= 8 {
        let modulus = 1_u32 << precision;
        let samples = (0..sample_count)
            .map(|index| {
                u8::try_from(
                    (u32::try_from(index).expect("small fixture index") * 37 + 11) % modulus,
                )
                .expect("8-bit fixture sample")
            })
            .collect::<Vec<_>>();
        (samples.clone(), NativeOracle::U8(samples))
    } else {
        let modulus = 1_u32 << precision;
        let samples = (0..sample_count)
            .map(|index| {
                u16::try_from(
                    (u32::try_from(index).expect("small fixture index") * 977 + 31) % modulus,
                )
                .expect("16-bit fixture sample")
            })
            .collect::<Vec<_>>();
        let bytes = samples
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect();
        (bytes, NativeOracle::U16(samples))
    };
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        use_mct: false,
        ..EncodeOptions::default()
    };
    let components_u16 = u16::try_from(components).expect("fixture component count");
    let encoded = match route {
        CodingRoute::Classic => encode(
            &bytes,
            WIDTH,
            HEIGHT,
            components_u16,
            precision,
            signed,
            &options,
        )
        .expect("encode classic matrix fixture"),
        CodingRoute::Htj2k => encode_htj2k(
            &bytes,
            WIDTH,
            HEIGHT,
            components_u16,
            precision,
            signed,
            &options,
        )
        .expect("encode HTJ2K matrix fixture"),
    };
    let route_name = match route {
        CodingRoute::Classic => "classic",
        CodingRoute::Htj2k => "htj2k",
    };
    NativeCase {
        name: format!(
            "{route_name}-{}-{sign}{precision}",
            if components == 1 { "gray" } else { "rgb" },
            sign = if signed { "s" } else { "u" },
        ),
        encoded: Arc::from(encoded),
        components,
        precision,
        signed,
        route: match route {
            CodingRoute::Classic => BatchCodecRoute::Classic,
            CodingRoute::Htj2k => BatchCodecRoute::Htj2k,
        },
        oracle,
    }
}

#[path = "owned_batch_fixtures/classic.rs"]
mod classic;
#[path = "owned_batch_fixtures/ht_matrix.rs"]
mod ht_matrix;
#[path = "owned_batch_fixtures/irreversible.rs"]
mod irreversible;
