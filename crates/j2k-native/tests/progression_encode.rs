// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_native::{
    encode, encode_j2k_packetization_scalar, encode_with_accelerator, DecodeSettings,
    EncodeOptions, EncodeProgressionOrder, Image, J2kEncodeStageAccelerator, J2kEncodeStageError,
    J2kEncodeStageResult, J2kPacketizationEncodeJob,
};

const ISSUE_WIDTH: u32 = 301;
const ISSUE_HEIGHT: u32 = 203;
const COMPACT_WIDTH: u32 = 129;
const COMPACT_HEIGHT: u32 = 97;

fn source_samples(width: u32, height: u32) -> Vec<u8> {
    (0..width * height * 3)
        .map(|index| u8::try_from(index * 7 % 251).expect("sample is less than 251"))
        .collect()
}

fn precincts(exponents: [(u8, u8); 4]) -> Vec<(u8, u8)> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(exponents.len())
        .expect("four precinct sizes");
    values.extend(exponents);
    values
}

fn issue_options(progression_order: EncodeProgressionOrder, num_layers: u8) -> EncodeOptions {
    EncodeOptions {
        reversible: true,
        num_decomposition_levels: 3,
        code_block_width_exp: 3,
        code_block_height_exp: 3,
        progression_order,
        num_layers,
        precinct_exponents: precincts([(5, 5), (6, 6), (6, 6), (6, 6)]),
        ..EncodeOptions::default()
    }
}

fn assert_issue_round_trip(progression_order: EncodeProgressionOrder, num_layers: u8) {
    let samples = source_samples(ISSUE_WIDTH, ISSUE_HEIGHT);
    let options = issue_options(progression_order, num_layers);
    let codestream = encode(&samples, ISSUE_WIDTH, ISSUE_HEIGHT, 3, 8, false, &options)
        .unwrap_or_else(|error| panic!("{progression_order:?}, {num_layers} layers: {error}"));
    let decoded = Image::new(&codestream, &DecodeSettings::default())
        .and_then(|image| image.decode())
        .unwrap_or_else(|error| panic!("{progression_order:?}, {num_layers} layers: {error}"));

    assert_eq!(
        decoded, samples,
        "{progression_order:?}, {num_layers} layers must reproduce the source samples"
    );
}

#[test]
fn position_major_progressions_round_trip_with_differing_precinct_spans() {
    for progression_order in [EncodeProgressionOrder::Pcrl, EncodeProgressionOrder::Cprl] {
        for num_layers in [1, 3] {
            assert_issue_round_trip(progression_order, num_layers);
        }
    }
}

#[test]
fn high_bit_partial_edge_tiles_round_trip_with_position_progression() {
    let samples = (0..ISSUE_WIDTH * ISSUE_HEIGHT)
        .flat_map(|index| (index.wrapping_mul(1_000_003) & ((1 << 29) - 1)).to_le_bytes())
        .collect::<Vec<_>>();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 3,
        code_block_width_exp: 2,
        code_block_height_exp: 2,
        progression_order: EncodeProgressionOrder::Cprl,
        use_mct: false,
        tile_size: Some((128, 128)),
        precinct_exponents: precincts([(4, 4), (5, 5), (5, 5), (5, 5)]),
        ..EncodeOptions::default()
    };

    let codestream = encode(&samples, ISSUE_WIDTH, ISSUE_HEIGHT, 1, 29, false, &options)
        .expect("encode 29-bit partial-edge multi-tile CPRL codestream");
    let decoded = Image::new(&codestream, &DecodeSettings::default())
        .and_then(|image| image.decode_native())
        .expect("decode 29-bit partial-edge multi-tile CPRL codestream");

    assert_eq!((decoded.width, decoded.height), (ISSUE_WIDTH, ISSUE_HEIGHT));
    assert_eq!(decoded.bit_depth, 29);
    assert_eq!(decoded.data, samples);
}

#[derive(Default)]
struct DescriptorCapturingAccelerator {
    descriptors: Vec<(u16, u32, u64, u8)>,
}

impl J2kEncodeStageAccelerator for DescriptorCapturingAccelerator {
    fn encode_packetization(
        &mut self,
        job: J2kPacketizationEncodeJob<'_>,
    ) -> J2kEncodeStageResult<Option<Vec<u8>>> {
        self.descriptors = job
            .packet_descriptors
            .iter()
            .map(|descriptor| {
                (
                    descriptor.component,
                    descriptor.resolution,
                    descriptor.precinct,
                    descriptor.layer,
                )
            })
            .collect();
        encode_j2k_packetization_scalar(job)
            .map(Some)
            .map_err(|source| {
                J2kEncodeStageError::backend(
                    "descriptor capture test",
                    "scalar packetization",
                    source,
                )
            })
    }
}

#[test]
fn packetization_accelerator_observes_reference_grid_progression_order() {
    let samples = source_samples(COMPACT_WIDTH, COMPACT_HEIGHT);
    let options = issue_options(EncodeProgressionOrder::Pcrl, 1);
    let mut accelerator = DescriptorCapturingAccelerator::default();
    let codestream = encode_with_accelerator(
        &samples,
        COMPACT_WIDTH,
        COMPACT_HEIGHT,
        3,
        8,
        false,
        &options,
        &mut accelerator,
    )
    .expect("encode PCRL through packetization accelerator");

    let expected_prefix = [
        (0, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 2, 0, 0),
        (0, 3, 0, 0),
        (1, 0, 0, 0),
        (1, 1, 0, 0),
        (1, 2, 0, 0),
        (1, 3, 0, 0),
        (2, 0, 0, 0),
        (2, 1, 0, 0),
        (2, 2, 0, 0),
        (2, 3, 0, 0),
        (0, 3, 1, 0),
    ];
    assert!(accelerator.descriptors.starts_with(&expected_prefix));

    let decoded = Image::new(&codestream, &DecodeSettings::default())
        .and_then(|image| image.decode())
        .expect("decode accelerator packetization codestream");
    assert_eq!(decoded, samples);
}
