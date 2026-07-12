// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use super::*;
use crate::j2c::codestream::{markers, read_header};
use crate::j2c::encode::single_tile::encode_single_tile_packets_impl;
use crate::j2c::encode::tile_parts::consume_packetized_tile_into_tile_parts;
use crate::j2c::encode::{
    NativeEncodePipelineError, NativeEncodePipelineResult, NativeEncodeRetainedInput,
};
use crate::reader::BitReader;
use crate::{CpuOnlyJ2kEncodeStageAccelerator, DecodeSettings, EncodeError};

fn pixels(width: u32, height: u32) -> Vec<u8> {
    (0..width * height)
        .map(|index| u8::try_from((index * 37 + index / 3) & 0xff).expect("masked sample"))
        .collect()
}

#[test]
fn isolated_child_returns_direct_packet_owners_with_separated_headers() {
    const WIDTH: u32 = 8;
    const HEIGHT: u32 = 8;
    let pixels = pixels(WIDTH, HEIGHT);
    let options = EncodeOptions {
        num_decomposition_levels: 1,
        reversible: true,
        write_plt: true,
        write_ppm: true,
        ..EncodeOptions::default()
    };
    let session = NativeEncodeSession::try_new(NativeEncodeRetainedInput::none())
        .expect("direct packet session");
    let mut accelerator = CpuOnlyJ2kEncodeStageAccelerator;
    let packetized = encode_single_tile_packets_impl(
        &pixels,
        WIDTH,
        HEIGHT,
        1,
        8,
        false,
        &options,
        BlockCodingMode::Classic,
        &[],
        &[],
        &session,
        &mut accelerator,
    )
    .expect("direct packetized tile");

    assert!(!packetized.packet_lengths.is_empty());
    assert_eq!(
        packetized.packet_headers.len(),
        packetized.packet_lengths.len()
    );
    let packet_bytes = packetized
        .packet_lengths
        .iter()
        .map(|&length| usize::try_from(length).expect("packet length fits usize"))
        .sum::<usize>();
    assert_eq!(packet_bytes, packetized.data.len());
    assert_ne!(packetized.data.get(..2), Some(&[0xff, 0x4f][..]));
    assert_ne!(
        packetized
            .data
            .get(packetized.data.len().saturating_sub(2)..),
        Some(&[0xff, 0xd9][..])
    );

    let data_ptr = packetized.data.as_ptr();
    let first_header_ptr = packetized.packet_headers[0].as_ptr();
    let parts = consume_packetized_tile_into_tile_parts(7, packetized, None, 0, &session)
        .expect("move direct packet owners into one parent tile-part");
    assert_eq!(parts.len(), 1);
    assert_eq!(parts[0].data.as_ptr(), data_ptr);
    assert_eq!(parts[0].packet_headers[0].as_ptr(), first_header_ptr);
}

#[test]
fn direct_packet_owners_match_single_tile_marker_serialization() {
    const WIDTH: u32 = 8;
    const HEIGHT: u32 = 8;
    let pixels = pixels(WIDTH, HEIGHT);
    let options = EncodeOptions {
        num_decomposition_levels: 1,
        reversible: true,
        write_plm: true,
        write_ppm: true,
        ..EncodeOptions::default()
    };

    let direct_session = NativeEncodeSession::try_new(NativeEncodeRetainedInput::none())
        .expect("direct packet session");
    let mut direct_accelerator = CpuOnlyJ2kEncodeStageAccelerator;
    let packetized = encode_single_tile_packets_impl(
        &pixels,
        WIDTH,
        HEIGHT,
        1,
        8,
        false,
        &options,
        BlockCodingMode::Classic,
        &[],
        &[],
        &direct_session,
        &mut direct_accelerator,
    )
    .expect("direct packetized tile");

    let serialized_session = NativeEncodeSession::try_new(NativeEncodeRetainedInput::none())
        .expect("single-tile serialization session");
    let mut serialized_accelerator = CpuOnlyJ2kEncodeStageAccelerator;
    let codestream = crate::j2c::encode::single_tile::encode_impl(
        &pixels,
        WIDTH,
        HEIGHT,
        1,
        8,
        false,
        &options,
        BlockCodingMode::Classic,
        &[],
        &[],
        &serialized_session,
        &mut serialized_accelerator,
    )
    .expect("single-tile marker serialization oracle");

    let mut reader = BitReader::new(&codestream);
    assert_eq!(reader.read_marker().expect("SOC marker"), markers::SOC);
    let header = read_header(&mut reader, &DecodeSettings::default(), 0)
        .expect("serialized single-tile header");
    assert_eq!(header.plm_packet_lengths, packetized.packet_lengths);
    assert_eq!(header.ppm_packets.len(), packetized.packet_headers.len());
    for (serialized, direct) in header.ppm_packets.iter().zip(&packetized.packet_headers) {
        assert_eq!(serialized.data, direct);
    }

    let sod = codestream
        .windows(2)
        .position(|marker| marker == [0xff, markers::SOD])
        .expect("serialized tile SOD");
    let eoc = codestream
        .windows(2)
        .rposition(|marker| marker == [0xff, markers::EOC])
        .expect("serialized tile EOC");
    assert_eq!(&codestream[sod + 2..eoc], packetized.data);
}

fn encode_two_tiles_with_cap(cap: usize) -> NativeEncodePipelineResult<Vec<u8>> {
    const WIDTH: u32 = 8;
    const HEIGHT: u32 = 4;
    let pixels = pixels(WIDTH, HEIGHT);
    let options = EncodeOptions {
        num_decomposition_levels: 1,
        reversible: true,
        tile_size: Some((4, 4)),
        tile_part_packet_limit: Some(1),
        write_ppt: true,
        ..EncodeOptions::default()
    };
    let session = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), cap)?;
    let mut accelerator = CpuOnlyJ2kEncodeStageAccelerator;
    encode_multitile_impl(
        &MultiTileEncodeRequest {
            pixels: &pixels,
            width: WIDTH,
            height: HEIGHT,
            num_components: 1,
            bit_depth: 8,
            signed: false,
            options: &options,
            block_coding_mode: BlockCodingMode::Classic,
            roi_regions: &[],
            component_sample_info: &[],
            session: &session,
            tile_width: 4,
            tile_height: 4,
        },
        &mut accelerator,
    )
}

fn cap_result(cap: usize) -> Result<bool, NativeEncodePipelineError> {
    match encode_two_tiles_with_cap(cap) {
        Ok(_) => Ok(true),
        Err(NativeEncodePipelineError::Typed(EncodeError::AllocationTooLarge { .. })) => Ok(false),
        Err(error) => Err(error),
    }
}

#[test]
fn direct_multitile_handoff_accepts_exact_peak_and_rejects_one_byte_less() {
    let mut upper = 1_024usize;
    while !cap_result(upper).expect("only the configured cap may reject discovery") {
        upper = upper.checked_mul(2).expect("small fixture peak fits usize");
    }
    let mut lower = 0usize;
    while lower + 1 < upper {
        let middle = lower + (upper - lower) / 2;
        if cap_result(middle).expect("only the configured cap may reject search") {
            upper = middle;
        } else {
            lower = middle;
        }
    }

    encode_two_tiles_with_cap(upper).expect("exact multi-tile peak");
    let error = encode_two_tiles_with_cap(upper - 1)
        .expect_err("one byte below the direct multi-tile peak must fail");
    assert!(matches!(
        error,
        NativeEncodePipelineError::Typed(EncodeError::AllocationTooLarge {
            requested,
            cap,
            ..
        }) if requested == upper && cap == upper - 1
    ));
}
