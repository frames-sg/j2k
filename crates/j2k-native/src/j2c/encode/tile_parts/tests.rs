// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::j2c::codestream_write::EncodeParams;
use crate::{EncodeError, NativeEncodeRetainedInput};

fn vector_with_capacity<T>(capacity: usize) -> Vec<T> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .expect("small finalization test allocation");
    values
}

fn packetized_fixture() -> packet_encode::PacketizedTileData {
    let mut data = vector_with_capacity::<u8>(4);
    data.extend_from_slice(&[1, 2, 3, 4]);
    let mut packet_lengths = vector_with_capacity::<u32>(2);
    packet_lengths.extend_from_slice(&[2, 2]);
    packet_encode::PacketizedTileData {
        data,
        packet_lengths,
        packet_headers: Vec::new(),
    }
}

#[test]
fn borrowed_finalization_accepts_exact_peak_and_rejects_one_byte_over() {
    let params = EncodeParams {
        width: 2,
        height: 2,
        tile_width: 2,
        tile_height: 2,
        num_components: 1,
        bit_depth: 8,
        num_decomposition_levels: 0,
        reversible: true,
        num_layers: 1,
        ..EncodeParams::default()
    };
    let packetized = packetized_fixture();
    let retained_phase_bytes = 7;
    let discovery = NativeEncodeSession::try_with_cap(
        NativeEncodeRetainedInput::none(),
        crate::DEFAULT_MAX_CODEC_BYTES,
    )
    .expect("discovery session");
    let ranges = borrowed_tile_part_ranges(
        packetized.data.len(),
        &packetized.packet_lengths,
        packetized.packet_headers.len(),
        Some(1),
        retained_phase_bytes
            + packet_encode::packetized_tile_retained_bytes(&packetized).expect("packetized bytes"),
        &discovery,
    )
    .expect("borrowed ranges");
    let range_bytes = ranges.capacity() * core::mem::size_of::<BorrowedTilePartRange>();
    let mut views = Vec::<codestream_write::TilePartData<'_>>::new();
    views
        .try_reserve_exact(ranges.len())
        .expect("small view test allocation");
    let view_bytes = views.capacity() * core::mem::size_of::<codestream_write::TilePartData<'_>>();
    let output = write_single_tile_packetized_codestream_for_session(
        &params,
        &packetized,
        &[(8, 0)],
        Some(1),
        retained_phase_bytes,
        &discovery,
    )
    .expect("discover output capacity");
    let base_bytes = retained_phase_bytes
        + packet_encode::packetized_tile_retained_bytes(&packetized).expect("packetized bytes");
    let owner_peak = base_bytes + range_bytes + view_bytes;
    let output_peak = base_bytes + view_bytes + output.capacity();
    let exact_cap = owner_peak.max(output_peak);

    let exact = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), exact_cap)
        .expect("exact finalization session");
    write_single_tile_packetized_codestream_for_session(
        &params,
        &packetized,
        &[(8, 0)],
        Some(1),
        retained_phase_bytes,
        &exact,
    )
    .expect("exact finalization peak");

    let over = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), exact_cap - 1)
        .expect("retained baseline remains below cap");
    let error = write_single_tile_packetized_codestream_for_session(
        &params,
        &packetized,
        &[(8, 0)],
        Some(1),
        retained_phase_bytes,
        &over,
    )
    .expect_err("final codestream is one byte over cap");
    let expected_what = if owner_peak >= output_peak {
        "single-tile codestream part views"
    } else {
        "single-tile codestream writer peak"
    };
    assert!(matches!(
        error,
        crate::j2c::encode::NativeEncodePipelineError::Typed(EncodeError::AllocationTooLarge {
            what,
            requested,
            cap,
        }) if what == expected_what && requested == exact_cap && cap == exact_cap - 1
    ));
}

#[test]
fn scratch_free_finalization_is_byte_exact_and_enforces_writer_peak() {
    let params = EncodeParams {
        width: 2,
        height: 2,
        tile_width: 2,
        tile_height: 2,
        num_components: 1,
        bit_depth: 8,
        num_decomposition_levels: 0,
        reversible: true,
        num_layers: 1,
        ..EncodeParams::default()
    };
    let packetized = packetized_fixture();
    let quantization = [(8, 0)];
    let retained_phase_bytes = 7;
    let expected = codestream_write::write_codestream(&params, &packetized.data, &quantization)
        .expect("reference writer");
    let discovery = NativeEncodeSession::try_with_cap(
        NativeEncodeRetainedInput::none(),
        crate::DEFAULT_MAX_CODEC_BYTES,
    )
    .expect("discovery session");
    let discovered = write_single_tile_packetized_codestream_for_session(
        &params,
        &packetized,
        &quantization,
        None,
        retained_phase_bytes,
        &discovery,
    )
    .expect("discover scratch-free writer peak");
    assert_eq!(discovered, expected);
    let exact_cap = retained_phase_bytes
        + packet_encode::packetized_tile_retained_bytes(&packetized).expect("packetized bytes")
        + discovered.capacity();

    let exact = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), exact_cap)
        .expect("exact writer session");
    let actual = write_single_tile_packetized_codestream_for_session(
        &params,
        &packetized,
        &quantization,
        None,
        retained_phase_bytes,
        &exact,
    )
    .expect("scratch-free writer at exact cap");
    assert_eq!(actual, expected);

    let over = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), exact_cap - 1)
        .expect("baseline below cap");
    let error = write_single_tile_packetized_codestream_for_session(
        &params,
        &packetized,
        &quantization,
        None,
        retained_phase_bytes,
        &over,
    )
    .expect_err("writer peak is one byte over cap");
    assert!(matches!(
        error,
        crate::j2c::encode::NativeEncodePipelineError::Typed(EncodeError::AllocationTooLarge {
            what: "single-tile codestream writer peak",
            requested,
            cap,
        }) if requested == exact_cap && cap == exact_cap - 1
    ));
}

#[test]
fn borrowed_tile_parts_cover_payload_without_copying_or_overlap() {
    let packetized = packetized_fixture();
    let session = NativeEncodeSession::try_with_cap(
        NativeEncodeRetainedInput::none(),
        crate::DEFAULT_MAX_CODEC_BYTES,
    )
    .expect("test session");
    let ranges = borrowed_tile_part_ranges(
        packetized.data.len(),
        &packetized.packet_lengths,
        packetized.packet_headers.len(),
        Some(1),
        packet_encode::packetized_tile_retained_bytes(&packetized).expect("packetized bytes"),
        &session,
    )
    .expect("borrowed ranges");

    assert_eq!(ranges.len(), 2);
    assert_eq!(ranges[0].data, 0..2);
    assert_eq!(ranges[1].data, 2..4);
    assert_eq!(
        packetized.data[ranges[0].data.clone()].as_ptr(),
        packetized.data.as_ptr()
    );
    assert_eq!(
        packetized.data[ranges[1].data.clone()].as_ptr(),
        packetized.data[2..].as_ptr()
    );
}
