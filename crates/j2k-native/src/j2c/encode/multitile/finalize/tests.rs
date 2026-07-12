// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::{EncodeError, NativeEncodeRetainedInput};

fn part(tile_index: u16, payload: &[u8], header: &[u8]) -> EncodedTilePart {
    let mut data = Vec::new();
    data.try_reserve_exact(payload.len())
        .expect("small payload");
    data.extend_from_slice(payload);
    let mut packet_lengths = Vec::new();
    packet_lengths
        .try_reserve_exact(1)
        .expect("small packet lengths");
    packet_lengths.push(u32::try_from(payload.len()).expect("small payload length"));
    let mut header_payload = Vec::new();
    header_payload
        .try_reserve_exact(header.len())
        .expect("small packet header");
    header_payload.extend_from_slice(header);
    let mut packet_headers = Vec::new();
    packet_headers
        .try_reserve_exact(1)
        .expect("small packet-header owner");
    packet_headers.push(header_payload);
    EncodedTilePart {
        tile_index,
        tile_part_index: 0,
        num_tile_parts: 1,
        data,
        packet_lengths,
        packet_headers,
    }
}

#[test]
fn marker_multitile_finalizer_counts_exact_writer_peak() {
    let params = codestream_write::EncodeParams {
        width: 4,
        height: 2,
        tile_width: 2,
        tile_height: 2,
        num_components: 1,
        bit_depth: 8,
        num_decomposition_levels: 0,
        reversible: true,
        num_layers: 1,
        write_tlm: true,
        write_plt: true,
        write_plm: true,
        write_ppm: true,
        ..codestream_write::EncodeParams::default()
    };
    let mut tile_bodies = Vec::new();
    tile_bodies
        .try_reserve_exact(2)
        .expect("small tile-part owners");
    tile_bodies.push(part(0, &[1, 2, 3], &[0xaa]));
    tile_bodies.push(part(1, &[4, 5], &[0xbb, 0xcc]));
    let quantization = [(8, 0)];
    let planning_bytes = quantization.len() * core::mem::size_of::<(u16, u16)>();
    let discovery = NativeEncodeSession::try_with_cap(
        NativeEncodeRetainedInput::none(),
        crate::DEFAULT_MAX_CODEC_BYTES,
    )
    .expect("discovery session");
    let output = finalize_multitile_codestream(
        &params,
        &tile_bodies,
        &quantization,
        planning_bytes,
        &discovery,
    )
    .expect("discover multi-tile writer peak");

    let base_bytes = encoded_tile_parts_retained_bytes(&tile_bodies, tile_bodies.capacity())
        .expect("tile bytes")
        + encode_params_retained_bytes(&params).expect("parameter bytes")
        + planning_bytes;
    let mut header_views = Vec::<&[Vec<u8>]>::new();
    header_views
        .try_reserve_exact(tile_bodies.len())
        .expect("small header views");
    let header_view_bytes = header_views.capacity() * core::mem::size_of::<&[Vec<u8>]>();
    let mut part_views = Vec::<codestream_write::TilePartData<'_>>::new();
    part_views
        .try_reserve_exact(tile_bodies.len())
        .expect("small part views");
    let part_view_bytes =
        part_views.capacity() * core::mem::size_of::<codestream_write::TilePartData<'_>>();
    let metadata_peak = base_bytes + header_view_bytes;
    let writer_peak = base_bytes + part_view_bytes + output.capacity();
    let exact_cap = metadata_peak.max(writer_peak);

    let exact = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), exact_cap)
        .expect("exact session");
    let exact_output =
        finalize_multitile_codestream(&params, &tile_bodies, &quantization, planning_bytes, &exact)
            .expect("multi-tile finalizer at exact cap");
    assert_eq!(exact_output, output);

    let over = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), exact_cap - 1)
        .expect("baseline below cap");
    let error =
        finalize_multitile_codestream(&params, &tile_bodies, &quantization, planning_bytes, &over)
            .expect_err("multi-tile finalizer is one byte over cap");
    let expected_what = if metadata_peak >= writer_peak {
        "multi-tile packet-header views"
    } else {
        "multi-tile codestream writer peak"
    };
    assert!(matches!(
        error,
        crate::j2c::encode::NativeEncodePipelineError::Typed(
            EncodeError::AllocationTooLarge {
                what,
                requested,
                cap,
            }
        ) if what == expected_what && requested == exact_cap && cap == exact_cap - 1
    ));
}
