// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::j2c::codestream::markers;
use crate::j2c::codestream_write::{
    write_codestream_tiles_accounted_with_peak_check, EncodeParams, TilePartData,
};
use crate::EncodeError;
use alloc::{vec, vec::Vec};

fn marker_offsets(codestream: &[u8], marker: u8) -> Vec<usize> {
    codestream
        .windows(2)
        .enumerate()
        .filter_map(|(offset, bytes)| (bytes == [0xff, marker]).then_some(offset))
        .collect()
}

fn marker_params() -> EncodeParams {
    EncodeParams {
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
        ..EncodeParams::default()
    }
}

fn assert_marker_payloads(codestream: &[u8]) {
    assert_eq!(marker_offsets(codestream, markers::TLM).len(), 2);
    assert_eq!(marker_offsets(codestream, markers::SOT).len(), 2);
    let plm = marker_offsets(codestream, markers::PLM);
    assert_eq!(plm.len(), 1);
    assert_eq!(
        &codestream[plm[0] + 9..plm[0] + 16],
        &[0x00, 0x7f, 0x81, 0x00, 0x81, 0x80, 0x00]
    );
    let ppm = marker_offsets(codestream, markers::PPM);
    assert_eq!(ppm.len(), 1);
    assert_eq!(
        &codestream[ppm[0] + 5..ppm[0] + 12],
        &[0x00, 0x02, 0xaa, 0xbb, 0x00, 0x01, 0xcc]
    );
    let plt = marker_offsets(codestream, markers::PLT);
    assert_eq!(plt.len(), 2);
    assert_eq!(
        &codestream[plt[0] + 5..plt[0] + 9],
        &[0x00, 0x7f, 0x81, 0x00]
    );
    assert_eq!(&codestream[plt[1] + 5..plt[1] + 8], &[0x81, 0x80, 0x00]);
}

#[test]
fn marker_multitile_writer_accepts_exact_peak_and_rejects_cap_minus_one() {
    let first_data = [0x11, 0x12, 0x13];
    let second_data = [0x21, 0x22];
    let first_lengths = [0, 127, 128];
    let second_lengths = [16_384];
    let first_headers = vec![vec![0xaa, 0xbb]];
    let second_headers = vec![vec![0xcc]];
    let tiles = [
        TilePartData {
            tile_index: 0,
            tile_part_index: 0,
            num_tile_parts: 1,
            data: &first_data,
            packet_lengths: &first_lengths,
            packet_headers: &first_headers,
        },
        TilePartData {
            tile_index: 1,
            tile_part_index: 0,
            num_tile_parts: 1,
            data: &second_data,
            packet_lengths: &second_lengths,
            packet_headers: &second_headers,
        },
    ];
    let params = marker_params();
    let quantization = [(8, 0)];
    let discovery = write_codestream_tiles_accounted_with_peak_check(
        &params,
        &tiles,
        &quantization,
        |_| Ok(()),
    )
    .expect("discover marker writer peak");
    assert_eq!(discovery.writer_peak_bytes, discovery.codestream.capacity());
    assert_marker_payloads(&discovery.codestream);

    let exact_cap = discovery.writer_peak_bytes;
    let exact = write_codestream_tiles_accounted_with_peak_check(
        &params,
        &tiles,
        &quantization,
        |writer_peak| {
            if writer_peak > exact_cap {
                return Err(EncodeError::AllocationTooLarge {
                    what: "marker writer test peak",
                    requested: writer_peak,
                    cap: exact_cap,
                });
            }
            Ok(())
        },
    )
    .expect("marker writer at exact cap");
    assert_eq!(exact.codestream, discovery.codestream);

    let cap = exact_cap - 1;
    let error = write_codestream_tiles_accounted_with_peak_check(
        &params,
        &tiles,
        &quantization,
        |writer_peak| {
            if writer_peak > cap {
                return Err(EncodeError::AllocationTooLarge {
                    what: "marker writer test peak",
                    requested: writer_peak,
                    cap,
                });
            }
            Ok(())
        },
    )
    .expect_err("marker writer is one byte over cap");
    assert!(matches!(
        error,
        EncodeError::AllocationTooLarge {
            what: "marker writer test peak",
            requested,
            cap: rejected_cap,
        } if requested == exact_cap && rejected_cap == cap
    ));
}
