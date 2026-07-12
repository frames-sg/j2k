// SPDX-License-Identifier: MIT OR Apache-2.0

//! Checked sizing for a scratch-free single-tile codestream writer.

use alloc::vec::Vec;

use super::packet_markers::{
    plm_marker_bytes, plt_marker_bytes, ppm_marker_bytes, ppt_marker_bytes,
};
use super::{EncodeParams, TilePartData};
use crate::j2c::encode::allocation::{checked_add_bytes, checked_element_bytes};
use crate::{EncodeError, EncodeResult};

mod header;
use header::main_header_prefix_len;

/// Completed codestream and the simultaneous writer-owned allocation peak.
#[derive(Debug)]
pub(crate) struct AccountedCodestream {
    pub(crate) codestream: Vec<u8>,
    pub(crate) writer_peak_bytes: usize,
}

pub(super) fn single_tile_output_len(
    params: &EncodeParams,
    tile_data_len: usize,
    quantization_step_sizes: &[(u16, u16)],
) -> EncodeResult<usize> {
    u32::try_from(tile_data_len)
        .ok()
        .and_then(|length| length.checked_add(14))
        .ok_or(EncodeError::InvalidInput {
            what: "tile-part length exceeds u32",
        })?;
    let mut bytes = main_header_prefix_len(params, quantization_step_sizes)?;
    if params.write_tlm {
        bytes = checked_add_bytes(bytes, 12, "codestream TLM marker bytes")?;
    }
    bytes = checked_add_bytes(bytes, 14, "codestream single tile-part bytes")?;
    bytes = checked_add_bytes(bytes, tile_data_len, "codestream tile payload bytes")?;
    checked_add_bytes(bytes, 2, "codestream EOC marker bytes")
}

pub(super) fn codestream_tiles_output_len(
    params: &EncodeParams,
    tiles: &[TilePartData<'_>],
    quantization_step_sizes: &[(u16, u16)],
) -> EncodeResult<usize> {
    if !(params.write_plt || params.write_plm || params.write_ppm || params.write_ppt) {
        if let [tile] = tiles {
            return single_tile_output_len(params, tile.data.len(), quantization_step_sizes);
        }
    }
    let mut bytes = main_header_prefix_len(params, quantization_step_sizes)?;
    if params.write_plm {
        bytes = checked_add_bytes(
            bytes,
            plm_marker_bytes(tiles)?,
            "codestream PLM marker bytes",
        )?;
    }
    if params.write_ppm {
        bytes = checked_add_bytes(
            bytes,
            ppm_marker_bytes(tiles)?,
            "codestream PPM marker bytes",
        )?;
    }
    if params.write_tlm {
        bytes = checked_add_bytes(
            bytes,
            checked_element_bytes::<[u8; 12]>(tiles.len(), "codestream TLM marker bytes")?,
            "codestream TLM marker bytes",
        )?;
    }
    for tile in tiles {
        bytes = checked_add_bytes(
            bytes,
            usize::try_from(tile_part_len(params, tile)?).map_err(|_| {
                EncodeError::ArithmeticOverflow {
                    what: "codestream tile-part bytes",
                }
            })?,
            "codestream tile-part bytes",
        )?;
    }
    checked_add_bytes(bytes, 2, "codestream EOC marker bytes")
}

pub(super) fn tile_part_len(params: &EncodeParams, tile: &TilePartData<'_>) -> EncodeResult<u32> {
    let mut bytes = 14usize;
    if params.write_plt {
        bytes = checked_add_bytes(
            bytes,
            plt_marker_bytes(tile.packet_lengths)?,
            "tile PLT bytes",
        )?;
    }
    if params.write_ppt {
        bytes = checked_add_bytes(
            bytes,
            ppt_marker_bytes(tile.packet_headers)?,
            "tile PPT bytes",
        )?;
    }
    bytes = checked_add_bytes(bytes, tile.data.len(), "tile-part payload bytes")?;
    u32::try_from(bytes).map_err(|_| EncodeError::InvalidInput {
        what: "tile-part length exceeds u32",
    })
}

#[cfg(test)]
#[path = "accounting/tests.rs"]
mod tests;
