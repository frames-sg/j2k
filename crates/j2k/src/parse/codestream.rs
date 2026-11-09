// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    allocation::{capacity_bytes, checked_add_bytes, ParseAllocationBudget},
    ParsedCod, ParsedSiz,
};
use crate::J2kComponentInfo;
use crate::J2kError;
use j2k_core::{InputError, TileLayout, Unsupported};
use j2k_native::{
    inspect_j2k_codestream_header, looks_like_j2k_codestream, J2kCodestreamComponentHeader,
    J2kCodestreamHeaderError, J2kCodestreamHeaderMetadata,
};

#[derive(Debug)]
pub(crate) struct CodestreamInfo {
    pub(crate) siz: ParsedSiz,
    pub(crate) cod: ParsedCod,
}

pub(crate) fn looks_like_codestream(input: &[u8]) -> bool {
    looks_like_j2k_codestream(input)
}

/// Where the image and tiles of raw codestream `input` lie on the reference
/// grid, from its main header alone.
pub(crate) fn parse_reference_grid(input: &[u8]) -> Result<crate::J2kReferenceGrid, J2kError> {
    let grid = j2k_native::inspect_j2k_reference_grid(input).map_err(map_header_error)?;
    Ok(crate::J2kReferenceGrid {
        image_origin: grid.image_origin,
        image_size: grid.image_size,
        tile_origin: grid.tile_origin,
        tile_size: grid.tile_size,
    })
}

/// The coding-style defaults of raw codestream `input`'s main-header COD
/// marker.
pub(crate) fn parse_coding_style(input: &[u8]) -> Result<crate::J2kCodingStyle, J2kError> {
    let style = j2k_native::inspect_j2k_coding_style(input).map_err(map_header_error)?;
    let progression_order = match style.progression_order {
        0 => crate::J2kProgressionOrder::Lrcp,
        1 => crate::J2kProgressionOrder::Rlcp,
        2 => crate::J2kProgressionOrder::Rpcl,
        3 => crate::J2kProgressionOrder::Pcrl,
        4 => crate::J2kProgressionOrder::Cprl,
        _ => {
            return Err(J2kError::InvalidCod {
                what: "progression order is outside Table A.16",
            })
        }
    };
    Ok(crate::J2kCodingStyle {
        progression_order,
        quality_layers: style.quality_layers,
        decomposition_levels: style.decomposition_levels,
        code_block_size: style.code_block_size,
    })
}

pub(crate) fn parse_codestream(
    input: &[u8],
    retained_bytes: usize,
) -> Result<CodestreamInfo, J2kError> {
    let header = inspect_j2k_codestream_header(input).map_err(map_header_error)?;
    let J2kCodestreamHeaderMetadata {
        dimensions,
        components,
        bit_depth,
        tile_size,
        tile_count,
        component_info,
        resolution_levels,
        has_mct,
        reversible,
        high_throughput,
    } = header;
    let source_capacity = component_info.capacity();
    let source_bytes = capacity_bytes::<J2kCodestreamComponentHeader>(
        source_capacity,
        "native codestream component metadata",
    )?;
    let mut live_bytes = retained_bytes;
    checked_add_bytes(
        &mut live_bytes,
        source_bytes,
        "JPEG 2000 inspection metadata",
    )?;
    let mut budget = ParseAllocationBudget::from_live_bytes(live_bytes)?;
    let mut facade_components =
        budget.try_vec(component_info.len(), "codestream component metadata")?;
    for component in component_info {
        facade_components.push(J2kComponentInfo {
            bit_depth: component.bit_depth,
            signed: component.signed,
            x_rsiz: component.x_rsiz,
            y_rsiz: component.y_rsiz,
        });
    }
    budget.release_capacity::<J2kCodestreamComponentHeader>(source_capacity)?;

    Ok(CodestreamInfo {
        siz: ParsedSiz {
            dimensions,
            components,
            bit_depth,
            tile_layout: TileLayout {
                tile_width: tile_size.0,
                tile_height: tile_size.1,
                tiles_x: tile_count.0,
                tiles_y: tile_count.1,
            },
            component_info: facade_components,
        },
        cod: ParsedCod {
            resolution_levels,
            has_mct,
            reversible,
            high_throughput,
        },
    })
}

fn map_header_error(error: J2kCodestreamHeaderError) -> J2kError {
    match error {
        J2kCodestreamHeaderError::TooShort { need, have } => {
            InputError::TooShort { need, have }.into()
        }
        J2kCodestreamHeaderError::TruncatedAt { offset, segment } => {
            InputError::TruncatedAt { offset, segment }.into()
        }
        J2kCodestreamHeaderError::InvalidMarker { offset, marker } => {
            J2kError::InvalidMarker { offset, marker }
        }
        J2kCodestreamHeaderError::MissingRequiredMarker { marker } => {
            J2kError::MissingRequiredMarker { marker }
        }
        J2kCodestreamHeaderError::InvalidSegment { offset, what } => {
            J2kError::InvalidBox { offset, what }
        }
        J2kCodestreamHeaderError::InvalidSiz { what } => J2kError::InvalidSiz { what },
        J2kCodestreamHeaderError::InvalidCod { what } => J2kError::InvalidCod { what },
        J2kCodestreamHeaderError::Unsupported { what } => {
            J2kError::Unsupported(Unsupported { what })
        }
        source => J2kError::CodestreamHeader {
            context: "JPEG 2000 codestream header inspection failed",
            source: crate::NativeBackendError::codestream_header(source),
        },
    }
}
