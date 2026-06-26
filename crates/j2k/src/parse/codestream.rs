// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{ParsedCod, ParsedComponentInfo, ParsedSiz};
use crate::J2kError;
use j2k_core::{InputError, TileLayout, Unsupported};
use j2k_native::{
    inspect_j2k_codestream_header, looks_like_j2k_codestream, J2kCodestreamHeaderError,
};

#[derive(Debug, Clone)]
pub(crate) struct CodestreamInfo {
    pub(crate) siz: ParsedSiz,
    pub(crate) cod: ParsedCod,
}

pub(crate) fn looks_like_codestream(input: &[u8]) -> bool {
    looks_like_j2k_codestream(input)
}

pub(crate) fn parse_codestream(input: &[u8]) -> Result<CodestreamInfo, J2kError> {
    let header = inspect_j2k_codestream_header(input).map_err(map_header_error)?;
    let components = u8::try_from(header.components).map_err(|_| {
        J2kError::Unsupported(Unsupported {
            what: "component count > 255",
        })
    })?;
    let component_info = header
        .component_info
        .into_iter()
        .map(|component| ParsedComponentInfo {
            bit_depth: component.bit_depth,
            signed: component.signed,
            x_rsiz: component.x_rsiz,
            y_rsiz: component.y_rsiz,
        })
        .collect();

    Ok(CodestreamInfo {
        siz: ParsedSiz {
            dimensions: header.dimensions,
            components,
            bit_depth: header.bit_depth,
            tile_layout: TileLayout {
                tile_width: header.tile_size.0,
                tile_height: header.tile_size.1,
                tiles_x: header.tile_count.0,
                tiles_y: header.tile_count.1,
            },
            component_info,
        },
        cod: ParsedCod {
            resolution_levels: header.resolution_levels,
            has_mct: header.has_mct,
            reversible: header.reversible,
            high_throughput: header.high_throughput,
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
        error => J2kError::Backend(error.to_string()),
    }
}
